/*
 * chartnet.js -- full maimai chart generator, in the browser, no server.
 *
 * Mirrors the Python training code 1:1:
 *   src/features.py  (STFT -> 32-band log-mel -> 83 features)
 *   src/model.py     (stride-4 dilated conv trunk + 8 heads)
 *   src/chartgen.py  (thresholds, button quota, 2-hand scheduling, simai writer)
 *
 * Works in the browser and under Node (used for the parity test).
 */
(function (root) {
  'use strict';

  var SR = 22050, N_FFT = 1024, HOP = 256, N_MEL = 32;
  var SLOTS_PER_BEAT = 24, SLOTS_PER_MEASURE = 96, BEATS_PER_MEASURE = 4;

  // ---------------------------------------------------------------- FFT
  function makeFFT(n) {
    var rev = new Uint32Array(n), bits = Math.round(Math.log2(n));
    for (var i = 0; i < n; i++) {
      var r = 0, x = i;
      for (var b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
      rev[i] = r;
    }
    var cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (var k = 0; k < n / 2; k++) {
      cos[k] = Math.cos(-2 * Math.PI * k / n);
      sin[k] = Math.sin(-2 * Math.PI * k / n);
    }
    return function fft(re, im) {
      for (var i = 0; i < n; i++) {
        var j = rev[i];
        if (j > i) {
          var tr = re[i]; re[i] = re[j]; re[j] = tr;
          var ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
      }
      for (var size = 2; size <= n; size <<= 1) {
        var half = size >> 1, step = n / size;
        for (var s = 0; s < n; s += size) {
          for (var k2 = 0; k2 < half; k2++) {
            var c = cos[k2 * step], sn = sin[k2 * step];
            var a = s + k2, bb = a + half;
            var xr = re[bb] * c - im[bb] * sn;
            var xi = re[bb] * sn + im[bb] * c;
            re[bb] = re[a] - xr; im[bb] = im[a] - xi;
            re[a] += xr; im[a] += xi;
          }
        }
      }
    };
  }

  // ------------------------------------------------------------- helpers
  function hz2mel(f) { return 2595.0 * Math.log10(1.0 + f / 700.0); }
  function mel2hz(m) { return 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0); }

  function melFilterbank(nMels, nFft, sr, fmin, fmax) {
    fmin = fmin || 30.0;
    fmax = fmax || sr / 2;
    var m0 = hz2mel(fmin), m1 = hz2mel(fmax);
    var hz = new Float64Array(nMels + 2);
    for (var i = 0; i < nMels + 2; i++) hz[i] = mel2hz(m0 + (m1 - m0) * i / (nMels + 1));
    var nb = nFft / 2 + 1;
    var fb = new Float32Array(nMels * nb);
    for (var m = 0; m < nMels; m++) {
      var lo = hz[m], ce = hz[m + 1], hi = hz[m + 2], sum = 0;
      for (var b = 0; b < nb; b++) {
        var f = b * sr / nFft;
        var left = (f - lo) / Math.max(ce - lo, 1e-9);
        var right = (hi - f) / Math.max(hi - ce, 1e-9);
        var v = Math.max(0, Math.min(left, right));
        fb[m * nb + b] = v;
        sum += v;
      }
      sum = sum > 1e-9 ? sum : 1e-9;
      for (var b2 = 0; b2 < nb; b2++) fb[m * nb + b2] /= sum;
    }
    return fb;
  }

  var _fb = null, _win = null, _freqs = null, _chromaIdx = null;
  function initTables() {
    if (_fb) return;
    _fb = melFilterbank(N_MEL, N_FFT, SR);
    _win = new Float32Array(N_FFT);
    for (var i = 0; i < N_FFT; i++) _win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N_FFT - 1));
    _freqs = new Float32Array(N_FFT / 2 + 1);
    _chromaIdx = new Int32Array(N_FFT / 2 + 1);
    for (var b = 0; b < N_FFT / 2 + 1; b++) {
      var f = b * SR / N_FFT;
      _freqs[b] = f;
      var lf = Math.max(f, 1.0);
      var midi = Math.round(12 * Math.log2(lf / 440.0) + 69);
      var idx = ((midi % 12) + 12) % 12;
      _chromaIdx[b] = idx;
    }
  }

  // --------------------------------------------------- frame-level features
  /** pcm: Float32Array mono @ 22050. Returns {feat: Float32Array [T*83], T} */
  function frameFeatures(pcm) {
    initTables();
    var nFrames = Math.max(1, 1 + Math.floor((pcm.length - N_FFT) / HOP));
    if (pcm.length < N_FFT) nFrames = 1;
    var nb = N_FFT / 2 + 1;
    var feat = new Float32Array(nFrames * 83);
    var re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
    var fft = makeFFT(N_FFT);
    var logmel = new Float32Array(nFrames * N_MEL);
    var prev = new Float32Array(N_MEL);
    var melBuf = new Float32Array(nb);
    var fbuf = new Float32Array(nb);

    for (var t = 0; t < nFrames; t++) {
      var off = t * HOP;
      for (var i = 0; i < N_FFT; i++) {
        var idx = off + i;
        re[i] = (idx < pcm.length ? pcm[idx] : 0) * _win[i];
        im[i] = 0;
      }
      fft(re, im);
      for (var b = 0; b < nb; b++) fbuf[b] = Math.hypot(re[b], im[b]);
      // mel
      for (var m = 0; m < N_MEL; m++) {
        var s = 0, base = m * nb;
        for (var b3 = 0; b3 < nb; b3++) s += _fb[base + b3] * fbuf[b3];
        melBuf[m] = s;
        logmel[t * N_MEL + m] = Math.log1p(s * 10.0);
      }
    }

    // dmel (prepend first frame), flux, rms, centroid, rolloff, chroma
    var e = new Float64Array(pcm.length);
    var K = HOP, h = 1.0 / K;
    // np.convolve(x^2, ones(K)/K, 'same') -> out[k] = sum_j x2[j]*h[k-j+K-1]
    var x2 = new Float64Array(pcm.length);
    for (var i2 = 0; i2 < pcm.length; i2++) x2[i2] = pcm[i2] * pcm[i2];
    var half = K >> 1;
    var acc = 0;
    for (var k = 0; k < pcm.length; k++) {
      // sliding window centred so that out[k] sums x2[k-half .. k+half-1]
      if (k === 0) {
        acc = 0;
        for (var j = Math.max(0, -half); j <= Math.min(pcm.length - 1, K - 1 - half); j++) acc += x2[j];
      } else {
        var addI = k + K - 1 - half, remI = k - 1 - half;
        if (addI < pcm.length) acc += x2[addI];
        if (remI >= 0) acc -= x2[remI];
      }
      e[k] = acc * h;
    }

    var prevRms = 0;
    for (var t2 = 0; t2 < nFrames; t2++) {
      var base2 = t2 * 83;
      var lbase = t2 * N_MEL;
      // logmel
      for (var m2 = 0; m2 < N_MEL; m2++) feat[base2 + m2] = logmel[lbase + m2];
      // dmel + flux
      var flux = 0, lo = 0, hi = 0;
      for (var m3 = 0; m3 < N_MEL; m3++) {
        var cur = logmel[lbase + m3];
        var pv = (t2 === 0) ? logmel[m3] : logmel[(t2 - 1) * N_MEL + m3];
        var d = cur - pv;
        feat[base2 + N_MEL + m3] = d;
        var dp = d > 0 ? d : 0;
        flux += dp;
        if (m3 < 8) lo += dp;
        if (m3 >= 22) hi += dp;
      }
      feat[base2 + 64] = flux;
      feat[base2 + 65] = lo;
      feat[base2 + 66] = hi;
      // rms
      var rmsIdx = t2 * HOP;
      var rms = Math.sqrt(rmsIdx < e.length ? e[rmsIdx] : 0);
      feat[base2 + 67] = rms;
      feat[base2 + 68] = rms - prevRms;
      prevRms = rms;
      // centroid / rolloff / chroma from the same frame
      var off2 = t2 * HOP;
      for (var i3 = 0; i3 < N_FFT; i3++) {
        var id3 = off2 + i3;
        re[i3] = (id3 < pcm.length ? pcm[id3] : 0) * _win[i3];
        im[i3] = 0;
      }
      fft(re, im);
      var tot = 1e-8, cen = 0;
      for (var b4 = 0; b4 < nb; b4++) {
        var mg = Math.hypot(re[b4], im[b4]);
        fbuf[b4] = mg;
        tot += mg;
        cen += mg * _freqs[b4];
      }
      feat[base2 + 69] = cen / tot;
      // rolloff at 0.85 of cumulative energy
      var cum = 0, target = 0.85 * tot, roll = 0;
      for (var b5 = 0; b5 < nb; b5++) {
        cum += fbuf[b5];
        if (cum >= target) { roll = _freqs[b5]; break; }
      }
      feat[base2 + 70] = roll;
      // chroma
      var ch = new Float32Array(12), chs = 1e-8;
      for (var b6 = 0; b6 < nb; b6++) {
        ch[_chromaIdx[b6]] += fbuf[b6];
        chs += fbuf[b6];
      }
      for (var c = 0; c < 12; c++) feat[base2 + 71 + c] = ch[c] / chs;
    }
    return {feat: feat, T: nFrames};
  }

  /** linear interpolation of frame features onto arbitrary slot times */
  function resampleToSlots(feat, T, fps, times) {
    var n = times.length, out = new Float32Array(n * 83);
    for (var s = 0; s < n; s++) {
      var pos = times[s] * fps;
      var i0 = Math.floor(pos);
      var w = pos - i0;
      if (i0 < 0) { i0 = 0; w = 0; }
      if (i0 > T - 1) { i0 = T - 1; w = 0; }
      var i1 = Math.min(i0 + 1, T - 1);
      var a = i0 * 83, b = i1 * 83, o = s * 83;
      for (var c = 0; c < 83; c++) out[o + c] = feat[a + c] * (1 - w) + feat[b + c] * w;
    }
    return out;
  }

  function phaseFeatures(nSlots) {
    var out = new Float32Array(nSlots * 8);
    for (var s = 0; s < nSlots; s++) {
      var ph = 2 * Math.PI * s / SLOTS_PER_MEASURE;
      var k = 0;
      for (var h = 1; h <= 4; h++) {
        out[s * 8 + k++] = Math.sin(h * ph);
        out[s * 8 + k++] = Math.cos(h * ph);
      }
    }
    return out;
  }

  // ------------------------------------------------------------- weights
  function b64ToF32(b64) {
    var bin = (typeof atob === 'function') ? atob(b64)
      : Buffer.from(b64, 'base64').toString('binary');
    var n = bin.length, bytes = new Uint8Array(n);
    for (var i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer, bytes.byteOffset, n >> 2);
  }

  function loadWeights(W) {
    var P = {};
    for (var k in W.w) P[k] = b64ToF32(W.w[k]);
    return {
      cfg: W.cfg, P: P,
      mean: b64ToF32(W.mean), std: b64ToF32(W.std),
      stats: W.level_stats, buttonTarget: W.button_target,
    };
  }

  // ---------------------------------------------------------------- model
  var GELU_C = Math.sqrt(2.0 / Math.PI);
  function geluInPlace(a) {
    for (var i = 0; i < a.length; i++) {
      var x = a[i];
      a[i] = 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
    }
  }
  function layernorm(x, T, D, g, b) {
    var out = new Float32Array(T * D);
    for (var t = 0; t < T; t++) {
      var off = t * D, mu = 0, v = 0, i;
      for (i = 0; i < D; i++) mu += x[off + i];
      mu /= D;
      for (i = 0; i < D; i++) { var d = x[off + i] - mu; v += d * d; }
      v = v / D;
      var inv = 1 / Math.sqrt(v + 1e-5);
      for (i = 0; i < D; i++) out[off + i] = (x[off + i] - mu) * inv * g[i] + b[i];
    }
    return out;
  }
  /** im2col conv1d, weights laid out as [k*Cin, Cout] */
  function conv1d(x, T, Cin, w, b, Cout, k, dil, stride, pad) {
    var Tout = Math.floor((T + 2 * pad - dil * (k - 1) - 1) / stride) + 1;
    var out = new Float32Array(Tout * Cout);
    var col = new Float32Array(k * Cin);
    for (var t = 0; t < Tout; t++) {
      for (var j = 0; j < k; j++) {
        var src = t * stride + j * dil - pad;
        var cbase = j * Cin;
        if (src >= 0 && src < T) {
          var sbase = src * Cin;
          for (var c = 0; c < Cin; c++) col[cbase + c] = x[sbase + c];
        } else {
          for (var c2 = 0; c2 < Cin; c2++) col[cbase + c2] = 0;
        }
      }
      var obase = t * Cout;
      for (var o = 0; o < Cout; o++) {
        var s2 = b[o], wbase = o;
        for (var i = 0; i < k * Cin; i++) s2 += col[i] * w[i * Cout + o];
        out[obase + o] = s2;
      }
    }
    return {y: out, T: Tout};
  }

  var MODEL = null;
  function setWeights(W) {
    MODEL = loadWeights(W);
    // an export from an older architecture has no grid mask and no level
    // statistics; running it silently produces an empty chart
    MODEL.archOk = !!(MODEL.cfg && MODEL.cfg.allowed && MODEL.cfg.allowed.length === 96
                      && MODEL.stats && MODEL.stats["5"]);
    return MODEL;
  }

  /** x: Float32Array [T*96] -> heads */
  function forward(x, T) {
    var P = MODEL.P, cfg = MODEL.cfg;
    var d = cfg.d_model, S = cfg.stride, fine = cfg.fine_dim;

    // stem
    var padT = (S - (T % S)) % S;
    var Tin = T + padT;
    var xin = x;
    if (padT) {
      xin = new Float32Array(Tin * 96);
      xin.set(x.subarray(0, T * 96));
    }
    var stemK = S;
    var stemPad = Math.floor((stemK - 1) / 2);   // (dil*(k-1))//2 with dil=1
    var h = conv1d(xin, Tin, 96, P['stem.w'], P['stem.b'], d, stemK, 1, S, stemPad);
    geluInPlace(h.y);
    var hT = h.T, hy = h.y;

    // residual blocks
    for (var bi = 0; bi < cfg.dilations.length; bi++) {
      var dil = cfg.dilations[bi];
      var ln = layernorm(hy, hT, d, P['b' + bi + '.g'], P['b' + bi + '.be']);
      var c1 = conv1d(ln, hT, d, P['b' + bi + '.w1'], P['b' + bi + '.b1'], d, 3, dil, 1, dil);
      geluInPlace(c1.y);
      var c2 = conv1d(c1.y, hT, d, P['b' + bi + '.w2'], P['b' + bi + '.b2'], d, 3, dil, 1, dil);
      for (var i = 0; i < hy.length; i++) hy[i] += c2.y[i];
    }
    var hn = layernorm(hy, hT, d, P['g'], P['be']);

    // upsample back to T and concat the raw-feature projection
    var up = new Float32Array(T * d);
    for (var t = 0; t < T; t++) {
      var src = Math.min(hT - 1, Math.floor(t / S));
      up.set(hn.subarray(src * d, src * d + d), t * d);
    }
    var fp = new Float32Array(T * fine);
    for (var t2 = 0; t2 < T; t2++) {
      var ob = t2 * fine, ib = t2 * 96;
      for (var o2 = 0; o2 < fine; o2++) {
        var s = P['fb'][o2];
        for (var c = 0; c < 96; c++) s += x[ib + c] * P['fw'][c * fine + o2];
        fp[ob + o2] = s;
      }
    }
    geluInPlace(fp);

    var hd = d + fine;
    var z = new Float32Array(T * hd);
    for (var t3 = 0; t3 < T; t3++) {
      z.set(up.subarray(t3 * d, t3 * d + d), t3 * hd);
      z.set(fp.subarray(t3 * fine, t3 * fine + fine), t3 * hd + d);
    }

    var heads = {};
    for (var hi = 0; hi < cfg.heads.length; hi++) {
      var name = cfg.heads[hi][0], outn = cfg.heads[hi][1];
      var w = P['h_' + name + '.w'], bb = P['h_' + name + '.b'];
      var o = new Float32Array(T * outn);
      for (var t4 = 0; t4 < T; t4++) {
        var zb = t4 * hd, ob2 = t4 * outn;
        for (var j = 0; j < outn; j++) {
          var s2 = bb[j];
          for (var i2 = 0; i2 < hd; i2++) s2 += z[zb + i2] * w[i2 * outn + j];
          o[ob2 + j] = s2;
        }
      }
      heads[name] = o;
    }
    // hard grid constraint: forbid the 48th / 96th positions
    var allowed = cfg.allowed;
    if (allowed) {
      var actH = heads.act;
      for (var tt = 0; tt < T; tt++) {
        if (allowed[tt % 96] > 0.5) continue;
        for (var bb2 = 0; bb2 < 8; bb2++) actH[tt * 8 + bb2] = -1e4;
      }
    }
    return { heads: heads, z: z, T: T, hd: hd };
  }

  /** pi0.5-style continuous head: integrate the flow field -> duration slots */
  function flowDuration(z, hd, kindIdx, T, steps) {
    var P = MODEL.P, cfg = MODEL.cfg;
    steps = steps || (cfg.fm_steps || 8);
    if (!P['fm.w1']) {
      // weights predate the flow-matching head -> flat fallback length
      var flat = new Float32Array(T);
      for (var i0 = 0; i0 < T; i0++) flat[i0] = 12;
      return flat;
    }
    var LOGMAX = Math.log(cfg.fm_max_dur || 288);
    var fmIn = hd + 8 + 3;
    var fh = P['fm.w1'].length / fmIn;
    var x = new Float32Array(T);
    var inp = new Float32Array(fmIn);
    var hid = new Float32Array(fh);
    for (var s = 0; s < steps; s++) {
      var t = (s + 0.5) / steps;
      for (var i = 0; i < T; i++) {
        for (var c = 0; c < hd; c++) inp[c] = z[i * hd + c];
        for (var k = 0; k < 4; k++) {
          inp[hd + k] = Math.sin(2 * Math.PI * (k + 1) * t);
          inp[hd + 4 + k] = Math.cos(2 * Math.PI * (k + 1) * t);
        }
        var kd = kindIdx[i] | 0;
        inp[hd + 8] = kd === 0 ? 1 : 0;
        inp[hd + 9] = kd === 1 ? 1 : 0;
        inp[hd + 10] = kd === 2 ? 1 : 0;
        var acc;
        for (var j = 0; j < fh; j++) {
          acc = P['fm.b1'][j];
          for (var c2 = 0; c2 < fmIn; c2++) acc += inp[c2] * P['fm.w1'][c2 * fh + j];
          acc = 0.5 * acc * (1 + Math.tanh(GELU_C * (acc + 0.044715 * acc * acc * acc)));
          hid[j] = acc;
        }
        var v = P['fm.b2'][0];
        for (var j2 = 0; j2 < fh; j2++) v += hid[j2] * P['fm.w2'][j2];
        x[i] += v / steps;
      }
    }
    var out = new Float32Array(T);
    for (var i2 = 0; i2 < T; i2++) {
      out[i2] = Math.min(192, Math.max(2, Math.exp(x[i2] * LOGMAX)));
    }
    return out;
  }

  // --------------------------------------------------------------- decode
  function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x)))); }

  function softmaxRow(arr, off, n) {
    var mx = -1e9, i;
    for (i = 0; i < n; i++) mx = Math.max(mx, arr[off + i]);
    var s = 0, out = new Float32Array(n);
    for (i = 0; i < n; i++) { out[i] = Math.exp(arr[off + i] - mx); s += out[i]; }
    for (i = 0; i < n; i++) out[i] /= s;
    return out;
  }

  /**
   * heads -> events.  Mirrors chartgen.decode + make_chart.decode_with_stats.
   * Returns {events, thr, thr2, stats:{...}}
   */
  /** one subdivision (4th/8th/16th) per HALF measure; all three must appear */
  function measureStrides(act, T, targetNotes) {
    var nHalf = Math.ceil(T / 48);
    var A = new Float32Array(T);
    for (var t0 = 0; t0 < T; t0++) {
      var s0 = 0;
      for (var b0 = 0; b0 < 8; b0++) s0 += act[t0 * 8 + b0];
      A[t0] = s0;
    }
    var mass = new Float32Array(nHalf);
    for (var t1 = 0; t1 < T; t1++) mass[(t1 / 48) | 0] += A[t1];
    var strides = new Int8Array(nHalf);
    for (var h = 0; h < nHalf; h++) strides[h] = 2;
    if (nHalf < 3) return strides;

    var ratio = new Float32Array(nHalf);
    var onPos = [0, 12, 24, 36], offPos = [6, 18, 30, 42];
    for (var h2 = 0; h2 < nHalf; h2++) {
      var base = h2 * 48, son = 0, soff = 0, non = 0, noff = 0;
      for (var k = 0; k < 4; k++) {
        if (base + onPos[k] < T) { son += A[base + onPos[k]]; non++; }
        if (base + offPos[k] < T) { soff += A[base + offPos[k]]; noff++; }
      }
      if (non && noff) ratio[h2] = (soff / noff + 1e-8) / (son / non + 1e-8);
    }

    if (targetNotes === undefined || targetNotes === null) targetNotes = 8 * nHalf / 2;
    var k4 = Math.max(1, Math.min(Math.round(0.22 * nHalf), nHalf - 2));
    var k16 = Math.round(0.28 * nHalf);
    var cap = 8 * k16 + 4 * (nHalf - k4 - k16) + 2 * k4;
    if (targetNotes > cap) k16 += Math.ceil((targetNotes - cap) / 4);
    k16 = Math.max(1, Math.min(k16, nHalf - k4 - 1));

    var o16 = [], all = [];
    for (var i = 0; i < nHalf; i++) { o16.push(i); all.push(i); }
    o16.sort(function (a, b) { return ratio[b] - ratio[a]; });
    for (var i2 = 0; i2 < k16; i2++) strides[o16[i2]] = 1;
    all.sort(function (a, b) { return mass[a] - mass[b]; });
    var done = 0;
    for (var i3 = 0; i3 < all.length && done < k4; i3++) {
      if (strides[all[i3]] === 2) { strides[all[i3]] = 4; done++; }
    }
    return strides;
  }

  function strideMask(strides, T) {
    var m = new Float32Array(T);
    for (var s = 0; s < T; s++) {
      var h = (s / 48) | 0;
      if (h >= strides.length) continue;
      var off = s % 48;
      if (off % 6 !== 0) continue;
      if (((off / 6) | 0) % strides[h] === 0) m[s] = 1;
    }
    return m;
  }
  function decode(model, T, opts) {
    opts = opts || {};
    var heads = model.heads, z = model.z, hd = model.hd;
    var cfg = MODEL.cfg, stats = MODEL.stats[String(opts.level || 5)] || {};
    var BT = MODEL.buttonTarget;
    var allow96 = cfg.allowed || (new Float32Array(96).fill(1));
    var minGap = opts.minGap || 6, minGapBtn = opts.minGapBtn || 12;

    // act probabilities
    var act = new Float32Array(T * 8);
    var _amin = 1e9, _amax = -1e9, _anan = 0;
    for (var i = 0; i < T * 8; i++) {
      var raw = heads.act[i];
      if (raw !== raw) _anan++; else { if (raw < _amin) _amin = raw; if (raw > _amax) _amax = raw; }
      act[i] = sigmoid(raw);
    }

    // one subdivision per measure -> the chart is rhythmically uniform
    // the net localises a note to ~2 slots but not to the exact slot, so
    // dilate the activity before thresholding (lands the peak on the grid)
    for (var dk = 1; dk <= 2; dk++) {
      for (var dt = T - 1; dt >= dk; dt--) {
        for (var db = 0; db < 8; db++) {
          act[dt * 8 + db] = Math.max(act[dt * 8 + db], act[(dt - dk) * 8 + db]);
        }
      }
      for (var dt2 = 0; dt2 < T - dk; dt2++) {
        for (var db2 = 0; db2 < 8; db2++) {
          act[dt2 * 8 + db2] = Math.max(act[dt2 * 8 + db2], act[(dt2 + dk) * 8 + db2]);
        }
      }
    }
    var smask = null, allow = new Float32Array(T);

    // best + second per slot (forbidden slots have p ~ 0 already)
    var best = new Float32Array(T), second = new Float32Array(T);
    for (var t = 0; t < T; t++) {
      var b1 = -1, b2 = -1, o = t * 8;
      for (var b = 0; b < 8; b++) {
        var v = act[o + b];
        if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) { b2 = v; }
      }
      best[t] = b1; second[t] = b2;
    }

    var dens = stats.density_mean || 8.0;
    var targetNotes = dens * T / 96;
    var dbl = stats.double_rate || 0.2;
    var strides = measureStrides(act, T, targetNotes);
    // prefer the coarser grid: a note on the 4th or 8th is cleaner than a
    // stray one on a 16th position the model only half-detected
    for (var bi2 = 0; bi2 < T; bi2++) {
      var p96 = bi2 % 96;
      var bo = (p96 % 24 === 0) ? 1.30 : ((p96 % 12 === 0) ? 1.15 : 1.0);
      if (bo !== 1.0) {
        for (var bb3 = 0; bb3 < 8; bb3++) act[bi2 * 8 + bb3] *= bo;
      }
    }
    var smask = strideMask(strides, T);
    var allow = new Float32Array(T);
    for (var i2 = 0; i2 < T; i2++) allow[i2] = allow96[i2 % 96] * smask[i2];
    function okSlot(tt) { return allow[tt] > 0.5; }

    function countAt(thr, thr2) {
      var n = 0;
      for (var t = 0; t < T; t++) {
        if (!okSlot(t)) continue;
        if (best[t] >= thr) { n++; if (second[t] >= thr2) n++; }
      }
      return n;
    }
    var thr2 = 1.0, thr = 0.5;
    for (var it = 0; it < 6; it++) {
      var lo = 0, hi = 1;
      for (var k = 0; k < 26; k++) {
        var mid = 0.5 * (lo + hi);
        if (countAt(mid, thr2) > targetNotes) lo = mid; else hi = mid;
      }
      thr = 0.5 * (lo + hi);
      var sec = [];
      for (var t2 = 0; t2 < T; t2++) {
        if (!okSlot(t2)) continue;
        if (best[t2] >= thr) sec.push(second[t2]);
      }
      if (sec.length < 8) break;
      sec.sort(function (a, b) { return a - b; });
      var q = sec[Math.max(0, Math.min(sec.length - 1,
        Math.floor((1 - Math.min(Math.max(dbl, 0.01), 0.95)) * sec.length)))];
      if (Math.abs(q - thr2) < 1e-3) { thr2 = q; break; }
      thr2 = q;
    }
    var lo2 = 0, hi2 = 1;
    for (var k2 = 0; k2 < 26; k2++) {
      var mid2 = 0.5 * (lo2 + hi2);
      if (countAt(mid2, thr2) > targetNotes) lo2 = mid2; else hi2 = mid2;
    }
    thr = 0.5 * (lo2 + hi2);

    var nNotes = countAt(thr, thr2);
    var quota = [], used = [0, 0, 0, 0, 0, 0, 0, 0];
    for (var q2 = 0; q2 < 8; q2++) quota.push(Math.max(2, Math.round(nNotes * BT[q2])));

    // per-slot continuous duration from the flow head
    var kindIdx = new Int32Array(T);
    for (var kk = 0; kk < T; kk++) {
      var kp = 0, kv = -1e9;
      for (var kb = 0; kb < 8; kb++) {
        var vv = heads.kind[kk * 24 + kb * 3];
        var vv2 = heads.kind[kk * 24 + kb * 3 + 1];
        var vv3 = heads.kind[kk * 24 + kb * 3 + 2];
        var mx = Math.max(vv, vv2, vv3);
        var kx = (mx === vv) ? 0 : (mx === vv2 ? 1 : 2);
        if (mx > kv) { kv = mx; kp = kx; }
      }
      kindIdx[kk] = kp;
    }
    var durArr = flowDuration(z, hd, kindIdx, T);

    var cand = [];
    for (var t3 = 0; t3 < T; t3++) {
      if (!okSlot(t3)) continue;
      if (best[t3] < thr) continue;
      var order = [];
      for (var b3 = 0; b3 < 8; b3++) order.push([act[t3 * 8 + b3], b3]);
      order.sort(function (a, b) { return b[0] - a[0]; });
      var want = 1;
      if (second[t3] >= thr2) want = 2;
      var chosen = [];
      for (var w2 = 0; w2 < want; w2++) {
        var picked = -1;
        for (var oi = 0; oi < 8; oi++) {
          var bidx = order[oi][1];
          if (chosen.indexOf(bidx) >= 0) continue;
          if (used[bidx] < quota[bidx]) { picked = bidx; break; }
        }
        if (picked < 0) {
          var bestB = -1, bestR = 1e9;
          for (var oi2 = 0; oi2 < 8; oi2++) {
            var b2 = order[oi2][1];
            if (chosen.indexOf(b2) >= 0) continue;
            var r2 = used[b2] / Math.max(quota[b2], 1);
            if (r2 < bestR) { bestR = r2; bestB = b2; }
          }
          picked = bestB;
        }
        if (picked >= 0) { chosen.push(picked); used[picked]++; }
      }
      for (var ci = 0; ci < chosen.length; ci++) {
        var bb2 = chosen[ci];
        var kindP = softmaxRow(heads.kind, t3 * 24 + bb2 * 3, 3);
        var kind = kindP[0] >= kindP[1] && kindP[0] >= kindP[2] ? 1
                 : (kindP[1] >= kindP[2] ? 2 : 3);
        var brk = sigmoid(heads.brk[t3 * 8 + bb2]) > 0.5;
        var ex = sigmoid(heads.ex[t3 * 8 + bb2]) > 0.6;
        var e = { slot: t3, button: bb2 + 1, kind: kind === 1 ? 'tap' : (kind === 2 ? 'hold' : 'slide'),
                  dur: 0, op: 0, target: bb2 + 1, brk: brk, ex: ex, conf: act[t3 * 8 + bb2] };
        if (kind === 2) {
          e.dur = Math.max(2, Math.min(192, Math.round(durArr[t3])));
        } else if (kind === 3) {
          e.dur = Math.max(6, Math.min(36, Math.round(durArr[t3])));
          var opp = softmaxRow(heads.op, t3 * 2, 2);
          e.op = opp[1] > opp[0] ? 1 : 0;
          var tp = softmaxRow(heads.tgt, t3 * 8, 8), ti = 0;
          for (var h4 = 1; h4 < 8; h4++) if (tp[h4] > tp[ti]) ti = h4;
          if (ti === bb2) {
            var sortedT = [];
            for (var h5 = 0; h5 < 8; h5++) sortedT.push([tp[h5], h5]);
            sortedT.sort(function (a, b) { return b[0] - a[0]; });
            for (var h6 = 0; h6 < 8; h6++) if (sortedT[h6][1] !== bb2) { ti = sortedT[h6][1]; break; }
          }
          e.target = ti + 1;
        }
        cand.push(e);
      }
    }
    cand.sort(function (a, b) { return a.slot - b.slot || b.conf - a.conf; });
    // ---- final planning: class mix, buttons, playability -------------------
  function _h01(slot, b, salt) {
    var x = ((slot * 73856093) ^ (b * 19349663) ^ ((salt + 1) * 83492791)) | 0;
    x = (x ^ (x >> 13)) | 0;
    x = Math.imul(x, 1274126177) | 0;
    return ((x >>> 0) % 100003) / 100003;
  }
  var LEFTK = [5, 6, 7], RIGHTK = [1, 2, 3];

  function _handOk(list) {
    if (list.length < 2) return true;
    var a = list[0], b = list[1];
    var la = LEFTK.indexOf(a) >= 0, lb = LEFTK.indexOf(b) >= 0;
    var ra = RIGHTK.indexOf(a) >= 0, rb = RIGHTK.indexOf(b) >= 0;
    if (la && lb) return false;
    if (ra && rb) return false;
    var d = Math.abs(a - b);
    if (d === 1 || d === 7) return false;
    return true;
  }
  function _clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  var HOLDQ = [12, 18, 24, 36, 48], HOLDW = [0.22, 0.14, 0.38, 0.16, 0.10];
  var SLIDEQ = [6, 8, 9, 12, 16], SLIDEW = [0.30, 0.22, 0.18, 0.22, 0.08];
  function _pickDur(tab, wt, slot, b) {
    var tot = 0, i;
    for (i = 0; i < wt.length; i++) tot += wt[i];
    var r = _h01(slot, b, 11), acc = 0;
    for (i = 0; i < tab.length; i++) {
      acc += wt[i] / tot;
      if (r <= acc) return tab[i];
    }
    return tab[tab.length - 1];
  }

  /** force the tap / hold / slide mix, stratified over TIME so the stars and
      holds do not all pile up in the first half of the song */
  function assignMix(events, kindp, op, tgt, slideRatio, holdRatio) {
    var n = events.length;
    if (!n) return events;
    slideRatio = (slideRatio === undefined || slideRatio === null) ? 0.169 : slideRatio;
    holdRatio = (holdRatio === undefined || holdRatio === null) ? 0.086 : holdRatio;
    var nSlide = Math.round(n * slideRatio), nHold = Math.round(n * holdRatio);
    function sc(e, c) { return kindp[(e.slot * 8 + (e.button - 1)) * 3 + (c - 1)] || 0; }
    var s0 = events[0].slot, s1 = events[0].slot, i;
    for (i = 0; i < n; i++) { if (events[i].slot < s0) s0 = events[i].slot; if (events[i].slot > s1) s1 = events[i].slot; }
    var span = Math.max(1, s1 - s0), NB = 8;
    var buckets = [], b;
    for (b = 0; b < NB; b++) buckets.push([]);
    for (i = 0; i < n; i++) {
      b = Math.min(NB - 1, Math.floor((events[i].slot - s0) * NB / span));
      buckets[b].push(i);
    }
    var isSlide = new Uint8Array(n);
    for (b = 0; b < NB; b++) {
      var ids = buckets[b];
      var want = Math.round(nSlide * ids.length / n);
      ids.sort(function (x, y) { return (sc(events[y], 3) - sc(events[y], 1)) - (sc(events[x], 3) - sc(events[x], 1)); });
      for (var k = 0; k < want && k < ids.length; k++) isSlide[ids[k]] = 1;
    }
    var rest = [];
    for (i = 0; i < n; i++) if (!isSlide[i]) rest.push(i);
    var isHold = new Uint8Array(n);
    var b2 = [], j;
    for (j = 0; j < NB; j++) b2.push([]);
    for (i = 0; i < rest.length; i++) {
      var ii = rest[i];
      b = Math.min(NB - 1, Math.floor((events[ii].slot - s0) * NB / span));
      b2[b].push(ii);
    }
    for (j = 0; j < NB; j++) {
      var ids2 = b2[j];
      var want2 = Math.round(nHold * ids2.length / Math.max(rest.length, 1));
      ids2.sort(function (x, y) { return (sc(events[y], 2) - sc(events[y], 1)) - (sc(events[x], 2) - sc(events[x], 1)); });
      for (var k2 = 0; k2 < want2 && k2 < ids2.length; k2++) isHold[ids2[k2]] = 1;
    }
    var out = [];
    for (var q = 0; q < n; q++) {
      var e = events[q], o = Object.assign({}, e);
      if (isSlide[q]) {
        o.kind = 'slide';
        o.dur = _pickDur(SLIDEQ, SLIDEW, e.slot, e.button);
        o.op = (op[e.slot * 2 + 1] > op[e.slot * 2]) ? 1 : 0;
        var base = e.slot * 8, bi = 0;
        for (var m = 1; m < 8; m++) if (tgt[base + m] > tgt[base + bi]) bi = m;
        var nt = bi + 1;
        if (nt === o.button) nt = (nt % 8) + 1;
        o.target = nt;
      } else if (isHold[q]) {
        o.kind = 'hold';
        o.dur = _pickDur(HOLDQ, HOLDW, e.slot, e.button);
      } else {
        o.kind = 'tap'; o.dur = 0;
      }
      out.push(o);
    }
    return out;
  }

  /** plan the buttons so the chart is playable AND spread over 8 keys */
  function assignButtons(events, hist, minGapBtn, pref) {
    var n = events.length;
    if (!n) return events;
    hist = hist || [0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12, 0.12];
    minGapBtn = minGapBtn || 12;
    var quota = [], counts = [], last = {};
    for (var b = 0; b < 8; b++) { quota.push(Math.max(2, n * hist[b])); counts.push(0); }
    var out = [], i = 0;
    while (i < n) {
      var j = i;
      while (j < n && events[j].slot === events[i].slot) j++;
      var group = events.slice(i, j), slot = events[i].slot, chosen = [];
      for (var g = 0; g < group.length; g++) {
        var cands = [], hard = [];
        for (var c = 0; c < 8; c++) if (chosen.indexOf(c) < 0) cands.push(c);
        if (!cands.length) break;
        for (var ci = 0; ci < cands.length; ci++) {
          var cb = cands[ci];
          if (last[cb] !== undefined && slot - last[cb] < minGapBtn) continue;
          if (!_handOk(chosen.concat([cb]))) continue;
          hard.push(cb);
        }
        var pool = hard.length ? hard : cands;
        var best = pool[0], bs = -1e9;
        for (var pi = 0; pi < pool.length; pi++) {
          var pbb = pool[pi];
          var pv = pref ? (pref[slot * 8 + pbb] || 0) : 0;
          var bal = 1 - counts[pbb] / quota[pbb];
          var rec = (last[pbb] === undefined) ? 1
            : 1 - Math.min(1, (slot - last[pbb]) / minGapBtn);
          var scb = 2.2 * pv + 1.1 * bal + 1.4 * rec + 0.9 * _h01(slot, pbb, g);
          if (scb > bs) { bs = scb; best = pbb; }
        }
        chosen.push(best); counts[best]++; last[best] = slot;
      }
      for (var q = 0; q < group.length; q++) {
        var ev = group[q];
        if (q >= chosen.length) { out.push(ev); continue; }
        var nb = chosen[q] + 1;
        var copy = Object.assign({}, ev);
        if (nb !== ev.button) copy.button = nb;
        if (copy.kind === 'slide') {
          var usedNow = chosen.map(function (x) { return x + 1; }).concat([nb]);
          for (var r = 0; r < 8 && usedNow.indexOf(copy.target) >= 0; r++) {
            copy.target = (copy.target % 8) + 1;
          }
        }
        out.push(copy);
      }
      i = j;
    }
    return out;
  }

  /** drop or move every note that lands on a still-occupied button (撞尾) */
  function fixCollisions(events, pref) {
    var n = events.length;
    if (!n) return events;
    var order = events.slice().sort(function (a, b) {
      return (a.slot - b.slot) || ((a.kind === 'tap' ? 0 : 1) - (b.kind === 'tap' ? 0 : 1));
    });
    var busy = {}, out = [], i = 0;
    while (i < n) {
      var j = i;
      while (j < n && order[j].slot === order[i].slot) j++;
      var slot = order[i].slot, taken = {}, keep = [];
      for (var g = i; g < j; g++) {
        var e = order[g], conflict = false;
        var bk = busy[e.button];
        if (bk && (slot < bk.end || (slot === bk.end && bk.kind !== 'tap'))) conflict = true;
        if (!conflict && !taken[e.button]) {
          keep.push(e); taken[e.button] = 1; continue;
        }
        var best = -1, bs = -1e9;
        for (var b = 0; b < 8; b++) {
          var nb = b + 1;
          if (taken[nb]) continue;
          var bb = busy[nb];
          if (bb && slot <= bb.end) continue;
          var s = (pref ? pref[slot * 8 + b] : 0) + 0.7 * _h01(slot, b, 3);
          if (s > bs) { bs = s; best = b; }
        }
        if (best >= 0) {
          var c = Object.assign({}, e);
          c.button = best + 1;
          if (c.kind === 'slide' && c.target === c.button) c.target = (c.button % 8) + 1;
          keep.push(c); taken[c.button] = 1;
        }
      }
      for (var q = 0; q < keep.length; q++) {
        var ev = keep[q];
        busy[ev.button] = { end: ev.slot + (ev.kind === 'tap' ? 0 : ev.dur), kind: ev.kind };
        out.push(ev);
      }
      i = j;
    }
    return out;
  }

  /** give every star a direction, an arc length and a kind (straight / rim) */
  function planSlides(events, rimRatio) {
    rimRatio = (rimRatio === undefined || rimRatio === null) ? 0.05 : rimRatio;
    var nSlide = 0;
    for (var i = 0; i < events.length; i++) {
      var e = events[i];
      if (e.kind !== 'slide') continue;
      var cw = (nSlide % 2 === 0);
      var span = [2, 3, 4, 4, 5, 3][Math.floor(_h01(e.slot, e.button, 22) * 6) % 6];
      nSlide++;
      var step = cw ? 1 : -1;
      // ~22% become a polyline (折线): they turn at an intermediate key
      var poly = (_h01(e.slot, e.button, 31) < 0.22) && span >= 2;
      var tb = null, path = null;
      if (poly) {
        var mid = ((e.button - 1 + step * Math.max(1, span - 1)) % 8 + 8) % 8 + 1;
        var back = (_h01(e.slot, e.button, 32) > 0.5) ? 1 : -1;
        var end = ((mid - 1 + back * 2) % 8 + 8) % 8 + 1;
        if (mid !== e.button && end !== e.button && end !== mid) {
          path = [mid, end]; tb = end;
        }
      }
      if (tb === null) tb = ((e.button - 1 + step * span) % 8 + 8) % 8 + 1;
      var busy = {};
      for (var j = 0; j < events.length; j++) if (events[j].slot === e.slot) busy[events[j].button] = 1;
      for (var k = 0; k < 8; k++) {
        if (!busy[tb] && tb !== e.button) break;
        tb = (tb % 8) + 1;
      }
      e.target = tb;
      e.path = path;
      e.op = (_h01(e.slot, e.button, 23) < rimRatio) ? 1 : 0;
    }
    return events;
  }

  /** hand budget: hold / straight star = 1 hand, wifi star (w) = 2 hands */
  function scheduleHands(events, maxConcurrent) {
    maxConcurrent = maxConcurrent || 2;
    if (!events.length) return events;
    var order = events.slice().sort(function (a, b) {
      return (a.slot - b.slot) || ((a.kind === 'tap' ? 0 : 1) - (b.kind === 'tap' ? 0 : 1));
    });
    var out = [], active = [], i = 0, n = order.length;
    while (i < n) {
      var j = i;
      while (j < n && order[j].slot === order[i].slot) j++;
      var slot = order[i].slot;
      active = active.filter(function (a) { return a.end > slot; });
      var used = 0, k;
      for (k = 0; k < active.length; k++) used += active[k].hands;
      var slidesRunning = false;
      for (k = 0; k < active.length; k++) if (active[k].slide) slidesRunning = true;
      var group = order.slice(i, j).sort(function (a, b) {
        return ((a.kind === 'tap' ? 0 : 1) - (b.kind === 'tap' ? 0 : 1)) || (b.conf - a.conf);
      });
      var keep = [];
      for (var g = 0; g < group.length; g++) {
        var e = group[g];
        var isSlide = (e.kind === 'slide');
        if (isSlide) {
          if (slidesRunning) continue;
          var nsl = 0, bad = false;
          for (var q = 0; q < keep.length; q++) {
            if (keep[q].kind === 'slide') {
              nsl++;
              if (keep[q].op === 1 || keep[q].path) bad = true;
            }
          }
          if (nsl >= 2) continue;
          // a double star must be two plain straights
          if (nsl === 1 && (e.op === 1 || e.path || bad)) continue;
        }
        var h = (isSlide && e.op === 1) ? 2 : 1;
        if (used + h > maxConcurrent) continue;
        keep.push(e);
        used += h;
      }
      for (var m = 0; m < keep.length; m++) {
        var ev = keep[m];
        active.push({ end: ev.slot + Math.max((ev.kind === 'tap' ? 0 : ev.dur), 1),
                      slide: ev.kind === 'slide',
                      hands: (ev.kind === 'slide' && ev.op === 1) ? 2 : 1 });
        out.push(ev);
      }
      i = j;
    }
    return out;
  }

  /** insert 扫键: 3-4 taps marching across neighbouring keys in an empty gap */
  function addSweeps(events, act, minGap, maxRuns, runLen) {
    minGap = minGap || 6; maxRuns = maxRuns || 8; runLen = runLen || 4;
    if (!events.length || !act) return events;
    var slots = [], seen = {};
    for (var i = 0; i < events.length; i++) if (!seen[events[i].slot]) { seen[events[i].slot] = 1; slots.push(events[i].slot); }
    slots.sort(function (a, b) { return a - b; });
    if (slots.length < 4) return events;
    var out = events.slice(), used = {}, added = 0;
    for (var u = 0; u < slots.length; u++) used[slots[u]] = 1;
    for (var k = 0; k + 1 < slots.length && added < maxRuns; k++) {
      var a = slots[k], b = slots[k + 1];
      if (b - a < runLen * minGap) continue;
      var start = a + minGap, ok = true, q;
      for (q = 0; q < runLen; q++) if (start + q * minGap >= b || used[start + q * minGap]) { ok = false; break; }
      if (!ok) continue;
      var bb = 0, bv = -1;
      for (var c = 1; c <= 8; c++) { var v = act[start * 8 + (c - 1)]; if (v > bv) { bv = v; bb = c; } }
      if (!bb) continue;
      var st = (_h01(start, bb, 41) > 0.5) ? 1 : -1;
      for (q = 0; q < runLen; q++) {
        var nb = ((bb - 1 + st * q) % 8 + 8) % 8 + 1;
        var s2 = start + q * minGap;
        out.push({ slot: s2, button: nb, kind: 'tap', dur: 0, op: 0, target: nb,
                   brk: false, ex: false, conf: act[s2 * 8 + (nb - 1)] || 0, path: null });
        used[s2] = 1;
      }
      added++;
    }
    return out;
  }

  /** the same key must not be hit twice within minGapBtn */
  function dedupeButtons(events, minGapBtn) {
    minGapBtn = minGapBtn || 12;
    var arr = events.slice().sort(function (a, b) { return (a.slot - b.slot) || (b.conf - a.conf); });
    var last = {}, out = [];
    for (var i = 0; i < arr.length; i++) {
      var e = arr[i], t0 = last[e.button];
      if (t0 !== undefined && e.slot - t0 < minGapBtn) continue;
      last[e.button] = e.slot;
      out.push(e);
    }
    return out;
  }

  /** restore the slide ratio after the gate dropped many stars */
  function boostMix(events, kindp, slideRatio) {
    var n = events.length;
    if (n < 8) return events;
    slideRatio = (slideRatio === undefined || slideRatio === null) ? 0.169 : slideRatio;
    var counts = {};
    for (var i = 0; i < n; i++) counts[events[i].slot] = (counts[events[i].slot] || 0) + 1;
    var cur = 0;
    for (var j = 0; j < n; j++) if (events[j].kind === 'slide') cur++;
    var need = Math.round(n * slideRatio) - cur;
    if (need <= 0) return events;
    var cand = [];
    for (var k = 0; k < n; k++) {
      var e = events[k];
      if (e.kind !== 'tap' || counts[e.slot] !== 1) continue;
      var idx = (e.slot * 8 + (e.button - 1)) * 3;
      cand.push([(kindp[idx + 2] || 0) - (kindp[idx] || 0), k]);
    }
    cand.sort(function (a, b) { return b[0] - a[0]; });
    for (var m = 0; m < need && m < cand.length; m++) {
      var ev = events[cand[m][1]];
      ev.kind = 'slide';
      ev.dur = _pickDur(SLIDEQ, SLIDEW, ev.slot, ev.button);
      ev.path = null; ev.target = (ev.button % 8) + 1; ev.op = 0;
    }
    return events;
  }

  /** force the 绝赞 (break) notes to the official ~5% instead of 0% or 28% */
  function assignBreaks(events, ratio, pref) {
    var n = events.length;
    if (!n) return events;
    ratio = (ratio === undefined || ratio === null) ? 0.05 : ratio;
    var nb = Math.max(1, Math.round(n * ratio));
    var sc = [];
    for (var i = 0; i < n; i++) {
      var e = events[i];
      if (e.kind === 'slide') continue;      // 绝赞星星 is never generated
      var p = pref ? (pref[e.slot * 8 + (e.button - 1)] || 0) : 0;
      var down = (e.slot % 96 === 0) ? 1 : ((e.slot % 24 === 0) ? 0.5 : 0);
      sc.push([p + 0.8 * down, i]);
    }
    sc.sort(function (a, b) { return b[0] - a[0]; });
    var pick = {};
    for (var k = 0; k < nb && k < sc.length; k++) pick[sc[k][1]] = 1;
    for (var j = 0; j < n; j++) events[j].brk = !!pick[j];
    return events;
  }

  var events = enforcePlayable(cand, minGap, minGapBtn);
    events = assignMix(events, heads.kind, heads.op, heads.tgt,
                       stats.slide_ratio, stats.hold_ratio);
    events = assignButtons(events, BT || stats.button_target, minGapBtn, act);
    events = planSlides(events, stats.rim_ratio);
    events = fixCollisions(events, act);
    events = scheduleHands(events, 2);
    events = enforcePlayable(events, minGap, minGapBtn);
    events = boostMix(events, heads.kind, stats.slide_ratio);
    events = planSlides(events, stats.rim_ratio);
    events = fixCollisions(events, act);
    events = addSweeps(events, act, minGap, 3, 4);
    events = scheduleHands(events, 2);
    events = fixCollisions(events, act);
    events = dedupeButtons(events, minGapBtn);
    events = sanitize(events, minGapBtn, minGap, 2);
    events = assignBreaks(events, stats.break_ratio || 0.05, heads.brk);
    events = limitSlowStars(events);
    var _max = 0, _nan = 0;
    for (var _i = 0; _i < T; _i++) { if (!(best[_i] === best[_i])) _nan++; else if (best[_i] > _max) _max = best[_i]; }
    return { events: events, thr: thr, thr2: thr2, targetNotes: targetNotes,
             maxP: _max, nanSlots: _nan,
             logitMin: _amin, logitMax: _amax, actNaN: _anan };
  }

  function limitSlowStars(events) {
    var SLOW = 18, MAXSLOW = 2;
    var idx = [];
    for (var i = 0; i < events.length; i++) {
      if (events[i].kind === 'slide' && events[i].dur > SLOW) idx.push(i);
    }
    if (idx.length <= MAXSLOW) return events;
    idx.sort(function (a, b) { return events[b].conf - events[a].conf; });
    for (var j = MAXSLOW; j < idx.length; j++) events[idx[j]].dur = SLOW;
    return events;
  }

  function endOf(e) { return e.slot + (e.kind === 'tap' ? 1 : e.dur); }

  function chordOk(a, b) {
    if (a === b) return false;
    var d = Math.abs(a - b);
    if (d === 1 || d === 7) return false;
    var LK = [5, 6, 7], RK = [1, 2, 3];
    var la = LK.indexOf(a - 1) >= 0, lb = LK.indexOf(b - 1) >= 0;
    var ra = RK.indexOf(a - 1) >= 0, rb = RK.indexOf(b - 1) >= 0;
    if (la && lb) return false;
    if (ra && rb) return false;
    return true;
  }

  function enforcePlayable(cand, minGap, minGapBtn) {
    minGap = minGap || 6;
    minGapBtn = minGapBtn || 12;
    var free = [0, 0], buttonFree = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    var buttonLast = [-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9];
    var out = [], lastSlot = -1e9;
    for (var i = 0; i < cand.length; i++) {
      var e = cand[i], br = e.button;
      if (e.slot !== lastSlot && e.slot - lastSlot < minGap) continue;
      if (e.slot < buttonFree[br]) continue;
      if (e.slot - buttonLast[br] < minGapBtn) continue;
      var h = free[0] <= free[1] ? 0 : 1;
      if (free[h] > e.slot) continue;
      var same = [];
      for (var s2 = 0; s2 < out.length; s2++) if (out[s2].slot === e.slot) same.push(out[s2]);
      if (same.length) {
        if (same.length >= 2) continue;
        if (!chordOk(same[0].button, e.button)) {
          var SYM = { 1: 5, 5: 1, 2: 8, 8: 2, 3: 7, 7: 3, 4: 6, 6: 4 };
          var mate = SYM[same[0].button] || 0;
          var freeMate = mate && buttonFree[mate] <= e.slot &&
                         (e.slot - buttonLast[mate] >= minGapBtn) &&
                         !same.some(function (o) { return o.button === mate; });
          if (!freeMate) continue;
          var rep = Object.assign({}, e);
          rep.button = mate; br = mate;
          e = rep;
        }
        // a star NEVER shares its chord with another note
        if ((same[0].kind === 'slide') !== (e.kind === 'slide')) continue;
      }
      var occ = e.kind === 'tap' ? 1 : e.dur;
      free[h] = e.slot + occ;
      buttonFree[br] = e.slot + occ;
      buttonLast[br] = e.slot;
      lastSlot = e.slot;
      out.push(e);
    }
    var tails = {}, final = [];
    for (var j = 0; j < out.length; j++) {
      var e2 = out[j];
      if (e2.kind === 'slide') {
        var key = e2.target + ':' + endOf(e2);
        if (tails[key]) continue;
        tails[key] = 1;
      }
      final.push(e2);
    }
    return final;
  }

  // ------------------------------------------------------------- simai out
  function evText(e) {
    var b = String(e.button);
    var x = e.ex ? 'x' : '', br = e.brk ? 'b' : '';
    if (e.kind === 'tap') return b + x + br;
    if (e.kind === 'hold') return b + 'h' + x + br + '[96:' + e.dur + ']';
    if (e.path && e.path.length) {
      var op = (e.op === 0 ? '-' : 'w');
      return b + x + op + e.path.join(op) + '[96:' + e.dur + ']' + br;
    }
    return b + x + (e.op === 0 ? '-' : 'w') + e.target + '[96:' + e.dur + ']' + br;
  }

  var LV_DEFAULT = { 2: 5.0, 3: 8.5, 4: 11.5, 5: 13.5, 6: 14.5 };

  /**
   * Write a complete maidata.txt.  Layout mirrors the official files exactly:
   *   &title / &artist / &first / &wholebpm / &shortid / &genreid / &genre /
   *   &versionid / &version / &demo_seek / &demo_len / &clock_count
   *   (blank) (blank)
   *   &lv_N= / &des_N= / &inote_N=(bpm){384},
   *   one line per measure (comma terminated)
   *   E
   *   (blank)
   * The "{384}," terminator costs 1/384 of a measure (~3ms) instead of the
   * conventional "{1}," which would push the whole chart one measure late.
   */
  function toSimai(events, bpm, level, meta) {
    level = level || 5;
    meta = meta || {};
    var nl = '\r\n';
    var lv = (meta.lv !== undefined) ? meta.lv : (LV_DEFAULT[level] || 0);
    var head = [
      '&title=' + (meta.title || ''),
      '&artist=' + (meta.artist || ''),
      '&first=' + (function (x) { var s = Number(x || 0).toFixed(3); if (s.indexOf('.') >= 0) s = s.replace(/0+$/, '').replace(/\\.$/, ''); return (s === '' || s === '-0') ? '0' : s; })(meta.first),
      '&wholebpm=' + (Math.round(bpm * 1000) / 1000),
      '&shortid=0',
      '&genreid=105',
      '&genre=' + (meta.genre || 'maimai'),
      '&versionid=23',
      '&version=' + (meta.version || 'PRiSM'),
      '&demo_seek=0',
      '&demo_len=0',
      '&clock_count=4'
    ];
    var measures = [];
    if (!events.length) {
      measures.push('{96}' + new Array(97).join(',') + ',');
    } else {
      var maxSlot = 0;
      for (var i = 0; i < events.length; i++) maxSlot = Math.max(maxSlot, events[i].slot);
      var nMeas = Math.floor(maxSlot / 96) + 1;
      var bySlot = {};
      for (var j = 0; j < events.length; j++) {
        var e = events[j];
        (bySlot[e.slot] = bySlot[e.slot] || []).push(e);
      }
      var curGrid = 384;
      for (var m = 0; m < nMeas; m++) {
        var base = m * 96, used = [];
        for (var s = base; s < base + 96; s++) if (bySlot[s]) used.push(s - base);
        if (!used.length) {
          // empty measure == one slot at grid {1}
          var pfx = (curGrid === 1) ? '' : '{1}';
          curGrid = 1;
          measures.push(pfx + ',');
          continue;
        }
        var g = 96;
        if (used.length) {
          var cands = [4, 8, 12, 16, 24, 32, 48, 96];
          for (var ci = 0; ci < cands.length; ci++) {
            var step = 96 / cands[ci], ok = true;
            for (var ui = 0; ui < used.length; ui++) if (used[ui] % step !== 0) { ok = false; break; }
            if (ok) { g = cands[ci]; break; }
          }
        }
        var step2 = 96 / g, row = [];
        for (var k = 0; k < g; k++) {
          var sl = base + k * step2, evs = bySlot[sl];
          if (!evs) { row.push(''); continue; }
          var parts = [];
          for (var q = 0; q < evs.length; q++) parts.push(evText(evs[q]));
          row.push(parts.join('/'));
        }
        var prefix = (g === curGrid) ? '' : '{' + g + '}';
        curGrid = g;
        measures.push(prefix + row.join(',') + ',');
      }
    }
    var out = head.concat(['', '']).concat([
      '&lv_' + level + '=' + lv,
      '&des_' + level + '=' + (meta.des || 'AI'),
      '&inote_' + level + '=(' + String(Math.round(bpm * 10000) / 10000) + '){384},'
    ]).concat(measures).concat(['E', '']);
    return out.join(nl) + nl;
  }

  // ---- retrieval helpers (module scope) -----------------------------------
  function _h01m(a, b, salt) {
    var x = ((a * 73856093) ^ (b * 19349663) ^ (salt * 83492791)) | 0;
    x = (x ^ (x >> 13)) | 0;
    x = Math.imul(x, 1274126177) | 0;
    return ((x >>> 0) % 100003) / 100003;
  }
  function _symKey(b, g) {
    if (g <= 0) return b;
    if (g < 4) return (b + g) % 8;
    return (-(b + (g - 4)) % 8 + 8) % 8;
  }
  // ---- retrieval generator ------------------------------------------------
  var TPL = null;
  function _b64ToU8(s) {
    var bin = (typeof atob === "function") ? atob(s) : Buffer.from(s, "base64").toString("binary");
    var n = bin.length, a = new Uint8Array(n);
    for (var i = 0; i < n; i++) a[i] = bin.charCodeAt(i);
    return a;
  }
  function _b64ToI16(s) {
    var u = _b64ToU8(s), n = u.length >> 1, a = new Int16Array(n);
    for (var i = 0; i < n; i++) a[i] = (u[2 * i] | (u[2 * i + 1] << 8)) << 16 >> 16;
    return a;
  }
  function _b64ToI32(s) {
    var u = _b64ToU8(s), n = u.length >> 2, a = new Int32Array(n);
    for (var i = 0; i < n; i++) a[i] = u[4 * i] | (u[4 * i + 1] << 8) | (u[4 * i + 2] << 16) | (u[4 * i + 3] << 24);
    return a;
  }

  function setTemplates(W) {
    if (!W || !W.feat) return null;
    var feat = _b64ToU8(W.feat), notes = _b64ToI16(W.notes), off = _b64ToI32(W.off);
    var D = W.dim || 48, n = W.n, unit = new Float32Array(n * D);
    for (var i = 0; i < n; i++) {
      var s = 0, k;
      for (k = 0; k < D; k++) s += feat[i * D + k] * feat[i * D + k];
      s = Math.sqrt(s) + 1e-6;
      for (k = 0; k < D; k++) unit[i * D + k] = feat[i * D + k] / s;
    }
    var ver = W.ver ? _b64ToI16(W.ver) : null;
    var bpmArr = W.bpm ? _b64ToI16(W.bpm) : null;
    var cnt = new Float32Array(n);
    for (var c2 = 0; c2 < n; c2++) cnt[c2] = Math.min(1, (off[c2 + 1] - off[c2]) / 8);
    TPL = { n: n, dim: D, unit: unit, notes: notes, off: off, scale: W.scale || 1,
            ver: ver, dxv: W.dxv || 13, cnt: cnt, bpm: bpmArr };
    return TPL;
  }

  /** 16-dim onset strength per measure, identical to the Python builder */
  var _BANDS = [[0, 10], [11, 21], [22, 32]];
  function _bandFlux(base, T) {
    var nb = _BANDS.length, e = new Float32Array(T * nb);
    for (var s = 0; s < T; s++) {
      for (var bi = 0; bi < nb; bi++) {
        var a = _BANDS[bi][0], b = _BANDS[bi][1], acc = 0;
        for (var c = a; c < b; c++) acc += base[s * 83 + c];
        e[s * nb + bi] = acc / (b - a);
      }
    }
    var f = new Float32Array(T * nb);
    for (var s2 = 1; s2 < T; s2++) {
      for (var bi2 = 0; bi2 < nb; bi2++) {
        var d = e[s2 * nb + bi2] - e[(s2 - 1) * nb + bi2];
        f[s2 * nb + bi2] = d > 0 ? d : 0;
      }
    }
    return f;
  }

  function _measureFeature(base, T) {
    var nb = _BANDS.length, D = 16 * nb;
    var nMeas = Math.floor(T / 96), out = new Float32Array(nMeas * D);
    var flux = _bandFlux(base, T);
    for (var m = 0; m < nMeas; m++) {
      for (var p = 0; p < 16; p++) {
        var a = m * 96 + p * 6, b = Math.min(a + 6, T);
        for (var bi = 0; bi < nb; bi++) {
          var mx = 0;
          for (var s = a; s < b; s++) { var v = flux[s * nb + bi]; if (v > mx) mx = v; }
          out[m * D + p * nb + bi] = mx;
        }
      }
    }
    return out;
  }

  /** per-measure loudness (mean log-mel) and "this bar is basically silent" */
  function _measureEnergy(base, T) {
    var nMeas = Math.floor(T / 96), e = new Float32Array(nMeas);
    for (var m = 0; m < nMeas; m++) {
      var acc = 0, n = 0;
      for (var s = m * 96; s < Math.min((m + 1) * 96, T); s++) {
        for (var c = 0; c < 32; c++) acc += base[s * 83 + c];
        n += 32;
      }
      e[m] = n ? acc / n : 0;
    }
    return e;
  }
  function _silentMask(energy, ratio) {
    ratio = ratio || 0.35;
    var mx = Array.prototype.slice.call(energy).sort(function (a, b) { return a - b; });
    var med = (mx.length ? mx[mx.length >> 1] : 0) + 1e-6;
    var out = [];
    for (var i = 0; i < energy.length; i++) out.push(energy[i] < ratio * med);
    return out;
  }
  /** no key may be hit more than `cap` times inside a two-beat window */
  function limitRepeats(events, window, cap) {
    window = window || 48; cap = cap || 2;
    var arr = events.slice().sort(function (a, b) {
      return (a.slot - b.slot) || (b.conf - a.conf);
    });
    var recent = {}, out = [];
    for (var i = 0; i < arr.length; i++) {
      var e = arr[i];
      if (e.kind === 'slide') { out.push(e); continue; }
      var lst = (recent[e.button] || []).filter(function (s) { return e.slot - s < window; });
      if (lst.length >= cap) continue;
      lst.push(e.slot);
      recent[e.button] = lst;
      out.push(e);
    }
    return out;
  }

  /** non-silent bars must not be left empty: build from the bar's own peaks */
  function fillSparse(events, qf, silent, nb, target, minNotes) {
    minNotes = minNotes || 2;
    if (!events.length || !qf) return events;
    var nMeas = Math.floor(qf.length / nb), by = {}, used = {};
    for (var i = 0; i < events.length; i++) {
      var e = events[i], mm = Math.floor(e.slot / 96);
      (by[mm] = by[mm] || []).push(e);
      if (e.kind !== 'slide') (used[e.button] = used[e.button] || []).push(e.slot);
    }
    var out = events.slice();
    for (var m = 0; m < nMeas; m++) {
      if (m < silent.length && silent[m]) continue;
      var grp = by[m] || [];
      var pos = new Float32Array(16), mx = 0;
      for (var p = 0; p < 16; p++) {
        var v = 0;
        for (var b = 0; b < nb; b++) { var x = qf[m * 16 * nb + p * nb + b]; if (x > v) v = x; }
        pos[p] = v; if (v > mx) mx = v;
      }
      if (mx < 0.02) continue;
      var need = minNotes - grp.length;
      if (need <= 0) continue;
      var have = {};
      for (var g = 0; g < grp.length; g++) if (grp[g].kind !== 'slide') have[grp[g].slot % 96] = 1;
      var order = [];
      for (var q = 0; q < 16; q++) {
        if (have[q * 6] || pos[q] <= 0.25 * mx) continue;
        order.push(q);
      }
      order.sort(function (a, b2) {
        var oa = (a % 2 === 0 ? 0 : 1), ob = (b2 % 2 === 0 ? 0 : 1);
        return (oa - ob) || (pos[b2] - pos[a]);
      });
      for (var k = 0; k < order.length && k < need; k++) {
        var slot = m * 96 + order[k] * 6, btn = 0;
        var gaps = [24, 12, 6, 0];
        for (var gi = 0; gi < gaps.length && !btn; gi++) {
          for (var cand = 1; cand <= 8; cand++) {
            var lst = used[cand] || [], bad = false;
            if (gaps[gi]) {
              for (var li = 0; li < lst.length; li++) if (Math.abs(slot - lst[li]) < gaps[gi]) { bad = true; break; }
            }
            if (!bad) { btn = cand; break; }
          }
        }
        if (!btn) continue;
        out.push({ slot: slot, button: btn, kind: 'tap', dur: 0, op: 0, target: btn,
                   brk: false, ex: false, conf: 0.9, path: null });
        (used[btn] = used[btn] || []).push(slot);
      }
    }
    return out;
  }

  function generateRetrieval(pcm, bpm, offset, level, onProgress, meta, k) {
    if (!TPL) throw new Error("模板库未加载");
    k = k || 12;
    var ff = frameFeatures(pcm);
    var fps = SR / HOP;
    var dur = pcm.length / SR;
    var nSlots = Math.ceil(dur * SLOTS_PER_BEAT * bpm / 60) + SLOTS_PER_MEASURE;
    var times = new Float32Array(nSlots);
    for (var t = 0; t < nSlots; t++) times[t] = offset + t * 60.0 / bpm / SLOTS_PER_BEAT;
    var base = resampleToSlots(ff.feat, ff.T, fps, times);
    var qf = _measureFeature(base, nSlots);
    var nMeas = Math.floor(qf.length / TPL.dim);
    var energy = _measureEnergy(base, nSlots);
    var silent = _silentMask(energy, 0.35);
    if (onProgress) onProgress(0.4, "retrieve");

    var events = [], recent = [], usedTpl = {};
    for (var m = 0; m < nMeas; m++) {
      if (silent[m]) continue;              // very quiet intro / outro
      var DQ = TPL.dim, q = new Float32Array(DQ), qs = 0, i;
      for (i = 0; i < DQ; i++) { q[i] = qf[m * DQ + i]; qs += q[i] * q[i]; }
      qs = Math.sqrt(qs) + 1e-6;
      for (i = 0; i < DQ; i++) q[i] /= qs;
      // top-k by cosine
      var best = [], bestIdx = [];
      for (var ti = 0; ti < TPL.n; ti++) {
        var dot = 0;
        for (i = 0; i < DQ; i++) dot += q[i] * TPL.unit[ti * DQ + i];
        if (TPL.ver && TPL.ver[ti] >= TPL.dxv) dot += 0.12;
        dot += 0.12 * TPL.cnt[ti];                 // busy bar -> fuller template
        if (TPL.bpm) {
          var db = Math.abs(TPL.bpm[ti] - bpm);
          dot += 0.12 * Math.max(0, 1 - db / 60);  // same tempo family
        }
        if (best.length < k || dot > best[best.length - 1]) {
          var pos = best.length;
          while (pos > 0 && best[pos - 1] < dot) pos--;
          best.splice(pos, 0, dot); bestIdx.splice(pos, 0, ti);
          if (best.length > k) { best.pop(); bestIdx.pop(); }
        }
      }
      // a configuration may be used at most twice in one chart
      var pick = -1, ps = -1e9;
      for (var j = 0; j < bestIdx.length; j++) {
        var id = bestIdx[j], sc = best[j];
        if ((usedTpl[id] || 0) >= 2) continue;
        if (recent.indexOf(id) >= 0) sc -= 0.6;
        if (sc > ps) { ps = sc; pick = id; }
      }
      if (pick < 0) {                       // everything capped -> least used
        for (var j2 = 0; j2 < bestIdx.length; j2++) {
          var id2 = bestIdx[j2];
          if (pick < 0 || (usedTpl[id2] || 0) < (usedTpl[pick] || 0)) pick = id2;
        }
      }
      usedTpl[pick] = (usedTpl[pick] || 0) + 1;
      recent.push(pick);
      if (recent.length > 8) recent.shift();
      // 变通: mirror / rotate the official configuration (45 deg steps)
      var hh = _h01m(m, pick, 51);
      var sym = (hh < 0.80) ? 0 : ((Math.floor(hh * 8) % 8) + 1) % 8;   // ~20% mirrored
      var a0 = TPL.off[pick], a1 = TPL.off[pick + 1], STRIDE = 7;
      for (var r = a0; r < a1; r++) {
        var p = TPL.notes[r * STRIDE], b0 = TPL.notes[r * STRIDE + 1], kd = TPL.notes[r * STRIDE + 2];
        var dd = TPL.notes[r * STRIDE + 3], op = TPL.notes[r * STRIDE + 4], tg = TPL.notes[r * STRIDE + 5];
        var bk = TPL.notes[r * STRIDE + 6];
        var slot = m * 96 + p * 6, btn = _symKey(b0, sym) + 1;
        var isBrk = !!bk;
        // 绝赞星星: never generated (too punishing to play)
        if (kd === 3) isBrk = false;
        var ev = { slot: slot, button: btn, kind: "tap", dur: 0, op: 0, target: btn,
                   brk: isBrk, ex: false, conf: 1, path: null };
        if (kd === 2) { ev.kind = "hold"; ev.dur = Math.max(6, dd); }
        else if (kd === 3) {
          ev.kind = "slide"; ev.dur = Math.max(6, Math.min(dd || 12, 24));
          ev.op = (op === 1) ? 1 : 0;
          var tt = (tg >= 0) ? (_symKey(tg, sym) + 1) : 0;
          if (tt < 1 || tt > 8 || tt === btn) tt = (btn % 8) + 1;
          if (tt === btn) tt = ((btn + 2) % 8) + 1;
          ev.target = tt;
        }
        events.push(ev);
      }
    }
    if (onProgress) onProgress(0.8, "repair");
    events = fillSparse(events, qf, silent, _BANDS.length, 6, 2);
    events = limitRepeats(events, 48, 2);
    events.sort(function (a, b) { return (a.slot - b.slot) || ((a.kind === "tap" ? 0 : 1) - (b.kind === "tap" ? 0 : 1)); });
    events = sanitize(events, 12, 6, 2);
    events = events.filter(function (e) {
      var mm = Math.floor(e.slot / 96);
      return !(mm < silent.length && silent[mm]);
    });
    var meta2 = Object.assign({}, meta || {});
    meta2.first = offset;
    var txt = toSimai(events, bpm, level, meta2);
    if (onProgress) onProgress(1.0, "done");
    return { maidata: txt, events: events, nSlots: nSlots, mode: "retrieval" };
  }

  /** last-resort guard: drop anything that still breaks a rule */
  function sanitize(events, minGapBtn) {
    minGapBtn = minGapBtn || 12;
    var arr = events.slice().sort(function (a, b) {
      return (a.slot - b.slot) || ((a.kind === "tap" ? 0 : 1) - (b.kind === "tap" ? 0 : 1));
    });
    var out = [], busy = {}, last = {};
    for (var i = 0; i < arr.length; i++) {
      var e = arr[i];
      if (e.kind === "slide" && e.target === e.button) continue;
      var bk = busy[e.button];
      if (bk && (e.slot < bk.end || (e.slot === bk.end && bk.kind !== "tap"))) continue;
      if (last[e.button] !== undefined && e.slot - last[e.button] < minGapBtn) continue;
      busy[e.button] = { end: e.slot + Math.max(e.kind === "tap" ? 0 : e.dur, 1), kind: e.kind };
      last[e.button] = e.slot;
      out.push(e);
    }
    return out;
  }

  /** complete final gate: whatever this returns is playable */
  function sanitize(events, minGapBtn, minGap, maxHands) {
    minGapBtn = minGapBtn || 12; minGap = minGap || 6; maxHands = maxHands || 2;
    if (!events.length) return events;
    var order = events.slice().sort(function (a, b) {
      return (a.slot - b.slot) || ((a.kind === 'tap' ? 0 : 1) - (b.kind === 'tap' ? 0 : 1)) || (b.conf - a.conf);
    });
    var out = [], busy = {}, last = {}, tails = {}, active = [], lastOnset = -1e9;
    for (var i = 0; i < order.length; i++) {
      var e = order[i];
      if (e.kind === 'slide' && e.target === e.button) continue;
      var bk = busy[e.button];
      if (bk && (e.slot < bk.end || (e.slot === bk.end && bk.kind !== 'tap'))) continue;
      if (last[e.button] !== undefined && e.slot - last[e.button] < minGapBtn) continue;
      if (e.slot !== lastOnset && e.slot - lastOnset < minGap) continue;
      active = active.filter(function (a) { return a.end > e.slot; });
      var used = 0;
      for (var q = 0; q < active.length; q++) used += active[q].hands;
      if (used >= maxHands) continue;
      var h = (e.kind === 'slide' && e.op === 1) ? 2 : 1;
      if (used + h > maxHands) continue;
      if (e.kind === 'slide') {
        var key = e.target + ':' + (e.slot + Math.max(e.dur, 1));
        if (tails[key]) continue;
        tails[key] = 1;
      }
      var occ = (e.kind === 'tap') ? 0 : e.dur;
      busy[e.button] = { end: e.slot + Math.max(occ, 1), kind: e.kind };
      last[e.button] = e.slot;
      lastOnset = e.slot;
      active.push({ end: e.slot + Math.max(occ, 1), hands: h });
      out.push(e);
    }
    return out;
  }

  // ------------------------------------------------------------- pipeline
  /** pcm: Float32Array mono 22050 Hz  ->  {maidata, events, nSlots, thr} */
  function generate(pcm, bpm, offset, level, onProgress, meta) {
    if (!MODEL) throw new Error('没有加载权重');
    if (MODEL.archOk === false) {
      throw new Error('权重版本过旧（旧架构 / 缺网格掩码），请先在训练页点「导出结果」重新生成 weights.js');
    }
    offset = offset || 0;
    level = level || 5;
    var ff = frameFeatures(pcm);
    var fps = SR / HOP;
    var dur = pcm.length / SR;
    var nSlots = Math.ceil(dur * SLOTS_PER_BEAT * bpm / 60) + SLOTS_PER_MEASURE;
    var times = new Float32Array(nSlots);
    for (var s = 0; s < nSlots; s++) times[s] = offset + s / SLOTS_PER_BEAT * 60 / bpm;
    if (onProgress) onProgress(0.15, 'resample');
    var base = resampleToSlots(ff.feat, ff.T, fps, times);
    var ph = phaseFeatures(nSlots);
    var lvIdx = [2, 3, 4, 5, 6].indexOf(level);
    if (lvIdx < 0) lvIdx = 3;

    var T = nSlots, x = new Float32Array(T * 96);
    var mean = MODEL.mean, std = MODEL.std;
    var badX = 0;
    for (var t = 0; t < T; t++) {
      var ob = t * 96, ib = t * 83;
      for (var c = 0; c < 83; c++) {
        var v = (base[ib + c] - mean[c]) / std[c];
        if (v !== v || v > 1e6 || v < -1e6) { v = 0; badX++; }
        x[ob + c] = v;
      }
      for (var p = 0; p < 8; p++) x[ob + 83 + p] = ph[t * 8 + p];
      for (var l = 0; l < 5; l++) x[ob + 91 + l] = (l === lvIdx) ? 1 : 0;
    }
    if (badX > 0) {
      // silence / DC-only stretches used to poison the whole network with NaN
      if (typeof console !== 'undefined') console.warn('chartnet: sanitised', badX, 'input values');
    }
    if (onProgress) onProgress(0.25, 'model');
    var model = forward(x, T);
    if (onProgress) onProgress(0.9, 'decode');
    var dec = decode(model, T, { level: level });
    var meta2 = Object.assign({}, meta || {});
    meta2.first = offset;
    var txt = toSimai(dec.events, bpm, level, meta2);
    if (onProgress) onProgress(1.0, 'done');
    return { maidata: txt, events: dec.events, nSlots: nSlots, thr: dec.thr, thr2: dec.thr2,
             maxP: dec.maxP, nanSlots: dec.nanSlots, targetNotes: dec.targetNotes,
             logitMin: dec.logitMin, logitMax: dec.logitMax, actNaN: dec.actNaN };
  }

  var API = {
    SR: SR, N_FFT: N_FFT, HOP: HOP,
    frameFeatures: frameFeatures, resampleToSlots: resampleToSlots,
    phaseFeatures: phaseFeatures, loadWeights: loadWeights, setWeights: setWeights,
    forward: forward, decode: decode, flowDuration: flowDuration, toSimai: toSimai, generate: generate,
    setTemplates: setTemplates, generateRetrieval: generateRetrieval, sanitize: sanitize,
    hasTemplates: function () { return !!TPL; },
    enforcePlayable: enforcePlayable,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  root.ChartNet = API;
})(typeof window !== 'undefined' ? window : globalThis);
