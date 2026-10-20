/* Browser (and Node) re-implementation of the Python pipeline in hw1/.
   preprocess: resample -> 4 kHz, Butterworth band-pass (sosfiltfilt), DC removal + peak norm, 20 s wrap/trim
   features:   STFT(512, hop 128, hann, centre, zero pad) -> mel(64, slaney, exported matrix) -> power_to_db(ref=max, top_db=80)
               -> DCT-II ortho (40 MFCC) -> savgol delta (width 9) -> mean & std  => 160 values
   models:     StandardScaler + calibrated RBF-SVM / Random Forest / MLP, parameters exported from the trained models. */
(function (root) {
  "use strict";

  // ------------------------------------------------------------ resampling (windowed sinc)
  function resample(x, srIn, srOut) {
    if (srIn === srOut) return Float64Array.from(x);
    const ratio = srOut / srIn;
    const nOut = Math.floor(x.length * ratio);
    const y = new Float64Array(nOut);
    // parameters tuned so the features match librosa's soxr_hq resampler closely (max |diff| ~0.03)
    const cutoff = 0.5 * Math.min(1, ratio) * 0.975;           // relative to input rate
    const half = 32 / Math.min(1, ratio);                         // half-width in input samples
    const hw = Math.ceil(half);
    for (let n = 0; n < nOut; n++) {
      const t = n / ratio;
      const k0 = Math.floor(t) - hw, k1 = Math.floor(t) + hw;
      let acc = 0, wsum = 0;
      for (let k = Math.max(0, k0); k <= Math.min(x.length - 1, k1); k++) {
        const d = t - k;
        const a = 2 * cutoff * d;
        const sinc = d === 0 ? 1 : Math.sin(Math.PI * a) / (Math.PI * a);
        const w = 0.5 + 0.5 * Math.cos(Math.PI * d / (half + 1));  // Hann window
        const h = 2 * cutoff * sinc * w;
        acc += x[k] * h; wsum += h;
      }
      y[n] = wsum !== 0 ? acc / wsum : 0;
    }
    return y;
  }

  // ------------------------------------------------------------ IIR filtering (scipy sosfilt / sosfiltfilt)
  function sosfilt(sos, x, zi) {
    const y = Float64Array.from(x);
    for (let s = 0; s < sos.length; s++) {
      const [b0, b1, b2, , a1, a2] = sos[s];
      let z0 = zi[s][0], z1 = zi[s][1];
      for (let n = 0; n < y.length; n++) {
        const xn = y[n];
        const yn = b0 * xn + z0;
        z0 = b1 * xn - a1 * yn + z1;
        z1 = b2 * xn - a2 * yn;
        y[n] = yn;
      }
    }
    return y;
  }
  function sosfiltfilt(sos, zi, x) {
    const nz = Math.min(sos.filter(r => r[2] === 0).length, sos.filter(r => r[5] === 0).length);
    const padlen = 3 * (2 * sos.length + 1 - nz);
    const n = x.length;
    const ext = new Float64Array(n + 2 * padlen);
    for (let i = 0; i < padlen; i++) ext[i] = 2 * x[0] - x[padlen - i];
    ext.set(x, padlen);
    for (let i = 0; i < padlen; i++) ext[padlen + n + i] = 2 * x[n - 1] - x[n - 2 - i];
    const scaled = (v) => zi.map(r => [r[0] * v, r[1] * v]);
    let y = sosfilt(sos, ext, scaled(ext[0]));
    y.reverse();
    y = sosfilt(sos, y, scaled(y[0]));
    y.reverse();
    return y.slice(padlen, padlen + n);
  }

  function normalize(y) {
    let m = 0; for (const v of y) m += v; m /= y.length;
    const out = new Float64Array(y.length); let peak = 0;
    for (let i = 0; i < y.length; i++) { out[i] = y[i] - m; peak = Math.max(peak, Math.abs(out[i])); }
    if (peak > 0) for (let i = 0; i < out.length; i++) out[i] /= peak;
    return out;
  }
  function fixDuration(y, n) {
    if (y.length >= n) return y.slice(0, n);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = y[i % y.length];
    return out;
  }

  // ------------------------------------------------------------ FFT (radix-2, in place)
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }

  function powerSpectrogram(y, nfft, hop) {
    const pad = nfft >> 1;
    const padded = new Float64Array(y.length + 2 * pad); padded.set(y, pad);   // centre=True, zero padding
    const nFrames = 1 + Math.floor((padded.length - nfft) / hop);
    const win = new Float64Array(nfft);
    for (let i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nfft);  // periodic Hann
    const nb = nfft / 2 + 1;
    const P = Array.from({ length: nb }, () => new Float64Array(nFrames));
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    for (let t = 0; t < nFrames; t++) {
      for (let i = 0; i < nfft; i++) { re[i] = padded[t * hop + i] * win[i]; im[i] = 0; }
      fft(re, im);
      for (let k = 0; k < nb; k++) P[k][t] = re[k] * re[k] + im[k] * im[k];
    }
    return P;   // [bins][frames]
  }

  function logMel(y, cfg) {
    const P = powerSpectrogram(y, cfg.n_fft, cfg.hop_length);
    const T = P[0].length, M = cfg.mel_basis;
    const S = M.map(row => {
      const out = new Float64Array(T);
      for (let k = 0; k < row.length; k++) { const w = row[k]; if (w === 0) continue; const pk = P[k]; for (let t = 0; t < T; t++) out[t] += w * pk[t]; }
      return out;
    });
    let ref = 0; for (const r of S) for (const v of r) if (v > ref) ref = v;
    const amin = 1e-10, refDb = 10 * Math.log10(Math.max(amin, ref));
    let mx = -Infinity;
    const D = S.map(r => r.map(v => { const d = 10 * Math.log10(Math.max(amin, v)) - refDb; if (d > mx) mx = d; return d; }));
    const floor = mx - 80;
    return D.map(r => r.map(v => Math.max(v, floor)));   // [mels][frames]
  }

  function mfccFromLogMel(L, nMfcc) {
    const N = L.length, T = L[0].length;
    const out = [];
    for (let k = 0; k < nMfcc; k++) {
      const scale = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
      const c = new Float64Array(T);
      for (let n = 0; n < N; n++) { const w = Math.cos(Math.PI * k * (2 * n + 1) / (2 * N)); const ln = L[n]; for (let t = 0; t < T; t++) c[t] += w * ln[t]; }
      for (let t = 0; t < T; t++) c[t] *= scale;
      out.push(c);
    }
    return out;   // [coef][frames]
  }

  function delta(row) {   // savgol_filter(width 9, polyorder 1, deriv 1, mode 'interp')
    const T = row.length, d = new Float64Array(T);
    const at = (t) => { let s = 0; for (let k = -4; k <= 4; k++) s += k * row[t + k]; return s / 60; };
    for (let t = 4; t < T - 4; t++) d[t] = at(t);
    for (let t = 0; t < 4; t++) d[t] = at(4);
    for (let t = T - 4; t < T; t++) d[t] = at(T - 5);
    return d;
  }

  function meanStd(rows) {
    const mean = [], std = [];
    for (const r of rows) {
      let m = 0; for (const v of r) m += v; m /= r.length;
      let s = 0; for (const v of r) s += (v - m) * (v - m);
      mean.push(m); std.push(Math.sqrt(s / r.length));
    }
    return mean.concat(std);
  }

  function preprocess(x, sr, cfg) {
    const y4 = resample(x, sr, cfg.target_sr);
    const yb = sosfiltfilt(cfg.sos, cfg.sos_zi, y4);
    const yp = fixDuration(normalize(yb), Math.round(cfg.target_sr * cfg.duration_s));
    return { y4, yb, yp };
  }

  function features(yp, cfg) {
    const L = logMel(yp, cfg);
    const M = mfccFromLogMel(L, cfg.n_mfcc);
    const D = M.map(delta);
    return { logmel: L, mfcc: M, vector: meanStd(M).concat(meanStd(D)) };
  }

  // ------------------------------------------------------------ models
  function scale(x, sc) { return x.map((v, i) => (v - sc.mean[i]) / sc.scale[i]); }
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  function predictSVM(m, x) {
    const z = scale(x, m.scaler);
    let f = m.intercept;
    for (let i = 0; i < m.sv.length; i++) {
      const sv = m.sv[i]; let d = 0;
      for (let j = 0; j < z.length; j++) { const e = z[j] - sv[j]; d += e * e; }
      f += m.dual[i] * Math.exp(-m.gamma * d);
    }
    return sigmoid(-(m.a * f + m.b));
  }
  function predictMLP(m, x) {
    let h = scale(x, m.scaler);
    m.W.forEach((W, l) => {
      const b = m.b[l], out = new Array(b.length).fill(0);
      for (let j = 0; j < b.length; j++) { let s = b[j]; for (let i = 0; i < h.length; i++) s += h[i] * W[i][j]; out[j] = s; }
      h = l < m.W.length - 1 ? out.map(v => Math.max(0, v)) : out;
    });
    return sigmoid(h[0]);
  }
  function predictRF(m, x) {
    const z = scale(x, m.scaler); let p = 0;
    for (const t of m.trees) {
      let n = 0;
      while (t.l[n] !== -1) n = z[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
      p += t.p[n];
    }
    return p / m.trees.length;
  }
  function predict(model, x) {
    return model.kind === "svm" ? predictSVM(model, x) : model.kind === "mlp" ? predictMLP(model, x) : predictRF(model, x);
  }

  // ------------------------------------------------------------ WAV decoding helper (PCM 16/24/32, float32)
  function decodeWav(buf) {
    const dv = new DataView(buf);
    const tag = (o) => String.fromCharCode(dv.getUint8(o), dv.getUint8(o + 1), dv.getUint8(o + 2), dv.getUint8(o + 3));
    if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("not a WAV file");
    let o = 12, fmt = null, data = null;
    while (o + 8 <= dv.byteLength) {
      const id = tag(o), size = dv.getUint32(o + 4, true);
      if (id === "fmt ") fmt = { format: dv.getUint16(o + 8, true), ch: dv.getUint16(o + 10, true), sr: dv.getUint32(o + 12, true), bits: dv.getUint16(o + 22, true) };
      if (id === "data") { data = { off: o + 8, size: Math.min(size, dv.byteLength - o - 8) }; break; }
      o += 8 + size + (size % 2);
    }
    if (!fmt || !data) throw new Error("WAV without fmt/data chunk");
    const bps = fmt.bits / 8, n = Math.floor(data.size / (bps * fmt.ch)), x = new Float64Array(n);
    const isFloat = fmt.format === 3 || (fmt.format === 0xFFFE && fmt.bits === 32 && false);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let c = 0; c < fmt.ch; c++) {
        const p = data.off + (i * fmt.ch + c) * bps;
        let v;
        if (isFloat) v = dv.getFloat32(p, true);
        else if (fmt.bits === 16) v = dv.getInt16(p, true) / 32768;
        else if (fmt.bits === 24) { let u = dv.getUint8(p) | (dv.getUint8(p + 1) << 8) | (dv.getUint8(p + 2) << 16); if (u & 0x800000) u -= 0x1000000; v = u / 8388608; }
        else if (fmt.bits === 32) v = dv.getInt32(p, true) / 2147483648;
        else if (fmt.bits === 8) v = (dv.getUint8(p) - 128) / 128;
        else throw new Error("unsupported bit depth " + fmt.bits);
        s += v;
      }
      x[i] = s / fmt.ch;
    }
    return { x, sr: fmt.sr };
  }

  function encodeWav16(y, sr) {
    const n = y.length, buf = new ArrayBuffer(44 + 2 * n), dv = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
    w(0, "RIFF"); dv.setUint32(4, 36 + 2 * n, true); w(8, "WAVE"); w(12, "fmt ");
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true); dv.setUint32(24, sr, true);
    dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true); w(36, "data"); dv.setUint32(40, 2 * n, true);
    let peak = 0; for (const v of y) peak = Math.max(peak, Math.abs(v)); const g = peak > 0 ? 0.95 / peak : 1;
    for (let i = 0; i < n; i++) dv.setInt16(44 + 2 * i, Math.max(-32767, Math.min(32767, Math.round(y[i] * g * 32767))), true);
    return buf;
  }

  root.DSP = { resample, sosfiltfilt, normalize, fixDuration, powerSpectrogram, logMel, mfccFromLogMel, delta,
               meanStd, preprocess, features, predict, decodeWav, encodeWav16 };
  if (typeof module !== "undefined") module.exports = root.DSP;
})(typeof window !== "undefined" ? window : globalThis);
