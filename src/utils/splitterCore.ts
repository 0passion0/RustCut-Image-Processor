export type SplitMode = 'auto' | 'xycut' | 'contour' | 'grid';

export interface SplitBox {
  id: string;
  index: number;
  row: number;
  col: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  strategy: 'xycut' | 'gutter' | 'contour' | 'grid';
  trimmedEdges: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

export interface SplitOptions {
  mode: SplitMode;
  sensitivity: number; // 0.1 to 1.0 (default 0.65)
  minRegionRatio: number; // minimum region dimension as fraction of min(w,h), default 0.08
  bleedTrim: number; // max anti-aliased seam bleed trim in px (0..6), default 2
  rows?: number; // for uniform grid mode
  cols?: number; // for uniform grid mode
  gutterPx?: number; // for uniform grid mode
  contourGapPx?: number; // dilation/merge gap for contour mode, default 8
}

export interface SplitAnalysisResult {
  imageWidth: number;
  imageHeight: number;
  detectedMode: 'xycut' | 'contour' | 'grid';
  boxes: SplitBox[];
  hProfile: number[];
  vProfile: number[];
  elapsedMs: number;
}

export interface RawImage {
  data: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
  channels: number; // 3 or 4
}

const DEFAULT_OPTIONS: SplitOptions = {
  mode: 'auto',
  sensitivity: 0.65,
  minRegionRatio: 0.08,
  bleedTrim: 2,
  rows: 3,
  cols: 3,
  gutterPx: 0,
  contourGapPx: 8,
};

function clamp(val: number, min: number, max: number): number {
  return val < min ? min : val > max ? max : val;
}

function rgbAt(img: RawImage, x: number, y: number): [number, number, number, number] {
  const c = img.channels;
  const idx = (y * img.width + x) * c;
  return [
    img.data[idx],
    img.data[idx + 1],
    img.data[idx + 2],
    c === 4 ? img.data[idx + 3] : 255,
  ];
}

function colorDist(a: [number, number, number, number], b: [number, number, number, number]): number {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]);
}

function estimateBorderBackground(img: RawImage): {
  rgb: [number, number, number, number];
  isTransparent: boolean;
  borderUniformity: number;
} {
  const samples: Array<[number, number, number, number]> = [];
  const stepX = Math.max(1, Math.floor(img.width / 80));
  const stepY = Math.max(1, Math.floor(img.height / 80));
  let transparentCount = 0;

  for (let x = 0; x < img.width; x += stepX) {
    const t = rgbAt(img, x, 0);
    const b = rgbAt(img, x, img.height - 1);
    samples.push(t, b);
    if (t[3] < 32) transparentCount++;
    if (b[3] < 32) transparentCount++;
  }
  for (let y = 0; y < img.height; y += stepY) {
    const l = rgbAt(img, 0, y);
    const r = rgbAt(img, img.width - 1, y);
    samples.push(l, r);
    if (l[3] < 32) transparentCount++;
    if (r[3] < 32) transparentCount++;
  }

  if (transparentCount > samples.length * 0.5) {
    return { rgb: [0, 0, 0, 0], isTransparent: true, borderUniformity: transparentCount / samples.length };
  }

  const counts = new Map<string, { count: number; sum: [number, number, number] }>();
  for (const s of samples) {
    const key = `${s[0] >> 3},${s[1] >> 3},${s[2] >> 3}`;
    const entry = counts.get(key);
    if (entry) {
      entry.count++;
      entry.sum[0] += s[0];
      entry.sum[1] += s[1];
      entry.sum[2] += s[2];
    } else {
      counts.set(key, { count: 1, sum: [s[0], s[1], s[2]] });
    }
  }

  let best = { count: 0, sum: [255, 255, 255] as [number, number, number] };
  for (const v of counts.values()) {
    if (v.count > best.count) best = v;
  }

  const avg: [number, number, number, number] = [
    Math.round(best.sum[0] / Math.max(1, best.count)),
    Math.round(best.sum[1] / Math.max(1, best.count)),
    Math.round(best.sum[2] / Math.max(1, best.count)),
    255,
  ];
  return {
    rgb: avg,
    isTransparent: false,
    borderUniformity: best.count / Math.max(1, samples.length),
  };
}

interface SeamCandidate {
  pos: number;
  score: number;
  coverage: number;
  smoothness: number;
  isGutter: boolean;
}

const COV_POW_LUT = new Float32Array(257);
for (let i = 0; i <= 256; i++) {
  COV_POW_LUT[i] = Math.pow(i / 256, 2.4);
}
function covPow(cov: number): number {
  const idx = Math.round(cov * 256);
  return COV_POW_LUT[idx < 0 ? 0 : idx > 256 ? 256 : idx];
}

function profileHorizontalSeams(
  img: RawImage,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  minDim: number,
  bgRgb: [number, number, number, number],
  bgUniform: boolean,
  isTransparent: boolean,
): SeamCandidate[] {
  const spanW = x1 - x0;
  const spanH = y1 - y0;
  const startY = y0 + minDim;
  const endY = y1 - minDim;
  if (endY <= startY || spanW < 8) return [];

  const data = img.data;
  const w = img.width;
  const c = img.channels;
  const stepX = Math.max(1, Math.floor(spanW / 200));
  const sampleCount = Math.ceil(spanW / stepX);

  const rowStep = new Float32Array(spanH);
  const rowLine = new Float32Array(spanH);
  const rowSmooth = new Float32Array(spanH);
  const rowBgMatch = new Float32Array(spanH);
  const rowActiveCount = new Uint16Array(spanH);
  const lastMarkedRy = new Int32Array(sampleCount).fill(-1);

  const bgR = bgRgb[0], bgG = bgRgb[1], bgB = bgRgb[2];
  const rowStride = w * c;
  const xStepBytes = stepX * c;

  for (let y = y0 + 1; y < y1 - 1; y++) {
    const ry = y - y0;
    const rHiLimit = ry + 2 < spanH - 1 ? ry + 2 : spanH - 2;
    const rLoBase = ry - 2 > 1 ? ry - 2 : 1;

    let sSum = 0;
    let lSum = 0;
    let smSum = 0;
    let bgHits = 0;
    let prevLuma = -1;
    let sxIdx = 0;

    let i1 = (y * w + x0) * c;
    const iEnd = (y * w + x1) * c;

    while (i1 < iEnd) {
      const i0 = i1 - rowStride;
      const i2 = i1 + rowStride;
      const r0 = data[i0], g0 = data[i0 + 1], b0 = data[i0 + 2];
      const r1 = data[i1], g1 = data[i1 + 1], b1 = data[i1 + 2];
      const r2 = data[i2], g2 = data[i2 + 1], b2 = data[i2 + 2];

      const ds = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);
      const dl = (Math.abs(2 * r1 - r0 - r2) + Math.abs(2 * g1 - g0 - g2) + Math.abs(2 * b1 - b0 - b2)) * 0.5;
      sSum += ds;
      lSum += dl;

      if (ds + dl > 16) {
        const last = lastMarkedRy[sxIdx];
        const rLo = last + 1 > rLoBase ? last + 1 : rLoBase;
        for (let r = rLo; r <= rHiLimit; r++) {
          rowActiveCount[r]++;
        }
        if (rHiLimit > last) lastMarkedRy[sxIdx] = rHiLimit;
      }
      sxIdx++;

      const luma = r1 + g1 + b1;
      if (prevLuma >= 0) smSum += Math.abs(luma - prevLuma);
      prevLuma = luma;

      if (bgUniform) {
        if (isTransparent ? (c === 4 && data[i1 + 3] < 24) : Math.abs(r1 - bgR) + Math.abs(g1 - bgG) + Math.abs(b1 - bgB) < 22) {
          bgHits++;
        }
      }
      i1 += xStepBytes;
    }
    rowStep[ry] = sSum / sampleCount;
    rowLine[ry] = lSum / sampleCount;
    rowSmooth[ry] = smSum / Math.max(1, sampleCount - 1);
    rowBgMatch[ry] = bgHits / sampleCount;
  }

  const rawCandidates: SeamCandidate[] = new Array(endY - startY + 1);
  let outIdx = 0;
  for (let y = startY; y <= endY; y++) {
    const ry = y - y0;
    const cov = rowActiveCount[ry] / sampleCount;
    const sPrev = rowSmooth[ry - 1 > 1 ? ry - 1 : 1];
    const sCur = rowSmooth[ry];
    const sNext = rowSmooth[ry + 1 < spanH - 1 ? ry + 1 : spanH - 2];
    const minSideSmooth = sPrev < sCur ? (sPrev < sNext ? sPrev : sNext) : (sCur < sNext ? sCur : sNext);

    const raw = rowStep[ry] + 0.7 * rowLine[ry];
    const seamScore = (raw * covPow(cov)) / (1 + minSideSmooth * 0.12);

    let gutterScore = 0;
    if (bgUniform && sCur < 2.5 && rowBgMatch[ry] > 0.95) {
      let contentAbove = false;
      let contentBelow = false;
      for (let dy = 3; dy < ry; dy += 3) {
        if (rowBgMatch[ry - dy] < 0.9) {
          contentAbove = true;
          break;
        }
      }
      for (let dy = 3; dy < spanH - ry; dy += 3) {
        if (rowBgMatch[ry + dy] < 0.9) {
          contentBelow = true;
          break;
        }
      }
      if (contentAbove && contentBelow) {
        gutterScore = 110 / (1 + sCur);
      }
    }

    const bestScore = seamScore > gutterScore ? seamScore : gutterScore;
    rawCandidates[outIdx++] = {
      pos: y,
      score: bestScore,
      coverage: cov,
      smoothness: minSideSmooth,
      isGutter: gutterScore > seamScore,
    };
  }

  return collapseGutterRunsInPlace(rawCandidates, 1);
}

function profileVerticalSeams(
  img: RawImage,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  minDim: number,
  bgRgb: [number, number, number, number],
  bgUniform: boolean,
  isTransparent: boolean,
): SeamCandidate[] {
  const spanH = y1 - y0;
  const spanW = x1 - x0;
  const startX = x0 + minDim;
  const endX = x1 - minDim;
  if (endX <= startX || spanH < 8) return [];

  const data = img.data;
  const w = img.width;
  const c = img.channels;
  const stepY = Math.max(1, Math.floor(spanH / 200));
  const sampleCount = Math.ceil(spanH / stepY);

  const colStep = new Float32Array(spanW);
  const colLine = new Float32Array(spanW);
  const colSmooth = new Float32Array(spanW);
  const colBgMatch = new Float32Array(spanW);
  const colActiveCount = new Uint16Array(spanW);
  const prevRowLuma = new Int32Array(spanW).fill(-1);

  const bgR = bgRgb[0], bgG = bgRgb[1], bgB = bgRgb[2];
  const innerSpanW = spanW - 2;

  // Row-major outer loop ensures sequential L1/L2 cache-friendly reads across the scanline
  for (let y = y0; y < y1; y += stepY) {
    let lastMarkedRx = -1;
    let i1 = (y * w + (x0 + 1)) * c;
    for (let rx = 1; rx <= innerSpanW; rx++) {
      const i0 = i1 - c;
      const i2 = i1 + c;
      const r0 = data[i0], g0 = data[i0 + 1], b0 = data[i0 + 2];
      const r1 = data[i1], g1 = data[i1 + 1], b1 = data[i1 + 2];
      const r2 = data[i2], g2 = data[i2 + 1], b2 = data[i2 + 2];

      const ds = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);
      const dl = (Math.abs(2 * r1 - r0 - r2) + Math.abs(2 * g1 - g0 - g2) + Math.abs(2 * b1 - b0 - b2)) * 0.5;
      colStep[rx] += ds;
      colLine[rx] += dl;

      if (ds + dl > 16) {
        const cLoBase = rx - 2 > 1 ? rx - 2 : 1;
        const cLo = lastMarkedRx + 1 > cLoBase ? lastMarkedRx + 1 : cLoBase;
        const cHi = rx + 2 < spanW - 1 ? rx + 2 : spanW - 2;
        for (let col = cLo; col <= cHi; col++) {
          colActiveCount[col]++;
        }
        if (cHi > lastMarkedRx) lastMarkedRx = cHi;
      }

      const luma = r1 + g1 + b1;
      const prev = prevRowLuma[rx];
      if (prev >= 0) colSmooth[rx] += Math.abs(luma - prev);
      prevRowLuma[rx] = luma;

      if (bgUniform) {
        if (isTransparent ? (c === 4 && data[i1 + 3] < 24) : Math.abs(r1 - bgR) + Math.abs(g1 - bgG) + Math.abs(b1 - bgB) < 22) {
          colBgMatch[rx]++;
        }
      }
      i1 += c;
    }
  }

  const smDenom = Math.max(1, sampleCount - 1);
  for (let rx = 1; rx <= innerSpanW; rx++) {
    colStep[rx] /= sampleCount;
    colLine[rx] /= sampleCount;
    colSmooth[rx] /= smDenom;
    colBgMatch[rx] /= sampleCount;
  }

  const rawCandidates: SeamCandidate[] = new Array(endX - startX + 1);
  let outIdx = 0;
  for (let x = startX; x <= endX; x++) {
    const rx = x - x0;
    const cov = colActiveCount[rx] / sampleCount;
    const sPrev = colSmooth[rx - 1 > 1 ? rx - 1 : 1];
    const sCur = colSmooth[rx];
    const sNext = colSmooth[rx + 1 < spanW - 1 ? rx + 1 : spanW - 2];
    const minSideSmooth = sPrev < sCur ? (sPrev < sNext ? sPrev : sNext) : (sCur < sNext ? sCur : sNext);

    const raw = colStep[rx] + 0.7 * colLine[rx];
    const seamScore = (raw * covPow(cov)) / (1 + minSideSmooth * 0.12);

    let gutterScore = 0;
    if (bgUniform && sCur < 2.5 && colBgMatch[rx] > 0.95) {
      let contentLeft = false;
      let contentRight = false;
      for (let dx = 3; dx < rx; dx += 3) {
        if (colBgMatch[rx - dx] < 0.9) {
          contentLeft = true;
          break;
        }
      }
      for (let dx = 3; dx < spanW - rx; dx += 3) {
        if (colBgMatch[rx + dx] < 0.9) {
          contentRight = true;
          break;
        }
      }
      if (contentLeft && contentRight) {
        gutterScore = 110 / (1 + sCur);
      }
    }

    const bestScore = seamScore > gutterScore ? seamScore : gutterScore;
    rawCandidates[outIdx++] = {
      pos: x,
      score: bestScore,
      coverage: cov,
      smoothness: minSideSmooth,
      isGutter: gutterScore > seamScore,
    };
  }

  return collapseGutterRunsInPlace(rawCandidates, 1);
}

/**
 * Collapses contiguous runs of gutter rows/cols into a single midpoint peak
 * so a 40px white gutter never splits into two cuts at its edges.
 */
function collapseGutterRunsInPlace(out: SeamCandidate[], step = 1): SeamCandidate[] {
  let i = 0;
  while (i < out.length) {
    if (!out[i].isGutter) {
      i++;
      continue;
    }
    let j = i;
    while (j + 1 < out.length && out[j + 1].isGutter && out[j + 1].pos <= out[j].pos + step) {
      j++;
    }
    const mid = (i + j) >> 1;
    let maxScore = 0;
    for (let k = i; k <= j; k++) {
      if (out[k].score > maxScore) maxScore = out[k].score;
    }
    for (let k = i; k <= j; k++) {
      if (k === mid) {
        out[k].score = maxScore;
      } else {
        out[k].score = 0;
        out[k].isGutter = false;
      }
    }
    i = j + 1;
  }
  return out;
}

function pickPeaksNMS(candidates: SeamCandidate[], threshold: number, minGap: number): SeamCandidate[] {
  const sorted = [...candidates]
    .filter((c) => c.score >= threshold && (c.isGutter || c.coverage >= 0.72))
    .sort((a, b) => b.score - a.score);

  const picked: SeamCandidate[] = [];
  for (const cand of sorted) {
    if (picked.every((p) => Math.abs(p.pos - cand.pos) >= minGap)) {
      picked.push(cand);
    }
  }
  picked.sort((a, b) => a.pos - b.pos);
  return picked;
}

function isRegionEmptyBackground(
  img: RawImage,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  bgRgb: [number, number, number, number],
  isTransparent: boolean,
): boolean {
  const stepX = Math.max(1, Math.floor((x1 - x0) / 32));
  const stepY = Math.max(1, Math.floor((y1 - y0) / 32));
  let total = 0;
  let fg = 0;
  for (let y = y0 + 2; y < y1 - 2; y += stepY) {
    for (let x = x0 + 2; x < x1 - 2; x += stepX) {
      const p = rgbAt(img, x, y);
      total++;
      if (isTransparent ? p[3] >= 24 : colorDist(p, bgRgb) > 20) {
        fg++;
      }
    }
  }
  return total > 0 && fg / total < 0.03;
}

function refineAndTrimCell(
  img: RawImage,
  rawBox: {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    row: number;
    col: number;
    conf: number;
    isGutter: boolean;
  },
  bleedTrim: number,
  bgRgb: [number, number, number, number],
  bgUniform: boolean,
  isTransparent: boolean,
): SplitBox {
  let { x0, y0, x1, y1 } = rawBox;
  const W = img.width;
  const H = img.height;
  const data = img.data;
  const c = img.channels;

  // Snap each interior split edge (+-8px) to the exact discontinuity for THIS panel's span
  if (!rawBox.isGutter) {
    if (y0 > 2) {
      let bestY = y0;
      let bestE = -1;
      const step = Math.max(1, Math.floor((x1 - x0) / 120));
      for (let y = Math.max(1, y0 - 8); y <= Math.min(y1 - 20, y0 + 8); y++) {
        let e = 0;
        let n = 0;
        for (let x = x0 + 4; x < x1 - 4; x += step) {
          const i0 = ((y - 1) * W + x) * c;
          const i1 = (y * W + x) * c;
          const i2 = ((y + 1) * W + x) * c;
          const ds = Math.abs(data[i1] - data[i2]) + Math.abs(data[i1 + 1] - data[i2 + 1]) + Math.abs(data[i1 + 2] - data[i2 + 2]);
          const dl = (Math.abs(2 * data[i1] - data[i0] - data[i2]) + Math.abs(2 * data[i1 + 1] - data[i0 + 1] - data[i2 + 1]) + Math.abs(2 * data[i1 + 2] - data[i0 + 2] - data[i2 + 2])) * 0.5;
          e += ds + dl;
          n++;
        }
        e /= Math.max(1, n);
        if (e > bestE) {
          bestE = e;
          bestY = y;
        }
      }
      if (bestE > 15) y0 = bestY + 1;
    }

    if (y1 < H - 2) {
      let bestY = y1;
      let bestE = -1;
      const step = Math.max(1, Math.floor((x1 - x0) / 120));
      for (let y = Math.max(y0 + 20, y1 - 8); y <= Math.min(H - 2, y1 + 8); y++) {
        let e = 0;
        let n = 0;
        for (let x = x0 + 4; x < x1 - 4; x += step) {
          const i0 = ((y - 1) * W + x) * c;
          const i1 = (y * W + x) * c;
          const i2 = ((y + 1) * W + x) * c;
          const ds = Math.abs(data[i0] - data[i1]) + Math.abs(data[i0 + 1] - data[i1 + 1]) + Math.abs(data[i0 + 2] - data[i1 + 2]);
          const dl = (Math.abs(2 * data[i1] - data[i0] - data[i2]) + Math.abs(2 * data[i1 + 1] - data[i0 + 1] - data[i2 + 1]) + Math.abs(2 * data[i1 + 2] - data[i0 + 2] - data[i2 + 2])) * 0.5;
          e += ds + dl;
          n++;
        }
        e /= Math.max(1, n);
        if (e > bestE) {
          bestE = e;
          bestY = y;
        }
      }
      if (bestE > 15) y1 = bestY;
    }

    if (x0 > 2) {
      let bestX = x0;
      let bestE = -1;
      const step = Math.max(1, Math.floor((y1 - y0) / 120));
      for (let x = Math.max(1, x0 - 8); x <= Math.min(x1 - 20, x0 + 8); x++) {
        let e = 0;
        let n = 0;
        for (let y = y0 + 4; y < y1 - 4; y += step) {
          const i0 = (y * W + (x - 1)) * c;
          const i1 = (y * W + x) * c;
          const i2 = (y * W + (x + 1)) * c;
          const ds = Math.abs(data[i1] - data[i2]) + Math.abs(data[i1 + 1] - data[i2 + 1]) + Math.abs(data[i1 + 2] - data[i2 + 2]);
          const dl = (Math.abs(2 * data[i1] - data[i0] - data[i2]) + Math.abs(2 * data[i1 + 1] - data[i0 + 1] - data[i2 + 1]) + Math.abs(2 * data[i1 + 2] - data[i0 + 2] - data[i2 + 2])) * 0.5;
          e += ds + dl;
          n++;
        }
        e /= Math.max(1, n);
        if (e > bestE) {
          bestE = e;
          bestX = x;
        }
      }
      if (bestE > 15) x0 = bestX + 1;
    }

    if (x1 < W - 2) {
      let bestX = x1;
      let bestE = -1;
      const step = Math.max(1, Math.floor((y1 - y0) / 120));
      for (let x = Math.max(x0 + 20, x1 - 8); x <= Math.min(W - 2, x1 + 8); x++) {
        let e = 0;
        let n = 0;
        for (let y = y0 + 4; y < y1 - 4; y += step) {
          const i0 = (y * W + (x - 1)) * c;
          const i1 = (y * W + x) * c;
          const i2 = (y * W + (x + 1)) * c;
          const ds = Math.abs(data[i0] - data[i1]) + Math.abs(data[i0 + 1] - data[i1 + 1]) + Math.abs(data[i0 + 2] - data[i1 + 2]);
          const dl = (Math.abs(2 * data[i1] - data[i0] - data[i2]) + Math.abs(2 * data[i1 + 1] - data[i0 + 1] - data[i2 + 1]) + Math.abs(2 * data[i1 + 2] - data[i0 + 2] - data[i2 + 2])) * 0.5;
          e += ds + dl;
          n++;
        }
        e /= Math.max(1, n);
        if (e > bestE) {
          bestE = e;
          bestX = x;
        }
      }
      if (bestE > 15) x1 = bestX;
    }
  }

  // Trim outer solid gutter if canvas has a uniform background margin around panels
  const trimmed = { top: 0, right: 0, bottom: 0, left: 0 };
  const bgR = bgRgb[0], bgG = bgRgb[1], bgB = bgRgb[2];
  if (bgUniform) {
    const isRowBg = (y: number) => {
      let hits = 0;
      let total = 0;
      const step = Math.max(1, Math.floor((x1 - x0) / 60));
      for (let x = x0; x < x1; x += step) {
        const i = (y * W + x) * c;
        if (isTransparent ? (c === 4 && data[i + 3] < 24) : Math.abs(data[i] - bgR) + Math.abs(data[i + 1] - bgG) + Math.abs(data[i + 2] - bgB) < 22) hits++;
        total++;
      }
      return hits / Math.max(1, total) > 0.94;
    };
    const isColBg = (x: number) => {
      let hits = 0;
      let total = 0;
      const step = Math.max(1, Math.floor((y1 - y0) / 60));
      for (let y = y0; y < y1; y += step) {
        const i = (y * W + x) * c;
        if (isTransparent ? (c === 4 && data[i + 3] < 24) : Math.abs(data[i] - bgR) + Math.abs(data[i + 1] - bgG) + Math.abs(data[i + 2] - bgB) < 22) hits++;
        total++;
      }
      return hits / Math.max(1, total) > 0.94;
    };
    while (y0 < y1 - 24 && isRowBg(y0)) { y0++; trimmed.top++; }
    while (y1 > y0 + 24 && isRowBg(y1 - 1)) { y1--; trimmed.bottom++; }
    while (x0 < x1 - 24 && isColBg(x0)) { x0++; trimmed.left++; }
    while (x1 > x0 + 24 && isColBg(x1 - 1)) { x1--; trimmed.right++; }
  }

  // Bleed Decontamination: remove 1..bleedTrim outer transition/hairline rows & cols on internal seams
  if (bleedTrim > 0) {
    const rowDiffToInterior = (y: number, refY: number) => {
      let d = 0;
      let n = 0;
      const step = Math.max(1, Math.floor((x1 - x0) / 80));
      for (let x = x0 + 8; x < x1 - 8; x += step) {
        const i1 = (y * W + x) * c;
        const i2 = (refY * W + x) * c;
        d += Math.abs(data[i1] - data[i2]) + Math.abs(data[i1 + 1] - data[i2 + 1]) + Math.abs(data[i1 + 2] - data[i2 + 2]);
        n++;
      }
      return d / Math.max(1, n);
    };
    const colDiffToInterior = (x: number, refX: number) => {
      let d = 0;
      let n = 0;
      const step = Math.max(1, Math.floor((y1 - y0) / 80));
      for (let y = y0 + 8; y < y1 - 8; y += step) {
        const i1 = (y * W + x) * c;
        const i2 = (y * W + refX) * c;
        d += Math.abs(data[i1] - data[i2]) + Math.abs(data[i1 + 1] - data[i2 + 1]) + Math.abs(data[i1 + 2] - data[i2 + 2]);
        n++;
      }
      return d / Math.max(1, n);
    };

    if (rawBox.y0 > 2 && y1 - y0 > 32) {
      const refY = Math.min(y1 - 1, y0 + bleedTrim + 2);
      for (let k = 0; k < bleedTrim; k++) {
        if (rowDiffToInterior(y0, refY) > 20) {
          y0++;
          trimmed.top++;
        } else break;
      }
    }
    if (rawBox.y1 < H - 2 && y1 - y0 > 32) {
      const refY = Math.max(y0, y1 - 1 - bleedTrim - 2);
      for (let k = 0; k < bleedTrim; k++) {
        if (rowDiffToInterior(y1 - 1, refY) > 20) {
          y1--;
          trimmed.bottom++;
        } else break;
      }
    }
    if (rawBox.x0 > 2 && x1 - x0 > 32) {
      const refX = Math.min(x1 - 1, x0 + bleedTrim + 2);
      for (let k = 0; k < bleedTrim; k++) {
        if (colDiffToInterior(x0, refX) > 20) {
          x0++;
          trimmed.left++;
        } else break;
      }
    }
    if (rawBox.x1 < W - 2 && x1 - x0 > 32) {
      const refX = Math.max(x0, x1 - 1 - bleedTrim - 2);
      for (let k = 0; k < bleedTrim; k++) {
        if (colDiffToInterior(x1 - 1, refX) > 20) {
          x1--;
          trimmed.right++;
        } else break;
      }
    }
  }

  return {
    id: `slice-${rawBox.row}-${rawBox.col}`,
    index: 0,
    row: rawBox.row,
    col: rawBox.col,
    x: x0,
    y: y0,
    width: Math.max(1, x1 - x0),
    height: Math.max(1, y1 - y0),
    confidence: Math.min(1, Number((rawBox.conf / 150).toFixed(3))),
    strategy: rawBox.isGutter ? 'gutter' : 'xycut',
    trimmedEdges: trimmed,
  };
}

export function detectRecursiveXYCut(
  img: RawImage,
  opts: Partial<SplitOptions> = {},
  precomputedBg?: { rgb: [number, number, number, number]; isTransparent: boolean; borderUniformity: number },
): { boxes: SplitBox[]; hProfile: number[]; vProfile: number[]; bg: { rgb: [number, number, number, number]; isTransparent: boolean; borderUniformity: number } } {
  const o = { ...DEFAULT_OPTIONS, ...opts };
  const minDim = Math.max(24, Math.round(Math.min(img.width, img.height) * clamp(o.minRegionRatio, 0.04, 0.35)));
  // Sensitivity 0.65 -> threshold ~ 73 (clearly separates true seams >100 from internal lines <40)
  const threshold = 125 - clamp(o.sensitivity, 0.1, 1.0) * 80;
  const bg = precomputedBg ?? estimateBorderBackground(img);
  const bgUniform = bg.borderUniformity > 0.72;

  const rootH = profileHorizontalSeams(img, 0, 0, img.width, img.height, minDim, bg.rgb, bgUniform, bg.isTransparent);
  const rootV = profileVerticalSeams(img, 0, 0, img.width, img.height, minDim, bg.rgb, bgUniform, bg.isTransparent);

  interface RawRegion {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    row: number;
    col: number;
    conf: number;
    isGutter: boolean;
  }

  const leaves: RawRegion[] = [];

  function recurse(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    depth: number,
    rowHint: number,
    colHint: number,
    parentConf: number,
    parentGutter: boolean,
  ) {
    const rw = x1 - x0;
    const rh = y1 - y0;
    if (depth >= 4 || (rw < minDim * 2 && rh < minDim * 2)) {
      leaves.push({ x0, y0, x1, y1, row: rowHint, col: colHint, conf: parentConf, isGutter: parentGutter });
      return;
    }

    const hCands =
      depth === 0 && x0 === 0 && x1 === img.width && y0 === 0 && y1 === img.height
        ? rootH
        : profileHorizontalSeams(img, x0, y0, x1, y1, minDim, bg.rgb, bgUniform, bg.isTransparent);
    const vCands =
      depth === 0 && x0 === 0 && x1 === img.width && y0 === 0 && y1 === img.height
        ? rootV
        : profileVerticalSeams(img, x0, y0, x1, y1, minDim, bg.rgb, bgUniform, bg.isTransparent);

    const hPeaks = pickPeaksNMS(hCands, threshold, minDim);
    const vPeaks = pickPeaksNMS(vCands, threshold, minDim);

    if (hPeaks.length === 0 && vPeaks.length === 0) {
      leaves.push({ x0, y0, x1, y1, row: rowHint, col: colHint, conf: parentConf, isGutter: parentGutter });
      return;
    }

    const maxH = hPeaks.reduce((m, p) => Math.max(m, p.score), 0);
    const maxV = vPeaks.reduce((m, p) => Math.max(m, p.score), 0);
    const chooseH = hPeaks.length > 0 && (vPeaks.length === 0 || maxH >= maxV * 0.72);

    if (chooseH) {
      const bounds = [y0, ...hPeaks.map((p) => p.pos), y1];
      for (let i = 0; i < bounds.length - 1; i++) {
        const conf = i === 0 ? hPeaks[0].score : hPeaks[i - 1].score;
        const g = i === 0 ? hPeaks[0].isGutter : hPeaks[i - 1].isGutter;
        recurse(x0, bounds[i], x1, bounds[i + 1], depth + 1, rowHint * 10 + i, colHint, Math.max(parentConf, conf), parentGutter || g);
      }
    } else {
      const bounds = [x0, ...vPeaks.map((p) => p.pos), x1];
      for (let i = 0; i < bounds.length - 1; i++) {
        const conf = i === 0 ? vPeaks[0].score : vPeaks[i - 1].score;
        const g = i === 0 ? vPeaks[0].isGutter : vPeaks[i - 1].isGutter;
        recurse(bounds[i], y0, bounds[i + 1], y1, depth + 1, rowHint, colHint * 10 + i, Math.max(parentConf, conf), parentGutter || g);
      }
    }
  }

  recurse(0, 0, img.width, img.height, 0, 0, 0, 120, false);

  // Sibling Grid-Line Propagation: zero-allocation direct typed-array scan
  const data = img.data;
  const imgW = img.width;
  const ch = img.channels;
  const rowStride = imgW * ch;
  let propagated = true;
  let propIter = 0;
  while (propagated && propIter < 3) {
    propagated = false;
    propIter++;
    const hSplitLines = Array.from(
      new Set(
        leaves
          .flatMap((l) => [l.y0, l.y1])
          .filter((y) => y > minDim && y < img.height - minDim),
      ),
    );
    const vSplitLines = Array.from(
      new Set(
        leaves
          .flatMap((l) => [l.x0, l.x1])
          .filter((x) => x > minDim && x < img.width - minDim),
      ),
    );

    for (let idx = 0; idx < leaves.length; idx++) {
      const leaf = leaves[idx];
      let splitDone = false;
      for (const yCut of hSplitLines) {
        if (yCut - leaf.y0 >= minDim && leaf.y1 - yCut >= minDim) {
          let bestY = -1;
          let bestScore = 0;
          const stepX = Math.max(1, Math.floor((leaf.x1 - leaf.x0) / 200));
          const xStepBytes = stepX * ch;
          const yLo = Math.max(leaf.y0 + minDim, yCut - 12);
          const yHi = Math.min(leaf.y1 - minDim, yCut + 12);
          for (let y = yLo; y <= yHi; y++) {
            let step = 0, line = 0, smooth = 0, active = 0, n = 0;
            let prevR = 0, prevG = 0, prevB = 0;
            let i1 = (y * imgW + leaf.x0 + 4) * ch;
            const iEnd = (y * imgW + leaf.x1 - 4) * ch;
            while (i1 < iEnd) {
              const i0 = i1 - rowStride;
              const i2 = i1 + rowStride;
              const r0 = data[i0], g0 = data[i0 + 1], b0 = data[i0 + 2];
              const r1 = data[i1], g1 = data[i1 + 1], b1 = data[i1 + 2];
              const r2 = data[i2], g2 = data[i2 + 1], b2 = data[i2 + 2];
              const ds = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);
              const dl = (Math.abs(2 * r1 - r0 - r2) + Math.abs(2 * g1 - g0 - g2) + Math.abs(2 * b1 - b0 - b2)) * 0.5;
              step += ds;
              line += dl;
              if (ds + dl > 6) active++;
              if (n > 0) smooth += Math.abs(r1 - prevR) + Math.abs(g1 - prevG) + Math.abs(b1 - prevB);
              prevR = r1; prevG = g1; prevB = b1;
              n++;
              i1 += xStepBytes;
            }
            const cov = active / Math.max(1, n);
            const avgRaw = (step + line) / Math.max(1, n);
            const avgSmooth = smooth / Math.max(1, n - 1);
            if (cov >= 0.78 && avgRaw >= 10 && avgSmooth < 2.2) {
              const sc = (avgRaw * cov) / (1 + avgSmooth * 0.2);
              if (sc > bestScore) {
                bestScore = sc;
                bestY = y;
              }
            }
          }
          if (bestY > 0) {
            leaves.splice(
              idx,
              1,
              { ...leaf, y1: bestY, conf: Math.max(leaf.conf, bestScore) },
              { ...leaf, y0: bestY, conf: Math.max(leaf.conf, bestScore) },
            );
            propagated = true;
            splitDone = true;
            break;
          }
        }
      }
      if (splitDone) break;

      for (const xCut of vSplitLines) {
        if (xCut - leaf.x0 >= minDim && leaf.x1 - xCut >= minDim) {
          let bestX = -1;
          let bestScore = 0;
          const stepY = Math.max(1, Math.floor((leaf.y1 - leaf.y0) / 200));
          const yStepBytes = stepY * rowStride;
          const xLo = Math.max(leaf.x0 + minDim, xCut - 12);
          const xHi = Math.min(leaf.x1 - minDim, xCut + 12);
          for (let x = xLo; x <= xHi; x++) {
            let step = 0, line = 0, smooth = 0, active = 0, n = 0;
            let prevR = 0, prevG = 0, prevB = 0;
            let i1 = ((leaf.y0 + 4) * imgW + x) * ch;
            const iEnd = ((leaf.y1 - 4) * imgW + x) * ch;
            while (i1 < iEnd) {
              const i0 = i1 - ch;
              const i2 = i1 + ch;
              const r0 = data[i0], g0 = data[i0 + 1], b0 = data[i0 + 2];
              const r1 = data[i1], g1 = data[i1 + 1], b1 = data[i1 + 2];
              const r2 = data[i2], g2 = data[i2 + 1], b2 = data[i2 + 2];
              const ds = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);
              const dl = (Math.abs(2 * r1 - r0 - r2) + Math.abs(2 * g1 - g0 - g2) + Math.abs(2 * b1 - b0 - b2)) * 0.5;
              step += ds;
              line += dl;
              if (ds + dl > 6) active++;
              if (n > 0) smooth += Math.abs(r1 - prevR) + Math.abs(g1 - prevG) + Math.abs(b1 - prevB);
              prevR = r1; prevG = g1; prevB = b1;
              n++;
              i1 += yStepBytes;
            }
            const cov = active / Math.max(1, n);
            const avgRaw = (step + line) / Math.max(1, n);
            const avgSmooth = smooth / Math.max(1, n - 1);
            if (cov >= 0.78 && avgRaw >= 10 && avgSmooth < 2.2) {
              const sc = (avgRaw * cov) / (1 + avgSmooth * 0.2);
              if (sc > bestScore) {
                bestScore = sc;
                bestX = x;
              }
            }
          }
          if (bestX > 0) {
            leaves.splice(
              idx,
              1,
              { ...leaf, x1: bestX, conf: Math.max(leaf.conf, bestScore) },
              { ...leaf, x0: bestX, conf: Math.max(leaf.conf, bestScore) },
            );
            propagated = true;
            break;
          }
        }
      }
      if (propagated) break;
    }
  }

  const nonEmptyLeaves = bgUniform
    ? leaves.filter((l) => !isRegionEmptyBackground(img, l.x0, l.y0, l.x1, l.y1, bg.rgb, bg.isTransparent))
    : leaves;

  const finalLeaves = nonEmptyLeaves.length > 0 ? nonEmptyLeaves : leaves;

  finalLeaves.sort((a, b) => {
    const rowTol = Math.min(a.y1 - a.y0, b.y1 - b.y0) * 0.35;
    if (Math.abs(a.y0 - b.y0) > rowTol) return a.y0 - b.y0;
    return a.x0 - b.x0;
  });

  let currentRow = 0;
  let currentCol = 0;
  let lastY = finalLeaves.length > 0 ? finalLeaves[0].y0 : 0;
  const boxes = finalLeaves.map((leaf, idx) => {
    if (idx > 0 && Math.abs(leaf.y0 - lastY) > (leaf.y1 - leaf.y0) * 0.35) {
      currentRow++;
      currentCol = 0;
      lastY = leaf.y0;
    } else if (idx > 0) {
      currentCol++;
    }
    const refined = refineAndTrimCell(
      img,
      { ...leaf, row: currentRow, col: currentCol },
      o.bleedTrim,
      bg.rgb,
      bgUniform,
      bg.isTransparent,
    );
    refined.index = idx + 1;
    refined.id = `slice-${idx + 1}`;
    return refined;
  });

  const downsample = (cands: SeamCandidate[], bins = 180): number[] => {
    if (cands.length === 0) return [];
    const out: number[] = [];
    const step = cands.length / bins;
    for (let i = 0; i < bins; i++) {
      const lo = Math.floor(i * step);
      const hi = Math.max(lo + 1, Math.floor((i + 1) * step));
      let mx = 0;
      for (let j = lo; j < hi && j < cands.length; j++) {
        if (cands[j].score > mx) mx = cands[j].score;
      }
      out.push(Number(mx.toFixed(1)));
    }
    return out;
  };

  return {
    boxes,
    hProfile: downsample(rootH),
    vProfile: downsample(rootV),
    bg,
  };
}

export function detectContours(
  img: RawImage,
  opts: Partial<SplitOptions> = {},
  precomputedBg?: { rgb: [number, number, number, number]; isTransparent: boolean; borderUniformity: number },
): SplitBox[] {
  const o = { ...DEFAULT_OPTIONS, ...opts };
  const bg = precomputedBg ?? estimateBorderBackground(img);
  const w = img.width;
  const h = img.height;
  const data = img.data;
  const c = img.channels;
  const bgR = bg.rgb[0], bgG = bg.rgb[1], bgB = bg.rgb[2];

  const scale = Math.max(1, Math.ceil(Math.max(w, h) / 720));
  const sw = Math.ceil(w / scale);
  const sh = Math.ceil(h / scale);
  const mask = new Uint8Array(sw * sh);
  const colorThresh = 32 - clamp(o.sensitivity, 0.1, 1.0) * 18;

  for (let sy = 0; sy < sh; sy++) {
    const y = Math.min(h - 1, sy * scale);
    const rowOff = sy * sw;
    const imgRowOff = y * w;
    for (let sx = 0; sx < sw; sx++) {
      const x = Math.min(w - 1, sx * scale);
      const idx = (imgRowOff + x) * c;
      if (bg.isTransparent) {
        mask[rowOff + sx] = c >= 4 && data[idx + 3] > 24 ? 1 : 0;
      } else {
        const d = Math.abs(data[idx] - bgR) + Math.abs(data[idx + 1] - bgG) + Math.abs(data[idx + 2] - bgB);
        mask[rowOff + sx] = d > colorThresh ? 1 : 0;
      }
    }
  }

  // Separable O(sw * sh) morphological box dilation (replaces O(sw * sh * gap^2) 2D loop)
  const gap = Math.max(1, Math.round((o.contourGapPx ?? 8) / scale));
  const horiz = new Uint8Array(sw * sh);
  for (let y = 0; y < sh; y++) {
    const off = y * sw;
    let rem = 0;
    // Forward & backward sweep for exact 1D dilation of radius `gap`
    for (let x = 0; x < sw; x++) {
      if (mask[off + x]) rem = gap + 1;
      if (rem > 0) {
        horiz[off + x] = 1;
        rem--;
      }
    }
    rem = 0;
    for (let x = sw - 1; x >= 0; x--) {
      if (mask[off + x]) rem = gap + 1;
      if (rem > 0) {
        horiz[off + x] = 1;
        rem--;
      }
    }
  }
  const dilated = new Uint8Array(sw * sh);
  for (let x = 0; x < sw; x++) {
    let rem = 0;
    for (let y = 0; y < sh; y++) {
      if (horiz[y * sw + x]) rem = gap + 1;
      if (rem > 0) {
        dilated[y * sw + x] = 1;
        rem--;
      }
    }
    rem = 0;
    for (let y = sh - 1; y >= 0; y--) {
      if (horiz[y * sw + x]) rem = gap + 1;
      if (rem > 0) {
        dilated[y * sw + x] = 1;
        rem--;
      }
    }
  }

  const visited = new Uint8Array(sw * sh);
  const minAreaScaled = Math.max(16, sw * sh * Math.pow(clamp(o.minRegionRatio, 0.02, 0.3), 2) * 0.45);
  const rawBoxes: Array<{ x0: number; y0: number; x1: number; y1: number; area: number }> = [];
  const queue = new Int32Array(sw * sh);

  const isFgPx = (x: number, y: number): boolean => {
    const i = (y * w + x) * c;
    if (bg.isTransparent) return c >= 4 && data[i + 3] > 24;
    return Math.abs(data[i] - bgR) + Math.abs(data[i + 1] - bgG) + Math.abs(data[i + 2] - bgB) > colorThresh;
  };
  const rowHasFg = (y: number, xa: number, xb: number): boolean => {
    const step = Math.max(1, Math.floor((xb - xa) / 160));
    for (let x = xa; x < xb; x += step) {
      if (isFgPx(x, y)) return true;
    }
    return false;
  };
  const colHasFg = (x: number, ya: number, yb: number): boolean => {
    const step = Math.max(1, Math.floor((yb - ya) / 160));
    for (let y = ya; y < yb; y += step) {
      if (isFgPx(x, y)) return true;
    }
    return false;
  };

  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const start = y * sw + x;
      if (!dilated[start] || visited[start]) continue;
      let head = 0;
      let tail = 0;
      queue[tail++] = start;
      visited[start] = 1;
      let minX = sw, minY = sh, maxX = 0, maxY = 0, fgCount = 0;

      while (head < tail) {
        const cur = queue[head++];
        const cx = cur % sw;
        const cy = (cur / sw) | 0;
        if (mask[cur]) {
          fgCount++;
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;
        }
        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= sh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx;
            if (nx < 0 || nx >= sw) continue;
            const nIdx = ny * sw + nx;
            if (dilated[nIdx] && !visited[nIdx]) {
              visited[nIdx] = 1;
              queue[tail++] = nIdx;
            }
          }
        }
      }

      if (fgCount >= minAreaScaled && maxX >= minX && maxY >= minY) {
        // 1:1 full-resolution tight edge refinement around downscaled bounding box
        let x0 = Math.max(0, minX * scale - scale);
        let y0 = Math.max(0, minY * scale - scale);
        let x1 = Math.min(w, (maxX + 1) * scale + scale);
        let y1 = Math.min(h, (maxY + 1) * scale + scale);

        while (y0 < y1 - 8 && !rowHasFg(y0, x0, x1)) y0++;
        while (y1 > y0 + 8 && !rowHasFg(y1 - 1, x0, x1)) y1--;
        while (x0 < x1 - 8 && !colHasFg(x0, y0, y1)) x0++;
        while (x1 > x0 + 8 && !colHasFg(x1 - 1, y0, y1)) x1--;

        // Allow 1px expansion back outward if foreground touches immediate neighbor
        if (y0 > 0 && rowHasFg(y0 - 1, x0, x1)) y0--;
        if (y1 < h && rowHasFg(y1, x0, x1)) y1++;
        if (x0 > 0 && colHasFg(x0 - 1, y0, y1)) x0--;
        if (x1 < w && colHasFg(x1, y0, y1)) x1++;

        rawBoxes.push({
          x0,
          y0,
          x1,
          y1,
          area: fgCount * scale * scale,
        });
      }
    }
  }

  rawBoxes.sort((a, b) => {
    const tol = Math.min(a.y1 - a.y0, b.y1 - b.y0) * 0.4;
    if (Math.abs(a.y0 - b.y0) > tol) return a.y0 - b.y0;
    return a.x0 - b.x0;
  });

  return rawBoxes.map((b, idx) => ({
    id: `slice-${idx + 1}`,
    index: idx + 1,
    row: idx,
    col: 0,
    x: b.x0,
    y: b.y0,
    width: b.x1 - b.x0,
    height: b.y1 - b.y0,
    confidence: 0.98,
    strategy: 'contour',
    trimmedEdges: { top: 0, right: 0, bottom: 0, left: 0 },
  }));
}

export function detectUniformGrid(
  img: RawImage,
  opts: Partial<SplitOptions> = {},
): SplitBox[] {
  const o = { ...DEFAULT_OPTIONS, ...opts };
  const rows = Math.max(1, Math.min(20, Math.round(o.rows ?? 3)));
  const cols = Math.max(1, Math.min(20, Math.round(o.cols ?? 3)));
  const gutter = Math.max(0, Math.round(o.gutterPx ?? 0));
  const halfG = Math.floor(gutter / 2);

  const boxes: SplitBox[] = [];
  let idx = 1;
  for (let r = 0; r < rows; r++) {
    const y0 = Math.round((r * img.height) / rows) + (r > 0 ? halfG : 0);
    const y1 = Math.round(((r + 1) * img.height) / rows) - (r < rows - 1 ? gutter - halfG : 0);
    for (let c = 0; c < cols; c++) {
      const x0 = Math.round((c * img.width) / cols) + (c > 0 ? halfG : 0);
      const x1 = Math.round(((c + 1) * img.width) / cols) - (c < cols - 1 ? gutter - halfG : 0);
      boxes.push({
        id: `slice-${idx}`,
        index: idx,
        row: r,
        col: c,
        x: x0,
        y: y0,
        width: Math.max(1, x1 - x0),
        height: Math.max(1, y1 - y0),
        confidence: 1,
        strategy: 'grid',
        trimmedEdges: { top: 0, right: 0, bottom: 0, left: 0 },
      });
      idx++;
    }
  }
  return boxes;
}

export function analyzeAndSplit(
  img: RawImage,
  opts: Partial<SplitOptions> = {},
): SplitAnalysisResult {
  const t0 = performance.now();
  const o = { ...DEFAULT_OPTIONS, ...opts };

  if (o.mode === 'grid') {
    const boxes = detectUniformGrid(img, o);
    return {
      imageWidth: img.width,
      imageHeight: img.height,
      detectedMode: 'grid',
      boxes,
      hProfile: [],
      vProfile: [],
      elapsedMs: Number((performance.now() - t0).toFixed(2)),
    };
  }

  if (o.mode === 'contour') {
    const boxes = detectContours(img, o);
    return {
      imageWidth: img.width,
      imageHeight: img.height,
      detectedMode: 'contour',
      boxes,
      hProfile: [],
      vProfile: [],
      elapsedMs: Number((performance.now() - t0).toFixed(2)),
    };
  }

  const xyRes = detectRecursiveXYCut(img, o);
  if (o.mode === 'xycut') {
    return {
      imageWidth: img.width,
      imageHeight: img.height,
      detectedMode: 'xycut',
      boxes: xyRes.boxes,
      hProfile: xyRes.hProfile,
      vProfile: xyRes.vProfile,
      elapsedMs: Number((performance.now() - t0).toFixed(2)),
    };
  }

  if (xyRes.bg.borderUniformity > 0.72) {
    const contourBoxes = detectContours(img, o, xyRes.bg);
    const contourAreaSum = contourBoxes.reduce((s, b) => s + b.width * b.height, 0);
    const totalArea = img.width * img.height;
    if (
      contourBoxes.length >= 2 &&
      (xyRes.boxes.length <= 1 || (contourAreaSum < totalArea * 0.65 && contourBoxes.length >= xyRes.boxes.length))
    ) {
      return {
        imageWidth: img.width,
        imageHeight: img.height,
        detectedMode: 'contour',
        boxes: contourBoxes,
        hProfile: xyRes.hProfile,
        vProfile: xyRes.vProfile,
        elapsedMs: Number((performance.now() - t0).toFixed(2)),
      };
    }
  }

  return {
    imageWidth: img.width,
    imageHeight: img.height,
    detectedMode: 'xycut',
    boxes: xyRes.boxes,
    hProfile: xyRes.hProfile,
    vProfile: xyRes.vProfile,
    elapsedMs: Number((performance.now() - t0).toFixed(2)),
  };
}
