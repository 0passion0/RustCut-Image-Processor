/**
 * Pure ES Module parity engine for SmartSplitter v0.2.4:
 * - Directional Step + 1px Hairline Energy Profiling
 * - Non-Linear Coverage (C^2.4) & Tangential Smoothness Penalty
 * - Gutter Valley Detection & Contiguous Run Collapsing
 * - Recursive XY-Cut BSP + Sibling Grid-Line Propagation (+-12px)
 * - Per-Cell +-8px Discontinuity Snap & Bleed Decontamination
 * - Separable O(N) Morphological Closing + Connected Components + 1:1 Edge Tightening
 * - Uniform Dimension Normalization (pad / resize / crop)
 */

const COV_POW_LUT = new Float32Array(257);
for (let i = 0; i <= 256; i++) {
  COV_POW_LUT[i] = Math.pow(i / 256, 2.4);
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function colorDistAt(data, idxA, idxB) {
  return (
    Math.abs(data[idxA] - data[idxB]) +
    Math.abs(data[idxA + 1] - data[idxB + 1]) +
    Math.abs(data[idxA + 2] - data[idxB + 2])
  );
}

function colorDistRgb(data, idx, bg) {
  return (
    Math.abs(data[idx] - bg[0]) +
    Math.abs(data[idx + 1] - bg[1]) +
    Math.abs(data[idx + 2] - bg[2])
  );
}

export function estimateBorderBg(data, w, h) {
  const sx = Math.max(1, Math.floor(w / 80));
  const sy = Math.max(1, Math.floor(h / 80));
  const samples = [];

  for (let x = 0; x < w; x += sx) {
    samples.push(x * 4);
    samples.push(((h - 1) * w + x) * 4);
  }
  for (let y = 0; y < h; y += sy) {
    samples.push(y * w * 4);
    samples.push((y * w + (w - 1)) * 4);
  }

  let transparent = 0;
  for (let i = 0; i < samples.length; i++) {
    if (data[samples[i] + 3] < 32) transparent++;
  }
  const n = Math.max(1, samples.length);
  if (transparent / n > 0.5) {
    return { bgRgb: [0, 0, 0], isTransparent: true, uniformity: transparent / n };
  }

  const bins = new Map();
  for (let i = 0; i < samples.length; i++) {
    const p = samples[i];
    const r = data[p];
    const g = data[p + 1];
    const b = data[p + 2];
    const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
    let entry = bins.get(key);
    if (!entry) {
      entry = [0, 0, 0, 0];
      bins.set(key, entry);
    }
    entry[0] += 1;
    entry[1] += r;
    entry[2] += g;
    entry[3] += b;
  }

  let best = [1, 255, 255, 255];
  for (const v of bins.values()) {
    if (v[0] > best[0]) best = v;
  }
  const cnt = Math.max(1, best[0]);
  return {
    bgRgb: [Math.round(best[1] / cnt), Math.round(best[2] / cnt), Math.round(best[3] / cnt)],
    isTransparent: false,
    uniformity: best[0] / n,
  };
}

function collapseGutterRuns(cands) {
  let i = 0;
  while (i < cands.length) {
    if (!cands[i].isGutter) {
      i++;
      continue;
    }
    let j = i;
    while (j + 1 < cands.length && cands[j + 1].isGutter && cands[j + 1].pos === cands[j].pos + 1) {
      j++;
    }
    const mid = (i + j) >> 1;
    let maxScore = 0;
    for (let k = i; k <= j; k++) {
      if (cands[k].score > maxScore) maxScore = cands[k].score;
    }
    for (let k = i; k <= j; k++) {
      if (k === mid) {
        cands[k].score = maxScore;
      } else {
        cands[k].score = 0;
        cands[k].isGutter = false;
      }
    }
    i = j + 1;
  }
}

function profileHorizontalSeams(data, w, x0, y0, x1, y1, minDim, bgRgb, bgUniform, isTrans) {
  const spanW = x1 - x0;
  const spanH = y1 - y0;
  if (spanH <= minDim * 2 || spanW < 8) return [];

  const stepX = Math.max(1, Math.floor(spanW / 200));
  const sampleCount = Math.ceil(spanW / stepX);
  const stride = w * 4;

  const avgStep = new Float32Array(spanH);
  const avgLine = new Float32Array(spanH);
  const avgSmooth = new Float32Array(spanH);
  const bgMatch = new Float32Array(spanH);
  const lastMarked = new Int32Array(sampleCount).fill(-10);
  const activeCount = new Uint16Array(spanH);

  for (let ry = 1; ry < spanH - 1; ry++) {
    const y = y0 + ry;
    const rowPtr = y * stride;
    const prevPtr = (y - 1) * stride;
    const nextPtr = (y + 1) * stride;
    let sSum = 0;
    let lSum = 0;
    let smSum = 0;
    let bgHits = 0;
    let prevLuma = 0;

    for (let ui = 0, x = x0; ui < sampleCount; ui++, x += stepX) {
      const colByte = x * 4;
      const p1 = rowPtr + colByte;
      const p0 = prevPtr + colByte;
      const p2 = nextPtr + colByte;
      const ds = colorDistAt(data, p1, p2);
      const dl =
        0.5 *
        (Math.abs(2 * data[p1] - data[p0] - data[p2]) +
          Math.abs(2 * data[p1 + 1] - data[p0 + 1] - data[p2 + 1]) +
          Math.abs(2 * data[p1 + 2] - data[p0 + 2] - data[p2 + 2]));
      sSum += ds;
      lSum += dl;

      if (ds + dl > 16) {
        const start = Math.max(lastMarked[ui] + 1, ry - 2);
        const end = Math.min(spanH - 2, ry + 2);
        for (let k = start; k <= end; k++) activeCount[k]++;
        lastMarked[ui] = end;
      }

      const luma = data[p1] + data[p1 + 1] + data[p1 + 2];
      if (ui > 0) smSum += Math.abs(luma - prevLuma);
      prevLuma = luma;

      if (bgUniform) {
        const isBg = isTrans ? data[p1 + 3] < 24 : colorDistRgb(data, p1, bgRgb) < 22;
        if (isBg) bgHits++;
      }
    }

    const sc = Math.max(1, sampleCount);
    avgStep[ry] = sSum / sc;
    avgLine[ry] = lSum / sc;
    avgSmooth[ry] = smSum / Math.max(1, sampleCount - 1);
    bgMatch[ry] = bgHits / sc;
  }

  const cands = [];
  const lo = minDim;
  const hi = spanH - minDim;
  for (let ry = lo; ry <= hi; ry++) {
    const cov = activeCount[ry] / Math.max(1, sampleCount);
    const covIdx = Math.min(256, Math.round(cov * 256));
    const minSm = Math.min(
      avgSmooth[Math.max(1, ry - 1)],
      avgSmooth[ry],
      avgSmooth[Math.min(spanH - 2, ry + 1)]
    );
    const seamScore = ((avgStep[ry] + 0.7 * avgLine[ry]) * COV_POW_LUT[covIdx]) / (1 + minSm * 0.12);

    let gutterScore = 0;
    if (bgUniform && avgSmooth[ry] < 2.5 && bgMatch[ry] > 0.95) {
      let hasBefore = false;
      for (let k = 2; k < ry; k += 3) {
        if (bgMatch[k] < 0.9) {
          hasBefore = true;
          break;
        }
      }
      let hasAfter = false;
      for (let k = ry + 3; k < spanH - 2; k += 3) {
        if (bgMatch[k] < 0.9) {
          hasAfter = true;
          break;
        }
      }
      if (hasBefore && hasAfter) {
        gutterScore = 110 / (1 + avgSmooth[ry]);
      }
    }
    const isGutter = gutterScore > seamScore;
    cands.push({
      pos: y0 + ry,
      score: Math.max(seamScore, gutterScore),
      coverage: cov,
      isGutter,
    });
  }

  collapseGutterRuns(cands);
  return cands;
}

function profileVerticalSeams(data, w, x0, y0, x1, y1, minDim, bgRgb, bgUniform, isTrans) {
  const spanW = x1 - x0;
  const spanH = y1 - y0;
  if (spanW <= minDim * 2 || spanH < 8) return [];

  const stepY = Math.max(1, Math.floor(spanH / 200));
  const sampleCount = Math.ceil(spanH / stepY);
  const stride = w * 4;

  const stepSum = new Float32Array(spanW);
  const lineSum = new Float32Array(spanW);
  const smoothSum = new Float32Array(spanW);
  const bgHitSum = new Uint32Array(spanW);
  const prevLuma = new Int32Array(spanW);
  const activeCount = new Uint16Array(spanW);

  // Row-major traversal so memory reads stay cache-friendly
  for (let ui = 0, y = y0; ui < sampleCount; ui++, y += stepY) {
    const rowPtr = y * stride;
    let lastX = -10;
    for (let rx = 1; rx < spanW - 1; rx++) {
      const p1 = rowPtr + (x0 + rx) * 4;
      const p0 = p1 - 4;
      const p2 = p1 + 4;
      const ds = colorDistAt(data, p1, p2);
      const dl =
        0.5 *
        (Math.abs(2 * data[p1] - data[p0] - data[p2]) +
          Math.abs(2 * data[p1 + 1] - data[p0 + 1] - data[p2 + 1]) +
          Math.abs(2 * data[p1 + 2] - data[p0 + 2] - data[p2 + 2]));
      stepSum[rx] += ds;
      lineSum[rx] += dl;

      if (ds + dl > 16) {
        const start = Math.max(lastX + 1, rx - 2);
        const end = Math.min(spanW - 2, rx + 2);
        for (let k = start; k <= end; k++) activeCount[k]++;
        lastX = end;
      }

      const luma = data[p1] + data[p1 + 1] + data[p1 + 2];
      if (ui > 0) smoothSum[rx] += Math.abs(luma - prevLuma[rx]);
      prevLuma[rx] = luma;

      if (bgUniform) {
        const isBg = isTrans ? data[p1 + 3] < 24 : colorDistRgb(data, p1, bgRgb) < 22;
        if (isBg) bgHitSum[rx]++;
      }
    }
  }

  const sc = Math.max(1, sampleCount);
  const smDenom = Math.max(1, sampleCount - 1);
  const avgSmooth = new Float32Array(spanW);
  const bgMatch = new Float32Array(spanW);
  for (let rx = 1; rx < spanW - 1; rx++) {
    avgSmooth[rx] = smoothSum[rx] / smDenom;
    bgMatch[rx] = bgHitSum[rx] / sc;
  }

  const cands = [];
  const lo = minDim;
  const hi = spanW - minDim;
  for (let rx = lo; rx <= hi; rx++) {
    const cov = activeCount[rx] / sc;
    const covIdx = Math.min(256, Math.round(cov * 256));
    const minSm = Math.min(
      avgSmooth[Math.max(1, rx - 1)],
      avgSmooth[rx],
      avgSmooth[Math.min(spanW - 2, rx + 1)]
    );
    const seamScore = (((stepSum[rx] / sc) + 0.7 * (lineSum[rx] / sc)) * COV_POW_LUT[covIdx]) / (1 + minSm * 0.12);

    let gutterScore = 0;
    if (bgUniform && avgSmooth[rx] < 2.5 && bgMatch[rx] > 0.95) {
      let hasBefore = false;
      for (let k = 2; k < rx; k += 3) {
        if (bgMatch[k] < 0.9) {
          hasBefore = true;
          break;
        }
      }
      let hasAfter = false;
      for (let k = rx + 3; k < spanW - 2; k += 3) {
        if (bgMatch[k] < 0.9) {
          hasAfter = true;
          break;
        }
      }
      if (hasBefore && hasAfter) {
        gutterScore = 110 / (1 + avgSmooth[rx]);
      }
    }
    const isGutter = gutterScore > seamScore;
    cands.push({
      pos: x0 + rx,
      score: Math.max(seamScore, gutterScore),
      coverage: cov,
      isGutter,
    });
  }

  collapseGutterRuns(cands);
  return cands;
}

function pickPeaksNms(cands, threshold, minGap) {
  const filtered = cands
    .filter((c) => c.score >= threshold && (c.isGutter || c.coverage >= 0.72))
    .sort((a, b) => b.score - a.score);
  const picked = [];
  for (const cand of filtered) {
    if (picked.every((p) => Math.abs(p.pos - cand.pos) >= minGap)) {
      picked.push(cand);
    }
  }
  picked.sort((a, b) => a.pos - b.pos);
  return picked;
}

function hasForeground(data, w, x0, y0, x1, y1, bgRgb, isTrans) {
  const stepX = Math.max(1, Math.floor((x1 - x0) / 32));
  const stepY = Math.max(1, Math.floor((y1 - y0) / 32));
  const stride = w * 4;
  let total = 0;
  let fg = 0;
  for (let y = y0 + 2; y < Math.max(y0 + 3, y1 - 2); y += stepY) {
    const rowPtr = y * stride;
    for (let x = x0 + 2; x < Math.max(x0 + 3, x1 - 2); x += stepX) {
      const p = rowPtr + x * 4;
      total++;
      if (isTrans ? data[p + 3] >= 24 : colorDistRgb(data, p, bgRgb) > 20) fg++;
    }
  }
  return total === 0 || fg / total >= 0.03;
}

function refineAndTrimCell(data, w, h, cell, bleedTrim, bgRgb, bgUniform, isTrans) {
  let { x0, y0, x1, y1, isGutter } = cell;
  const origX0 = x0;
  const origY0 = y0;
  const origX1 = x1;
  const origY1 = y1;
  const stride = w * 4;

  if (!isGutter) {
    if (y0 > 2 && x1 - x0 > 8) {
      const sx = Math.max(1, Math.floor((x1 - x0) / 120));
      let bestY = y0;
      let bestE = -1;
      for (let y = Math.max(1, y0 - 8); y <= Math.min(y1 - 20, y0 + 8); y++) {
        let e = 0;
        let n = 0;
        for (let x = x0 + 4; x < x1 - 4; x += sx) {
          const p0 = (y - 1) * stride + x * 4;
          const p1 = y * stride + x * 4;
          const p2 = (y + 1) * stride + x * 4;
          e +=
            colorDistAt(data, p1, p2) +
            0.5 *
              (Math.abs(2 * data[p1] - data[p0] - data[p2]) +
                Math.abs(2 * data[p1 + 1] - data[p0 + 1] - data[p2 + 1]) +
                Math.abs(2 * data[p1 + 2] - data[p0 + 2] - data[p2 + 2]));
          n++;
        }
        const avg = n > 0 ? e / n : 0;
        if (avg > bestE) {
          bestE = avg;
          bestY = y;
        }
      }
      if (bestE > 15) y0 = bestY + 1;
    }

    if (y1 < h - 2 && x1 - x0 > 8) {
      const sx = Math.max(1, Math.floor((x1 - x0) / 120));
      let bestY = y1;
      let bestE = -1;
      for (let y = Math.max(y0 + 20, y1 - 8); y <= Math.min(h - 2, y1 + 8); y++) {
        let e = 0;
        let n = 0;
        for (let x = x0 + 4; x < x1 - 4; x += sx) {
          const p0 = (y - 1) * stride + x * 4;
          const p1 = y * stride + x * 4;
          const p2 = (y + 1) * stride + x * 4;
          e +=
            colorDistAt(data, p0, p1) +
            0.5 *
              (Math.abs(2 * data[p1] - data[p0] - data[p2]) +
                Math.abs(2 * data[p1 + 1] - data[p0 + 1] - data[p2 + 1]) +
                Math.abs(2 * data[p1 + 2] - data[p0 + 2] - data[p2 + 2]));
          n++;
        }
        const avg = n > 0 ? e / n : 0;
        if (avg > bestE) {
          bestE = avg;
          bestY = y;
        }
      }
      if (bestE > 15) y1 = bestY;
    }

    if (x0 > 2 && y1 - y0 > 8) {
      const sy = Math.max(1, Math.floor((y1 - y0) / 120));
      let bestX = x0;
      let bestE = -1;
      for (let x = Math.max(1, x0 - 8); x <= Math.min(x1 - 20, x0 + 8); x++) {
        let e = 0;
        let n = 0;
        for (let y = y0 + 4; y < y1 - 4; y += sy) {
          const p1 = y * stride + x * 4;
          const p0 = p1 - 4;
          const p2 = p1 + 4;
          e +=
            colorDistAt(data, p1, p2) +
            0.5 *
              (Math.abs(2 * data[p1] - data[p0] - data[p2]) +
                Math.abs(2 * data[p1 + 1] - data[p0 + 1] - data[p2 + 1]) +
                Math.abs(2 * data[p1 + 2] - data[p0 + 2] - data[p2 + 2]));
          n++;
        }
        const avg = n > 0 ? e / n : 0;
        if (avg > bestE) {
          bestE = avg;
          bestX = x;
        }
      }
      if (bestE > 15) x0 = bestX + 1;
    }

    if (x1 < w - 2 && y1 - y0 > 8) {
      const sy = Math.max(1, Math.floor((y1 - y0) / 120));
      let bestX = x1;
      let bestE = -1;
      for (let x = Math.max(x0 + 20, x1 - 8); x <= Math.min(w - 2, x1 + 8); x++) {
        let e = 0;
        let n = 0;
        for (let y = y0 + 4; y < y1 - 4; y += sy) {
          const p1 = y * stride + x * 4;
          const p0 = p1 - 4;
          const p2 = p1 + 4;
          e +=
            colorDistAt(data, p0, p1) +
            0.5 *
              (Math.abs(2 * data[p1] - data[p0] - data[p2]) +
                Math.abs(2 * data[p1 + 1] - data[p0 + 1] - data[p2 + 1]) +
                Math.abs(2 * data[p1 + 2] - data[p0 + 2] - data[p2 + 2]));
          n++;
        }
        const avg = n > 0 ? e / n : 0;
        if (avg > bestE) {
          bestE = avg;
          bestX = x;
        }
      }
      if (bestE > 15) x1 = bestX;
    }
  }

  let tt = 0;
  let tr = 0;
  let tb = 0;
  let tl = 0;

  if (bgUniform) {
    const isRowBg = (y) => {
      const step = Math.max(1, Math.floor((x1 - x0) / 60));
      let hits = 0;
      let tot = 0;
      for (let x = x0; x < x1; x += step) {
        const p = y * stride + x * 4;
        if (isTrans ? data[p + 3] < 24 : colorDistRgb(data, p, bgRgb) < 22) hits++;
        tot++;
      }
      return tot > 0 && hits / tot > 0.94;
    };
    const isColBg = (x) => {
      const step = Math.max(1, Math.floor((y1 - y0) / 60));
      let hits = 0;
      let tot = 0;
      for (let y = y0; y < y1; y += step) {
        const p = y * stride + x * 4;
        if (isTrans ? data[p + 3] < 24 : colorDistRgb(data, p, bgRgb) < 22) hits++;
        tot++;
      }
      return tot > 0 && hits / tot > 0.94;
    };

    while (y0 < y1 - 24 && isRowBg(y0)) {
      y0++;
      tt++;
    }
    while (y1 > y0 + 24 && isRowBg(y1 - 1)) {
      y1--;
      tb++;
    }
    while (x0 < x1 - 24 && isColBg(x0)) {
      x0++;
      tl++;
    }
    while (x1 > x0 + 24 && isColBg(x1 - 1)) {
      x1--;
      tr++;
    }
  }

  if (bleedTrim > 0 && y1 - y0 > 32 && x1 - x0 > 32) {
    const sx = Math.max(1, Math.floor((x1 - x0) / 80));
    const sy = Math.max(1, Math.floor((y1 - y0) / 80));

    const rowDiff = (y, ry) => {
      let d = 0;
      let n = 0;
      for (let x = x0 + 8; x < x1 - 8; x += sx) {
        d += colorDistAt(data, y * stride + x * 4, ry * stride + x * 4);
        n++;
      }
      return n > 0 ? d / n : 0;
    };
    const colDiff = (x, rx) => {
      let d = 0;
      let n = 0;
      for (let y = y0 + 8; y < y1 - 8; y += sy) {
        d += colorDistAt(data, y * stride + x * 4, y * stride + rx * 4);
        n++;
      }
      return n > 0 ? d / n : 0;
    };

    if (origY0 > 2) {
      const refY = Math.min(y1 - 1, y0 + bleedTrim + 2);
      for (let k = 0; k < bleedTrim; k++) {
        if (rowDiff(y0, refY) > 20) {
          y0++;
          tt++;
        } else break;
      }
    }
    if (origY1 < h - 2) {
      const refY = Math.max(y0, y1 - 1 - bleedTrim - 2);
      for (let k = 0; k < bleedTrim; k++) {
        if (rowDiff(y1 - 1, refY) > 20) {
          y1--;
          tb++;
        } else break;
      }
    }
    if (origX0 > 2) {
      const refX = Math.min(x1 - 1, x0 + bleedTrim + 2);
      for (let k = 0; k < bleedTrim; k++) {
        if (colDiff(x0, refX) > 20) {
          x0++;
          tl++;
        } else break;
      }
    }
    if (origX1 < w - 2) {
      const refX = Math.max(x0, x1 - 1 - bleedTrim - 2);
      for (let k = 0; k < bleedTrim; k++) {
        if (colDiff(x1 - 1, refX) > 20) {
          x1--;
          tr++;
        } else break;
      }
    }
  }

  return {
    x0,
    y0,
    x1,
    y1,
    trimmedEdges: [tt, tr, tb, tl],
  };
}

function detectRecursiveXyCut(data, w, h, sensitivity, minRegionRatio, bleedTrim, bgRgb, bgUniform, isTrans) {
  const minDim = Math.max(24, Math.round(Math.min(w, h) * clamp(minRegionRatio, 0.03, 0.35)));
  const threshold = 125 - clamp(sensitivity, 0.1, 1.0) * 80;
  let leaves = [];

  function recurse(x0, y0, x1, y1, depth, conf, isGutter) {
    if (depth >= 4 || (x1 - x0 < minDim * 2 && y1 - y0 < minDim * 2)) {
      leaves.push({ x0, y0, x1, y1, conf, isGutter });
      return;
    }
    const hCands = profileHorizontalSeams(data, w, x0, y0, x1, y1, minDim, bgRgb, bgUniform, isTrans);
    const vCands = profileVerticalSeams(data, w, x0, y0, x1, y1, minDim, bgRgb, bgUniform, isTrans);
    const hPeaks = pickPeaksNms(hCands, threshold, minDim);
    const vPeaks = pickPeaksNms(vCands, threshold, minDim);

    if (hPeaks.length === 0 && vPeaks.length === 0) {
      leaves.push({ x0, y0, x1, y1, conf, isGutter });
      return;
    }

    const maxH = hPeaks.reduce((m, p) => Math.max(m, p.score), 0);
    const maxV = vPeaks.reduce((m, p) => Math.max(m, p.score), 0);
    if (hPeaks.length > 0 && (vPeaks.length === 0 || maxH >= maxV * 0.72)) {
      const bounds = [y0, ...hPeaks.map((p) => p.pos), y1];
      for (let i = 0; i < bounds.length - 1; i++) {
        const pk = i === 0 ? hPeaks[0] : hPeaks[i - 1];
        recurse(x0, bounds[i], x1, bounds[i + 1], depth + 1, Math.max(conf, pk.score), isGutter || pk.isGutter);
      }
    } else {
      const bounds = [x0, ...vPeaks.map((p) => p.pos), x1];
      for (let i = 0; i < bounds.length - 1; i++) {
        const pk = i === 0 ? vPeaks[0] : vPeaks[i - 1];
        recurse(bounds[i], y0, bounds[i + 1], y1, depth + 1, Math.max(conf, pk.score), isGutter || pk.isGutter);
      }
    }
  }

  recurse(0, 0, w, h, 0, 120, false);

  const stride = w * 4;
  // Sibling Grid-Line Propagation (+-12px)
  for (let pass = 0; pass < 3; pass++) {
    let propagated = false;
    const hSet = new Set();
    const vSet = new Set();
    for (const l of leaves) {
      if (l.y0 > minDim && l.y0 < h - minDim) hSet.add(l.y0);
      if (l.y1 > minDim && l.y1 < h - minDim) hSet.add(l.y1);
      if (l.x0 > minDim && l.x0 < w - minDim) vSet.add(l.x0);
      if (l.x1 > minDim && l.x1 < w - minDim) vSet.add(l.x1);
    }

    for (let idx = 0; idx < leaves.length; idx++) {
      const leaf = leaves[idx];
      let splitDone = false;
      for (const yCut of hSet) {
        if (yCut - leaf.y0 >= minDim && leaf.y1 - yCut >= minDim) {
          const sx = Math.max(1, Math.floor((leaf.x1 - leaf.x0) / 200));
          const yLo = Math.max(leaf.y0 + minDim, yCut - 12);
          const yHi = Math.min(leaf.y1 - minDim, yCut + 12);
          let bestY = -1;
          let bestSc = 0;
          for (let y = yLo; y <= yHi; y++) {
            let stepS = 0;
            let lineS = 0;
            let smoothS = 0;
            let act = 0;
            let n = 0;
            let prev = 0;
            for (let x = leaf.x0 + 4; x < leaf.x1 - 4; x += sx) {
              const p1 = y * stride + x * 4;
              const p0 = p1 - stride;
              const p2 = p1 + stride;
              const ds = colorDistAt(data, p1, p2);
              const dl =
                0.5 *
                (Math.abs(2 * data[p1] - data[p0] - data[p2]) +
                  Math.abs(2 * data[p1 + 1] - data[p0 + 1] - data[p2 + 1]) +
                  Math.abs(2 * data[p1 + 2] - data[p0 + 2] - data[p2 + 2]));
              stepS += ds;
              lineS += dl;
              if (ds + dl > 6) act++;
              const luma = data[p1] + data[p1 + 1] + data[p1 + 2];
              if (n > 0) smoothS += Math.abs(luma - prev);
              prev = luma;
              n++;
            }
            if (n >= 2) {
              const cov = act / n;
              const avgRaw = (stepS + lineS) / n;
              const avgSm = smoothS / Math.max(1, n - 1);
              if (cov >= 0.78 && avgRaw >= 10 && avgSm < 2.2) {
                const sc = (avgRaw * cov) / (1 + avgSm * 0.2);
                if (sc > bestSc) {
                  bestSc = sc;
                  bestY = y;
                }
              }
            }
          }
          if (bestY > 0) {
            leaves.splice(
              idx,
              1,
              { x0: leaf.x0, y0: leaf.y0, x1: leaf.x1, y1: bestY, conf: Math.max(leaf.conf, bestSc), isGutter: leaf.isGutter },
              { x0: leaf.x0, y0: bestY, x1: leaf.x1, y1: leaf.y1, conf: Math.max(leaf.conf, bestSc), isGutter: leaf.isGutter }
            );
            propagated = true;
            splitDone = true;
            break;
          }
        }
      }
      if (splitDone) break;

      for (const xCut of vSet) {
        if (xCut - leaf.x0 >= minDim && leaf.x1 - xCut >= minDim) {
          const sy = Math.max(1, Math.floor((leaf.y1 - leaf.y0) / 200));
          const xLo = Math.max(leaf.x0 + minDim, xCut - 12);
          const xHi = Math.min(leaf.x1 - minDim, xCut + 12);
          let bestX = -1;
          let bestSc = 0;
          for (let x = xLo; x <= xHi; x++) {
            let stepS = 0;
            let lineS = 0;
            let smoothS = 0;
            let act = 0;
            let n = 0;
            let prev = 0;
            for (let y = leaf.y0 + 4; y < leaf.y1 - 4; y += sy) {
              const p1 = y * stride + x * 4;
              const p0 = p1 - 4;
              const p2 = p1 + 4;
              const ds = colorDistAt(data, p1, p2);
              const dl =
                0.5 *
                (Math.abs(2 * data[p1] - data[p0] - data[p2]) +
                  Math.abs(2 * data[p1 + 1] - data[p0 + 1] - data[p2 + 1]) +
                  Math.abs(2 * data[p1 + 2] - data[p0 + 2] - data[p2 + 2]));
              stepS += ds;
              lineS += dl;
              if (ds + dl > 6) act++;
              const luma = data[p1] + data[p1 + 1] + data[p1 + 2];
              if (n > 0) smoothS += Math.abs(luma - prev);
              prev = luma;
              n++;
            }
            if (n >= 2) {
              const cov = act / n;
              const avgRaw = (stepS + lineS) / n;
              const avgSm = smoothS / Math.max(1, n - 1);
              if (cov >= 0.78 && avgRaw >= 10 && avgSm < 2.2) {
                const sc = (avgRaw * cov) / (1 + avgSm * 0.2);
                if (sc > bestSc) {
                  bestSc = sc;
                  bestX = x;
                }
              }
            }
          }
          if (bestX > 0) {
            leaves.splice(
              idx,
              1,
              { x0: leaf.x0, y0: leaf.y0, x1: bestX, y1: leaf.y1, conf: Math.max(leaf.conf, bestSc), isGutter: leaf.isGutter },
              { x0: bestX, y0: leaf.y0, x1: leaf.x1, y1: leaf.y1, conf: Math.max(leaf.conf, bestSc), isGutter: leaf.isGutter }
            );
            propagated = true;
            break;
          }
        }
      }
      if (propagated) break;
    }
    if (!propagated) break;
  }

  if (bgUniform) {
    const nonEmpty = leaves.filter((l) => hasForeground(data, w, l.x0, l.y0, l.x1, l.y1, bgRgb, isTrans));
    if (nonEmpty.length > 0) leaves = nonEmpty;
  }

  leaves.sort((a, b) => {
    const tol = Math.min(a.y1 - a.y0, b.y1 - b.y0) * 0.35;
    return Math.abs(a.y0 - b.y0) > tol ? a.y0 - b.y0 : a.x0 - b.x0;
  });

  const boxes = [];
  let curR = 0;
  let curC = 0;
  let lastY = leaves.length > 0 ? leaves[0].y0 : 0;

  leaves.forEach((leaf, idx) => {
    if (idx > 0 && Math.abs(leaf.y0 - lastY) > (leaf.y1 - leaf.y0) * 0.35) {
      curR++;
      curC = 0;
      lastY = leaf.y0;
    } else if (idx > 0) {
      curC++;
    }
    const trimmed = refineAndTrimCell(data, w, h, leaf, bleedTrim, bgRgb, bgUniform, isTrans);
    boxes.push({
      index: idx + 1,
      row: curR,
      col: curC,
      x: trimmed.x0,
      y: trimmed.y0,
      width: Math.max(1, trimmed.x1 - trimmed.x0),
      height: Math.max(1, trimmed.y1 - trimmed.y0),
      confidence: Math.min(1, leaf.conf / 150),
      strategy: leaf.isGutter ? "gutter" : "xycut",
      trimmedEdges: trimmed.trimmedEdges,
    });
  });

  return boxes;
}

function dilateSeparable(mask, sw, sh, radius) {
  if (radius === 0) return new Uint8Array(mask);
  const horiz = new Uint8Array(sw * sh);
  for (let sy = 0; sy < sh; sy++) {
    const base = sy * sw;
    let rem = 0;
    for (let sx = 0; sx < sw; sx++) {
      if (mask[base + sx] !== 0) rem = radius + 1;
      if (rem > 0) {
        horiz[base + sx] = 1;
        rem--;
      }
    }
    rem = 0;
    for (let sx = sw - 1; sx >= 0; sx--) {
      if (mask[base + sx] !== 0) rem = radius + 1;
      if (rem > 0) {
        horiz[base + sx] = 1;
        rem--;
      }
    }
  }
  const vert = new Uint8Array(sw * sh);
  for (let sx = 0; sx < sw; sx++) {
    let rem = 0;
    for (let sy = 0; sy < sh; sy++) {
      const idx = sy * sw + sx;
      if (horiz[idx] !== 0) rem = radius + 1;
      if (rem > 0) {
        vert[idx] = 1;
        rem--;
      }
    }
    rem = 0;
    for (let sy = sh - 1; sy >= 0; sy--) {
      const idx = sy * sw + sx;
      if (horiz[idx] !== 0) rem = radius + 1;
      if (rem > 0) {
        vert[idx] = 1;
        rem--;
      }
    }
  }
  return vert;
}

function erodeSeparable(mask, sw, sh, radius) {
  if (radius === 0) return new Uint8Array(mask);
  const horiz = new Uint8Array(sw * sh);
  for (let sy = 0; sy < sh; sy++) {
    const base = sy * sw;
    let rem = 0;
    for (let sx = 0; sx < sw; sx++) {
      if (mask[base + sx] === 0) rem = radius + 1;
      if (rem > 0) {
        horiz[base + sx] = 0;
        rem--;
      } else {
        horiz[base + sx] = 1;
      }
    }
    rem = 0;
    for (let sx = sw - 1; sx >= 0; sx--) {
      if (mask[base + sx] === 0) rem = radius + 1;
      if (rem > 0) {
        horiz[base + sx] = 0;
        rem--;
      }
    }
  }
  const vert = new Uint8Array(sw * sh);
  for (let sx = 0; sx < sw; sx++) {
    let rem = 0;
    for (let sy = 0; sy < sh; sy++) {
      const idx = sy * sw + sx;
      if (horiz[idx] === 0) rem = radius + 1;
      if (rem > 0) {
        vert[idx] = 0;
        rem--;
      } else {
        vert[idx] = 1;
      }
    }
    rem = 0;
    for (let sy = sh - 1; sy >= 0; sy--) {
      const idx = sy * sw + sx;
      if (horiz[idx] === 0) rem = radius + 1;
      if (rem > 0) {
        vert[idx] = 0;
        rem--;
      }
    }
  }
  return vert;
}

function detectContours(data, w, h, sensitivity, minRegionRatio, contourGapPx, bgRgb, isTrans) {
  const stride = w * 4;
  const scale = Math.max(1, Math.ceil(Math.max(w, h) / 720));
  const sw = Math.ceil(w / scale);
  const sh = Math.ceil(h / scale);
  const colorThresh = 32 - clamp(sensitivity, 0.1, 1.0) * 18;

  const mask = new Uint8Array(sw * sh);
  for (let sy = 0; sy < sh; sy++) {
    const y = Math.min(h - 1, sy * scale);
    const rowPtr = y * stride;
    for (let sx = 0; sx < sw; sx++) {
      const x = Math.min(w - 1, sx * scale);
      const p = rowPtr + x * 4;
      mask[sy * sw + sx] = isTrans
        ? data[p + 3] > 24
          ? 1
          : 0
        : colorDistRgb(data, p, bgRgb) > colorThresh
          ? 1
          : 0;
    }
  }

  const rScaled = Math.min(20, Math.max(1, Math.round(Math.max(1, contourGapPx) / scale)));
  const dilated = dilateSeparable(mask, sw, sh, rScaled);
  const closed = erodeSeparable(dilated, sw, sh, rScaled);

  const labels = new Int32Array(sw * sh);
  const parent = [0];
  const find = (a) => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  const union = (a, b) => {
    const ra = find(a);
    const rb = find(b);
    if (ra === rb) return;
    if (ra < rb) parent[rb] = ra;
    else parent[ra] = rb;
  };

  for (let sy = 0; sy < sh; sy++) {
    for (let sx = 0; sx < sw; sx++) {
      const idx = sy * sw + sx;
      if (closed[idx] === 0) continue;
      const nbrs = [];
      if (sx > 0 && labels[idx - 1] > 0) nbrs.push(labels[idx - 1]);
      if (sy > 0 && labels[idx - sw] > 0) nbrs.push(labels[idx - sw]);
      if (sx > 0 && sy > 0 && labels[idx - sw - 1] > 0) nbrs.push(labels[idx - sw - 1]);
      if (sx + 1 < sw && sy > 0 && labels[idx - sw + 1] > 0) nbrs.push(labels[idx - sw + 1]);

      if (nbrs.length === 0) {
        const lbl = parent.length;
        parent.push(lbl);
        labels[idx] = lbl;
      } else {
        const best = Math.min(...nbrs);
        labels[idx] = best;
        for (const n of nbrs) union(best, n);
      }
    }
  }

  const stats = new Map();
  for (let sy = 0; sy < sh; sy++) {
    for (let sx = 0; sx < sw; sx++) {
      const idx = sy * sw + sx;
      const l = labels[idx];
      if (l === 0) continue;
      const root = find(l);
      let s = stats.get(root);
      if (!s) {
        s = [sx, sy, sx, sy, 0];
        stats.set(root, s);
      }
      if (sx < s[0]) s[0] = sx;
      if (sy < s[1]) s[1] = sy;
      if (sx > s[2]) s[2] = sx;
      if (sy > s[3]) s[3] = sy;
      s[4] += 1;
    }
  }

  const minArea = Math.max(16, sw * sh * Math.pow(clamp(minRegionRatio, 0.02, 0.3), 2) * 0.45);
  const isFg = (x, y) => {
    const p = y * stride + x * 4;
    return isTrans ? data[p + 3] > 24 : colorDistRgb(data, p, bgRgb) > colorThresh;
  };
  const rowHasFg = (y, x0, x1) => {
    const step = Math.max(1, Math.floor((x1 - x0) / 96));
    for (let x = x0; x <= x1; x += step) {
      if (isFg(x, y)) return true;
    }
    return isFg(x1, y);
  };
  const colHasFg = (x, y0, y1) => {
    const step = Math.max(1, Math.floor((y1 - y0) / 96));
    for (let y = y0; y <= y1; y += step) {
      if (isFg(x, y)) return true;
    }
    return isFg(x, y1);
  };

  const rawBoxes = [];
  for (const [mx0, my0, mx1, my1, area] of stats.values()) {
    if (area < minArea) continue;
    let x0 = mx0 * scale;
    let y0 = my0 * scale;
    let x1 = Math.min(w - 1, (mx1 + 1) * scale - 1);
    let y1 = Math.min(h - 1, (my1 + 1) * scale - 1);

    const pad = scale + 2;
    for (let k = 0; k < pad && y0 > 0; k++) {
      if (rowHasFg(y0 - 1, x0, x1)) y0--;
      else break;
    }
    for (let k = 0; k < pad && y1 + 1 < h; k++) {
      if (rowHasFg(y1 + 1, x0, x1)) y1++;
      else break;
    }
    for (let k = 0; k < pad && x0 > 0; k++) {
      if (colHasFg(x0 - 1, y0, y1)) x0--;
      else break;
    }
    for (let k = 0; k < pad && x1 + 1 < w; k++) {
      if (colHasFg(x1 + 1, y0, y1)) x1++;
      else break;
    }

    while (y0 < y1 && !rowHasFg(y0, x0, x1)) y0++;
    while (y1 > y0 && !rowHasFg(y1, x0, x1)) y1--;
    while (x0 < x1 && !colHasFg(x0, y0, y1)) x0++;
    while (x1 > x0 && !colHasFg(x1, y0, y1)) x1--;

    rawBoxes.push({ x0, y0, x1: x1 + 1, y1: y1 + 1 });
  }

  const band = Math.max(1, Math.floor(Math.min(w, h) * 0.1));
  rawBoxes.sort((a, b) => {
    const ba = Math.floor(a.y0 / band);
    const bb = Math.floor(b.y0 / band);
    return ba !== bb ? ba - bb : a.x0 - b.x0;
  });

  return rawBoxes.map((b, idx) => ({
    index: idx + 1,
    row: idx,
    col: 0,
    x: b.x0,
    y: b.y0,
    width: Math.max(1, b.x1 - b.x0),
    height: Math.max(1, b.y1 - b.y0),
    confidence: 0.98,
    strategy: "contour",
    trimmedEdges: [0, 0, 0, 0],
  }));
}

function splitUniformGrid(w, h, rows, cols, gutterPx) {
  const rCnt = Math.max(1, rows | 0);
  const cCnt = Math.max(1, cols | 0);
  const g = Math.max(0, gutterPx | 0);
  const halfG = g >> 1;
  const boxes = [];
  let idx = 1;
  for (let r = 0; r < rCnt; r++) {
    const y0 = Math.round((r * h) / rCnt) + (r > 0 ? halfG : 0);
    const y1 = Math.round(((r + 1) * h) / rCnt) - (r < rCnt - 1 ? g - halfG : 0);
    for (let c = 0; c < cCnt; c++) {
      const x0 = Math.round((c * w) / cCnt) + (c > 0 ? halfG : 0);
      const x1 = Math.round(((c + 1) * w) / cCnt) - (c < cCnt - 1 ? g - halfG : 0);
      boxes.push({
        index: idx++,
        row: r,
        col: c,
        x: x0,
        y: y0,
        width: Math.max(1, x1 - x0),
        height: Math.max(1, y1 - y0),
        confidence: 1.0,
        strategy: "grid",
        trimmedEdges: [0, 0, 0, 0],
      });
    }
  }
  return boxes;
}

export function analyzeAndSplit(data, width, height, options = {}) {
  const {
    mode = "auto",
    sensitivity = 0.65,
    minRegionRatio = 0.08,
    bleedTrim = 2,
    contourGapPx = 8,
    rows = 3,
    cols = 3,
    gutterPx = 0,
    uniformSize = false,
    targetSize = null,
    uniformMode = "pad",
  } = options;

  const t0 = performance.now();
  const bgInfo = estimateBorderBg(data, width, height);
  const bgUniform = bgInfo.uniformity > 0.72;
  let detectedMode = mode;
  let boxes;

  if (mode === "grid") {
    boxes = splitUniformGrid(width, height, rows, cols, gutterPx);
  } else if (mode === "contour") {
    boxes = detectContours(data, width, height, sensitivity, minRegionRatio, contourGapPx, bgInfo.bgRgb, bgInfo.isTransparent);
  } else {
    const xyBoxes = detectRecursiveXyCut(
      data,
      width,
      height,
      sensitivity,
      minRegionRatio,
      bleedTrim,
      bgInfo.bgRgb,
      bgUniform,
      bgInfo.isTransparent
    );
    boxes = xyBoxes;
    detectedMode = "xycut";

    if (mode === "auto" && bgUniform) {
      const ctBoxes = detectContours(
        data,
        width,
        height,
        sensitivity,
        minRegionRatio,
        contourGapPx,
        bgInfo.bgRgb,
        bgInfo.isTransparent
      );
      const ctArea = ctBoxes.reduce((s, b) => s + b.width * b.height, 0);
      const totalArea = width * height;
      if (ctBoxes.length >= 2 && (xyBoxes.length <= 1 || (ctArea * 100 < totalArea * 65 && ctBoxes.length >= xyBoxes.length))) {
        boxes = ctBoxes;
        detectedMode = "contour";
      }
    }
  }

  const effectiveUniform = uniformSize || targetSize != null;
  if (effectiveUniform && uniformMode === "crop" && boxes.length > 0) {
    const tw = targetSize ? targetSize[0] : Math.max(...boxes.map((b) => b.width));
    const th = targetSize ? targetSize[1] : Math.max(...boxes.map((b) => b.height));
    const cw = Math.max(1, Math.min(tw, width));
    const ch = Math.max(1, Math.min(th, height));
    boxes = boxes.map((b) => {
      const cx = b.x + Math.floor(b.width / 2);
      const cy = b.y + Math.floor(b.height / 2);
      const nx = Math.min(Math.max(0, cx - Math.floor(cw / 2)), Math.max(0, width - cw));
      const ny = Math.min(Math.max(0, cy - Math.floor(ch / 2)), Math.max(0, height - ch));
      return { ...b, x: nx, y: ny, width: cw, height: ch };
    });
  }

  let resolvedTargetSize = null;
  if (effectiveUniform && uniformMode !== "crop" && boxes.length > 0) {
    resolvedTargetSize = targetSize
      ? [targetSize[0], targetSize[1]]
      : [Math.max(...boxes.map((b) => b.width)), Math.max(...boxes.map((b) => b.height))];
  }

  return {
    imageWidth: width,
    imageHeight: height,
    detectedMode,
    elapsedMs: performance.now() - t0,
    boxes,
    bgInfo,
    resolvedTargetSize,
    uniformMode,
  };
}

export function resizeRgbaBilinear(src, sw, sh, tw, th) {
  if (sw === tw && sh === th) return new Uint8ClampedArray(src);
  const out = new Uint8ClampedArray(tw * th * 4);
  for (let y = 0; y < th; y++) {
    const gy = ((y + 0.5) * sh) / th - 0.5;
    const y0 = Math.max(0, Math.min(sh - 1, Math.floor(gy)));
    const y1 = Math.min(sh - 1, y0 + 1);
    const wy = gy - Math.floor(gy);
    for (let x = 0; x < tw; x++) {
      const gx = ((x + 0.5) * sw) / tw - 0.5;
      const x0 = Math.max(0, Math.min(sw - 1, Math.floor(gx)));
      const x1 = Math.min(sw - 1, x0 + 1);
      const wx = gx - Math.floor(gx);
      const p00 = (y0 * sw + x0) * 4;
      const p10 = (y0 * sw + x1) * 4;
      const p01 = (y1 * sw + x0) * 4;
      const p11 = (y1 * sw + x1) * 4;
      const dst = (y * tw + x) * 4;
      for (let c = 0; c < 4; c++) {
        const top = src[p00 + c] * (1 - wx) + src[p10 + c] * wx;
        const bot = src[p01 + c] * (1 - wx) + src[p11 + c] * wx;
        out[dst + c] = Math.round(top * (1 - wy) + bot * wy);
      }
    }
  }
  return out;
}

export function extractNormalizedTile(srcData, imgW, box, resolvedTargetSize, uniformMode, bgInfo) {
  const bw = box.width;
  const bh = box.height;
  const crop = new Uint8ClampedArray(bw * bh * 4);
  for (let y = 0; y < bh; y++) {
    const srcStart = ((box.y + y) * imgW + box.x) * 4;
    crop.set(srcData.subarray(srcStart, srcStart + bw * 4), y * bw * 4);
  }

  if (!resolvedTargetSize || uniformMode === "crop") {
    return { data: crop, width: bw, height: bh };
  }

  const [tw, th] = resolvedTargetSize;
  if (bw === tw && bh === th) {
    return { data: crop, width: bw, height: bh };
  }
  if (uniformMode === "resize") {
    return { data: resizeRgbaBilinear(crop, bw, bh, tw, th), width: tw, height: th };
  }

  // pad mode: shrink proportionally if larger than target canvas, then center over fill background
  let fitted = crop;
  let fw = bw;
  let fh = bh;
  if (bw > tw || bh > th) {
    const scale = Math.min(tw / bw, th / bh);
    fw = Math.max(1, Math.round(bw * scale));
    fh = Math.max(1, Math.round(bh * scale));
    fitted = resizeRgbaBilinear(crop, bw, bh, fw, fh);
  }

  const canvas = new Uint8ClampedArray(tw * th * 4);
  const fill = bgInfo && bgInfo.isTransparent ? [0, 0, 0, 0] : [bgInfo?.bgRgb?.[0] ?? 255, bgInfo?.bgRgb?.[1] ?? 255, bgInfo?.bgRgb?.[2] ?? 255, 255];
  for (let i = 0; i < tw * th; i++) {
    const o = i * 4;
    canvas[o] = fill[0];
    canvas[o + 1] = fill[1];
    canvas[o + 2] = fill[2];
    canvas[o + 3] = fill[3];
  }

  const ox = Math.floor((tw - fw) / 2);
  const oy = Math.floor((th - fh) / 2);
  for (let y = 0; y < fh; y++) {
    const srcStart = y * fw * 4;
    const dstStart = ((oy + y) * tw + ox) * 4;
    canvas.set(fitted.subarray(srcStart, srcStart + fw * 4), dstStart);
  }
  return { data: canvas, width: tw, height: th };
}
