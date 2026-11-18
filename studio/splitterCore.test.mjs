import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeAndSplit, extractNormalizedTile } from './splitterCore.mjs';

function makeRgba(w, h, r, g, b, a = 255) {
  const buf = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    buf[o] = r;
    buf[o + 1] = g;
    buf[o + 2] = b;
    buf[o + 3] = a;
  }
  return buf;
}

function fillRect(buf, imgW, x0, y0, x1, y1, r, g, b, a = 255) {
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const o = (y * imgW + x) * 4;
      buf[o] = r;
      buf[o + 1] = g;
      buf[o + 2] = b;
      buf[o + 3] = a;
    }
  }
}

test('Asymmetric Bento layout splits into 3 clean panels', () => {
  const w = 600;
  const h = 400;
  const buf = makeRgba(w, h, 255, 255, 255);
  fillRect(buf, w, 0, 0, 298, 400, 24, 52, 110);
  fillRect(buf, w, 299, 0, 600, 198, 215, 72, 65);
  fillRect(buf, w, 299, 200, 600, 400, 32, 145, 112);

  const res = analyzeAndSplit(buf, w, h, { mode: 'xycut', sensitivity: 0.65, bleedTrim: 1 });
  assert.equal(res.boxes.length, 3);
  assert.ok(res.boxes[0].height > 390);
});

test('White-guttered 2x2 collage isolates 4 panels with zero white gutters', () => {
  const w = 800;
  const h = 600;
  const buf = makeRgba(w, h, 255, 255, 255);
  fillRect(buf, w, 20, 20, 380, 280, 30, 64, 175);
  fillRect(buf, w, 420, 20, 780, 280, 16, 185, 129);
  fillRect(buf, w, 20, 320, 380, 580, 225, 29, 72);
  fillRect(buf, w, 420, 320, 780, 580, 245, 158, 11);

  const res = analyzeAndSplit(buf, w, h, { mode: 'xycut', sensitivity: 0.65, bleedTrim: 2 });
  assert.equal(res.boxes.length, 4);
  for (const box of res.boxes) {
    assert.ok(Math.abs(box.width - 360) <= 2);
    assert.ok(Math.abs(box.height - 260) <= 2);
  }
});

test('Bleed decontamination peels 1px gray divider cleanly', () => {
  const w = 600;
  const h = 400;
  const buf = makeRgba(w, h, 252, 252, 252);
  fillRect(buf, w, 0, 0, 299, 400, 18, 36, 74);
  fillRect(buf, w, 299, 0, 300, 400, 213, 214, 217);
  fillRect(buf, w, 300, 0, 600, 400, 250, 251, 253);

  const res = analyzeAndSplit(buf, w, h, { mode: 'xycut', sensitivity: 0.7, bleedTrim: 2 });
  assert.equal(res.boxes.length, 2);
  const sorted = [...res.boxes].sort((a, b) => a.x - b.x);
  assert.ok(sorted[0].x + sorted[0].width <= 299);
  assert.ok(sorted[1].x >= 300);
});

test('Contour segmentation isolates scattered targets with 1:1 tight accuracy', () => {
  const w = 1500;
  const h = 950;
  const buf = makeRgba(w, h, 248, 248, 248);
  fillRect(buf, w, 80, 70, 400, 350, 29, 78, 216);
  fillRect(buf, w, 540, 90, 920, 330, 4, 120, 87);
  fillRect(buf, w, 1060, 65, 1400, 375, 190, 18, 60);
  fillRect(buf, w, 170, 490, 630, 830, 15, 23, 42);
  fillRect(buf, w, 810, 470, 1320, 840, 217, 119, 6);

  const res = analyzeAndSplit(buf, w, h, { mode: 'auto', sensitivity: 0.7 });
  assert.equal(res.detectedMode, 'contour');
  assert.equal(res.boxes.length, 5);
  assert.equal(res.boxes[0].x, 80);
  assert.equal(res.boxes[0].y, 70);
  assert.equal(res.boxes[0].width, 320);
  assert.equal(res.boxes[0].height, 280);
});

test('Uniform sizing normalizes slices identically across pad, resize, and crop modes', () => {
  const w = 900;
  const h = 600;
  const buf = makeRgba(w, h, 255, 255, 255);
  fillRect(buf, w, 20, 20, 320, 220, 24, 52, 110);
  fillRect(buf, w, 360, 20, 760, 280, 215, 72, 65);
  fillRect(buf, w, 20, 320, 420, 540, 32, 145, 112);

  const padRes = analyzeAndSplit(buf, w, h, { mode: 'auto', uniformSize: true, uniformMode: 'pad' });
  const padTiles = padRes.boxes.map((b) =>
    extractNormalizedTile(buf, w, b, padRes.resolvedTargetSize, 'pad', padRes.bgInfo)
  );
  const padDims = new Set(padTiles.map((t) => `${t.width}x${t.height}`));
  assert.equal(padDims.size, 1);

  const resizeRes = analyzeAndSplit(buf, w, h, {
    mode: 'auto',
    targetSize: [256, 192],
    uniformMode: 'resize',
  });
  const resizeTiles = resizeRes.boxes.map((b) =>
    extractNormalizedTile(buf, w, b, resizeRes.resolvedTargetSize, 'resize', resizeRes.bgInfo)
  );
  assert.ok(resizeTiles.every((t) => t.width === 256 && t.height === 192));

  const cropRes = analyzeAndSplit(buf, w, h, { mode: 'auto', uniformSize: true, uniformMode: 'crop' });
  const cropDims = new Set(cropRes.boxes.map((b) => `${b.width}x${b.height}`));
  assert.equal(cropDims.size, 1);
});
