import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import { analyzeAndSplit, type RawImage } from './splitterCore';

function makeCanvas(w: number, h: number, fill: [number, number, number, number]): RawImage {
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = fill[0];
    data[i * 4 + 1] = fill[1];
    data[i * 4 + 2] = fill[2];
    data[i * 4 + 3] = fill[3];
  }
  return { data, width: w, height: h, channels: 4 };
}

function paintRect(
  img: RawImage,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: [number, number, number, number],
) {
  for (let y = Math.max(0, y0); y < Math.min(img.height, y1); y++) {
    for (let x = Math.max(0, x0); x < Math.min(img.width, x1); x++) {
      const idx = (y * img.width + x) * 4;
      img.data[idx] = color[0];
      img.data[idx + 1] = color[1];
      img.data[idx + 2] = color[2];
      img.data[idx + 3] = color[3];
    }
  }
}

test('Real sample 2848x1600 3x3 slide sheet splits into 9 exact seam-aligned slides', async () => {
  const { data, info } = await sharp(
    'public/assets/aistudio/e3e01d85a65848fa4acdfaa7bcd4ac0d20a578b05b20476af8d929802f2366e9_e3e01d85.png',
  )
    .raw()
    .ensureAlpha()
    .toBuffer({ resolveWithObject: true });

  const result = analyzeAndSplit(
    { data, width: info.width, height: info.height, channels: info.channels },
    { mode: 'auto', sensitivity: 0.65, bleedTrim: 2 },
  );

  assert.equal(result.detectedMode, 'xycut');
  assert.equal(result.boxes.length, 9, `Expected 9 slides, got ${result.boxes.length}`);

  // Verify Row 0..2 right column local seam adaptation (x=1824 in rows 0-1 vs x=1812 in row 2)
  const topRight = result.boxes[2];
  const bottomRight = result.boxes[8];
  assert.ok(topRight.x >= 1823 && topRight.x <= 1827, `Top-right x=${topRight.x}`);
  assert.ok(bottomRight.x >= 1810 && bottomRight.x <= 1815, `Bottom-right x=${bottomRight.x}`);
  assert.ok(bottomRight.y >= 1046 && bottomRight.y <= 1050, `Bottom-right y=${bottomRight.y}`);

  // Verify zero outer-edge bleed on previously vulnerable seams (#2 right, #5/#8 left, #9 top/left)
  const pxAt = (x: number, y: number) => {
    const i = (y * info.width + x) * info.channels;
    return [data[i], data[i + 1], data[i + 2]];
  };
  const b2 = result.boxes[1];
  const b2Right = pxAt(b2.x + b2.width - 1, b2.y + (b2.height >> 1));
  assert.ok(b2Right[0] > 240 && b2Right[1] > 240 && b2Right[2] > 240, `#2 right edge should be clean white, got ${b2Right}`);

  const b5 = result.boxes[4];
  const b5Left = pxAt(b5.x, b5.y + (b5.height >> 1));
  assert.ok(b5Left[0] > 240 && b5Left[1] > 240 && b5Left[2] > 240, `#5 left edge should strip 1px divider, got ${b5Left}`);

  const b8 = result.boxes[7];
  const b8Left = pxAt(b8.x, b8.y + (b8.height >> 1));
  assert.ok(b8Left[0] > 240 && b8Left[1] > 240 && b8Left[2] > 240, `#8 left edge should strip 1px divider, got ${b8Left}`);

  const b9Left = pxAt(bottomRight.x, bottomRight.y + (bottomRight.height >> 1));
  assert.ok(b9Left[0] < 25 && b9Left[2] > 70, `#9 left edge should be deep navy without transition bleed, got ${b9Left}`);
});

test('Asymmetric Bento layout (1 tall left + 2 stacked right) splits into 3 panels', () => {
  const img = makeCanvas(600, 400, [255, 255, 255, 255]);
  paintRect(img, 0, 0, 298, 400, [24, 52, 110, 255]); // left dark panel
  paintRect(img, 299, 0, 600, 198, [215, 72, 65, 255]); // top-right coral panel
  paintRect(img, 299, 200, 600, 400, [32, 145, 112, 255]); // bottom-right green panel

  const result = analyzeAndSplit(img, { mode: 'xycut', sensitivity: 0.65, bleedTrim: 1 });
  assert.equal(result.boxes.length, 3);
  assert.ok(result.boxes[0].height > 390, 'Left panel should remain full height');
});

test('White-guttered 2x2 collage separates into 4 clean panels without gutters', () => {
  const img = makeCanvas(640, 440, [255, 255, 255, 255]);
  paintRect(img, 20, 20, 300, 200, [40, 80, 160, 255]);
  paintRect(img, 340, 20, 620, 200, [160, 60, 90, 255]);
  paintRect(img, 20, 240, 300, 420, [50, 140, 90, 255]);
  paintRect(img, 340, 240, 620, 420, [200, 140, 30, 255]);

  const result = analyzeAndSplit(img, { mode: 'xycut', sensitivity: 0.65, bleedTrim: 1 });
  assert.equal(result.boxes.length, 4);
  for (const b of result.boxes) {
    assert.ok(b.width >= 278 && b.width <= 282, `Trimmed width=${b.width}`);
    assert.ok(b.height >= 178 && b.height <= 182, `Trimmed height=${b.height}`);
  }
});

test('Contour segmentation isolates high-res scattered objects with 1:1 tight edge precision', () => {
  const img = makeCanvas(1500, 950, [248, 250, 252, 255]);
  paintRect(img, 80, 70, 400, 350, [29, 78, 216, 255]);
  paintRect(img, 540, 90, 920, 330, [4, 120, 87, 255]);
  paintRect(img, 1060, 65, 1400, 375, [190, 18, 60, 255]);
  paintRect(img, 170, 490, 630, 830, [15, 23, 42, 255]);
  paintRect(img, 810, 470, 1320, 840, [217, 119, 6, 255]);

  const result = analyzeAndSplit(img, { mode: 'auto', sensitivity: 0.7 });
  assert.equal(result.detectedMode, 'contour');
  assert.equal(result.boxes.length, 5);
  assert.ok(Math.abs(result.boxes[0].x - 80) <= 2, `Expected x~80, got ${result.boxes[0].x}`);
  assert.ok(Math.abs(result.boxes[0].width - 320) <= 4, `Expected w~320, got ${result.boxes[0].width}`);
  assert.ok(Math.abs(result.boxes[4].width - 510) <= 4, `Expected w~510, got ${result.boxes[4].width}`);
});

test('Transparent RGBA sprite sheet isolates alpha sprites cleanly', () => {
  const img = makeCanvas(720, 480, [0, 0, 0, 0]);
  paintRect(img, 40, 40, 220, 200, [230, 70, 60, 255]);
  paintRect(img, 360, 60, 640, 210, [40, 180, 120, 255]);
  paintRect(img, 120, 260, 460, 430, [60, 110, 230, 255]);

  const result = analyzeAndSplit(img, { mode: 'auto', sensitivity: 0.65 });
  assert.equal(result.detectedMode, 'contour');
  assert.equal(result.boxes.length, 3);
  assert.ok(Math.abs(result.boxes[0].width - 180) <= 2);
});

test('Uniform grid splits exact 2x4 tiles', () => {
  const img = makeCanvas(800, 400, [200, 200, 200, 255]);
  const result = analyzeAndSplit(img, { mode: 'grid', rows: 2, cols: 4, gutterPx: 0 });
  assert.equal(result.boxes.length, 8);
  assert.equal(result.boxes[0].width, 200);
  assert.equal(result.boxes[0].height, 200);
});
