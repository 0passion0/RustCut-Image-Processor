import { analyzeAndSplit, extractNormalizedTile } from './splitterCore.mjs';
import { buildZipBlob } from './zip.mjs';

const state = {
  preset: 'slides',
  sourceName: 'sample_slides.png',
  width: 0,
  height: 0,
  rgba: null,
  sourceCanvas: null,
  mode: 'auto',
  sensitivity: 0.65,
  minRegionRatio: 0.08,
  bleedTrim: 2,
  contourGapPx: 8,
  rows: 3,
  cols: 3,
  gutterPx: 0,
  uniformSize: false,
  uniformMode: 'pad',
  targetW: '',
  targetH: '',
  format: 'png',
  showOverlay: true,
  selectedIndex: -1,
  result: null,
  tiles: [],
  lightboxIndex: -1,
};

const els = {
  presetBtns: [...document.querySelectorAll('[data-preset]')],
  modeBtns: [...document.querySelectorAll('[data-mode]')],
  uniformModeBtns: [...document.querySelectorAll('[data-uniform]')],
  fileInput: document.getElementById('fileInput'),
  dropZone: document.getElementById('dropZone'),
  sourceDimensions: document.getElementById('sourceDimensions'),
  sensitivityInput: document.getElementById('sensitivityInput'),
  sensitivityValue: document.getElementById('sensitivityValue'),
  minRatioInput: document.getElementById('minRatioInput'),
  minRatioValue: document.getElementById('minRatioValue'),
  bleedTrimInput: document.getElementById('bleedTrimInput'),
  bleedTrimValue: document.getElementById('bleedTrimValue'),
  contourGapInput: document.getElementById('contourGapInput'),
  contourGapValue: document.getElementById('contourGapValue'),
  gridControls: document.getElementById('gridControls'),
  rowsInput: document.getElementById('rowsInput'),
  colsInput: document.getElementById('colsInput'),
  gutterInput: document.getElementById('gutterInput'),
  uniformToggle: document.getElementById('uniformToggle'),
  uniformOptions: document.getElementById('uniformOptions'),
  targetWInput: document.getElementById('targetWInput'),
  targetHInput: document.getElementById('targetHInput'),
  formatSelect: document.getElementById('formatSelect'),
  overlayToggle: document.getElementById('overlayToggle'),
  pythonSnippet: document.getElementById('pythonSnippet'),
  copyCodeBtn: document.getElementById('copyCodeBtn'),
  analysisSummary: document.getElementById('analysisSummary'),
  gallerySummary: document.getElementById('gallerySummary'),
  previewCanvas: document.getElementById('previewCanvas'),
  slicesGrid: document.getElementById('slicesGrid'),
  downloadZipBtn: document.getElementById('downloadZipBtn'),
  lightboxDialog: document.getElementById('lightboxDialog'),
  lightboxTitle: document.getElementById('lightboxTitle'),
  lightboxMeta: document.getElementById('lightboxMeta'),
  lightboxCanvas: document.getElementById('lightboxCanvas'),
  zoomToggle: document.getElementById('zoomToggle'),
  lightboxDownloadBtn: document.getElementById('lightboxDownloadBtn'),
  closeLightboxBtn: document.getElementById('closeLightboxBtn'),
};

function generatePresetCanvas(name) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');

  if (name === 'bento') {
    c.width = 1200;
    c.height = 800;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = '#18346e';
    ctx.fillRect(0, 0, 594, 800);
    ctx.fillStyle = '#d74841';
    ctx.fillRect(597, 0, 603, 396);
    ctx.fillStyle = '#209170';
    ctx.fillRect(597, 399, 603, 401);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 34px sans-serif';
    ctx.fillText('Panel A (Tall)', 48, 84);
    ctx.fillText('Panel B (Top Right)', 640, 84);
    ctx.fillText('Panel C (Bottom Right)', 640, 480);
  } else if (name === 'gutters') {
    c.width = 1000;
    c.height = 760;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, c.width, c.height);
    const cards = [
      [36, 36, 444, 324, '#1e40af', 'Card 01'],
      [520, 36, 444, 324, '#059669', 'Card 02'],
      [36, 400, 444, 324, '#be123c', 'Card 03'],
      [520, 400, 444, 324, '#d97706', 'Card 04'],
    ];
    for (const [x, y, w, h, color, label] of cards) {
      ctx.fillStyle = color;
      ctx.fillRect(x, y, w, h);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 28px sans-serif';
      ctx.fillText(label, x + 28, y + 60);
    }
  } else {
    c.width = 1500;
    c.height = 950;
    ctx.fillStyle = '#f8f8f8';
    ctx.fillRect(0, 0, c.width, c.height);
    const blobs = [
      [80, 70, 320, 280, '#1d4ed8', 'Object #1'],
      [540, 90, 380, 240, '#047857', 'Object #2'],
      [1060, 65, 340, 310, '#be123c', 'Object #3'],
      [170, 490, 460, 340, '#0f172a', 'Object #4'],
      [810, 470, 510, 370, '#d97706', 'Object #5'],
    ];
    for (const [x, y, w, h, color, label] of blobs) {
      ctx.fillStyle = color;
      ctx.fillRect(x, y, w, h);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 26px sans-serif';
      ctx.fillText(label, x + 24, y + 52);
    }
  }
  return c;
}

async function loadSampleImage() {
  const img = new Image();
  img.decoding = 'async';
  img.src = '/examples/sample_slides.png';
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
  });
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  setSourceCanvas(c, 'sample_slides.png');
}

function setSourceCanvas(canvas, name) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  state.sourceCanvas = canvas;
  state.width = canvas.width;
  state.height = canvas.height;
  state.rgba = imageData.data;
  state.sourceName = name;
  state.selectedIndex = -1;
  els.sourceDimensions.textContent = `${canvas.width} × ${canvas.height}`;
  runAnalysis();
}

function parseTargetSize() {
  const tw = Number.parseInt(els.targetWInput.value, 10);
  const th = Number.parseInt(els.targetHInput.value, 10);
  if (Number.isFinite(tw) && tw > 0 && Number.isFinite(th) && th > 0) {
    return [tw, th];
  }
  return null;
}

function updatePythonSnippet() {
  const target = parseTargetSize();
  const lines = [
    'from smart_splitter import SmartSplitter',
    '',
    'splitter = SmartSplitter(',
    `    mode="${state.mode}",`,
    `    sensitivity=${state.sensitivity.toFixed(2)},`,
    `    min_region_ratio=${state.minRegionRatio.toFixed(2)},`,
    `    bleed_trim=${state.bleedTrim},`,
    `    contour_gap_px=${state.contourGapPx},`,
  ];
  if (state.uniformSize || target) {
    lines.push(`    uniform_size=${state.uniformSize ? 'True' : 'False'},`);
    if (target) lines.push(`    target_size=(${target[0]}, ${target[1]}),`);
    lines.push(`    uniform_mode="${state.uniformMode}",`);
  }
  lines.push(')');
  if (state.mode === 'grid') {
    lines.push(
      `res, paths = splitter.split_file("${state.sourceName}", "output_slices/", fmt="${state.format}", rows=${state.rows}, cols=${state.cols}, gutter_px=${state.gutterPx})`
    );
  } else {
    lines.push(`res, paths = splitter.split_file("${state.sourceName}", "output_slices/", fmt="${state.format}")`);
  }
  els.pythonSnippet.textContent = lines.join('\n');
}

function runAnalysis() {
  if (!state.rgba) return;
  const targetSize = parseTargetSize();
  const res = analyzeAndSplit(state.rgba, state.width, state.height, {
    mode: state.mode,
    sensitivity: state.sensitivity,
    minRegionRatio: state.minRegionRatio,
    bleedTrim: state.bleedTrim,
    contourGapPx: state.contourGapPx,
    rows: state.rows,
    cols: state.cols,
    gutterPx: state.gutterPx,
    uniformSize: state.uniformSize,
    targetSize,
    uniformMode: state.uniformMode,
  });

  state.result = res;
  state.tiles = res.boxes.map((box) =>
    extractNormalizedTile(
      state.rgba,
      state.width,
      box,
      res.resolvedTargetSize,
      res.uniformMode,
      res.bgInfo
    )
  );

  const uniformTag = res.resolvedTargetSize
    ? ` · 统一 ${res.resolvedTargetSize[0]}×${res.resolvedTargetSize[1]} (${state.uniformMode})`
    : state.uniformSize && state.uniformMode === 'crop' && res.boxes.length > 0
      ? ` · 原图同框 ${res.boxes[0].width}×${res.boxes[0].height}`
      : '';

  els.analysisSummary.textContent = `${state.sourceName} · ${state.width}×${state.height} · 命中 ${res.detectedMode} · ${res.boxes.length} 张切片 · ${res.elapsedMs.toFixed(1)} ms${uniformTag}`;
  els.gallerySummary.textContent = `${res.boxes.length} 张切片就绪`;

  renderPreviewCanvas();
  renderSlicesGrid();
  updatePythonSnippet();
}

function renderPreviewCanvas() {
  if (!state.sourceCanvas || !state.result) return;
  const canvas = els.previewCanvas;
  const maxRenderW = 1440;
  const scale = state.width > maxRenderW ? maxRenderW / state.width : 1;
  const rw = Math.round(state.width * scale);
  const rh = Math.round(state.height * scale);
  canvas.width = rw;
  canvas.height = rh;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(state.sourceCanvas, 0, 0, rw, rh);

  if (!state.showOverlay) return;
  const boxes = state.result.boxes;
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    const isSel = i === state.selectedIndex;
    const x = b.x * scale;
    const y = b.y * scale;
    const w = b.width * scale;
    const h = b.height * scale;

    ctx.fillStyle = isSel ? 'rgba(37, 99, 235, 0.22)' : 'rgba(37, 99, 235, 0.08)';
    ctx.fillRect(x, y, w, h);

    ctx.lineWidth = isSel ? 3 : 2;
    ctx.strokeStyle = isSel ? '#1d4ed8' : '#2563eb';
    ctx.strokeRect(x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2));

    const label = `#${b.index} (${b.width}×${b.height})`;
    ctx.font = '600 11px "JetBrains Mono", monospace';
    const textW = ctx.measureText(label).width;
    const badgeW = textW + 10;
    const badgeH = 20;
    const bx = w > badgeW + 10 ? x + w - badgeW - 5 : x + 3;
    const by = h > badgeH + 10 ? y + h - badgeH - 5 : y + 3;
    ctx.fillStyle = isSel ? 'rgba(29, 78, 216, 0.94)' : 'rgba(15, 23, 42, 0.82)';
    ctx.fillRect(bx, by, badgeW, badgeH);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, bx + 5, by + 14);
  }
}

function tileToCanvas(tile) {
  const c = document.createElement('canvas');
  c.width = tile.width;
  c.height = tile.height;
  const ctx = c.getContext('2d');
  ctx.putImageData(new ImageData(tile.data, tile.width, tile.height), 0, 0);
  return c;
}

function renderSlicesGrid() {
  const grid = els.slicesGrid;
  grid.innerHTML = '';
  const boxes = state.result?.boxes || [];
  if (boxes.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'meta-line';
    empty.textContent = '未检测到可分割子图，请尝试提高灵敏度或切换切割模式。';
    grid.appendChild(empty);
    return;
  }

  boxes.forEach((box, idx) => {
    const tile = state.tiles[idx];
    const card = document.createElement('article');
    card.className = `slice-card${idx === state.selectedIndex ? ' is-selected' : ''}`;

    const thumbWrap = document.createElement('div');
    thumbWrap.className = 'slice-thumb-wrap';
    const thumbCanvas = tileToCanvas(tile);
    thumbWrap.appendChild(thumbCanvas);

    const info = document.createElement('div');
    info.className = 'slice-card-info';
    const title = document.createElement('span');
    title.className = 'slice-index';
    title.textContent = `#${String(box.index).padStart(2, '0')} · ${box.strategy}`;
    const dims = document.createElement('span');
    dims.className = 'slice-dims';
    dims.textContent = `${tile.width}×${tile.height}`;
    info.append(title, dims);

    const coords = document.createElement('div');
    coords.className = 'meta-line';
    const [tt, tr, tb, tl] = box.trimmedEdges;
    coords.textContent = `(${box.x}, ${box.y}) · 修边 ${tt}/${tr}/${tb}/${tl}`;

    const actions = document.createElement('div');
    actions.className = 'slice-card-actions';

    const inspectBtn = document.createElement('button');
    inspectBtn.type = 'button';
    inspectBtn.className = 'btn-secondary slice-inspect-btn';
    inspectBtn.textContent = '1:1 检验';
    inspectBtn.setAttribute('aria-label', `Inspect slice ${box.index}`);
    inspectBtn.addEventListener('click', () => openLightbox(idx));

    const saveBtn = document.createElement('button');
    saveBtn.type = 'button';
    saveBtn.className = 'btn-secondary';
    saveBtn.textContent = '下载';
    saveBtn.setAttribute('aria-label', `Download slice ${box.index}`);
    saveBtn.addEventListener('click', () => downloadSingleSlice(idx));

    actions.append(inspectBtn, saveBtn);
    card.append(thumbWrap, info, coords, actions);
    grid.appendChild(card);
  });
}

async function encodeTileBytes(tile, format) {
  const c = tileToCanvas(tile);
  const mime = format === 'jpeg' ? 'image/jpeg' : format === 'webp' ? 'image/webp' : 'image/png';
  const quality = format === 'png' ? undefined : 0.95;
  const blob = await new Promise((resolve) => c.toBlob(resolve, mime, quality));
  const buf = await blob.arrayBuffer();
  return { blob, bytes: new Uint8Array(buf) };
}

async function downloadSingleSlice(idx) {
  const box = state.result.boxes[idx];
  const tile = state.tiles[idx];
  const { blob } = await encodeTileBytes(tile, state.format);
  const ext = state.format === 'jpeg' ? 'jpg' : state.format;
  const stem = state.sourceName.replace(/\.[^.]+$/, '');
  triggerBlobDownload(blob, `${stem}_slice_${String(box.index).padStart(2, '0')}.${ext}`);
}

async function downloadAllZip() {
  if (!state.tiles.length) return;
  const origText = els.downloadZipBtn.textContent;
  els.downloadZipBtn.textContent = '打包 ZIP 中...';
  els.downloadZipBtn.disabled = true;
  try {
    const ext = state.format === 'jpeg' ? 'jpg' : state.format;
    const stem = state.sourceName.replace(/\.[^.]+$/, '');
    const entries = [];
    for (let i = 0; i < state.tiles.length; i++) {
      const box = state.result.boxes[i];
      const { bytes } = await encodeTileBytes(state.tiles[i], state.format);
      entries.push({
        name: `${stem}_slice_${String(box.index).padStart(2, '0')}.${ext}`,
        bytes,
      });
    }
    const zipBlob = buildZipBlob(entries);
    triggerBlobDownload(zipBlob, `${stem}_slices.zip`);
  } finally {
    els.downloadZipBtn.textContent = origText;
    els.downloadZipBtn.disabled = false;
  }
}

function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function openLightbox(idx) {
  state.lightboxIndex = idx;
  state.selectedIndex = idx;
  renderPreviewCanvas();
  renderSlicesGrid();
  drawLightboxTile();
  if (!els.lightboxDialog.open) els.lightboxDialog.showModal();
}

function drawLightboxTile() {
  if (state.lightboxIndex < 0) return;
  const box = state.result.boxes[state.lightboxIndex];
  const tile = state.tiles[state.lightboxIndex];
  const zoom = els.zoomToggle.checked ? 2 : 1;
  const srcC = tileToCanvas(tile);

  els.lightboxCanvas.width = tile.width * zoom;
  els.lightboxCanvas.height = tile.height * zoom;
  const ctx = els.lightboxCanvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(srcC, 0, 0, tile.width * zoom, tile.height * zoom);

  els.lightboxTitle.textContent = `切片 #${String(box.index).padStart(2, '0')} 边缘检验`;
  const [tt, tr, tb, tl] = box.trimmedEdges;
  els.lightboxMeta.textContent = `输出 ${tile.width}×${tile.height} · 原图起点 (${box.x}, ${box.y}) · 修边 ${tt}/${tr}/${tb}/${tl}`;
}

function handleUploadedFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas');
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext('2d').drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    els.presetBtns.forEach((b) => b.classList.remove('is-active'));
    setSourceCanvas(c, file.name);
  };
  img.src = url;
}

// Event Wiring
els.presetBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    const preset = btn.dataset.preset;
    state.preset = preset;
    els.presetBtns.forEach((b) => b.classList.toggle('is-active', b === btn));
    if (preset === 'slides') {
      loadSampleImage();
    } else {
      setSourceCanvas(generatePresetCanvas(preset), `${preset}_preset.png`);
    }
  });
});

els.modeBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    state.mode = btn.dataset.mode;
    els.modeBtns.forEach((b) => b.classList.toggle('is-active', b === btn));
    els.gridControls.classList.toggle('is-hidden', state.mode !== 'grid');
    runAnalysis();
  });
});

els.uniformModeBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    state.uniformMode = btn.dataset.uniform;
    els.uniformModeBtns.forEach((b) => b.classList.toggle('is-active', b === btn));
    runAnalysis();
  });
});

els.sensitivityInput.addEventListener('input', (e) => {
  state.sensitivity = Number(e.target.value);
  els.sensitivityValue.textContent = state.sensitivity.toFixed(2);
  runAnalysis();
});

els.minRatioInput.addEventListener('input', (e) => {
  state.minRegionRatio = Number(e.target.value);
  els.minRatioValue.textContent = state.minRegionRatio.toFixed(2);
  runAnalysis();
});

els.bleedTrimInput.addEventListener('input', (e) => {
  state.bleedTrim = Number.parseInt(e.target.value, 10);
  els.bleedTrimValue.textContent = `${state.bleedTrim} px`;
  runAnalysis();
});

els.contourGapInput.addEventListener('input', (e) => {
  state.contourGapPx = Number.parseInt(e.target.value, 10);
  els.contourGapValue.textContent = `${state.contourGapPx} px`;
  runAnalysis();
});

[els.rowsInput, els.colsInput, els.gutterInput].forEach((input) => {
  input.addEventListener('change', () => {
    state.rows = Math.max(1, Number.parseInt(els.rowsInput.value, 10) || 3);
    state.cols = Math.max(1, Number.parseInt(els.colsInput.value, 10) || 3);
    state.gutterPx = Math.max(0, Number.parseInt(els.gutterInput.value, 10) || 0);
    runAnalysis();
  });
});

els.uniformToggle.addEventListener('change', (e) => {
  state.uniformSize = e.target.checked;
  els.uniformOptions.classList.toggle('is-disabled', !state.uniformSize);
  runAnalysis();
});

[els.targetWInput, els.targetHInput].forEach((input) => {
  input.addEventListener('input', runAnalysis);
});

els.formatSelect.addEventListener('change', (e) => {
  state.format = e.target.value;
  updatePythonSnippet();
});

els.overlayToggle.addEventListener('change', (e) => {
  state.showOverlay = e.target.checked;
  renderPreviewCanvas();
});

els.previewCanvas.addEventListener('click', (e) => {
  if (!state.result) return;
  const rect = els.previewCanvas.getBoundingClientRect();
  const scaleX = state.width / rect.width;
  const scaleY = state.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;
  const hitIdx = state.result.boxes.findIndex(
    (b) => px >= b.x && px <= b.x + b.width && py >= b.y && py <= b.y + b.height
  );
  if (hitIdx >= 0) {
    state.selectedIndex = hitIdx;
    renderPreviewCanvas();
    renderSlicesGrid();
  }
});

els.fileInput.addEventListener('change', (e) => {
  if (e.target.files?.[0]) handleUploadedFile(e.target.files[0]);
});

els.dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  els.dropZone.classList.add('is-dragover');
});
els.dropZone.addEventListener('dragleave', () => {
  els.dropZone.classList.remove('is-dragover');
});
els.dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  els.dropZone.classList.remove('is-dragover');
  if (e.dataTransfer?.files?.[0]) handleUploadedFile(e.dataTransfer.files[0]);
});

els.copyCodeBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(els.pythonSnippet.textContent);
    els.copyCodeBtn.textContent = '已复制';
    setTimeout(() => {
      els.copyCodeBtn.textContent = '复制代码';
    }, 1500);
  } catch {
    els.copyCodeBtn.textContent = '请手动复制';
  }
});

els.downloadZipBtn.addEventListener('click', downloadAllZip);
els.zoomToggle.addEventListener('change', drawLightboxTile);
els.lightboxDownloadBtn.addEventListener('click', () => {
  if (state.lightboxIndex >= 0) downloadSingleSlice(state.lightboxIndex);
});
els.closeLightboxBtn.addEventListener('click', () => {
  els.lightboxDialog.close();
});
els.lightboxDialog.addEventListener('click', (e) => {
  if (e.target === els.lightboxDialog) els.lightboxDialog.close();
});

loadSampleImage();
