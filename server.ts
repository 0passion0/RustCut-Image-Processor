import express from 'express';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import multer from 'multer';
import sharp from 'sharp';
import JSZip from 'jszip';
import { createServer as createViteServer } from 'vite';
import { analyzeAndSplit, type SplitBox, type SplitOptions } from './src/utils/splitterCore';

const SAMPLE_IMAGE_PATH = path.resolve(
  process.cwd(),
  'public/assets/aistudio/e3e01d85a65848fa4acdfaa7bcd4ac0d20a578b05b20476af8d929802f2366e9_e3e01d85.png',
);

interface CachedJob {
  id: string;
  sourceBuffer: Buffer;
  sourceMime: string;
  rawData: Buffer;
  channels: 1 | 2 | 3 | 4;
  width: number;
  height: number;
  filename: string;
  createdAt: number;
}

const jobCache = new Map<string, CachedJob>();
const MAX_CACHED_JOBS = 12;

function cacheJob(job: CachedJob) {
  jobCache.set(job.id, job);
  if (jobCache.size > MAX_CACHED_JOBS) {
    const oldestKey = jobCache.keys().next().value;
    if (oldestKey) jobCache.delete(oldestKey);
  }
}

async function generatePresetBuffer(presetId: string): Promise<{ buffer: Buffer; filename: string }> {
  if (presetId === 'preset-bento') {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000">
      <rect x="0" y="0" width="740" height="1000" fill="#0f294a"/>
      <circle cx="370" cy="420" r="170" fill="#1d4ed8" opacity="0.55"/>
      <text x="80" y="140" font-family="sans-serif" font-size="48" font-weight="bold" fill="#ffffff">BENTO HERO PANEL</text>
      <text x="80" y="200" font-family="sans-serif" font-size="24" fill="#93c5fd">Full-height left showcase region</text>
      <rect x="742" y="0" width="858" height="498" fill="#f8fafc"/>
      <rect x="810" y="90" width="240" height="240" rx="16" fill="#0ea5e9"/>
      <text x="1090" y="170" font-family="sans-serif" font-size="38" font-weight="bold" fill="#0f172a">TOP RIGHT METRIC</text>
      <text x="1090" y="225" font-family="sans-serif" font-size="22" fill="#475569">Asymmetric horizontal cut</text>
      <rect x="742" y="500" width="858" height="500" fill="#1e293b"/>
      <rect x="810" y="590" width="720" height="300" rx="12" fill="#334155"/>
      <text x="850" y="680" font-family="sans-serif" font-size="36" font-weight="bold" fill="#f8fafc">BOTTOM RIGHT MODULE</text>
      <text x="850" y="735" font-family="sans-serif" font-size="22" fill="#cbd5e1">Independent seam alignment</text>
    </svg>`;
    const buffer = await sharp(Buffer.from(svg)).png().toBuffer();
    return { buffer, filename: 'bento_asymmetric_3panel.png' };
  }

  if (presetId === 'preset-gutter') {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1080">
      <rect width="1600" height="1080" fill="#ffffff"/>
      <rect x="40" y="40" width="740" height="480" fill="#1e3a8a"/>
      <text x="90" y="130" font-family="sans-serif" font-size="36" font-weight="bold" fill="#ffffff">CARD 01 · LAB REPORT</text>
      <circle cx="410" cy="310" r="95" fill="#3b82f6"/>
      <rect x="820" y="40" width="740" height="480" fill="#0f766e"/>
      <text x="870" y="130" font-family="sans-serif" font-size="36" font-weight="bold" fill="#ffffff">CARD 02 · BIO ASSAY</text>
      <rect x="920" y="220" width="540" height="180" rx="12" fill="#14b8a6"/>
      <rect x="40" y="560" width="740" height="480" fill="#b45309"/>
      <text x="90" y="650" font-family="sans-serif" font-size="36" font-weight="bold" fill="#ffffff">CARD 03 · FORMULATION</text>
      <rect x="140" y="740" width="540" height="180" rx="12" fill="#f59e0b"/>
      <rect x="820" y="560" width="740" height="480" fill="#334155"/>
      <text x="870" y="650" font-family="sans-serif" font-size="36" font-weight="bold" fill="#ffffff">CARD 04 · SUMMARY</text>
      <circle cx="1190" cy="830" r="95" fill="#64748b"/>
    </svg>`;
    const buffer = await sharp(Buffer.from(svg)).png().toBuffer();
    return { buffer, filename: 'gutter_2x2_cards.png' };
  }

  // preset-contours: 5 isolated objects on clean studio backdrop
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="950">
    <rect width="1500" height="950" fill="#f8fafc"/>
    <rect x="80" y="70" width="320" height="280" rx="28" fill="#1d4ed8"/>
    <circle cx="240" cy="210" r="70" fill="#93c5fd"/>
    <rect x="540" y="90" width="380" height="240" rx="20" fill="#047857"/>
    <rect x="610" y="150" width="240" height="120" rx="12" fill="#6ee7b7"/>
    <rect x="1060" y="65" width="340" height="310" rx="32" fill="#be123c"/>
    <polygon points="1230,130 1320,290 1140,290" fill="#fda4af"/>
    <rect x="170" y="490" width="460" height="340" rx="24" fill="#0f172a"/>
    <circle cx="400" cy="660" r="90" fill="#38bdf8"/>
    <rect x="810" y="470" width="510" height="370" rx="28" fill="#d97706"/>
    <rect x="910" y="570" width="310" height="170" rx="18" fill="#fde68a"/>
  </svg>`;
  const buffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return { buffer, filename: 'isolated_contours_5objs.png' };
}

function collectDirFiles(baseDir: string, currentDir = baseDir): Array<{ path: string; content: string }> {
  const entries = fs.readdirSync(currentDir, { withFileTypes: true });
  const results: Array<{ path: string; content: string }> = [];
  for (const entry of entries) {
    const full = path.join(currentDir, entry.name);
    if (entry.isDirectory()) {
      results.push(...collectDirFiles(baseDir, full));
    } else {
      results.push({
        path: path.relative(baseDir, full).replace(/\\/g, '/'),
        content: fs.readFileSync(full, 'utf8'),
      });
    }
  }
  results.sort((a, b) => a.path.localeCompare(b.path));
  return results;
}

async function startServer() {
  const app = express();
  const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 40 * 1024 * 1024 } });

  app.use(express.json({ limit: '10mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true });
  });

  app.post('/api/analyze', upload.single('image'), async (req, res) => {
    try {
      let options: Partial<SplitOptions> = {};
      const hasFile = Boolean(req.file);
      const reqSource = typeof req.body?.source === 'string' ? req.body.source : undefined;
      const reqJobId = typeof req.body?.jobId === 'string' ? req.body.jobId : undefined;

      if (hasFile && req.file) {
        if (typeof req.body.options === 'string') {
          options = JSON.parse(req.body.options);
        }
      } else {
        options = req.body?.options || {};
      }

      // Fast-path: re-analyze directly from cached raw RAM pixels (0ms image decode/re-encode!)
      if (!hasFile && !reqSource && reqJobId && jobCache.has(reqJobId)) {
        const cached = jobCache.get(reqJobId)!;
        const analysis = analyzeAndSplit(
          {
            data: cached.rawData,
            width: cached.width,
            height: cached.height,
            channels: cached.channels,
          },
          options,
        );
        res.json({
          jobId: cached.id,
          filename: cached.filename,
          imageUrl: `/api/source-image/${cached.id}`,
          analysis,
        });
        return;
      }

      let imageBuffer: Buffer;
      let filename = 'uploaded_image.png';

      if (hasFile && req.file) {
        imageBuffer = req.file.buffer;
        filename = req.file.originalname || filename;
      } else if (reqSource && reqSource.startsWith('preset-')) {
        const preset = await generatePresetBuffer(reqSource);
        imageBuffer = preset.buffer;
        filename = preset.filename;
      } else {
        imageBuffer = fs.readFileSync(SAMPLE_IMAGE_PATH);
        filename = 'bayer_slides_3x3_sample.png';
      }

      const meta = await sharp(imageBuffer).metadata();
      const needsRotation = Boolean(meta.orientation && meta.orientation > 1);
      const { data, info } = await sharp(imageBuffer)
        .rotate()
        .raw()
        .ensureAlpha()
        .toBuffer({ resolveWithObject: true });

      const analysis = analyzeAndSplit(
        { data, width: info.width, height: info.height, channels: info.channels },
        options,
      );

      let sourceBuffer = imageBuffer;
      let sourceMime =
        meta.format === 'jpeg'
          ? 'image/jpeg'
          : meta.format === 'webp'
            ? 'image/webp'
            : 'image/png';

      // Only re-encode when EXIF orientation rotation changed pixel coordinates or format isn't browser-native
      if (needsRotation || (meta.format !== 'png' && meta.format !== 'jpeg' && meta.format !== 'webp')) {
        sourceBuffer = await sharp(data, {
          raw: { width: info.width, height: info.height, channels: info.channels },
        })
          .png({ compressionLevel: 6 })
          .toBuffer();
        sourceMime = 'image/png';
      }

      const newJobId = crypto.randomUUID();
      cacheJob({
        id: newJobId,
        sourceBuffer,
        sourceMime,
        rawData: data,
        channels: info.channels,
        width: info.width,
        height: info.height,
        filename,
        createdAt: Date.now(),
      });

      res.json({
        jobId: newJobId,
        filename,
        imageUrl: `/api/source-image/${newJobId}`,
        analysis,
      });
    } catch (err) {
      console.error('Analyze error:', err);
      res.status(500).json({ error: err instanceof Error ? err.message : 'Failed to analyze image' });
    }
  });

  app.get('/api/source-image/:jobId', (req, res) => {
    const job = jobCache.get(req.params.jobId);
    if (!job) {
      res.status(404).send('Job expired');
      return;
    }
    res.setHeader('Content-Type', job.sourceMime);
    res.setHeader('Cache-Control', 'private, max-age=3600');
    res.send(job.sourceBuffer);
  });

  app.get('/api/crop/:jobId', async (req, res) => {
    const job = jobCache.get(req.params.jobId);
    if (!job) {
      res.status(404).send('Job expired');
      return;
    }
    try {
      const left = Math.max(0, Math.min(job.width - 1, parseInt(String(req.query.x || '0'), 10)));
      const top = Math.max(0, Math.min(job.height - 1, parseInt(String(req.query.y || '0'), 10)));
      const width = Math.max(1, Math.min(job.width - left, parseInt(String(req.query.w || '100'), 10)));
      const height = Math.max(1, Math.min(job.height - top, parseInt(String(req.query.h || '100'), 10)));
      const fmt = String(req.query.format || 'png').toLowerCase();
      const download = req.query.download === '1';
      const idx = String(req.query.index || '1').padStart(2, '0');

      let pipeline = sharp(job.rawData, {
        raw: { width: job.width, height: job.height, channels: job.channels },
      }).extract({ left, top, width, height });
      let mime = 'image/png';
      let ext = 'png';
      if (fmt === 'webp') {
        pipeline = pipeline.webp({ quality: 96 });
        mime = 'image/webp';
        ext = 'webp';
      } else if (fmt === 'jpg' || fmt === 'jpeg') {
        pipeline = pipeline.flatten({ background: '#ffffff' }).jpeg({ quality: 95 });
        mime = 'image/jpeg';
        ext = 'jpg';
      } else {
        pipeline = pipeline.png({ compressionLevel: 6 });
      }

      const out = await pipeline.toBuffer();
      res.setHeader('Content-Type', mime);
      if (download) {
        const stem = path.parse(job.filename).name;
        res.setHeader('Content-Disposition', `attachment; filename="${stem}_slice_${idx}.${ext}"`);
      }
      res.send(out);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : 'Crop failed' });
    }
  });

  app.post('/api/export-zip', async (req, res) => {
    try {
      const { jobId, boxes, format = 'png' } = req.body as {
        jobId: string;
        boxes: SplitBox[];
        format?: 'png' | 'webp' | 'jpg';
      };
      const job = jobCache.get(jobId);
      if (!job) {
        res.status(404).json({ error: 'Image session expired, please re-analyze.' });
        return;
      }

      const zip = new JSZip();
      const stem = path.parse(job.filename).name;
      const ext = format === 'jpg' ? 'jpg' : format === 'webp' ? 'webp' : 'png';

      await Promise.all(
        boxes.map(async (box, idx) => {
          const left = Math.max(0, Math.min(job.width - 1, Math.round(box.x)));
          const top = Math.max(0, Math.min(job.height - 1, Math.round(box.y)));
          const width = Math.max(1, Math.min(job.width - left, Math.round(box.width)));
          const height = Math.max(1, Math.min(job.height - top, Math.round(box.height)));

          let pipe = sharp(job.rawData, {
            raw: { width: job.width, height: job.height, channels: job.channels },
          }).extract({ left, top, width, height });
          if (ext === 'webp') pipe = pipe.webp({ quality: 96 });
          else if (ext === 'jpg') pipe = pipe.flatten({ background: '#ffffff' }).jpeg({ quality: 95 });
          else pipe = pipe.png({ compressionLevel: 6 });

          const buf = await pipe.toBuffer();
          const fileName = `${stem}_slice_${String(idx + 1).padStart(2, '0')}.${ext}`;
          zip.file(fileName, buf);
        }),
      );

      const zipBuf = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
      res.setHeader('Content-Type', 'application/zip');
      res.setHeader('Content-Disposition', `attachment; filename="${stem}_slices.zip"`);
      res.send(zipBuf);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : 'ZIP export failed' });
    }
  });

  app.get('/api/python-rust/files', (_req, res) => {
    const dir = path.resolve(process.cwd(), 'python_rust_splitter');
    if (!fs.existsSync(dir)) {
      res.status(404).json({ error: 'Engine directory not found' });
      return;
    }
    res.json({ files: collectDirFiles(dir) });
  });

  app.get('/api/python-rust/download', async (_req, res) => {
    const dir = path.resolve(process.cwd(), 'python_rust_splitter');
    const files = collectDirFiles(dir);
    const zip = new JSZip();
    const root = zip.folder('smart-image-splitter')!;
    for (const f of files) {
      root.file(f.path, f.content);
    }
    const zipBuf = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', 'attachment; filename="smart-image-splitter-rust-python.zip"');
    res.send(zipBuf);
  });

  if (process.env.NODE_ENV === 'production') {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(3000, '0.0.0.0', () => {
    console.log('SliceCraft Studio listening on http://0.0.0.0:3000');
  });
}

startServer();
