import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const PORT = Number(process.env.PORT) || 3000;

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
};

const ROUTES = {
  '/': path.join(__dirname, 'index.html'),
  '/index.html': path.join(__dirname, 'index.html'),
  '/styles.css': path.join(__dirname, 'styles.css'),
  '/app.js': path.join(__dirname, 'app.js'),
  '/splitterCore.mjs': path.join(__dirname, 'splitterCore.mjs'),
  '/zip.mjs': path.join(__dirname, 'zip.mjs'),
  '/examples/sample_slides.png': path.join(ROOT_DIR, 'examples', 'sample_slides.png'),
};

const server = http.createServer((req, res) => {
  const urlPath = (req.url || '/').split('?')[0];
  const filePath = ROUTES[urlPath];
  if (!filePath || !fs.existsSync(filePath)) {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  res.writeHead(200, {
    'Content-Type': MIME_TYPES[ext] || 'application/octet-stream',
    'Cache-Control': 'no-cache',
  });
  fs.createReadStream(filePath).pipe(res);
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`SliceCraft Test Studio running at http://0.0.0.0:${PORT}`);
});
