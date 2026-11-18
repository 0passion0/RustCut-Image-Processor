import { useEffect, useState } from 'react';
import { Copy, Check, Download, Terminal } from 'lucide-react';
import type { EngineSourceFile } from '../types/splitter';

export function CodeEngineView() {
  const [files, setFiles] = useState<EngineSourceFile[]>([]);
  const [selectedPath, setSelectedPath] = useState<string>('src/lib.rs');
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/python-rust/files')
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.files)) {
          setFiles(data.files);
          const libRs = data.files.find((f: EngineSourceFile) => f.path === 'src/lib.rs');
          if (libRs) setSelectedPath(libRs.path);
          else if (data.files[0]) setSelectedPath(data.files[0].path);
        }
      })
      .finally(() => setLoading(false));
  }, []);

  const currentFile = files.find((f) => f.path === selectedPath);

  const handleCopy = async () => {
    if (!currentFile) return;
    await navigator.clipboard.writeText(currentFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="font-display text-2xl font-bold text-slate-900">
            Python + Rust (PyO3 / Rayon / OpenCV) 生产级切割引擎
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            底层采用 Rust + Rayon 并行计算像素能量剖面与无锁裁剪，通过 PyO3 + maturin 暴露为零拷贝 Python 原生扩展模块，并内置 OpenCV 轮廓分割与 SAM2 / ONNX 机器学习分割钩子。
          </p>
        </div>

        <a
          href="/api/python-rust/download"
          download
          className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 whitespace-nowrap shrink-0"
        >
          <Download className="h-4 w-4" aria-hidden="true" />
          <span>下载完整 Rust + Python 源码包 (.zip)</span>
        </a>
      </div>

      {/* Quick build guide */}
      <div className="rounded-xl border border-slate-200 bg-slate-900 p-4 text-slate-100">
        <div className="flex items-center gap-2 text-xs font-medium text-slate-300 mb-2">
          <Terminal className="h-4 w-4 text-blue-400" aria-hidden="true" />
          <span>本地一键编译 Rust 原生扩展并运行切割 CLI</span>
        </div>
        <pre className="font-mono text-xs leading-relaxed text-slate-100 overflow-x-auto">
{`unzip smart-image-splitter-rust-python.zip && cd smart-image-splitter
pip install maturin opencv-python-headless numpy pillow
maturin develop --release
smart-split input_grid.png -o ./output_slices --mode auto --bleed-trim 2 --format png`}
        </pre>
      </div>

      {/* File browser */}
      <div className="grid grid-cols-1 lg:grid-cols-12 rounded-xl border border-slate-200 bg-white overflow-hidden">
        <div className="lg:col-span-3 border-b lg:border-b-0 lg:border-r border-slate-200 bg-slate-50 p-3 space-y-1">
          <p className="px-2 py-1 text-xs font-semibold text-slate-500">工程文件结构</p>
          {loading ? (
            <p className="px-2 py-2 text-xs text-slate-500">加载源码中...</p>
          ) : (
            files.map((f) => (
              <button
                key={f.path}
                type="button"
                onClick={() => setSelectedPath(f.path)}
                className={`w-full rounded-md px-2.5 py-2 text-left font-mono text-xs transition-colors truncate ${
                  selectedPath === f.path
                    ? 'bg-slate-900 text-white'
                    : 'text-slate-700 hover:bg-slate-200/70'
                }`}
              >
                {f.path}
              </button>
            ))
          )}
        </div>

        <div className="lg:col-span-9 flex flex-col">
          <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2.5 bg-white">
            <span className="font-mono text-xs font-medium text-slate-800">{selectedPath}</span>
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" />
                  <span>已复制</span>
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>复制代码</span>
                </>
              )}
            </button>
          </div>
          <pre className="p-4 font-mono text-xs leading-relaxed text-slate-800 overflow-auto max-h-[580px] bg-white">
            {currentFile?.content || ''}
          </pre>
        </div>
      </div>
    </div>
  );
}
