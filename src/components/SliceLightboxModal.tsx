import { useState } from 'react';
import { X, Download } from 'lucide-react';
import type { ExportFormat, SplitBox } from '../types/splitter';

interface SliceLightboxModalProps {
  jobId: string;
  box: SplitBox;
  exportFormat: ExportFormat;
  onClose: () => void;
}

export function SliceLightboxModal({ jobId, box, exportFormat, onClose }: SliceLightboxModalProps) {
  const [zoomed, setZoomed] = useState(false);
  const cropUrl = `/api/crop/${jobId}?x=${box.x}&y=${box.y}&w=${box.width}&h=${box.height}&format=${exportFormat}&index=${box.index}`;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/70 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={`切片 #${box.index} 边缘像素检查`}
    >
      <div className="w-full max-w-5xl rounded-xl border border-slate-200 bg-white overflow-hidden shadow-xl flex flex-col max-h-[90vh]">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 px-5 py-3.5">
          <div className="flex items-center gap-2 text-sm">
            <h3 className="font-display font-semibold text-slate-900">
              切片 #{String(box.index).padStart(2, '0')} 边缘检查
            </h3>
            <span className="text-slate-400" aria-hidden="true">·</span>
            <span className="font-mono text-xs text-slate-600 tabular-nums">
              {box.width} × {box.height} px · 坐标 ({box.x}, {box.y})
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setZoomed((z) => !z)}
              className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
            >
              {zoomed ? '适应窗口' : '1:1 原始像素'}
            </button>
            <a
              href={`${cropUrl}&download=1`}
              download
              className="inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
            >
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              <span>下载单张</span>
            </a>
            <button
              type="button"
              onClick={onClose}
              aria-label="关闭预览"
              className="rounded-md p-1.5 text-slate-500 hover:bg-slate-100 hover:text-slate-900"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto bg-slate-100 p-6 flex items-center justify-center">
          <img
            src={cropUrl}
            alt={`切片 ${box.index} 原图`}
            referrerPolicy="no-referrer"
            className={`border border-slate-300 bg-white shadow-sm ${
              zoomed ? 'max-w-none' : 'max-h-[68vh] max-w-full object-contain'
            }`}
          />
        </div>
      </div>
    </div>
  );
}
