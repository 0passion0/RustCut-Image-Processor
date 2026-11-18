import { Download, ZoomIn, CheckSquare, Square } from 'lucide-react';
import type { ExportFormat, SplitBox } from '../types/splitter';

interface SliceGalleryProps {
  jobId: string | null;
  boxes: SplitBox[];
  selectedBoxId: string | null;
  exportFormat: ExportFormat;
  onSelectBox: (id: string) => void;
  onHoverBox: (id: string | null) => void;
  onToggleInclude: (id: string) => void;
  onInspectBox: (box: SplitBox) => void;
}

export function SliceGallery({
  jobId,
  boxes,
  selectedBoxId,
  exportFormat,
  onSelectBox,
  onHoverBox,
  onToggleInclude,
  onInspectBox,
}: SliceGalleryProps) {
  if (!jobId || boxes.length === 0) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-8 text-center">
        <p className="text-sm text-slate-600">暂未检测到可切割的子图区域，可尝试提高左侧“边界检测灵敏度”或切换至“均匀网格”模式。</p>
      </section>
    );
  }

  const includedCount = boxes.filter((b) => b.included !== false).length;

  return (
    <section className="space-y-4" aria-label="切割输出子图列表" data-testid="slice-gallery">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-sm">
          <h2 className="font-display text-base font-semibold text-slate-900">切割输出子图</h2>
          <span className="text-slate-400" aria-hidden="true">·</span>
          <span className="font-mono text-xs text-slate-600 tabular-nums">
            已选 {includedCount} / {boxes.length} 张
          </span>
        </div>
        <p className="text-xs text-slate-500">点击任意切片可在画布定位，双击或点击放大镜可 1:1 检查边缘零杂边效果</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
        {boxes.map((box) => {
          const cropUrl = `/api/crop/${jobId}?x=${box.x}&y=${box.y}&w=${box.width}&h=${box.height}&format=${exportFormat}&index=${box.index}`;
          const downloadUrl = `${cropUrl}&download=1`;
          const isSelected = box.id === selectedBoxId;
          const isIncluded = box.included !== false;
          const totalTrimmed =
            box.trimmedEdges.top +
            box.trimmedEdges.right +
            box.trimmedEdges.bottom +
            box.trimmedEdges.left;

          return (
            <article
              key={box.id}
              onMouseEnter={() => onHoverBox(box.id)}
              onMouseLeave={() => onHoverBox(null)}
              className={`rounded-xl border bg-white overflow-hidden transition-colors ${
                isSelected ? 'border-blue-600 ring-1 ring-blue-600' : 'border-slate-200'
              } ${!isIncluded ? 'opacity-60' : ''}`}
            >
              {/* Thumbnail (100% unobstructed) */}
              <div
                onClick={() => onSelectBox(box.id)}
                onDoubleClick={() => onInspectBox(box)}
                className="relative aspect-[16/10] bg-slate-100 flex items-center justify-center overflow-hidden cursor-pointer border-b border-slate-100"
              >
                <img
                  src={cropUrl}
                  alt={`切片 ${box.index}`}
                  loading="lazy"
                  referrerPolicy="no-referrer"
                  className="max-h-full max-w-full object-contain"
                />
              </div>

              {/* Clean unboxed metadata + actions */}
              <div className="p-3.5 flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => onToggleInclude(box.id)}
                      aria-label={isIncluded ? `取消勾选切片 #${box.index}` : `勾选切片 #${box.index}`}
                      className="text-slate-600 hover:text-blue-600 shrink-0"
                    >
                      {isIncluded ? (
                        <CheckSquare className="h-4 w-4 text-blue-600" aria-hidden="true" />
                      ) : (
                        <Square className="h-4 w-4" aria-hidden="true" />
                      )}
                    </button>
                    <span className="text-sm font-semibold text-slate-900 tabular-nums">
                      切片 #{String(box.index).padStart(2, '0')}
                    </span>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-1.5 font-mono text-[11px] text-slate-500 tabular-nums">
                    <span>{box.width}×{box.height}</span>
                    <span aria-hidden="true">·</span>
                    <span>({box.x},{box.y})</span>
                    {totalTrimmed > 0 && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="text-emerald-700">去杂边 {totalTrimmed}px</span>
                      </>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    type="button"
                    onClick={() => onInspectBox(box)}
                    aria-label={`放大检查切片 #${box.index}`}
                    title="1:1 边缘检查"
                    className="inline-flex items-center justify-center rounded-md border border-slate-200 p-1.5 text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                  >
                    <ZoomIn className="h-3.5 w-3.5" aria-hidden="true" />
                  </button>
                  <a
                    href={downloadUrl}
                    download
                    aria-label={`下载切片 #${box.index}`}
                    className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap"
                  >
                    <Download className="h-3.5 w-3.5" aria-hidden="true" />
                    <span>下载</span>
                  </a>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
