import { Eye, EyeOff, Maximize2 } from 'lucide-react';
import type { SplitAnalysisResult, SplitBox } from '../types/splitter';

interface CanvasStageProps {
  imageUrl: string | null;
  filename: string;
  analysis: SplitAnalysisResult | null;
  boxes: SplitBox[];
  selectedBoxId: string | null;
  hoveredBoxId: string | null;
  showOverlay: boolean;
  onToggleOverlay: () => void;
  onSelectBox: (id: string) => void;
  onHoverBox: (id: string | null) => void;
  onInspectBox: (box: SplitBox) => void;
  loading: boolean;
}

export function CanvasStage({
  imageUrl,
  filename,
  analysis,
  boxes,
  selectedBoxId,
  hoveredBoxId,
  showOverlay,
  onToggleOverlay,
  onSelectBox,
  onHoverBox,
  onInspectBox,
  loading,
}: CanvasStageProps) {
  const modeLabel =
    analysis?.detectedMode === 'xycut'
      ? '递归 XY-Cut 局部对齐'
      : analysis?.detectedMode === 'contour'
        ? '轮廓连通域分割'
        : '均匀网格切分';

  return (
    <section className="rounded-xl border border-slate-200 bg-white overflow-hidden" aria-label="画布预览与切割线叠加">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
          <span className="font-semibold text-slate-900 truncate max-w-[260px]">{filename}</span>
          {analysis && (
            <>
              <span aria-hidden="true">·</span>
              <span className="font-mono tabular-nums">
                {analysis.imageWidth} × {analysis.imageHeight} px
              </span>
              <span aria-hidden="true">·</span>
              <span>{modeLabel}</span>
              <span aria-hidden="true">·</span>
              <span className="font-mono tabular-nums">{boxes.length} 个切片</span>
              <span aria-hidden="true">·</span>
              <span className="font-mono tabular-nums">{analysis.elapsedMs} ms</span>
            </>
          )}
        </div>

        <button
          type="button"
          onClick={onToggleOverlay}
          className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap shrink-0"
        >
          {showOverlay ? (
            <>
              <EyeOff className="h-3.5 w-3.5" aria-hidden="true" />
              <span>隐藏切割框</span>
            </>
          ) : (
            <>
              <Eye className="h-3.5 w-3.5" aria-hidden="true" />
              <span>显示切割框</span>
            </>
          )}
        </button>
      </div>

      {/* Viewport */}
      <div className="relative bg-slate-100/80 p-4 sm:p-6 flex items-center justify-center min-h-[360px]">
        {loading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/70 backdrop-blur-[1px]">
            <div className="rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-800 shadow-sm">
              正在执行边缘能量分析与无缝切割...
            </div>
          </div>
        )}

        {imageUrl && analysis ? (
          <div
            className="relative inline-block max-w-full shadow-sm border border-slate-200 bg-white"
            data-testid="canvas-overlay-container"
          >
            <img
              src={imageUrl}
              alt={filename}
              referrerPolicy="no-referrer"
              className="block max-h-[560px] w-auto max-w-full select-none"
            />

            {showOverlay && (
              <div className="absolute inset-0">
                {boxes.map((box) => {
                  const leftPct = (box.x / analysis.imageWidth) * 100;
                  const topPct = (box.y / analysis.imageHeight) * 100;
                  const widthPct = (box.width / analysis.imageWidth) * 100;
                  const heightPct = (box.height / analysis.imageHeight) * 100;
                  const isSelected = box.id === selectedBoxId;
                  const isHovered = box.id === hoveredBoxId;
                  const included = box.included !== false;

                  return (
                    <button
                      key={box.id}
                      type="button"
                      onClick={() => onSelectBox(box.id)}
                      onDoubleClick={() => onInspectBox(box)}
                      onMouseEnter={() => onHoverBox(box.id)}
                      onMouseLeave={() => onHoverBox(null)}
                      aria-label={`选择切片 #${box.index} (${box.width}×${box.height})`}
                      style={{
                        left: `${leftPct}%`,
                        top: `${topPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                      }}
                      className={`group absolute transition-colors focus-visible:outline-2 focus-visible:outline-blue-600 ${
                        !included
                          ? 'border border-dashed border-slate-400 bg-slate-900/25 opacity-55'
                          : isSelected
                            ? 'border-2 border-blue-600 bg-blue-600/12 z-10'
                            : isHovered
                              ? 'border-2 border-blue-500 bg-blue-500/10 z-10'
                              : 'border border-blue-500/80 bg-transparent hover:bg-blue-500/8'
                      }`}
                    >
                      <span
                        className={`pointer-events-none absolute left-1.5 bottom-1.5 inline-flex items-center rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold leading-none ${
                          isSelected || isHovered
                            ? 'bg-blue-600 text-white'
                            : 'bg-slate-900/65 text-white'
                        }`}
                      >
                        #{box.index}
                      </span>

                      {(isSelected || isHovered) && (
                        <span className="pointer-events-none absolute right-1.5 bottom-1.5 inline-flex items-center gap-1 rounded bg-slate-900/85 px-1.5 py-0.5 font-mono text-[10px] text-white">
                          {box.width}×{box.height}
                          <Maximize2 className="h-2.5 w-2.5" aria-hidden="true" />
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div className="py-16 text-center text-sm text-slate-500">
            请上传图像或从左侧选择测试样本开始自动切割
          </div>
        )}
      </div>
    </section>
  );
}
