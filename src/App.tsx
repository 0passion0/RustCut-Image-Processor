/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { HeaderBar, type ActiveTab } from './components/HeaderBar';
import { ControlSidebar } from './components/ControlSidebar';
import { CanvasStage } from './components/CanvasStage';
import { SliceGallery } from './components/SliceGallery';
import { SliceLightboxModal } from './components/SliceLightboxModal';
import { CodeEngineView } from './components/CodeEngineView';
import { AlgorithmView } from './components/AlgorithmView';
import { PresetsView } from './components/PresetsView';
import type {
  AnalyzeResponse,
  ExportFormat,
  SplitAnalysisResult,
  SplitBox,
  SplitOptions,
} from './types/splitter';

const INITIAL_OPTIONS: SplitOptions = {
  mode: 'auto',
  sensitivity: 0.65,
  minRegionRatio: 0.08,
  bleedTrim: 2,
  rows: 3,
  cols: 3,
  gutterPx: 0,
  contourGapPx: 8,
};

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('workspace');
  const [options, setOptions] = useState<SplitOptions>(INITIAL_OPTIONS);
  const [exportFormat, setExportFormat] = useState<ExportFormat>('png');
  const [activeSource, setActiveSource] = useState<string>('sample');
  const [jobId, setJobId] = useState<string | null>(null);
  const [filename, setFilename] = useState<string>('bayer_slides_3x3_sample.png');
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<SplitAnalysisResult | null>(null);
  const [boxes, setBoxes] = useState<SplitBox[]>([]);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [hoveredBoxId, setHoveredBoxId] = useState<string | null>(null);
  const [inspectedBox, setInspectedBox] = useState<SplitBox | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const jobIdRef = useRef<string | null>(null);
  jobIdRef.current = jobId;

  const runAnalysis = useCallback(
    async (payload: { file?: File; source?: string; useCachedJob?: boolean; opts?: SplitOptions }) => {
      setLoading(true);
      setErrorMsg(null);
      try {
        const opts = payload.opts ?? options;
        let res: Response;

        if (payload.file) {
          const form = new FormData();
          form.append('image', payload.file);
          form.append('options', JSON.stringify(opts));
          res = await fetch('/api/analyze', { method: 'POST', body: form });
        } else {
          res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              source: payload.source,
              jobId: payload.useCachedJob ? jobIdRef.current : undefined,
              options: opts,
            }),
          });
        }

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.error || `分析失败 (${res.status})`);
        }

        const data: AnalyzeResponse = await res.json();
        setJobId(data.jobId);
        setFilename(data.filename);
        setImageUrl(data.imageUrl);
        setAnalysis(data.analysis);
        const nextBoxes = data.analysis.boxes.map((b) => ({ ...b, included: true }));
        setBoxes(nextBoxes);
        setSelectedBoxId(nextBoxes[0]?.id ?? null);
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : '图像切割请求失败');
      } finally {
        setLoading(false);
      }
    },
    [options],
  );

  // Initial mount: automatically analyze attached sample image
  useEffect(() => {
    void runAnalysis({ source: 'sample', opts: INITIAL_OPTIONS });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-analyze when options change after initial load
  const handleOptionsChange = (next: SplitOptions) => {
    setOptions(next);
    if (jobId) {
      void runAnalysis({ useCachedJob: true, opts: next });
    }
  };

  const handleUploadFile = (file: File) => {
    setActiveSource('custom');
    setActiveTab('workspace');
    void runAnalysis({ file, opts: options });
  };

  const handleLoadSource = (sourceId: string) => {
    setActiveSource(sourceId);
    setActiveTab('workspace');
    void runAnalysis({ source: sourceId, opts: options });
  };

  const handleUpdateBox = (id: string, patch: Partial<SplitBox>) => {
    setBoxes((prev) =>
      prev.map((b) => {
        if (b.id !== id) return b;
        const maxW = analysis?.imageWidth ?? 9999;
        const maxH = analysis?.imageHeight ?? 9999;
        const nx = Math.max(0, Math.min(maxW - 8, patch.x ?? b.x));
        const ny = Math.max(0, Math.min(maxH - 8, patch.y ?? b.y));
        const nw = Math.max(8, Math.min(maxW - nx, patch.width ?? b.width));
        const nh = Math.max(8, Math.min(maxH - ny, patch.height ?? b.height));
        return { ...b, ...patch, x: nx, y: ny, width: nw, height: nh };
      }),
    );
  };

  const handleDeleteBox = (id: string) => {
    setBoxes((prev) => {
      const next = prev.filter((b) => b.id !== id).map((b, i) => ({ ...b, index: i + 1 }));
      if (selectedBoxId === id) setSelectedBoxId(next[0]?.id ?? null);
      return next;
    });
  };

  const handleAddManualBox = () => {
    if (!analysis) return;
    const w = Math.max(40, Math.round(analysis.imageWidth * 0.25));
    const h = Math.max(40, Math.round(analysis.imageHeight * 0.25));
    const newId = `slice-manual-${Date.now()}`;
    const newBox: SplitBox = {
      id: newId,
      index: boxes.length + 1,
      row: 0,
      col: 0,
      x: Math.round((analysis.imageWidth - w) / 2),
      y: Math.round((analysis.imageHeight - h) / 2),
      width: w,
      height: h,
      confidence: 1,
      strategy: 'grid',
      trimmedEdges: { top: 0, right: 0, bottom: 0, left: 0 },
      included: true,
    };
    setBoxes((prev) => [...prev, newBox]);
    setSelectedBoxId(newId);
  };

  const handleToggleInclude = (id: string) => {
    setBoxes((prev) => prev.map((b) => (b.id === id ? { ...b, included: b.included === false } : b)));
  };

  const handleExportZip = async () => {
    if (!jobId) return;
    const selected = boxes.filter((b) => b.included !== false);
    if (selected.length === 0) return;

    setExporting(true);
    setErrorMsg(null);
    try {
      const res = await fetch('/api/export-zip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId, boxes: selected, format: exportFormat }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'ZIP 导出失败');
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const stem = filename.replace(/\.[^.]+$/, '');
      a.href = url;
      a.download = `${stem}_slices.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'ZIP 导出失败');
    } finally {
      setExporting(false);
    }
  };

  const includedBoxes = boxes.filter((b) => b.included !== false);
  const selectedBox = boxes.find((b) => b.id === selectedBoxId) ?? null;

  return (
    <div id="top" className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <HeaderBar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onExportZip={handleExportZip}
        exporting={exporting}
        sliceCount={includedBoxes.length}
      />

      {/* Mobile tab strip */}
      <div className="flex md:hidden items-center gap-1 overflow-x-auto border-b border-slate-200 bg-white px-4 py-2">
        {(
          [
            { id: 'workspace', label: '工作台' },
            { id: 'presets', label: '样本' },
            { id: 'engine', label: 'Python/Rust' },
            { id: 'algorithm', label: '原理' },
          ] as Array<{ id: ActiveTab; label: string }>
        ).map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActiveTab(t.id)}
            className={`rounded-md px-3 py-1.5 text-xs font-medium whitespace-nowrap ${
              activeTab === t.id ? 'bg-slate-900 text-white' : 'text-slate-600'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {activeTab === 'workspace' && (
        <div className="flex-1 flex flex-col lg:flex-row">
          <ControlSidebar
            options={options}
            onChangeOptions={handleOptionsChange}
            onUploadFile={handleUploadFile}
            onLoadSource={handleLoadSource}
            activeSource={activeSource}
            loading={loading}
            exportFormat={exportFormat}
            onChangeFormat={setExportFormat}
            selectedBox={selectedBox}
            onUpdateBox={handleUpdateBox}
            onDeleteBox={handleDeleteBox}
            onAddManualBox={handleAddManualBox}
            onReanalyze={() => {
              if (jobId) void runAnalysis({ useCachedJob: true, opts: options });
            }}
          />

          <main className="flex-1 min-w-0 p-5 sm:p-6 space-y-6">
            {errorMsg && (
              <div
                role="alert"
                className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
              >
                {errorMsg}
              </div>
            )}

            <CanvasStage
              imageUrl={imageUrl}
              filename={filename}
              analysis={analysis}
              boxes={boxes}
              selectedBoxId={selectedBoxId}
              hoveredBoxId={hoveredBoxId}
              showOverlay={showOverlay}
              onToggleOverlay={() => setShowOverlay((s) => !s)}
              onSelectBox={setSelectedBoxId}
              onHoverBox={setHoveredBoxId}
              onInspectBox={setInspectedBox}
              loading={loading}
            />

            <SliceGallery
              jobId={jobId}
              boxes={boxes}
              selectedBoxId={selectedBoxId}
              exportFormat={exportFormat}
              onSelectBox={setSelectedBoxId}
              onHoverBox={setHoveredBoxId}
              onToggleInclude={handleToggleInclude}
              onInspectBox={setInspectedBox}
            />
          </main>
        </div>
      )}

      {activeTab === 'presets' && <PresetsView onSelectPreset={handleLoadSource} />}
      {activeTab === 'engine' && <CodeEngineView />}
      {activeTab === 'algorithm' && <AlgorithmView />}

      {inspectedBox && jobId && (
        <SliceLightboxModal
          jobId={jobId}
          box={inspectedBox}
          exportFormat={exportFormat}
          onClose={() => setInspectedBox(null)}
        />
      )}
    </div>
  );
}
