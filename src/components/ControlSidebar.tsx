import React, { useRef } from 'react';
import { Upload, RotateCcw, Plus, Trash2 } from 'lucide-react';
import type { ExportFormat, SplitBox, SplitMode, SplitOptions } from '../types/splitter';

interface ControlSidebarProps {
  options: SplitOptions;
  onChangeOptions: (next: SplitOptions) => void;
  onUploadFile: (file: File) => void;
  onLoadSource: (source: string) => void;
  activeSource: string;
  loading: boolean;
  exportFormat: ExportFormat;
  onChangeFormat: (fmt: ExportFormat) => void;
  selectedBox: SplitBox | null;
  onUpdateBox: (id: string, patch: Partial<SplitBox>) => void;
  onDeleteBox: (id: string) => void;
  onAddManualBox: () => void;
  onReanalyze: () => void;
}

const MODES: Array<{ id: SplitMode; label: string }> = [
  { id: 'auto', label: '智能 Auto' },
  { id: 'xycut', label: '递归 XY-Cut' },
  { id: 'contour', label: '轮廓分割' },
  { id: 'grid', label: '均匀网格' },
];

const SOURCES: Array<{ id: string; label: string; desc: string }> = [
  { id: 'sample', label: '示例 3×3 PPT 组图', desc: '2848×1600 非均匀错位边界' },
  { id: 'preset-bento', label: '非对称 Bento 拼图', desc: '左1右2 异形多面板' },
  { id: 'preset-gutter', label: '白边 2×2 卡片图', desc: '自动剔除留白间隙' },
  { id: 'preset-contours', label: '独立多目标图集', desc: '纯色背景轮廓聚类提取' },
];

export function ControlSidebar({
  options,
  onChangeOptions,
  onUploadFile,
  onLoadSource,
  activeSource,
  loading,
  exportFormat,
  onChangeFormat,
  selectedBox,
  onUpdateBox,
  onDeleteBox,
  onAddManualBox,
  onReanalyze,
}: ControlSidebarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadFile(file);
      e.target.value = '';
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onUploadFile(file);
    }
  };

  return (
    <aside className="w-full lg:w-[300px] shrink-0 border-b lg:border-b-0 lg:border-r border-slate-200 bg-white p-5 flex flex-col gap-6 lg:sticky lg:top-[61px] lg:h-[calc(100vh-61px)] lg:overflow-y-auto">
      {/* 1. Image Input */}
      <section aria-label="输入图像">
        <div className="flex items-center justify-between mb-2.5">
          <h2 className="text-xs font-semibold tracking-wide text-slate-500">输入图像</h2>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="text-xs font-medium text-blue-600 hover:text-blue-700 whitespace-nowrap"
          >
            选择本地文件
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={handleFileChange}
          className="hidden"
          aria-label="上传待切割图像"
        />

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="w-full rounded-lg border border-dashed border-slate-300 bg-slate-50/70 px-4 py-4 text-center transition-colors hover:border-blue-500 hover:bg-blue-50/40 focus-visible:outline-2 focus-visible:outline-blue-600"
        >
          <Upload className="mx-auto h-5 w-5 text-slate-400 mb-1.5" aria-hidden="true" />
          <p className="text-sm font-medium text-slate-800">拖拽或点击上传图像</p>
          <p className="mt-0.5 text-xs text-slate-500">PNG · JPG · WebP（最高 40MB）</p>
        </button>

        <div className="mt-3 space-y-1">
          {SOURCES.map((src) => {
            const active = activeSource === src.id;
            return (
              <button
                key={src.id}
                type="button"
                disabled={loading}
                onClick={() => onLoadSource(src.id)}
                className={`w-full rounded-md px-2.5 py-2 text-left transition-colors flex items-center justify-between gap-2 ${
                  active
                    ? 'bg-slate-900 text-white'
                    : 'text-slate-700 hover:bg-slate-100'
                }`}
              >
                <span className="text-xs font-medium truncate">{src.label}</span>
                <span className={`text-[11px] shrink-0 ${active ? 'text-slate-300' : 'text-slate-400'}`}>
                  {active ? '当前' : '加载'}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      <hr className="border-slate-200" />

      {/* 2. Detection Strategy */}
      <section aria-label="切割策略">
        <div className="flex items-center justify-between mb-2.5">
          <h2 className="text-xs font-semibold tracking-wide text-slate-500">切割算法模式</h2>
          <button
            type="button"
            onClick={onReanalyze}
            disabled={loading}
            className="inline-flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-slate-900"
          >
            <RotateCcw className="h-3 w-3" aria-hidden="true" />
            <span>重新识别</span>
          </button>
        </div>

        <div className="grid grid-cols-2 gap-1 rounded-lg bg-slate-100 p-1">
          {MODES.map((m) => {
            const active = options.mode === m.id;
            return (
              <button
                key={m.id}
                type="button"
                onClick={() => onChangeOptions({ ...options, mode: m.id })}
                className={`rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap ${
                  active
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {m.label}
              </button>
            );
          })}
        </div>

        {options.mode === 'grid' ? (
          <div className="mt-4 grid grid-cols-3 gap-2.5">
            <label className="block">
              <span className="text-xs text-slate-600">行数 Rows</span>
              <input
                type="number"
                min={1}
                max={20}
                value={options.rows}
                onChange={(e) => onChangeOptions({ ...options, rows: Math.max(1, Number(e.target.value) || 1) })}
                className="mt-1 w-full rounded-md border border-slate-200 px-2.5 py-1.5 font-mono text-xs text-slate-900"
              />
            </label>
            <label className="block">
              <span className="text-xs text-slate-600">列数 Cols</span>
              <input
                type="number"
                min={1}
                max={20}
                value={options.cols}
                onChange={(e) => onChangeOptions({ ...options, cols: Math.max(1, Number(e.target.value) || 1) })}
                className="mt-1 w-full rounded-md border border-slate-200 px-2.5 py-1.5 font-mono text-xs text-slate-900"
              />
            </label>
            <label className="block">
              <span className="text-xs text-slate-600">间距 px</span>
              <input
                type="number"
                min={0}
                max={200}
                value={options.gutterPx}
                onChange={(e) => onChangeOptions({ ...options, gutterPx: Math.max(0, Number(e.target.value) || 0) })}
                className="mt-1 w-full rounded-md border border-slate-200 px-2.5 py-1.5 font-mono text-xs text-slate-900"
              />
            </label>
          </div>
        ) : (
          <div className="mt-4 space-y-3.5">
            <label className="block">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-slate-600">边界检测灵敏度</span>
                <span className="font-mono font-medium text-slate-900">{Math.round(options.sensitivity * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.15}
                max={0.95}
                step={0.05}
                value={options.sensitivity}
                onChange={(e) => onChangeOptions({ ...options, sensitivity: Number(e.target.value) })}
                className="w-full accent-blue-600"
              />
            </label>

            <label className="block">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-slate-600">边缘去杂边 (Bleed Trim)</span>
                <span className="font-mono font-medium text-slate-900">{options.bleedTrim} px</span>
              </div>
              <input
                type="range"
                min={0}
                max={6}
                step={1}
                value={options.bleedTrim}
                onChange={(e) => onChangeOptions({ ...options, bleedTrim: Number(e.target.value) })}
                className="w-full accent-blue-600"
              />
            </label>

            <label className="block">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-slate-600">最小子图占比</span>
                <span className="font-mono font-medium text-slate-900">{Math.round(options.minRegionRatio * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.04}
                max={0.22}
                step={0.02}
                value={options.minRegionRatio}
                onChange={(e) => onChangeOptions({ ...options, minRegionRatio: Number(e.target.value) })}
                className="w-full accent-blue-600"
              />
            </label>
          </div>
        )}
      </section>

      <hr className="border-slate-200" />

      {/* 3. Export Format & Manual Region */}
      <section aria-label="输出格式与区域微调">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xs font-semibold tracking-wide text-slate-500">输出格式</h2>
          <button
            type="button"
            onClick={onAddManualBox}
            className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700"
          >
            <Plus className="h-3.5 w-3.5" aria-hidden="true" />
            <span>新增选区</span>
          </button>
        </div>

        <div className="grid grid-cols-3 gap-1 rounded-lg bg-slate-100 p-1">
          {(['png', 'webp', 'jpg'] as ExportFormat[]).map((fmt) => (
            <button
              key={fmt}
              type="button"
              onClick={() => onChangeFormat(fmt)}
              className={`rounded-md py-1.5 text-xs font-medium uppercase transition-colors ${
                exportFormat === fmt
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {fmt}
            </button>
          ))}
        </div>

        {selectedBox && (
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-900">
                选中区域 #{selectedBox.index}
              </span>
              <button
                type="button"
                onClick={() => onDeleteBox(selectedBox.id)}
                aria-label={`删除选区 #${selectedBox.index}`}
                className="inline-flex items-center gap-1 text-xs text-red-600 hover:text-red-700"
              >
                <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                <span>移除</span>
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="text-[11px] text-slate-500">X 坐标</span>
                <input
                  type="number"
                  value={selectedBox.x}
                  onChange={(e) => onUpdateBox(selectedBox.id, { x: Math.max(0, Number(e.target.value) || 0) })}
                  className="mt-0.5 w-full rounded border border-slate-200 bg-white px-2 py-1 font-mono text-xs"
                />
              </label>
              <label className="block">
                <span className="text-[11px] text-slate-500">Y 坐标</span>
                <input
                  type="number"
                  value={selectedBox.y}
                  onChange={(e) => onUpdateBox(selectedBox.id, { y: Math.max(0, Number(e.target.value) || 0) })}
                  className="mt-0.5 w-full rounded border border-slate-200 bg-white px-2 py-1 font-mono text-xs"
                />
              </label>
              <label className="block">
                <span className="text-[11px] text-slate-500">宽度 W</span>
                <input
                  type="number"
                  value={selectedBox.width}
                  onChange={(e) => onUpdateBox(selectedBox.id, { width: Math.max(8, Number(e.target.value) || 8) })}
                  className="mt-0.5 w-full rounded border border-slate-200 bg-white px-2 py-1 font-mono text-xs"
                />
              </label>
              <label className="block">
                <span className="text-[11px] text-slate-500">高度 H</span>
                <input
                  type="number"
                  value={selectedBox.height}
                  onChange={(e) => onUpdateBox(selectedBox.id, { height: Math.max(8, Number(e.target.value) || 8) })}
                  className="mt-0.5 w-full rounded border border-slate-200 bg-white px-2 py-1 font-mono text-xs"
                />
              </label>
            </div>
          </div>
        )}
      </section>
    </aside>
  );
}
