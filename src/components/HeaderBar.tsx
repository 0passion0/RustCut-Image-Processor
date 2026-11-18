import { Download } from 'lucide-react';

export type ActiveTab = 'workspace' | 'presets' | 'engine' | 'algorithm';

interface HeaderBarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  onExportZip: () => void;
  exporting: boolean;
  sliceCount: number;
}

export function HeaderBar({
  activeTab,
  onSelectTab,
  onExportZip,
  exporting,
  sliceCount,
}: HeaderBarProps) {
  const navItems: Array<{ id: ActiveTab; label: string }> = [
    { id: 'workspace', label: '切割工作台' },
    { id: 'presets', label: '测试样本' },
    { id: 'engine', label: 'Python / Rust 源码' },
    { id: 'algorithm', label: '架构原理' },
  ];

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between gap-8 border-b border-slate-200 bg-white px-6 py-3.5">
      <a
        href="#top"
        onClick={(e) => {
          e.preventDefault();
          onSelectTab('workspace');
        }}
        className="font-display text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
      >
        SliceCraft
      </a>

      <nav className="hidden md:flex items-center gap-6" aria-label="Primary navigation">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectTab(item.id)}
              className={`relative py-1 text-sm font-medium transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-blue-600 ${
                isActive ? 'text-slate-900' : 'text-slate-500 hover:text-slate-900'
              }`}
            >
              {item.label}
              {isActive && (
                <span className="absolute inset-x-0 -bottom-[15px] h-0.5 bg-blue-600" aria-hidden="true" />
              )}
            </button>
          );
        })}
      </nav>

      <div className="flex items-center shrink-0">
        <button
          type="button"
          onClick={onExportZip}
          disabled={exporting || sliceCount === 0}
          className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
        >
          <Download className="h-4 w-4" aria-hidden="true" />
          <span>{exporting ? '打包导出中...' : `导出全部切片 (${sliceCount})`}</span>
        </button>
      </div>
    </header>
  );
}
