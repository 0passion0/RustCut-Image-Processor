import { ArrowRight } from 'lucide-react';

interface PresetsViewProps {
  onSelectPreset: (sourceId: string) => void;
}

const PRESET_CARDS = [
  {
    id: 'sample',
    title: '示例 3×3 演讲幻灯片组图',
    specs: '2848 × 1600 · 非均匀错位边界 · 含同色白底相邻页',
    description:
      '用户提供的真实样本：第 1-2 行右侧边界在 x=1824，第 3 行右侧边界偏移至 x=1812，且左下两张均为浅白底。测试递归 XY-Cut 局部吸附与同级网格线传播能力。',
  },
  {
    id: 'preset-bento',
    title: '非对称 Bento 多面板拼图',
    specs: '1600 × 1000 · 左 1 整栏 + 右 2 堆叠',
    description:
      '测试非均等宽高布局：左半区保持完整纵向高度不被误切，右半区独立识别横向分割线并输出 3 张无杂边子图。',
  },
  {
    id: 'preset-gutter',
    title: '带纯白留白间隙的 2×2 卡片图',
    specs: '1600 × 1080 · 40px 纯白通道',
    description:
      '测试连续空白通道折叠为单一切割中点，并自动剥离每张子图四周的纯白画布外边距。',
  },
  {
    id: 'preset-contours',
    title: '纯色画布上的离散多目标图集',
    specs: '1500 × 950 · 5 个非网格分布目标',
    description:
      '测试 Auto 模式自动识别离散前景目标并切换至连通域轮廓分割，精准包围每个独立图形元素。',
  },
];

export function PresetsView({ onSelectPreset }: PresetsViewProps) {
  return (
    <div className="mx-auto max-w-5xl p-6 space-y-6">
      <div className="border-b border-slate-200 pb-5">
        <h1 className="font-display text-2xl font-bold text-slate-900">多场景图像切割测试样本库</h1>
        <p className="mt-1 text-sm text-slate-600">
          点击任意测试场景即可载入工作台，实时验证递归 XY-Cut、留白通道剥离与连通域轮廓分割效果。
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {PRESET_CARDS.map((card) => (
          <div
            key={card.id}
            className="rounded-xl border border-slate-200 bg-white p-6 flex flex-col justify-between gap-4"
          >
            <div>
              <h2 className="font-display text-base font-semibold text-slate-900">{card.title}</h2>
              <p className="mt-1 font-mono text-xs text-slate-500 tabular-nums">{card.specs}</p>
              <p className="mt-3 text-sm text-slate-600 leading-relaxed">{card.description}</p>
            </div>

            <div>
              <button
                type="button"
                onClick={() => onSelectPreset(card.id)}
                className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-medium text-white hover:bg-slate-800 transition-colors"
              >
                <span>在工作台载入并切割</span>
                <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
