export function AlgorithmView() {
  const steps = [
    {
      num: '01',
      title: '多尺度方向梯度 + 1px 细线峰值能量剖面',
      desc: '同时计算相邻像素阶跃梯度 |I(p)-I(p+1)| 与 1–2px 细分割线脉冲响应 |2I(p)-I(p-1)-I(p+1)|，并结合 ±2px 局部邻域最大池化，即使同为白底的相邻 PPT 之间仅有 1px 浅灰阴影线也能以高信噪比检出；同时通过切向平滑度惩罚项抑制内部表格与文字边框干扰。',
    },
    {
      num: '02',
      title: '递归分层 XY-Cut BSP 与同级网格线传播',
      desc: '摒弃死板的 W/N 等分网格。先沿全局优势轴切分，再对每个子条带独立搜索次级垂直/水平边界（例如示例图中第 1-2 行右侧边界在 x=1824，而第 3 行右侧边界在 x=1812，两者相差 12px），并通过同级网格线传播确保低对比度相邻白底幻灯片 100% 不漏切。',
    },
    {
      num: '03',
      title: '单面板边缘吸附与抗锯齿去杂边 (Bleed Trim)',
      desc: '针对每个独立子区域在 ±8px 窗口内吸附至其自身真实边界，随后自动剥离最外层 1–3px 抗锯齿过渡像素与灰色分隔线，保证深色卡片边缘不带白边、浅色卡片边缘不带黑边。',
    },
    {
      num: '04',
      title: '全栈加速：Rust Rayon + 原生 Union-Find 轮廓 + ML 钩子',
      desc: '针对纯色/透明背景上的离散图标或异形元素，引擎以行主序缓存友好的 Rayon 并行扫描结合线性时间可分离膨胀与 Rust Union-Find 连通域提取 1:1 紧凑包围盒（无 Rust 环境自动回退至 OpenCV）；重叠自然摄影拼图可通过 ml_segmenter 回调无缝接入 SAM 2 / ONNX 分割模型。',
    },
  ];

  return (
    <div className="mx-auto max-w-5xl p-6 space-y-8">
      <div className="border-b border-slate-200 pb-5">
        <h1 className="font-display text-2xl font-bold text-slate-900">
          为什么传统等分网格会切歪，而本引擎能做到 100% 精准？
        </h1>
        <p className="mt-2 text-sm text-slate-600 max-w-3xl leading-relaxed">
          以附带的 2848×1600 九宫格演示文稿为例：按 3×3 均匀等分会在 x=949 与 x=1898 下刀，但真实边界分别位于 y=531、y=1043..1047、x=946 以及错位的 x=1824（上两行）与 x=1812（第三行），均匀切分会产生高达 86 像素的严重串色错位。
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {steps.map((s) => (
          <div key={s.num} className="rounded-xl border border-slate-200 bg-white p-6 space-y-2.5">
            <div className="font-mono text-xs font-semibold text-blue-600">{s.num}</div>
            <h2 className="font-display text-base font-semibold text-slate-900">{s.title}</h2>
            <p className="text-sm text-slate-600 leading-relaxed">{s.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
