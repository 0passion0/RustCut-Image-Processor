# smart-image-splitter · 高精度 Rust + Python 智能图像切割包

**简体中文** | [English](./README.en.md)

[![PyPI](https://img.shields.io/pypi/v/smart-image-splitter?color=blue&label=PyPI)](https://pypi.org/project/smart-image-splitter/)
[![Python](https://img.shields.io/pypi/pyversions/smart-image-splitter)](https://pypi.org/project/smart-image-splitter/)
[![CI](https://github.com/0passion0/RustCut-Image-Processor/actions/workflows/ci.yml/badge.svg)](https://github.com/0passion0/RustCut-Image-Processor/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-green.svg)](./LICENSE)

**smart-image-splitter** 是一个可直接通过 `pip install` 安装的像素级复合图像自动切割 Python 工具包，底层由 **原生 Rust (`PyO3` + `Rayon`) 并行计算内核** 与 **全特性对齐的向量化 OpenCV / NumPy 回退引擎** 驱动。

系统专为解决“传统 `W/N × H/M` 等分网格切歪、串色、漏切”等痛点而设计，能够高精度处理**错位排版 PPT 组图**、**非对称 Bento 拼图**、**白边隔离卡片集**、**纯色/透明背景离散图标与精灵图**，并支持通过回调接入 **SAM 2 / YOLO / ONNX** 模型处理自然重叠场景。

---

## 目录

1. [为什么传统等分网格必然切歪？](#1-为什么传统等分网格必然切歪)
2. [切割模式选择与架构总览](#2-切割模式选择与架构总览)
3. [算法理论推导与数学建模](#3-算法理论推导与数学建模)
   - [3.1 多尺度方向阶跃梯度 + 1px 细线脉冲能量剖面](#31-多尺度方向阶跃梯度--1px-细线脉冲能量剖面)
   - [3.2 非线性覆盖率加权与切向平滑度惩罚](#32-非线性覆盖率加权与切向平滑度惩罚)
   - [3.3 留白沟壑谷值检测与连续区间中点坍缩](#33-留白沟壑谷值检测与连续区间中点坍缩)
   - [3.4 递归分层 XY-Cut BSP 与同级网格线传播](#34-递归分层-xy-cut-bsp-与同级网格线传播)
   - [3.5 单面板 ±8px 局部边缘吸附与抗锯齿去杂边](#35-单面板-8px-局部边缘吸附与抗锯齿去杂边)
   - [3.6 线性时间可分离膨胀 + Union-Find 轮廓提取与 1:1 边缘收紧](#36-线性时间可分离膨胀--union-find-轮廓提取与-11-边缘收紧)
4. [工程落地实践与极限性能优化](#4-工程落地实践与极限性能优化)
5. [快速开始（PyPI / GitHub / CLI）](#5-快速开始pypi--github--cli)
6. [Python API 与 CLI 参数速查](#6-python-api-与-cli-参数速查)
7. [仓库目录结构](#7-仓库目录结构)
8. [开源协议](#8-开源协议)

---

## 1. 为什么传统等分网格必然切歪？

以仓库内置的 `2848 × 1600` 九宫格演示文稿示例图（[`examples/sample_slides.png`](./examples/sample_slides.png)）为例：
- 传统 `W/3 × H/3` 均匀等分会在 `x = 949, 1898` 与 `y = 533, 1066` 下刀。
- 实际上，该图真实子图边界存在显著的非均匀与局部错位：
  - 水平边界：`y = 531` 与 `y = 1043..1047`；
  - 左侧垂直边界：`x = 945..946`（且 `x = 946` 处存在一条 `1px` 灰色分隔线 `(213, 214, 217)`）；
  - 右侧垂直边界：第 1–2 行位于 **`x = 1824`**，而第 3 行整体**向左错位 12 像素至 `x = 1812`**！
- 机械等分会导致最高 **86 像素的跨面板串色裁切错误**；同时第 2 行与第 3 行左侧均为白底幻灯片，二者之间仅隔一条极微弱的 `1px` 浅灰阴影线，常规全局阈值极易漏切。

---

## 2. 切割模式选择与架构总览

```text
输入图像 (np.ndarray / 文件路径)
  │
  ├─► 若注入 ml_segmenter 回调 ──► 调度外部 SAM 2 / YOLO / ONNX 掩膜框 ("ml-hook")
  │
  └─► 默认调度原生 Rust 核心 (`smart_splitter._core`，无扩展时无缝切换 OpenCV/NumPy 内核)
        ├─ mode = "grid"    ──► 均匀行列网格 + 可选固定留白扣除
        ├─ mode = "contour" ──► 可分离形态学闭运算 + 并查集 8-连通域 + 1:1 边缘收紧
        ├─ mode = "xycut"   ──► 递归分层 XY-Cut BSP + 同级网格线传播 + ±8px 吸附 + 去杂边
        └─ mode = "auto"    ──► 自适应判定：统一纯色/透明背景且分散对象占比 < 65% 时走 contour，否则走 xycut
```

| 模式 | 典型适用场景 | 核心算法策略 |
|---|---|---|
| `auto`（默认） | 绝大多数混合拼图、PPT 组图、商品图标集 | 先检测边缘背景一致性，在 `xycut` 与 `contour` 间自动择优 |
| `xycut` | 无缝拼接九宫格、错位 Bento 排版、带留白沟壑的卡片墙 | 递归分层 XY-Cut BSP + 同级网格传播 + 抗锯齿 `bleed_trim` |
| `contour` | 纯色或透明 RGBA 底上自由散布的贴纸、Icon、非矩形组件 | 降采样线性闭运算 + 8-连通并查集 + 原图 1:1 紧框收紧 |
| `grid` | 已知固定行列数且严格等分（可选 `gutter_px` 间距）的图片 | 数学精确均匀网格切分 |

---

## 3. 算法理论推导与数学建模

### 3.1 多尺度方向阶跃梯度 + 1px 细线脉冲能量剖面

对于任意候选扫描线位置 $p$（例如水平扫描行 $y$，沿正交方向采样 $u \in [x_0, x_1)$），真实的版面分割边界必然表现为以下两类信号之一：
1. **阶跃跳变边（Step Edge）**：相邻子图底色或边缘存在色差；
2. **1px 细线脉冲（Hairline Impulse）**：即使上下两侧同为纯白背景，其间的 `1px` 细分隔线或投影也会产生显著的二阶拉普拉斯峰值。

我们在 RGB $L_1$ 空间同时计算两项方向能量响应：

$$E_{\text{step}}(p, u) = \|I(p, u) - I(p+1, u)\|_1$$

$$E_{\text{line}}(p, u) = \frac{1}{2} \|2I(p, u) - I(p-1, u) - I(p+1, u)\|_1$$

局部综合边缘能量定义为：

$$E(p, u) = E_{\text{step}}(p, u) + E_{\text{line}}(p, u)$$

**理论意义**：当深蓝卡片遇上纯白卡片时，$E_{\text{step}} \approx 600$ 占主导；当两张白底 PPT 仅靠 `1px` 浅灰线（每通道差 $\Delta = 15$）分隔时，$E_{\text{step}} \approx 45$ 且 $E_{\text{line}} \approx 45$ 形成双重叠加，信噪比远高于单一差分算子。

### 3.2 非线性覆盖率加权与切向平滑度惩罚

面板内部的文字行、表格框线、折线图坐标轴同样具有局部高梯度——**如何保证绝对不把图表或标题从中间腰斩？**

1. **$\pm 2\text{px}$ 邻域覆盖率 $C(p)$**：
   受抗锯齿与 JPEG 压缩影响，物理边界可能产生 $\pm 1\text{–}2\text{px}$ 抖动。只要 $[p-2, p+2]$ 局部窗口内存在满足 $E(r, u) > 16$ 的响应，即记该采样列处于激活态：
   $$C(p) = \frac{1}{N} \sum_{u=1}^{N} \mathbb{I}\!\left[\max_{r \in [p-2, p+2]} E(r, u) > 16\right]$$
   内部表格横线通常只覆盖局部宽度的 $30\%\text{–}60\%$，而真实跨栏分隔线的覆盖率 $C(p) \in [0.85, 1.00]$。引入非线性幂次权重 $C(p)^{2.4}$ 后，局部短线会被强力压制（$0.4^{2.4} = 0.11$ vs. $0.95^{2.4} = 0.88$）。

2. **切向平滑度惩罚 $S_{\text{tan}}(p)$**：
   真实子图边框沿切线方向颜色高度连续，而穿过文字或图标的扫描线在切线方向高频剧烈震荡：
   $$S_{\text{tan}}(p) = \frac{1}{N-1} \sum_{u=2}^{N} \left| L(p, u) - L(p, u-1) \right|, \quad L = R + G + B$$
   算法取 $[p-1, p, p+1]$ 中的最小平滑度 $S_{\min}(p)$，避免高纹理卡片贴近纯色卡片时边缘被误伤惩罚。

3. **综合分割评分公式**：
   $$\text{Score}_{\text{seam}}(p) = \frac{\left(\bar{E}_{\text{step}}(p) + 0.7\,\bar{E}_{\text{line}}(p)\right) \cdot C(p)^{2.4}}{1 + 0.12 \cdot S_{\min}(p)}$$
   候选峰值通过**一维非极大值抑制（NMS）**提取，最小峰间距为 $d_{\min} = \max(24, \text{min\_region\_ratio} \cdot \min(W, H))$，并要求 $C(p) \ge 0.72$。

### 3.3 留白沟壑谷值检测与连续区间中点坍缩

针对带有**宽纯色/透明留白间距**的卡片图（例如各卡片之间留有 `40px` 白色缝隙）：
1. 基于图像四周边缘像素构建 $8\times 8\times 8$ 量化直方图估计主背景色 $\mathbf{b}$ 与边缘一致性 $U_{\text{border}}$（透明图直接检测 $\alpha < 32$ 比例）。
2. 当 $U_{\text{border}} > 0.72$ 时，若某一内部分割带满足背景匹配率 $> 95\%$、切向抖动 $S_{\text{tan}} < 2.5$ 且**两侧均存在前景内容**，则赋予留白谷值分：
   $$\text{Score}_{\text{gutter}}(p) = \frac{110}{1 + S_{\text{tan}}(p)}$$
3. **连续沟壑区间坍缩**：宽 `40px` 的白色过道会产生 40 条相邻高分线。将连续沟壑区间 $[i, j]$ 坍缩到唯一几何中点 $\lfloor (i + j)/2 \rfloor$，彻底杜绝“同一条宽白边被切成两条线并产生空白废图”的问题；叶子区域前景占比低于 $3\%$ 的纯背景块自动剔除。

### 3.4 递归分层 XY-Cut BSP 与同级网格线传播

1. 在当前区域 $[x_0, y_0, x_1, y_1]$ 内同时评估 H/V 两个方向的最优峰值，优先沿强分割轴（$\max \text{Score}_H \ge 0.72 \cdot \max \text{Score}_V$）切分为若干条带。
2. **递归独立求解**：每个子条带独立重新统计次级边界，从而天然适应上两行 `x=1824` 与第三行 `x=1812` 之间 `12px` 的错位拼图以及非对称 Bento 布局。
3. **同级网格线传播（Sibling Grid-Line Propagation）**：
   若相邻兄弟区域已在 $p_{\text{cut}}$ 切分，而跨过该线段的低对比度区域（例如两张无缝贴合的白底幻灯片）主阈值未触发，则在 $p_{\text{cut}} \pm 12\text{px}$ 窄窗口内启用高覆盖率低阈值精搜（$\text{cov} \ge 0.78$、$\bar{E}_{\text{raw}} \ge 10$、$S_{\text{tan}} < 2.2$），100% 召回微弱阴影分割线且零误切。

### 3.5 单面板 ±8px 局部边缘吸附与抗锯齿去杂边

BSP 划分出独立子区域后，执行两步精修：
1. **±8px 局部边界吸附**：针对单个子图的上下左右四条内边在 $\pm 8\text{px}$ 内独立对齐至本区域真实像素跃迁点。
2. **外缘背景裁剪与防串色去杂边（`bleed_trim`）**：
   - 先自动剥离四边残留的整行/整列统一背景（$> 94\%$ 匹配率）。
   - 再沿内部切割边取最外侧 `0..bleed_trim`（默认 `2px`）像素，与内侧参考行/列 `ref = edge ± (bleed_trim + 2)` 计算平均 $L_1$ 色差。当色差 $> 20$ 时剥离该过渡像素或 `1px` 灰线——做到**深色卡片不带白边、白底卡片不带灰线黑边**。

### 3.6 线性时间可分离膨胀 + Union-Find 轮廓提取与 1:1 边缘收紧

针对纯色或透明背景上散落的独立图标、贴纸、商品图集：
1. 降采样至最大边 $\le 720\text{px}$ 网格（`scale = ceil(max(W, H) / 720)`）生成前景二值掩膜。
2. **可分离 $O(s_w \cdot s_h)$ 双向扫描形态学闭运算**：水平 + 垂直两趟正反向游程扫描实现任意 `contour_gap_px` 闭运算合并，复杂度与核半径无关。
3. **8-连通域提取**：Rust 端（[`src/lib.rs`](./src/lib.rs)）采用无递归**并查集（DSU + 路径减半）**，OpenCV 回退端采用 `cv2.connectedComponentsWithStats`。
4. **1:1 原图分辨率边缘收紧**：消除降采样带来的 $\pm \text{scale}$ 像素量化误差，直接在 1:1 原始像素矩阵上收紧/微扩四边包围盒，实现 **0px 坐标误差**。

---

## 4. 工程落地实践与极限性能优化

| 工程优化项 | 解决的性能/精度瓶颈 | 具体实现方案 |
|---|---|---|
| **1. 行主序 Rayon 并行扫描** | 传统列剖面 `for x { for y }` 每次跨行跳转 `W * C` 字节，导致严重 CPU L1/L2 Cache Miss | 在 Rust 端改为 `ys.par_iter().fold().reduce()` 行主序并行归约，所有内存访问均命中连续缓存行 |
| **2. 单遍邻域激活追踪** | 每个递归子区域分配二维 `O(span × 200)` 能量矩阵造成大量内存读写 | 用一维单调游标实现 $\pm 2\text{px}$ 邻域最大值追踪，消除二维能量矩阵分配 |
| **3. 正交 200 采样 + 主轴 1px 步进** | 主轴跳行采样会漏检位于奇数坐标（如 `y=531, 1043`）的 `1px` 细线 | 严格保持主扫描轴 **`1px` 逐行/逐列扫描**不漏线，仅对正交方向按 $\approx 200$ 个点步进采样 |
| **4. 零拷贝 NumPy 借用与释放 GIL** | 大图跨 FFI 内存拷贝与阻塞 Python 主线程 | Rust 端对 C-contiguous NumPy 数组通过 `Cow<[u8]>` **零拷贝借用**，并通过 `py.allow_threads(...)` 释放 Python GIL |
| **5. 内存级并发裁剪编码** | 服务端接口或深度学习 DataLoader 写磁盘产生高 I/O 开销 | `batch_crop_to_memory_rs` 在内存中通过 Rayon 多线程并发编码裁剪字节流，零磁盘临时文件 |

---

## 5. 快速开始（PyPI / GitHub / CLI）

### 安装方式（支持 Python `>= 3.9`，全平台 `abi3` 预编译 Wheel）

```bash
# 方式 1：直接从 PyPI 安装（推荐，自动匹配 Linux / macOS / Windows 预编译包）
pip install smart-image-splitter

# 方式 2：通过 Git + pip 直接从 GitHub 安装
pip install git+https://github.com/0passion0/RustCut-Image-Processor.git

# 方式 3：通过 GitHub Releases 预编译 abi3 Wheel 安装（免本机 Rust 工具链）
pip install https://github.com/0passion0/RustCut-Image-Processor/releases/download/v0.2.2/smart_image_splitter-0.2.2-cp39-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

> **零配置双后端自适应**：
> - 从 PyPI 安装或本机已配备 `cargo` 时，自动启用原生多核加速扩展 `smart_splitter._core`（`PyO3` + `Rayon`）；
> - 目标环境仅安装纯 Python 依赖且无 `cargo` 从源码安装时，`setup.py` 会自动跳过编译并无缝切换至全特性对齐的 OpenCV / NumPy 内核。

### 命令行与 Python API 示例

```bash
# 命令行批量切割单图或目录
smart-split examples/sample_slides.png -o ./output_slices --mode auto --bleed-trim 2 --format png
```

```python
import cv2
from smart_splitter import SmartSplitter

splitter = SmartSplitter(mode="auto", sensitivity=0.65, bleed_trim=2)

# 1. 文件级直接切割保存
result, paths = splitter.split_file("examples/sample_slides.png", "output_slices/")
print(f"后端 {result.backend} ({result.detected_mode}) 在 {result.elapsed_ms:.1f}ms 内输出 {len(paths)} 张切片")

# 2. 零磁盘内存字节流输出（适用于 FastAPI / PyTorch 数据流水线）
img = cv2.imread("examples/sample_slides.png")
result, png_blobs = splitter.split_to_bytes(img, fmt="png")

# 3. 可选机器学习分割回调（可接入 SAM 2 / YOLO / ONNX）
ml_splitter = SmartSplitter(ml_segmenter=my_sam2_box_predictor)
```

---

## 6. Python API 与 CLI 参数速查

### `SmartSplitter` 构造参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mode` | `"auto"` | 检测策略：`"auto"` / `"xycut"` / `"contour"` / `"grid"` |
| `sensitivity` | `0.65` | 边缘阶跃与前景阈值灵敏度（裁剪至 `[0.1, 1.0]`） |
| `min_region_ratio` | `0.08` | 最小子图尺寸占图像短边比例下限（裁剪至 `[0.03, 0.35]`） |
| `bleed_trim` | `2` | 内部接缝抗锯齿去杂边最大剥离像素数（px） |
| `contour_gap_px` | `8` | `contour` 模式下形态学闭运算弥合组件间隙的半径（px） |
| `ml_segmenter` | `None` | 可选回调 `Callable[[np.ndarray], List[Tuple[int, int, int, int]]]`，返回 `(x, y, w, h)` 列表 |

### 返回结构 `SplitResult` & `SplitBox`

- **`SplitResult`**：`image_width`, `image_height`, `detected_mode`, `backend`（`"rust-rayon"` / `"opencv-numpy"` / `"ml-hook"`）, `elapsed_ms`, `boxes`
- **`SplitBox`**：`index`（1-based）, `row`, `col`, `x`, `y`, `width`, `height`, `confidence`, `strategy`, `trimmed_edges`（`(top, right, bottom, left)`）

### 本地开发与测试

```bash
pip install setuptools wheel setuptools-rust maturin pytest numpy opencv-python-headless pillow
maturin develop --release
pytest -v tests/
cargo clippy --all-targets -- -D warnings
```

---

## 7. 仓库目录结构

```text
.
├── Cargo.toml                        # Rust PyO3 + Rayon Crate 配置（含 abi3-py39 与 Release LTO）
├── pyproject.toml                    # PEP 517 + setuptools / maturin 构建配置
├── setup.py                          # 自适应 Rust 编译（无 cargo 时无感回退纯 Python）
├── MANIFEST.in                       # 源码分发清单
├── LICENSE / CONTRIBUTING.md         # Apache-2.0 OR MIT 双重协议与贡献指南
├── README.md / README.en.md          # 中文（默认）/ 英文双语文档
├── .github/workflows/
│   ├── ci.yml                        # 根目录 pip install + Cargo + maturin + pytest 自动化测试
│   └── release-wheels.yml            # Linux(x64/ARM64) / macOS(Intel/M系列) / Windows PyPI 自动发布
├── src/lib.rs                        # 原生 Rust 核心：XY-Cut BSP、并查集轮廓提取与内存并发裁剪
├── python/smart_splitter/
│   ├── __init__.py                   # 对外导出 (SmartSplitter, SplitBox, SplitResult)
│   ├── engine.py                     # Rust / 全特性对齐 OpenCV 双引擎与 ML 钩子
│   └── cli.py                        # `smart-split` 命令行入口
├── tests/test_splitter.py            # pytest 单元测试（XY-Cut、留白沟壑、去杂边、轮廓提取）
└── examples/sample_slides.png        # 2848x1600 九宫格真实测试样本图
```

---

## 8. 开源协议

本项目采用 **[Apache-2.0 OR MIT](LICENSE)** 双重协议开源。
