# Smart Image Splitter (`smart_splitter`) — Rust PyO3 + Rayon + OpenCV

[English](./README.md) | **简体中文**

高性能像素级 Python 复合图像切割库，底层基于 **编译期优化的 Rust（`PyO3` + `rayon` + `image`）原生扩展**，提供 **OpenCV（`cv2`）无缝回退内核** 与 **可选机器学习分割钩子**（SAM 2 / YOLO / ONNX）。

---

## 1. 算法理论体系

本引擎针对错位多宫格（如不同行垂直分割线错位 `12px`）、非对称 Bento 拼图、白边卡片以及散落图标/透明精灵图，设计了五阶段联合数学模型：

1. **方向阶跃梯度 + `1px` 细线脉冲响应**：
   $$E(p, u) = \|I(p, u) - I(p+1, u)\|_1 + \frac{1}{2}\|2I(p, u) - I(p-1, u) - I(p+1, u)\|_1$$
   同时捕获色块边界跃迁与同白底幻灯片之间微弱的 `1px` 阴影分隔线。
2. **非线性覆盖率与切向平滑度抑制**：
   $$\text{Score}_{\text{seam}}(p) = \frac{(\bar{E}_{\text{step}}(p) + 0.7\,\bar{E}_{\text{line}}(p)) \cdot C(p)^{2.4}}{1 + 0.12\,S_{\min}(p)}$$
   通过 $\pm 2\text{px}$ 邻域激活覆盖率 $C(p)$ 与切向亮度抖动惩罚 $S_{\min}(p)$ 过滤文字行、表格与图表坐标轴干扰。
3. **留白沟壑检测与连续区间中点坍缩**：
   基于边缘像素 $8\times 8\times 8$ 量化直方图识别统一背景/透明通道，将宽留白走廊 $[i, j]$ 坍缩为单条中点切线，并自动剔除纯背景残块。
4. **递归分层 XY-Cut BSP + 同级网格线传播**：
   按优势轴递归分割并在子条带内独立求解次级边界；通过 $\pm 12\text{px}$ 邻域同级网格线传播（`cov >= 0.78`, `avg_raw >= 10`, `avg_smooth < 2.2`）精准找回低对比度白底相邻面板分割线。
5. **单面板边缘吸附、防串色去杂边与 1:1 轮廓收紧**：
   每块子图在 $\pm 8\text{px}$ 内单独对齐边缘，剥离外侧 `0..bleed_trim` 过渡/灰线像素（$L_1$ 差 $> 20.0$）；`contour` 模式通过可分离 $O(N)$ 闭运算膨胀 + 非递归并查集（DSU）提取连通域，并在 1:1 原图上收紧至 **0px 误差**。

---

## 2. Rust 与 Python 工程实践优化

- **零拷贝 NumPy 内存借用**：对连续（C-contiguous）`HxWx3` / `HxWx4` `uint8` 数组通过 `Cow<[u8]>` 直接借用内存指针，并通过 `py.allow_threads` 彻底释放 Python GIL。
- **行主序缓存友好并行归约**：垂直方向能量扫描采用外层按行遍历 + `ys.par_iter().fold().reduce()` 并行归约，避免大跨度垂直读取导致的 CPU Cache Miss。
- **正交 200 点采样 + 主轴 1px 全扫描**：保证不漏检任何奇数坐标 `1px` 细线，同时极大压缩像素运算量。
- **全模式 Rust 原生调度与零磁盘内存导出**：`detect_auto_rs` 原生覆盖 `auto / xycut / contour / grid` 四种模式，`batch_crop_to_memory_rs` 支持多线程直接输出内存 `bytes` 列表，无需落盘中转。

---

## 3. 快速编译与安装

```bash
# 1. 安装 Rust 工具链 (rustup) 与 Python 依赖
pip install maturin opencv-python-headless numpy pillow pytest

# 2. 编译 Release 级别 Rust 原生扩展
maturin develop --release

# 3. 命令行批量切割
smart-split slides_3x3.png -o ./slides_out --mode auto --bleed-trim 2 --format png

# 4. 运行单元测试
pytest -v tests/
```

---

## 4. Python API 使用指南

```python
import cv2
from smart_splitter import SmartSplitter

splitter = SmartSplitter(
    mode="auto",            # "auto" | "xycut" | "contour" | "grid"
    sensitivity=0.65,       # 边界灵敏度 (0.1 .. 1.0)
    min_region_ratio=0.08,  # 最小子图占总尺寸比例
    bleed_trim=2,           # 外缘抗锯齿/杂边剥离像素数
    contour_gap_px=8,       # 轮廓模式相邻元素合并间距
)

# 从文件读取并由 Rust 多线程并行裁切保存
result, output_paths = splitter.split_file("sheet.png", "output_slices/", fmt="png")
print(f"成功输出 {len(output_paths)} 张切片 [{result.detected_mode}/{result.backend}]，耗时 {result.elapsed_ms:.1f}ms")

# 零磁盘内存导出（适合 FastAPI 服务或 PyTorch 训练流水线）
img = cv2.imread("sheet.png", cv2.IMREAD_UNCHANGED)
result, png_blobs = splitter.split_to_bytes(img, fmt="png")
```

### 接入自定义深度学习分割模型（处理重叠自然摄影拼图）

```python
def sam2_segmenter(img_rgb):
    # 返回目标框列表: [(x, y, width, height), ...]
    return [(10, 20, 400, 300), (430, 20, 380, 300)]

ml_splitter = SmartSplitter(ml_segmenter=sam2_segmenter)
```
