# Smart Image Splitter (`smart_splitter`) — Rust PyO3 + Rayon + OpenCV

**English** | [简体中文](./README.zh-CN.md)

High-performance, pixel-accurate composite image cutting library with a **compiled Rust (`PyO3` + `rayon` + `image`) native core**, **OpenCV (`cv2`) C++ fallback**, and an **optional ML segmentation callback hook** (SAM 2 / YOLO / ONNX).

---

## 1. Theoretical Algorithm Pipeline

Smart Image Splitter replaces fragile uniform grids (`W/N × H/M`) with a five-stage mathematical pipeline capable of handling misaligned slide sheets (e.g., `12px` column offset across rows), asymmetric Bento collages, wide white-guttered cards, and scattered icons/alpha sprites:

1. **Directional Step + `1px` Hairline Energy**:
   $$E(p, u) = \|I(p, u) - I(p+1, u)\|_1 + \frac{1}{2}\|2I(p, u) - I(p-1, u) - I(p+1, u)\|_1$$
   Captures both region color boundaries and faint `1px` shadow/divider lines between white-on-white slides.
2. **Non-Linear Coverage & Tangential Smoothness Gate**:
   $$\text{Score}_{\text{seam}}(p) = \frac{(\bar{E}_{\text{step}}(p) + 0.7\,\bar{E}_{\text{line}}(p)) \cdot C(p)^{2.4}}{1 + 0.12\,S_{\min}(p)}$$
   where $C(p)$ is the fraction of orthogonal samples active within $[p-2, p+2]$ ($E > 16$), suppressing interior text lines and chart axes.
3. **Gutter Valley Detection & Run-Length Midpoint Collapsing**:
   Estimates border background via $8\times 8\times 8$ color quantization; detects low-variance interior gutters separating content on both sides and collapses contiguous gutter runs $[i, j]$ to a single midpoint cut.
4. **Recursive Hierarchical XY-Cut BSP + Sibling Grid-Line Propagation**:
   Splits along the dominant axis and recurses independently per band, followed by $\pm 12\text{px}$ sibling grid-line propagation (`cov >= 0.78`, `avg_raw >= 10`, `avg_smooth < 2.2`) so adjacent low-contrast panels never remain merged.
5. **Per-Cell Discontinuity Snap, Bleed Decontamination & 1:1 Contour Refinement**:
   Snaps each leaf edge within $\pm 8\text{px}$, peels `0..bleed_trim` anti-aliased transition/divider rows/columns where interior $L_1$ difference exceeds `20.0`, and—in `contour` mode—runs separable $O(N)$ dilation + Union-Find DSU connected components with **1:1 full-resolution tight edge trimming**.

---

## 2. Engineering & Rust Systems Optimizations

- **Zero-Copy NumPy Slice Borrowing**: Contiguous `HxWx3` / `HxWx4` `uint8` NumPy arrays are borrowed directly via `Cow<[u8]>` without copying memory, releasing the Python GIL (`py.allow_threads`).
- **Cache-Friendly Row-Major Vertical Profiling**: Vertical seam scans traverse rows on the outer loop (`ys.par_iter().fold().reduce()`), avoiding multi-kilobyte vertical stride L1/L2 cache misses.
- **200-Sample Orthogonal Stride + Compact Bitflag Tracking**: Keeps `1px` resolution along the primary split axis while striding orthogonal checks to $\approx 200$ points.
- **Unified Rust Dispatch (`detect_auto_rs`) & Zero-Disk Encoding (`batch_crop_to_memory_rs`)**: Executes `auto`, `xycut`, `contour`, and `grid` entirely in native Rust, with optional direct in-memory PNG/JPEG/WebP encoding for web servers and PyTorch pipelines.

---

## 3. Quick Start

```bash
# 1. Install Rust toolchain (rustup) and Python build dependencies
pip install maturin opencv-python-headless numpy pillow pytest

# 2. Build & install the release Rust extension
maturin develop --release

# 3. CLI batch splitting
smart-split slides_3x3.png -o ./slides_out --mode auto --bleed-trim 2 --format png

# 4. Run tests
pytest -v tests/
```

---

## 4. Python API Reference

```python
import cv2
from smart_splitter import SmartSplitter

splitter = SmartSplitter(
    mode="auto",            # "auto" | "xycut" | "contour" | "grid"
    sensitivity=0.65,       # 0.1 .. 1.0
    min_region_ratio=0.08,  # minimum sub-region ratio
    bleed_trim=2,           # strip 1..N px seam bleed / hairlines
    contour_gap_px=8,       # morphological merge radius for scattered objects
)

# File-to-disk parallel Rust cropping
result, output_paths = splitter.split_file("sheet.png", "output_slices/", fmt="png")
print(f"Extracted {len(output_paths)} slices [{result.detected_mode}/{result.backend}] in {result.elapsed_ms:.1f}ms")

# Zero-disk in-memory splitting
img = cv2.imread("sheet.png", cv2.IMREAD_UNCHANGED)
result, png_blobs = splitter.split_to_bytes(img, fmt="png")
```

### Optional ML Segmentation Hook (Natural Overlapping Collages)

```python
def sam2_segmenter(img_rgb):
    # Return list of (x, y, width, height) bounding boxes from SAM 2 / YOLO / ONNX
    return [(10, 20, 400, 300), (430, 20, 380, 300)]

ml_splitter = SmartSplitter(ml_segmenter=sam2_segmenter)
```
