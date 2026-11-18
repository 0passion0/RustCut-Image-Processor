# smart-image-splitter · High-Precision Rust + Python Image Splitter

**English** | [简体中文](./README.zh-CN.md)

**smart-image-splitter** is a high-performance, pixel-accurate composite image segmentation Python package powered by a **native Rust (`PyO3` + `Rayon`) core** and a **full-parity vectorized OpenCV / NumPy fallback engine**.

It is engineered for real-world composite images where naive `W / N × H / M` grid slicing fails—including **slide sheets with misaligned seams**, **asymmetric Bento layouts**, **white-guttered collages**, **scattered objects on solid/transparent backgrounds**, and **overlapping photo grids (via optional ML segmentation hooks)**.

---

## Table of Contents
1. [Why Naive Grid Slicing Fails](#1-why-naive-grid-slicing-fails)
2. [Algorithm Theory & Mathematical Formulation](#2-algorithm-theory--mathematical-formulation)
   - [2.1 Directional Step + 1px Hairline Energy Profiling](#21-directional-step--1px-hairline-energy-profiling)
   - [2.2 Non-Linear Coverage Weighting & Tangential Smoothness Penalty](#22-non-linear-coverage-weighting--tangential-smoothness-penalty)
   - [2.3 Gutter Valley Detection & Contiguous Run Collapsing](#23-gutter-valley-detection--contiguous-run-collapsing)
   - [2.4 Recursive Hierarchical XY-Cut BSP & Sibling Grid-Line Propagation](#24-recursive-hierarchical-xy-cut-bsp--sibling-grid-line-propagation)
   - [2.5 Per-Cell Discontinuity Snap & Bleed Decontamination](#25-per-cell-discontinuity-snap--bleed-decontamination)
   - [2.6 Separable O(N) DSU Contour Segmentation & 1:1 Tight Edge Refinement](#26-separable-on-dsu-contour-segmentation--11-tight-edge-refinement)
3. [Engineering Practice & Extreme Performance Optimization](#3-engineering-practice--extreme-performance-optimization)
4. [Quick Start (`pip install` from PyPI or GitHub)](#4-quick-start-pip-install-from-pypi-or-github)
5. [Repository Structure](#5-repository-structure)
6. [License](#6-license)

---

## 1. Why Naive Grid Slicing Fails

Consider the included `2848 × 1600` 3×3 presentation slide sheet (`examples/sample_slides.png`):
- A uniform 3×3 mathematical grid slices at `x = 949, 1898` and `y = 533, 1066`.
- In reality, the true panel boundaries sit at:
  - Horizontal seams: `y = 531`, `y = 1043..1047`
  - Left column seam: `x = 945..946` (with a `1px` gray divider `(213, 214, 217)` at `x = 946`)
  - Right column seam: **`x = 1824`** in Rows 0–1, shifting **12px left to `x = 1812`** in Row 2!
- Naive uniform slicing produces up to **86 pixels of cross-panel bleed** and splits slide content down the middle.
- Worse yet, adjacent white-background slides (e.g., Row 1 Col 0 vs. Row 2 Col 0) are separated only by a faint `1px` shadow hairline—invisible to naive thresholding, yet crisp when modeled with a combined **step + impulse gradient** and **sibling grid-line propagation**.

---

## 2. Algorithm Theory & Mathematical Formulation

### 2.1 Directional Step + 1px Hairline Energy Profiling

For any candidate axis position $p$ (e.g., row $y$ with orthogonal samples $x \in [x_0, x_1)$), a true seam exhibits either:
1. **A color step transition** across two neighboring regions, or
2. **A `1px` hairline divider / drop shadow** where pixel $I(p)$ deviates sharply from both neighbors $I(p-1)$ and $I(p+1)$ (even when both panels share the exact same white background!).

We compute two $L_1$ color-space responses at each sample point $u$:

$$E_{\text{step}}(p, u) = \|I(p, u) - I(p+1, u)\|_1$$

$$E_{\text{line}}(p, u) = \frac{1}{2} \|2I(p, u) - I(p-1, u) - I(p+1, u)\|_1$$

Total local edge energy at $(p, u)$ is:

$$E(p, u) = E_{\text{step}}(p, u) + E_{\text{line}}(p, u)$$

Why this matters in practice:
- If Slide A (navy) meets Slide B (white), $E_{\text{step}}$ fires strongly ($\approx 600$).
- If Slide A (white) meets Slide B (white) across a `1px` light-gray shadow line $\Delta = 15$ per channel, $E_{\text{step}} \approx 45$ and $E_{\text{line}} \approx 45$, reinforcing each other while suppressing uniform interior noise.

### 2.2 Non-Linear Coverage Weighting & Tangential Smoothness Penalty

Text, charts, and tables inside a panel also create strong local gradients—**how do we avoid cutting through a chart title or table header?**

1. **$\pm 2\text{px}$ Neighborhood Coverage $C(p)$**:
   Real layout seams are rarely strictly horizontal to `0px` under JPEG/anti-aliasing compression; they may wobble by $\pm 1\text{–}2\text{px}$. A sample $u$ is marked active if any row/column within $[p-2, p+2]$ satisfies $E(r, u) > 16$:
   $$C(p) = \frac{1}{N} \sum_{u=1}^{N} \mathbb{I}\!\left[\max_{r \in [p-2, p+2]} E(r, u) > 16\right]$$
   Text lines rarely span $> 70\%$ of the full panel width, whereas true panel boundaries have $C(p) \in [0.85, 1.00]$. Applying the non-linear power $C(p)^{2.4}$ strongly attenuates partial interior lines (e.g., $0.4^{2.4} = 0.11$ vs. $0.95^{2.4} = 0.88$).

2. **Tangential Smoothness Penalty $S_{\text{tan}}(p)$**:
   Along a true divider or clean slide edge, pixel luminance varies smoothly in the tangential direction $u$:
   $$S_{\text{tan}}(p) = \frac{1}{N-1} \sum_{u=2}^{N} \left| L(p, u) - L(p, u-1) \right|, \quad L = R + G + B$$
   Taking the minimum smoothness across $[p-1, p, p+1]$ ensures a sharp edge right next to a clean background isn't penalized by texture on the opposite side.

3. **Combined Seam Score**:
   $$\text{Score}_{\text{seam}}(p) = \frac{\left(\bar{E}_{\text{step}}(p) + 0.7\,\bar{E}_{\text{line}}(p)\right) \cdot C(p)^{2.4}}{1 + 0.12 \cdot S_{\min}(p)}$$
   Candidate peaks are selected via **1D Non-Maximum Suppression (NMS)** with minimum spacing $d_{\min} = \max(24, \text{minRegionRatio} \cdot \min(W, H))$ and coverage gate $C(p) \ge 0.72$.

### 2.3 Gutter Valley Detection & Contiguous Run Collapsing

When panels are separated by **wide solid or transparent gutters** (e.g., a `40px` white margin between cards), edge gradients fire only on the gutter walls, not inside the empty gap.
1. Border background color $\mathbf{b}$ and border uniformity $U_{\text{border}}$ are estimated via $8\times 8\times 8$-quantized histogram binning over 4 image perimeters (with alpha transparency check $\alpha < 32$).
2. When $U_{\text{border}} > 0.72$, any interior strip with $> 95\%$ background match, low tangential variance ($S_{\text{tan}} < 2.5$), and foreground content present on **both sides** receives a high gutter score:
   $$\text{Score}_{\text{gutter}}(p) = \frac{110}{1 + S_{\text{tan}}(p)}$$
3. **Contiguous Run Collapsing**: A `40px` wide gutter generates 40 adjacent high-scoring rows/columns. Collapsing each contiguous gutter run $[i, j]$ to its single midpoint $\lfloor (i + j)/2 \rfloor$ guarantees cutting right down the center of the gutter rather than fracturing wide gutters into empty slivers. Empty background-only leaf regions ($< 3\%$ foreground density) are discarded automatically.

### 2.4 Recursive Hierarchical XY-Cut BSP & Sibling Grid-Line Propagation

Rather than slicing globally along X and Y simultaneously:
1. At each region $[x_0, y_0, x_1, y_1]$, profile both H and V seam scores.
2. Select the dominant split axis (choosing H if $\max \text{Score}_H \ge 0.72 \cdot \max \text{Score}_V$, matching natural row-first slide flow).
3. Partition into horizontal/vertical bands and **recursively profile each sub-band independently**.
   - *Result*: Row 0–1 naturally finds `x = 1824` while Row 2 independently finds `x = 1812`—zero global misalignment!
4. **Sibling Grid-Line Propagation**:
   Suppose Column 1|2 has a faint white-on-white boundary below threshold $T$ on one sub-band while sibling columns above/below split cleanly around `y_cut`. For every internal boundary line $p_{\text{cut}}$ produced by siblings, any adjacent leaf spanning $p_{\text{cut}}$ searches a narrow $\pm 12\text{px}$ window under relaxed thresholds ($\text{cov} \ge 0.78$, $\bar{E}_{\text{raw}} \ge 10$, $S_{\text{tan}} < 2.2$). This recovers low-contrast white-on-white seams with zero false positives.

### 2.5 Per-Cell Discontinuity Snap & Bleed Decontamination

After BSP leaves are finalized, each cell undergoes two refinement stages:
1. **Local $\pm 8\text{px}$ Edge Snap**: Each interior edge snaps independently within $\pm 8\text{px}$ to the exact pixel transition of *that specific cell* (correcting minor diagonal tilt or row-by-row drift).
2. **Outer Margin & Anti-Aliased Bleed Decontamination (`bleed_trim`)**:
   - First strips any remaining uniform canvas background rows/columns (`> 94%` background match).
   - Then inspects up to `bleed_trim` (default `2px`) outer pixel rows/columns along internal cut edges against an interior reference line `ref = edge ± (bleed_trim + 2)`. Whenever the average $L_1$ difference exceeds `20`, the outer `1px` transition/divider pixel (e.g., `1px` gray border or JPEG chroma ringing) is peeled off—guaranteeing **zero white halo on dark slides and zero dark edge bleed on light slides**.

### 2.6 Separable $O(N)$ DSU Contour Segmentation & 1:1 Tight Edge Refinement

When an image consists of scattered objects, UI stickers, or transparent RGBA sprites ($U_{\text{border}} > 0.72$ and contour coverage $< 65\%$ of total area with $\ge 2$ components, or explicit `contour` mode):
1. Downsample onto a canonical $\le 720\text{px}$ grid (`scale = ceil(max(W, H) / 720)`) and threshold foreground against $\mathbf{b}$ (or $\alpha > 24$).
2. Apply **separable 1D forward/backward sweep morphological closing/dilation** in $O(s_w \cdot s_h)$ time (independent of `contour_gap_px`, avoiding $O(N \cdot r^2)$ nested loops).
3. Extract 8-connected components using two-pass **Disjoint-Set Union (DSU) with path halving** in Rust (`src/lib.rs`) or `cv2.connectedComponentsWithStats` in OpenCV (`python/smart_splitter/engine.py`).
4. **1:1 Full-Resolution Edge Tightening**: Because downsampling introduces up to $\pm \text{scale}$ quantization error, every component bounding box is shrunk/expanded directly against 1:1 raw image pixels until all 4 borders sit on exact foreground boundaries (`0px` integer error).

---

## 3. Engineering Practice & Extreme Performance Optimization

| Optimization | Problem Solved | Concrete Implementation |
|---|---|---|
| **1. Row-Major Parallel Scanning (`Rayon`)** | Vertical column loops `for x { for y }` stride `W * C` bytes per pixel read, thrashing CPU L1/L2 caches | Inverted vertical profiling to row-major `ys.par_iter().fold().reduce()` in Rust so every CPU thread streams contiguous cache lines |
| **2. Single-Pass Active Tracking** | Allocating `O(span × 200)` 2D energy matrices per recursive band wasted memory bandwidth | Replaced 2D matrices with a monotonic 1D cursor tracker proving exact equivalence to $\max_{r \in [p-2,p+2]} E(r,u) > 16$ |
| **3. 200-Sample Orthogonal Stride** | Scanning all 2,848 pixels horizontally at every row is redundant; skipping primary-axis rows misses `1px` odd hairlines | Kept primary-axis step strictly **`1px`** while striding the orthogonal axis to $\approx 200$ samples |
| **4. Zero-Copy NumPy Borrow + GIL Release** | Copying multi-megapixel NumPy buffers across FFI and blocking Python threads | Borrows C-contiguous NumPy arrays zero-copy (`Cow<[u8]>`) and releases the GIL via `py.allow_threads(...)` during detection and encoding |
| **5. Parallel In-Memory Batch Cropping** | Disk I/O bottlenecks when serving slices via FastAPI or feeding PyTorch DataLoaders | `batch_crop_to_memory_rs` encodes all slices concurrently in RAM across Rayon worker threads without touching disk |

---

## 4. Quick Start (`pip install` from PyPI or GitHub)

### Installation (Zero Extra Setup)

```bash
# Option 1: Install from PyPI (recommended)
pip install smart-image-splitter

# Option 2: Install directly via Git + pip
pip install git+https://github.com/0passion0/RustCut-Image-Processor.git

# Option 3: Install via GitHub source archive URL (works even without git CLI installed)
pip install https://github.com/0passion0/RustCut-Image-Processor/archive/refs/heads/main.zip

# Option 4: Install a prebuilt binary wheel from GitHub Releases (precompiled Rust PyO3, no compiler needed)
pip install https://github.com/0passion0/RustCut-Image-Processor/releases/latest/download/smart_image_splitter-0.2.1-cp311-cp311-manylinux_2_17_x86_64.whl
```

> **Note on GitHub URLs & `pip`**: Standard `pip` cannot install a bare GitHub webpage URL (`pip install https://github.com/owner/repo`) because GitHub serves HTML there. Always use `git+https://github.com/owner/repo` or `/archive/refs/heads/main.zip` (or a Release `.whl`).
>
> **Automatic Rust / OpenCV Dual-Backend**:
> - If `cargo` (Rust toolchain) is installed on the target machine, `pip install` automatically compiles `smart_splitter._core` (`PyO3` + `Rayon`) in release mode.
> - If `cargo` is **not** installed, `setup.py` skips Rust compilation gracefully with **zero errors** and uses the full-parity vectorized OpenCV / NumPy backend.
> - Pushing a version tag (`git tag v0.2.1 && git push --tags`) triggers `.github/workflows/release-wheels.yml` (`PyO3/maturin-action`) to publish prebuilt native Rust wheels for Linux, macOS, and Windows.

### CLI & Python API Usage

```bash
# CLI batch splitting
smart-split examples/sample_slides.png -o ./output_slices --mode auto --bleed-trim 2 --format png
```

```python
import cv2
from smart_splitter import SmartSplitter

splitter = SmartSplitter(mode="auto", sensitivity=0.65, bleed_trim=2)

# 1. File-to-directory splitting
result, paths = splitter.split_file("examples/sample_slides.png", "output_slices/")
print(f"Split into {len(paths)} slices via {result.backend} in {result.elapsed_ms:.1f}ms")

# 2. Zero-disk in-memory splitting (FastAPI / PyTorch dataloaders)
img = cv2.imread("examples/sample_slides.png")
result, png_blobs = splitter.split_to_bytes(img, fmt="png")

# 3. Optional ML segmentation hook (SAM 2 / YOLO / ONNX) for overlapping natural photos
ml_splitter = SmartSplitter(ml_segmenter=my_sam2_box_predictor)
```

### Local Development & Testing

```bash
pip install setuptools wheel setuptools-rust maturin pytest numpy opencv-python-headless pillow
maturin develop --release
pytest -v tests/
```

---

## 5. Repository Structure

```text
.
├── Cargo.toml                        # Rust PyO3 + Rayon crate manifest
├── pyproject.toml                    # PEP 517 + setuptools / maturin build configuration
├── setup.py                          # Optional Rust extension builder with graceful fallback
├── MANIFEST.in                       # Source distribution manifest (Python + Rust sources only)
├── LICENSE / CONTRIBUTING.md
├── README.md / README.zh-CN.md
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Root pip install + Cargo + maturin + pytest CI
│   │   └── release-wheels.yml        # Multi-OS prebuilt wheel publisher (Linux/macOS/Windows)
│   ├── ISSUE_TEMPLATE/bug_report.md
│   └── pull_request_template.md
├── src/
│   └── lib.rs                        # Native Rust (PyO3 + Rayon) engine: XY-Cut, DSU Contour & Memory Crop
├── python/
│   └── smart_splitter/
│       ├── __init__.py               # Public exports (SmartSplitter, SplitBox, SplitResult)
│       ├── engine.py                 # Hybrid Rust / full-parity OpenCV engine & ML hook
│       └── cli.py                    # `smart-split` CLI entrypoint
├── tests/
│   └── test_splitter.py              # Pytest unit tests (XY-Cut, White Gutter, Bleed Trim, Contour)
└── examples/
    └── sample_slides.png             # Real-world 2848x1600 3x3 slide benchmark image
```

---

## 6. License

Dual-licensed under [Apache-2.0 OR MIT](LICENSE).
