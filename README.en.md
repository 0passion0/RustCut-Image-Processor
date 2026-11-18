# smart-image-splitter · High-Precision Rust + Python Image Splitter

[简体中文](./README.md) | **English**

[![PyPI](https://img.shields.io/pypi/v/smart-image-splitter?color=blue&label=PyPI)](https://pypi.org/project/smart-image-splitter/)
[![Python](https://img.shields.io/pypi/pyversions/smart-image-splitter)](https://pypi.org/project/smart-image-splitter/)
[![CI](https://github.com/0passion0/RustCut-Image-Processor/actions/workflows/ci.yml/badge.svg)](https://github.com/0passion0/RustCut-Image-Processor/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-green.svg)](./LICENSE)

**smart-image-splitter** is a high-performance, pixel-accurate composite image segmentation Python package powered **exclusively by a native Rust (`PyO3` + `Rayon` + `image`) parallel core**.

It is engineered for real-world composite images where naive `W/N × H/M` grid slicing fails—including **slide sheets with misaligned seams**, **asymmetric Bento layouts**, **white-guttered collages**, **scattered objects on solid/transparent backgrounds**, **uniform output dimension normalization (`pad` / `resize` / `crop` natively in Rust)**, and **overlapping photo grids (via optional ML segmentation hooks)**.

---

## Table of Contents

1. [Why Naive Grid Slicing Fails](#1-why-naive-grid-slicing-fails)
2. [Splitting Modes & Architecture](#2-splitting-modes--architecture)
3. [Algorithm Theory & Mathematical Formulation](#3-algorithm-theory--mathematical-formulation)
4. [Engineering Practice & Performance Optimization](#4-engineering-practice--performance-optimization)
5. [Quick Start (PyPI / GitHub / CLI)](#5-quick-start-pypi--github--cli)
6. [Python API & CLI Reference](#6-python-api--cli-reference)
7. [Repository Structure](#7-repository-structure)
8. [License](#8-license)

---

## 1. Why Naive Grid Slicing Fails

Consider the included `2848 × 1600` 3×3 presentation slide sheet ([`examples/sample_slides.png`](./examples/sample_slides.png)):
- A uniform 3×3 mathematical grid slices at `x = 949, 1898` and `y = 533, 1066`.
- In reality, the true panel boundaries sit at:
  - Horizontal seams: `y = 531`, `y = 1043..1047`
  - Left column seam: `x = 945..946` (with a `1px` gray divider `(213, 214, 217)` at `x = 946`)
  - Right column seam: **`x = 1824`** in Rows 0–1, shifting **12px left to `x = 1812`** in Row 2!
- Naive uniform slicing produces up to **86 pixels of cross-panel bleed** and fails on faint `1px` shadow hairlines between adjacent white slides.

---

## 2. Splitting Modes & Architecture

```text
Input Image (np.ndarray / File Path)
  │
  ├─► If ml_segmenter hook provided ──► External SAM 2 / YOLO / ONNX boxes ("ml-hook")
  │
  └─► Native Rust Core (`smart_splitter._core`, GIL-released Rayon execution)
        ├─ mode = "grid"    ──► Uniform grid with optional gutter deduction
        ├─ mode = "contour" ──► Separable closing + DSU 8-connected components + 1:1 edge tightening
        ├─ mode = "xycut"   ──► Recursive XY-Cut BSP + sibling grid propagation + ±8px snap + bleed trim
        └─ mode = "auto"    ──► Selects contour on uniform backgrounds with scattered foreground (< 65% area), else xycut
              │
              ▼ [Optional: uniform_size=True / target_size=(W, H)]
        └─ Uniform Dimension Normalization (Native Rust Rayon)
              ├─ uniform_mode = "pad"    ──► Center-pad with detected border background/alpha (default)
              ├─ uniform_mode = "resize" ──► Resize every slice directly to target dimensions
              └─ uniform_mode = "crop"   ──► Align raw bounding boxes on canvas without scaling/padding
```

---

## 3. Algorithm Theory & Mathematical Formulation

- **Directional Step + `1px` Hairline Energy**:
  ```text
  E_step(p, u) = || I(p, u) - I(p + 1, u) ||₁
  E_line(p, u) = 0.5 × || 2·I(p, u) - I(p - 1, u) - I(p + 1, u) ||₁
  ```
  Combines step jumps and second-order `1px` shadow hairlines.
- **Non-Linear Coverage & Tangential Smoothness**:
  ```text
  Score_seam(p) = [ (mean(E_step(p)) + 0.7 × mean(E_line(p))) × C(p)^2.4 ] / (1 + 0.12 × S_min(p))
  ```
  Suppresses internal text/charts while selecting true cross-panel seams via 1D NMS (`C(p) ≥ 0.72`).
- **Gutter Valley Collapsing**: Collapses wide solid/transparent gutters `[i, j]` to their single geometric midpoint `⌊(i + j) / 2⌋` and discards empty background leaves (`< 3%` foreground).
- **Recursive XY-Cut BSP & Sibling Propagation**: Handles shifted rows (`x = 1824` vs. `x = 1812`) independently and recovers faint adjacent boundaries in a `±12px` window around sibling cut lines.
- **Per-Cell Snap & `bleed_trim`**: Independently snaps each leaf edge within `±8px` and trims uniform margins plus up to `bleed_trim` anti-aliased edge bleed.
- **Separable `O(N)` DSU Contours**: Downsamples to `≤ 720px`, runs linear-time separable 1D sweep closing + Disjoint-Set Union connected components, and tightens bounds back at 1:1 resolution.

---

## 4. Engineering Practice & Performance Optimization

| Optimization | Problem Solved | Concrete Implementation |
|---|---|---|
| **Row-Major Rayon Scans** | Column-wise loops stride `W * C` bytes and thrash L1/L2 caches | `ys.par_iter().fold().reduce()` keeps every thread streaming contiguous cache lines |
| **1-Pass Neighborhood Cursor** | Eliminates 2D `O(span × 200)` energy matrix allocations | Monotonic 1D cursor tracks `±2px` coverage peaks in-place |
| **Zero-Copy NumPy + GIL Release** | Avoids full-image copies and Python GIL contention | Borrows C-contiguous buffers via `Cow<[u8]>` inside `py.allow_threads(...)` |
| **In-Memory Parallel Normalization** | Eliminates disk I/O & Python padding bottlenecks | `batch_crop_to_memory_rs` / `batch_crop_and_save_rs` run `pad`/`resize`/`crop` & encoding concurrently in Rust Rayon |

---

## 5. Quick Start (PyPI / GitHub / CLI)

```bash
# Option 1: Install from PyPI (recommended, prebuilt abi3 wheels for Linux / macOS / Windows)
pip install smart-image-splitter

# Option 2: Install directly from GitHub source
pip install git+https://github.com/0passion0/RustCut-Image-Processor.git

# Option 3: Install prebuilt abi3 wheel from GitHub Releases (example: v0.2.3 Linux x86_64)
pip install https://github.com/0passion0/RustCut-Image-Processor/releases/download/v0.2.3/smart_image_splitter-0.2.3-cp39-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

```bash
# Standard CLI usage
smart-split examples/sample_slides.png -o ./output_slices --mode auto --bleed-trim 2 --format png

# Enforce uniform output dimensions across all slices (default: center-pad)
smart-split examples/sample_slides.png -o ./output_uniform --uniform-size --uniform-mode pad

# Or explicit target size with direct equal-sized canvas cropping
smart-split examples/sample_slides.png -o ./output_fixed --target-size 800x500 --uniform-mode crop
```

```python
import cv2
from smart_splitter import SmartSplitter

splitter = SmartSplitter(mode="auto", sensitivity=0.65, bleed_trim=2)

# 1. Split image file directly to disk
result, paths = splitter.split_file("examples/sample_slides.png", "output_slices/")
print(f"{result.backend} ({result.detected_mode}) wrote {len(paths)} slices in {result.elapsed_ms:.1f}ms")

# 2. Enforce uniform output dimensions natively in Rust Rayon (pad / resize / crop)
uniform_splitter = SmartSplitter(
    mode="auto",
    uniform_size=True,       # or target_size=(512, 512)
    uniform_mode="pad",      # "pad" | "resize" | "crop"
)
img = cv2.imread("examples/sample_slides.png")
result, png_blobs = uniform_splitter.split_to_bytes(img, fmt="png")

# 3. Optional ML segmentation callback (SAM 2 / YOLO / ONNX)
ml_splitter = SmartSplitter(ml_segmenter=my_sam2_box_predictor)
```

---

## 6. Python API & CLI Reference

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"auto"` | `"auto"` \| `"xycut"` \| `"contour"` \| `"grid"` |
| `sensitivity` | `0.65` | Edge & foreground detection sensitivity (`[0.1, 1.0]`) |
| `min_region_ratio` | `0.08` | Minimum panel ratio relative to `min(W, H)` (`[0.03, 0.35]`) |
| `bleed_trim` | `2` | Maximum seam anti-alias trim in pixels |
| `contour_gap_px` | `8` | Morphological closing radius in `contour` mode (px) |
| `uniform_size` | `False` | Force all output slices to share identical dimensions (max width × max height) |
| `target_size` | `None` | Explicit `(width, height)` target dimensions (implies `uniform_size=True`) |
| `uniform_mode` | `"pad"` | Normalization mode: `"pad"` (background fill), `"resize"`, or `"crop"` (canvas box alignment) |
| `ml_segmenter` | `None` | Optional callback returning `(x, y, w, h)` bounding boxes |

```bash
pip install setuptools wheel setuptools-rust maturin pytest numpy opencv-python-headless pillow
maturin develop --release
pytest -v tests/
cargo clippy --all-targets -- -D warnings
```

---

## 7. Repository Structure

```text
.
├── Cargo.toml                        # Rust PyO3 + Rayon crate config (abi3-py39 + Release LTO)
├── pyproject.toml                    # PEP 517 + setuptools / maturin config
├── setup.py / MANIFEST.in            # Adaptive build (OpenCV fallback when cargo is absent)
├── LICENSE / CONTRIBUTING.md         # Apache-2.0 OR MIT dual license & guidelines
├── README.md / README.en.md          # Chinese (default) / English docs
├── .github/workflows/                # CI & multi-platform abi3 wheel / PyPI release workflows
├── src/lib.rs                        # Native Rust core (XY-Cut BSP, DSU contours, uniform sizing, parallel encoding)
├── python/smart_splitter/            # Python package (__init__.py, engine.py, cli.py)
├── tests/test_splitter.py            # Pytest suite
└── examples/sample_slides.png        # 2848x1600 multi-panel test sheet
```

---

## 8. License

Dual-licensed under **[Apache-2.0 OR MIT](LICENSE)**.
