# Smart Image Splitter (Rust PyO3 + Rayon + OpenCV)

High-performance Python image cutting library with a compiled **Rust** (`PyO3` + `rayon` + `image`) core and **OpenCV** (`cv2`) contour/fallback kernels. Designed for multi-panel slide grids, irregular Bento collages, guttered layouts, and isolated sprite/object extraction.

## Architecture

1. **Rust Native Extension (`smart_splitter._core`)**:
   - **Zero-Copy NumPy Borrowing**: Borrows contiguous `HxWx3` / `HxWx4` NumPy buffers directly without cloning; releases Python GIL via `py.allow_threads`.
   - **Cache-Friendly Row-Major Rayon Profiling**: Orthogonal 200-sample stride + single-pass active neighborhood tracking + sibling grid-line propagation (`detect_recursive_xy_cut_rs`).
   - **Native Rust Union-Find Contour Segmentation (`detect_contours_rs`)**: Separable $O(N)$ morphological closing + DSU connected components + 1:1 tight edge refinement.
   - **Unified Mode Dispatch (`detect_auto_rs`) & Zero-Disk In-Memory Slicing (`batch_crop_to_memory_rs` / `batch_crop_and_save_rs`)**.

2. **OpenCV + Optional ML Hook (`smart_splitter.engine`)**:
   - Tight-trimmed OpenCV connected-component fallback (`cv2.connectedComponentsWithStats`) when Rust extension is not compiled.
   - Drop-in `ml_segmenter` callback hook for SAM 2 / ONNX / PyTorch models when cutting overlapping natural scenes.

## Quick Start

```bash
# 1. Install Rust toolchain & maturin
pip install maturin opencv-python-headless numpy pillow

# 2. Build the Rust release extension into your Python environment
cd smart-image-splitter
maturin develop --release

# 3. Run on any image or folder
smart-split slides_3x3.png -o ./slides_out --mode auto --bleed-trim 2 --format png
```

## Python API

```python
from smart_splitter import SmartSplitter

splitter = SmartSplitter(mode="auto", sensitivity=0.65, bleed_trim=2)
result, output_paths = splitter.split_file("sheet.png", "output_slices/")
print(f"Extracted {len(output_paths)} clean images via {result.backend} in {result.elapsed_ms:.1f}ms")

# Zero-disk in-memory splitting for streaming servers / PyTorch dataloaders:
result, png_blobs = splitter.split_to_bytes(image_array, fmt="png")
```
