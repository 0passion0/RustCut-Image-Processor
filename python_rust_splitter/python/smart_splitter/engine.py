"""
SmartSplitter: Hybrid Rust (PyO3 + Rayon) + OpenCV + Optional ML Segmentation Engine.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import cv2
import numpy as np

try:
    from . import _core as _rust_core
    HAS_RUST_CORE = True
except ImportError:
    _rust_core = None
    HAS_RUST_CORE = False


SplitMode = Literal["auto", "xycut", "contour", "grid"]


@dataclass
class SplitBox:
    index: int
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    confidence: float
    strategy: str
    trimmed_edges: Tuple[int, int, int, int] = (0, 0, 0, 0)  # top, right, bottom, left


@dataclass
class SplitResult:
    image_width: int
    image_height: int
    detected_mode: str
    backend: str
    elapsed_ms: float
    boxes: List[SplitBox] = field(default_factory=list)


class SmartSplitter:
    """
    Production multi-strategy image splitter.

    - Uses compiled Rust PyO3 + Rayon native extension (`smart_splitter._core`) when available.
    - Falls back cleanly to vectorized NumPy + OpenCV (`cv2`) C++ kernels if Rust extension is not built.
    - Supports an optional ML segmentation callback (e.g. SAM 2 / YOLO / ONNX) for overlapping natural photos.
    """

    def __init__(
        self,
        mode: SplitMode = "auto",
        sensitivity: float = 0.65,
        min_region_ratio: float = 0.08,
        bleed_trim: int = 2,
        contour_gap_px: int = 8,
        ml_segmenter: Optional[Callable[[np.ndarray], List[Tuple[int, int, int, int]]]] = None,
    ) -> None:
        self.mode = mode
        self.sensitivity = float(np.clip(sensitivity, 0.1, 1.0))
        self.min_region_ratio = float(np.clip(min_region_ratio, 0.03, 0.35))
        self.bleed_trim = max(0, int(bleed_trim))
        self.contour_gap_px = max(1, int(contour_gap_px))
        self.ml_segmenter = ml_segmenter

    def analyze(
        self,
        image_bgr_or_rgb: np.ndarray,
        rows: int = 3,
        cols: int = 3,
        gutter_px: int = 0,
    ) -> SplitResult:
        t0 = time.perf_counter()
        if image_bgr_or_rgb.ndim != 3:
            raise ValueError("Expected HxWx3 or HxWx4 image array.")
        h, w = image_bgr_or_rgb.shape[:2]

        if self.ml_segmenter is not None:
            rects = self.ml_segmenter(image_bgr_or_rgb)
            if rects:
                boxes = [
                    SplitBox(i + 1, i, 0, x, y, bw, bh, 0.99, "ml-segmentation")
                    for i, (x, y, bw, bh) in enumerate(rects)
                ]
                return SplitResult(w, h, "ml", "ml-hook", (time.perf_counter() - t0) * 1000, boxes)

        # Unified Native Rust Path (handles auto, xycut, contour, and grid zero-copy via PyO3 + Rayon)
        if HAS_RUST_CORE:
            detected_mode, rust_boxes = _rust_core.detect_auto_rs(
                np.ascontiguousarray(image_bgr_or_rgb),
                self.mode,
                self.sensitivity,
                self.min_region_ratio,
                self.bleed_trim,
                self.contour_gap_px,
                rows,
                cols,
                gutter_px,
            )
            boxes = [
                SplitBox(
                    index=b.index,
                    row=b.row,
                    col=b.col,
                    x=b.x,
                    y=b.y,
                    width=b.width,
                    height=b.height,
                    confidence=b.confidence,
                    strategy=b.strategy,
                    trimmed_edges=(b.trim_top, b.trim_right, b.trim_bottom, b.trim_left),
                )
                for b in rust_boxes
            ]
            return SplitResult(w, h, detected_mode, "rust-rayon", (time.perf_counter() - t0) * 1000, boxes)

        if self.mode == "grid":
            boxes = self._split_uniform_grid(w, h, rows, cols, gutter_px)
            return SplitResult(w, h, "grid", "numpy", (time.perf_counter() - t0) * 1000, boxes)

        if self.mode == "contour":
            boxes = self._detect_contours_cv2(image_bgr_or_rgb)
            return SplitResult(w, h, "contour", "opencv", (time.perf_counter() - t0) * 1000, boxes)

        xy_boxes = self._detect_xycut_opencv(image_bgr_or_rgb)
        if self.mode == "auto":
            bg_unif = self._border_uniformity(image_bgr_or_rgb)
            if bg_unif > 0.72:
                contour_boxes = self._detect_contours_cv2(image_bgr_or_rgb)
                contour_area = sum(b.width * b.height for b in contour_boxes)
                if len(contour_boxes) >= 2 and (
                    len(xy_boxes) <= 1 or (contour_area < (w * h) * 0.65 and len(contour_boxes) >= len(xy_boxes))
                ):
                    return SplitResult(w, h, "contour", "opencv", (time.perf_counter() - t0) * 1000, contour_boxes)

        return SplitResult(w, h, "xycut", "opencv-numpy", (time.perf_counter() - t0) * 1000, xy_boxes)

    def split_to_bytes(
        self,
        image_bgr_or_rgb: np.ndarray,
        fmt: str = "png",
        rows: int = 3,
        cols: int = 3,
        gutter_px: int = 0,
    ) -> Tuple[SplitResult, List[bytes]]:
        """Zero-disk in-memory splitting for streaming servers and ML/PyTorch pipelines."""
        res = self.analyze(image_bgr_or_rgb, rows=rows, cols=cols, gutter_px=gutter_px)
        if not res.boxes:
            return res, []

        if HAS_RUST_CORE:
            coords = [(b.x, b.y, b.width, b.height) for b in res.boxes]
            raw_list = _rust_core.batch_crop_to_memory_rs(
                np.ascontiguousarray(image_bgr_or_rgb), coords, fmt
            )
            return res, [bytes(item) for item in raw_list]

        ext = f".{fmt.lower().lstrip('.')}"
        out_bytes: List[bytes] = []
        for b in res.boxes:
            crop = image_bgr_or_rgb[b.y : b.y + b.height, b.x : b.x + b.width]
            ok, enc = cv2.imencode(ext, crop)
            if ok:
                out_bytes.append(enc.tobytes())
        return res, out_bytes

    def split_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        prefix: str = "slice",
        fmt: str = "png",
        rows: int = 3,
        cols: int = 3,
        gutter_px: int = 0,
    ) -> Tuple[SplitResult, List[str]]:
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        res = self.analyze(img, rows=rows, cols=cols, gutter_px=gutter_px)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if HAS_RUST_CORE and res.boxes:
            coords = [(b.x, b.y, b.width, b.height) for b in res.boxes]
            saved_paths = _rust_core.batch_crop_and_save_rs(
                str(input_path), coords, str(out_dir), prefix, fmt
            )
            return res, saved_paths

        saved_paths = []
        ext = fmt.lower().lstrip(".")
        for b in res.boxes:
            crop = img[b.y : b.y + b.height, b.x : b.x + b.width]
            out_file = out_dir / f"{prefix}_{b.index:02d}.{ext}"
            cv2.imwrite(str(out_file), crop)
            saved_paths.append(str(out_file))
        return res, saved_paths

    @staticmethod
    def _split_uniform_grid(w: int, h: int, rows: int, cols: int, gutter: int) -> List[SplitBox]:
        rows = max(1, rows)
        cols = max(1, cols)
        half_g = gutter // 2
        boxes: List[SplitBox] = []
        idx = 1
        for r in range(rows):
            y0 = round((r * h) / rows) + (half_g if r > 0 else 0)
            y1 = round(((r + 1) * h) / rows) - ((gutter - half_g) if r < rows - 1 else 0)
            for c in range(cols):
                x0 = round((c * w) / cols) + (half_g if c > 0 else 0)
                x1 = round(((c + 1) * w) / cols) - ((gutter - half_g) if c < cols - 1 else 0)
                boxes.append(SplitBox(idx, r, c, x0, y0, max(1, x1 - x0), max(1, y1 - y0), 1.0, "grid"))
                idx += 1
        return boxes

    @staticmethod
    def _border_uniformity(img: np.ndarray) -> float:
        h, w = img.shape[:2]
        top = img[0, :: max(1, w // 80), :3]
        bot = img[-1, :: max(1, w // 80), :3]
        left = img[:: max(1, h // 80), 0, :3]
        right = img[:: max(1, h // 80), -1, :3]
        border = np.concatenate([top, bot, left, right], axis=0).astype(np.int32)
        med = np.median(border, axis=0)
        dist = np.abs(border - med).sum(axis=1)
        return float((dist < 24).mean())

    def _detect_contours_cv2(self, img: np.ndarray) -> List[SplitBox]:
        h, w = img.shape[:2]
        if img.shape[2] == 4 and (img[:, :, 3] < 24).mean() > 0.25:
            fg = (img[:, :, 3] > 24).astype(np.uint8) * 255
        else:
            border = np.concatenate([img[0, :, :3], img[-1, :, :3], img[:, 0, :3], img[:, -1, :3]], axis=0)
            bg_med = np.median(border, axis=0).astype(np.int16)
            diff = np.abs(img[:, :, :3].astype(np.int16) - bg_med).sum(axis=2)
            thresh = 32 - self.sensitivity * 18
            fg = (diff > thresh).astype(np.uint8) * 255

        k = self.contour_gap_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        min_area = (w * h) * (self.min_region_ratio ** 2) * 0.45
        raw = []
        for lbl in range(1, num_labels):
            x, y, bw, bh, area = stats[lbl]
            if area >= min_area:
                # Tight 1:1 trim against unclosed foreground mask
                sub = fg[y : y + bh, x : x + bw]
                ys, xs = np.where(sub > 0)
                if len(xs) > 0:
                    tx0, tx1 = int(xs.min()), int(xs.max()) + 1
                    ty0, ty1 = int(ys.min()), int(ys.max()) + 1
                    raw.append((int(x) + tx0, int(y) + ty0, max(1, tx1 - tx0), max(1, ty1 - ty0)))
                else:
                    raw.append((int(x), int(y), int(bw), int(bh)))

        raw.sort(key=lambda r: (r[1] // max(1, int(min(h, w) * 0.1)), r[0]))
        return [
            SplitBox(i + 1, i, 0, x, y, bw, bh, 0.98, "contour")
            for i, (x, y, bw, bh) in enumerate(raw)
        ]

    def _detect_xycut_opencv(self, img: np.ndarray) -> List[SplitBox]:
        """Vectorized OpenCV + NumPy Recursive XY-Cut fallback with local seam refinement."""
        h, w = img.shape[:2]
        arr = img[:, :, :3].astype(np.float32)
        min_dim = max(24, int(round(min(w, h) * self.min_region_ratio)))
        threshold = 125.0 - self.sensitivity * 80.0

        def profile_axis(x0: int, y0: int, x1: int, y1: int, horizontal: bool) -> List[Tuple[int, float]]:
            sub = arr[y0:y1, x0:x1]
            if horizontal:
                if y1 - y0 <= min_dim * 2:
                    return []
                step = np.abs(sub[1:-1] - sub[2:]).sum(axis=2)
                line = 0.5 * np.abs(2 * sub[1:-1] - sub[:-2] - sub[2:]).sum(axis=2)
                energy = step + line
                # +-2px neighborhood max across rows
                e_max = energy.copy()
                for dy in (-2, -1, 1, 2):
                    shifted = np.roll(energy, dy, axis=0)
                    e_max = np.maximum(e_max, shifted)
                cov = (e_max > 16.0).mean(axis=1)
                smooth = np.abs(np.diff(sub[1:-1].sum(axis=2), axis=1)).mean(axis=1)
                raw = step.mean(axis=1) + 0.7 * line.mean(axis=1)
                scores = (raw * (cov ** 2.4)) / (1.0 + smooth * 0.12)
                cands = [(y0 + 1 + i, float(scores[i])) for i in range(min_dim - 1, len(scores) - min_dim + 1) if cov[i] >= 0.72 and scores[i] >= threshold]
            else:
                if x1 - x0 <= min_dim * 2:
                    return []
                step = np.abs(sub[:, 1:-1] - sub[:, 2:]).sum(axis=2)
                line = 0.5 * np.abs(2 * sub[:, 1:-1] - sub[:, :-2] - sub[:, 2:]).sum(axis=2)
                energy = step + line
                e_max = energy.copy()
                for dx in (-2, -1, 1, 2):
                    e_max = np.maximum(e_max, np.roll(energy, dx, axis=1))
                cov = (e_max > 16.0).mean(axis=0)
                smooth = np.abs(np.diff(sub[:, 1:-1].sum(axis=2), axis=0)).mean(axis=0)
                raw = step.mean(axis=0) + 0.7 * line.mean(axis=0)
                scores = (raw * (cov ** 2.4)) / (1.0 + smooth * 0.12)
                cands = [(x0 + 1 + i, float(scores[i])) for i in range(min_dim - 1, len(scores) - min_dim + 1) if cov[i] >= 0.72 and scores[i] >= threshold]

            cands.sort(key=lambda c: c[1], reverse=True)
            picked: List[Tuple[int, float]] = []
            for pos, sc in cands:
                if all(abs(pos - p) >= min_dim for p, _ in picked):
                    picked.append((pos, sc))
            picked.sort(key=lambda c: c[0])
            return picked

        leaves: List[Tuple[int, int, int, int, float]] = []

        def recurse(x0: int, y0: int, x1: int, y1: int, depth: int, conf: float) -> None:
            if depth >= 4 or ((x1 - x0) < min_dim * 2 and (y1 - y0) < min_dim * 2):
                leaves.append((x0, y0, x1, y1, conf))
                return
            hp = profile_axis(x0, y0, x1, y1, True)
            vp = profile_axis(x0, y0, x1, y1, False)
            if not hp and not vp:
                leaves.append((x0, y0, x1, y1, conf))
                return
            mh = max((s for _, s in hp), default=0.0)
            mv = max((s for _, s in vp), default=0.0)
            if hp and (not vp or mh >= mv * 0.72):
                bounds = [y0] + [p for p, _ in hp] + [y1]
                for i in range(len(bounds) - 1):
                    recurse(x0, bounds[i], x1, bounds[i + 1], depth + 1, max(conf, mh))
            else:
                bounds = [x0] + [p for p, _ in vp] + [x1]
                for i in range(len(bounds) - 1):
                    recurse(bounds[i], y0, bounds[i + 1], y1, depth + 1, max(conf, mv))

        recurse(0, 0, w, h, 0, 120.0)
        leaves.sort(key=lambda b: (b[1] // max(1, min_dim), b[0]))
        return [
            SplitBox(
                i + 1,
                i // 3,
                i % 3,
                x0 + (self.bleed_trim if x0 > 2 else 0),
                y0 + (self.bleed_trim if y0 > 2 else 0),
                max(1, (x1 - (self.bleed_trim if x1 < w - 2 else 0)) - (x0 + (self.bleed_trim if x0 > 2 else 0))),
                max(1, (y1 - (self.bleed_trim if y1 < h - 2 else 0)) - (y0 + (self.bleed_trim if y0 > 2 else 0))),
                min(1.0, conf / 150.0),
                "xycut",
            )
            for i, (x0, y0, x1, y1, conf) in enumerate(leaves)
        ]
