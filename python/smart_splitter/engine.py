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

    @staticmethod
    def _estimate_border_bg(img: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        h, w = img.shape[:2]
        sx = max(1, w // 80)
        sy = max(1, h // 80)
        if img.shape[2] == 4:
            borders_a = np.concatenate([img[0, ::sx, 3], img[-1, ::sx, 3], img[::sy, 0, 3], img[::sy, -1, 3]])
            trans_ratio = float((borders_a < 32).mean())
            if trans_ratio > 0.5:
                return np.array([0, 0, 0], dtype=np.float32), True, trans_ratio

        borders = np.concatenate(
            [img[0, ::sx, :3], img[-1, ::sx, :3], img[::sy, 0, :3], img[::sy, -1, :3]],
            axis=0,
        )
        bins: dict[tuple[int, int, int], list] = {}
        for px in borders:
            key = (int(px[0]) >> 3, int(px[1]) >> 3, int(px[2]) >> 3)
            entry = bins.get(key)
            if entry is None:
                bins[key] = [1, int(px[0]), int(px[1]), int(px[2])]
            else:
                entry[0] += 1
                entry[1] += int(px[0])
                entry[2] += int(px[1])
                entry[3] += int(px[2])
        best = max(bins.values(), key=lambda v: v[0])
        cnt = max(1, best[0])
        bg_rgb = np.array([best[1] / cnt, best[2] / cnt, best[3] / cnt], dtype=np.float32)
        return bg_rgb, False, float(best[0] / max(1, len(borders)))

    def _detect_xycut_opencv(self, img: np.ndarray) -> List[SplitBox]:
        """Full-parity vectorized NumPy/OpenCV Recursive XY-Cut with sibling propagation & bleed trim."""
        h, w = img.shape[:2]
        arr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3] if img.shape[2] == 4 else None
        min_dim = max(24, int(round(min(w, h) * self.min_region_ratio)))
        threshold = 125.0 - self.sensitivity * 80.0
        bg_rgb, is_trans, bg_unif = self._estimate_border_bg(img)
        bg_uniform = bg_unif > 0.72

        def profile_axis(x0: int, y0: int, x1: int, y1: int, horizontal: bool) -> List[Tuple[int, float, bool]]:
            if horizontal:
                if y1 - y0 <= min_dim * 2 or x1 - x0 < 8:
                    return []
                step_u = max(1, (x1 - x0) // 200)
                sub = arr[y0:y1, x0:x1:step_u]
                step = np.abs(sub[1:-1] - sub[2:]).sum(axis=2)
                line = 0.5 * np.abs(2 * sub[1:-1] - sub[:-2] - sub[2:]).sum(axis=2)
                energy = step + line
                e_max = energy.copy()
                for dy in (-2, -1, 1, 2):
                    shifted = np.roll(energy, dy, axis=0)
                    if dy > 0:
                        shifted[:dy] = 0
                    else:
                        shifted[dy:] = 0
                    e_max = np.maximum(e_max, shifted)
                cov = (e_max > 16.0).mean(axis=1)
                smooth = np.abs(np.diff(sub[1:-1].sum(axis=2), axis=1)).mean(axis=1)
                min_sm = np.minimum(smooth, np.minimum(np.roll(smooth, 1), np.roll(smooth, -1)))
                raw = step.mean(axis=1) + 0.7 * line.mean(axis=1)
                seam_scores = (raw * (cov ** 2.4)) / (1.0 + min_sm * 0.12)

                if bg_uniform:
                    if is_trans and alpha is not None:
                        bg_match = (alpha[y0 + 1 : y1 - 1, x0:x1:step_u] < 24).mean(axis=1)
                    else:
                        bg_match = (np.abs(sub[1:-1] - bg_rgb).sum(axis=2) < 22.0).mean(axis=1)
                else:
                    bg_match = np.zeros(len(seam_scores), dtype=np.float32)

                base_pos = y0 + 1
            else:
                if x1 - x0 <= min_dim * 2 or y1 - y0 < 8:
                    return []
                step_u = max(1, (y1 - y0) // 200)
                sub = arr[y0:y1:step_u, x0:x1]
                step = np.abs(sub[:, 1:-1] - sub[:, 2:]).sum(axis=2)
                line = 0.5 * np.abs(2 * sub[:, 1:-1] - sub[:, :-2] - sub[:, 2:]).sum(axis=2)
                energy = step + line
                e_max = energy.copy()
                for dx in (-2, -1, 1, 2):
                    shifted = np.roll(energy, dx, axis=1)
                    if dx > 0:
                        shifted[:, :dx] = 0
                    else:
                        shifted[:, dx:] = 0
                    e_max = np.maximum(e_max, shifted)
                cov = (e_max > 16.0).mean(axis=0)
                smooth = np.abs(np.diff(sub[:, 1:-1].sum(axis=2), axis=0)).mean(axis=0)
                min_sm = np.minimum(smooth, np.minimum(np.roll(smooth, 1), np.roll(smooth, -1)))
                raw = step.mean(axis=0) + 0.7 * line.mean(axis=0)
                seam_scores = (raw * (cov ** 2.4)) / (1.0 + min_sm * 0.12)

                if bg_uniform:
                    if is_trans and alpha is not None:
                        bg_match = (alpha[y0:y1:step_u, x0 + 1 : x1 - 1] < 24).mean(axis=0)
                    else:
                        bg_match = (np.abs(sub[:, 1:-1] - bg_rgb).sum(axis=2) < 22.0).mean(axis=0)
                else:
                    bg_match = np.zeros(len(seam_scores), dtype=np.float32)

                base_pos = x0 + 1

            lo_i = min_dim - 1
            hi_i = len(seam_scores) - min_dim + 1
            if hi_i <= lo_i:
                return []

            cands: List[List] = []
            for i in range(lo_i, hi_i):
                sc = float(seam_scores[i])
                gs = 0.0
                if bg_uniform and smooth[i] < 2.5 and bg_match[i] > 0.95:
                    has_before = bool(np.any(bg_match[:i:3] < 0.90))
                    has_after = bool(np.any(bg_match[i + 3 :: 3] < 0.90))
                    if has_before and has_after:
                        gs = float(110.0 / (1.0 + smooth[i]))
                is_g = gs > sc
                cands.append([base_pos + i, max(sc, gs), float(cov[i]), is_g])

            # Collapse contiguous gutter runs to single midpoint peak
            idx = 0
            while idx < len(cands):
                if not cands[idx][3]:
                    idx += 1
                    continue
                j = idx
                while j + 1 < len(cands) and cands[j + 1][3] and cands[j + 1][0] <= cands[j][0] + 1:
                    j += 1
                mid = (idx + j) // 2
                mx = max(c[1] for c in cands[idx : j + 1])
                for k in range(idx, j + 1):
                    if k == mid:
                        cands[k][1] = mx
                    else:
                        cands[k][1] = 0.0
                        cands[k][3] = False
                idx = j + 1

            valid = [c for c in cands if c[1] >= threshold and (c[3] or c[2] >= 0.72)]
            valid.sort(key=lambda c: c[1], reverse=True)
            picked: List[Tuple[int, float, bool]] = []
            for pos, sc, _, is_g in valid:
                if all(abs(pos - p) >= min_dim for p, _, _ in picked):
                    picked.append((pos, sc, is_g))
            picked.sort(key=lambda c: c[0])
            return picked

        # [x0, y0, x1, y1, conf, is_gutter]
        leaves: List[Tuple[int, int, int, int, float, bool]] = []

        def recurse(x0: int, y0: int, x1: int, y1: int, depth: int, conf: float, is_g: bool) -> None:
            if depth >= 4 or ((x1 - x0) < min_dim * 2 and (y1 - y0) < min_dim * 2):
                leaves.append((x0, y0, x1, y1, conf, is_g))
                return
            hp = profile_axis(x0, y0, x1, y1, True)
            vp = profile_axis(x0, y0, x1, y1, False)
            if not hp and not vp:
                leaves.append((x0, y0, x1, y1, conf, is_g))
                return
            mh = max((s for _, s, _ in hp), default=0.0)
            mv = max((s for _, s, _ in vp), default=0.0)
            if hp and (not vp or mh >= mv * 0.72):
                bounds = [y0] + [p for p, _, _ in hp] + [y1]
                for i in range(len(bounds) - 1):
                    p_sc, p_g = hp[0][1:] if i == 0 else hp[i - 1][1:]
                    recurse(x0, bounds[i], x1, bounds[i + 1], depth + 1, max(conf, p_sc), is_g or p_g)
            else:
                bounds = [x0] + [p for p, _, _ in vp] + [x1]
                for i in range(len(bounds) - 1):
                    p_sc, p_g = vp[0][1:] if i == 0 else vp[i - 1][1:]
                    recurse(bounds[i], y0, bounds[i + 1], y1, depth + 1, max(conf, p_sc), is_g or p_g)

        recurse(0, 0, w, h, 0, 120.0, False)

        # Sibling Grid-Line Propagation (+-12px) for low-contrast adjacent panels
        for _ in range(3):
            propagated = False
            h_lines = sorted({y for l in leaves for y in (l[1], l[3]) if min_dim < y < h - min_dim})
            v_lines = sorted({x for l in leaves for x in (l[0], l[2]) if min_dim < x < w - min_dim})

            for idx, leaf in enumerate(leaves):
                lx0, ly0, lx1, ly1, lconf, lg = leaf
                split_done = False
                for y_cut in h_lines:
                    if y_cut - ly0 >= min_dim and ly1 - y_cut >= min_dim:
                        sx = max(1, (lx1 - lx0) // 200)
                        y_lo = max(ly0 + min_dim, y_cut - 12)
                        y_hi = min(ly1 - min_dim, y_cut + 12)
                        best_y, best_sc = -1, 0.0
                        for y in range(y_lo, y_hi + 1):
                            row_c = arr[y, lx0 + 4 : lx1 - 4 : sx]
                            if len(row_c) < 2:
                                continue
                            ds = np.abs(row_c - arr[y + 1, lx0 + 4 : lx1 - 4 : sx]).sum(axis=1)
                            dl = 0.5 * np.abs(
                                2 * row_c - arr[y - 1, lx0 + 4 : lx1 - 4 : sx] - arr[y + 1, lx0 + 4 : lx1 - 4 : sx]
                            ).sum(axis=1)
                            cov = float(((ds + dl) > 6.0).mean())
                            avg_raw = float((ds + dl).mean())
                            avg_sm = float(np.abs(np.diff(row_c, axis=0)).sum(axis=1).mean())
                            if cov >= 0.78 and avg_raw >= 10.0 and avg_sm < 2.2:
                                sc = (avg_raw * cov) / (1.0 + avg_sm * 0.2)
                                if sc > best_sc:
                                    best_sc, best_y = sc, y
                        if best_y > 0:
                            leaves[idx : idx + 1] = [
                                (lx0, ly0, lx1, best_y, max(lconf, best_sc), lg),
                                (lx0, best_y, lx1, ly1, max(lconf, best_sc), lg),
                            ]
                            propagated = split_done = True
                            break
                if split_done:
                    break

                for x_cut in v_lines:
                    if x_cut - lx0 >= min_dim and lx1 - x_cut >= min_dim:
                        sy = max(1, (ly1 - ly0) // 200)
                        x_lo = max(lx0 + min_dim, x_cut - 12)
                        x_hi = min(lx1 - min_dim, x_cut + 12)
                        best_x, best_sc = -1, 0.0
                        for x in range(x_lo, x_hi + 1):
                            col_c = arr[ly0 + 4 : ly1 - 4 : sy, x]
                            if len(col_c) < 2:
                                continue
                            ds = np.abs(col_c - arr[ly0 + 4 : ly1 - 4 : sy, x + 1]).sum(axis=1)
                            dl = 0.5 * np.abs(
                                2 * col_c - arr[ly0 + 4 : ly1 - 4 : sy, x - 1] - arr[ly0 + 4 : ly1 - 4 : sy, x + 1]
                            ).sum(axis=1)
                            cov = float(((ds + dl) > 6.0).mean())
                            avg_raw = float((ds + dl).mean())
                            avg_sm = float(np.abs(np.diff(col_c, axis=0)).sum(axis=1).mean())
                            if cov >= 0.78 and avg_raw >= 10.0 and avg_sm < 2.2:
                                sc = (avg_raw * cov) / (1.0 + avg_sm * 0.2)
                                if sc > best_sc:
                                    best_sc, best_x = sc, x
                        if best_x > 0:
                            leaves[idx : idx + 1] = [
                                (lx0, ly0, best_x, ly1, max(lconf, best_sc), lg),
                                (best_x, ly0, lx1, ly1, max(lconf, best_sc), lg),
                            ]
                            propagated = True
                            break
                if propagated:
                    break
            if not propagated:
                break

        # Filter empty background leaves when canvas has uniform border
        if bg_uniform:
            kept = []
            for lx0, ly0, lx1, ly1, lconf, lg in leaves:
                sx = max(1, (lx1 - lx0) // 32)
                sy = max(1, (ly1 - ly0) // 32)
                if is_trans and alpha is not None:
                    sub_a = alpha[ly0 + 2 : ly1 - 2 : sy, lx0 + 2 : lx1 - 2 : sx]
                    fg_ratio = float((sub_a >= 24).mean()) if sub_a.size else 1.0
                else:
                    sub = arr[ly0 + 2 : ly1 - 2 : sy, lx0 + 2 : lx1 - 2 : sx]
                    fg_ratio = float((np.abs(sub - bg_rgb).sum(axis=2) > 20.0).mean()) if sub.size else 1.0
                if fg_ratio >= 0.03:
                    kept.append((lx0, ly0, lx1, ly1, lconf, lg))
            if kept:
                leaves = kept

        leaves.sort(key=lambda b: (b[1] // max(1, min_dim), b[0]))

        out: List[SplitBox] = []
        cur_r, cur_c = 0, 0
        last_y = leaves[0][1] if leaves else 0
        for idx, (rx0, ry0, rx1, ry1, conf, is_g) in enumerate(leaves):
            if idx > 0 and abs(ry0 - last_y) > (ry1 - ry0) * 0.35:
                cur_r += 1
                cur_c = 0
                last_y = ry0
            elif idx > 0:
                cur_c += 1

            x0, y0, x1, y1 = rx0, ry0, rx1, ry1
            if not is_g:
                if y0 > 2 and x1 - x0 > 8:
                    sx = max(1, (x1 - x0) // 120)
                    by, be = y0, -1.0
                    for y in range(max(1, y0 - 8), min(y1 - 20, y0 + 8) + 1):
                        r0, r1, r2 = arr[y - 1, x0 + 4 : x1 - 4 : sx], arr[y, x0 + 4 : x1 - 4 : sx], arr[y + 1, x0 + 4 : x1 - 4 : sx]
                        e = float((np.abs(r1 - r2).sum(axis=1) + 0.5 * np.abs(2 * r1 - r0 - r2).sum(axis=1)).mean())
                        if e > be:
                            be, by = e, y
                    if be > 15.0:
                        y0 = by + 1
                if y1 < h - 2 and x1 - x0 > 8:
                    sx = max(1, (x1 - x0) // 120)
                    by, be = y1, -1.0
                    for y in range(max(y0 + 20, y1 - 8), min(h - 2, y1 + 8) + 1):
                        r0, r1, r2 = arr[y - 1, x0 + 4 : x1 - 4 : sx], arr[y, x0 + 4 : x1 - 4 : sx], arr[y + 1, x0 + 4 : x1 - 4 : sx]
                        e = float((np.abs(r0 - r1).sum(axis=1) + 0.5 * np.abs(2 * r1 - r0 - r2).sum(axis=1)).mean())
                        if e > be:
                            be, by = e, y
                    if be > 15.0:
                        y1 = by
                if x0 > 2 and y1 - y0 > 8:
                    sy = max(1, (y1 - y0) // 120)
                    bx, be = x0, -1.0
                    for x in range(max(1, x0 - 8), min(x1 - 20, x0 + 8) + 1):
                        c0, c1, c2 = arr[y0 + 4 : y1 - 4 : sy, x - 1], arr[y0 + 4 : y1 - 4 : sy, x], arr[y0 + 4 : y1 - 4 : sy, x + 1]
                        e = float((np.abs(c1 - c2).sum(axis=1) + 0.5 * np.abs(2 * c1 - c0 - c2).sum(axis=1)).mean())
                        if e > be:
                            be, bx = e, x
                    if be > 15.0:
                        x0 = bx + 1
                if x1 < w - 2 and y1 - y0 > 8:
                    sy = max(1, (y1 - y0) // 120)
                    bx, be = x1, -1.0
                    for x in range(max(x0 + 20, x1 - 8), min(w - 2, x1 + 8) + 1):
                        c0, c1, c2 = arr[y0 + 4 : y1 - 4 : sy, x - 1], arr[y0 + 4 : y1 - 4 : sy, x], arr[y0 + 4 : y1 - 4 : sy, x + 1]
                        e = float((np.abs(c0 - c1).sum(axis=1) + 0.5 * np.abs(2 * c1 - c0 - c2).sum(axis=1)).mean())
                        if e > be:
                            be, bx = e, x
                    if be > 15.0:
                        x1 = bx

            tt = tr = tb = tl = 0
            if bg_uniform:
                def is_row_bg(y_val: int) -> bool:
                    sx = max(1, (x1 - x0) // 60)
                    if is_trans and alpha is not None:
                        return float((alpha[y_val, x0:x1:sx] < 24).mean()) > 0.94
                    return float((np.abs(arr[y_val, x0:x1:sx] - bg_rgb).sum(axis=1) < 22.0).mean()) > 0.94

                def is_col_bg(x_val: int) -> bool:
                    sy = max(1, (y1 - y0) // 60)
                    if is_trans and alpha is not None:
                        return float((alpha[y0:y1:sy, x_val] < 24).mean()) > 0.94
                    return float((np.abs(arr[y0:y1:sy, x_val] - bg_rgb).sum(axis=1) < 22.0).mean()) > 0.94

                while y0 < y1 - 24 and is_row_bg(y0):
                    y0 += 1; tt += 1
                while y1 > y0 + 24 and is_row_bg(y1 - 1):
                    y1 -= 1; tb += 1
                while x0 < x1 - 24 and is_col_bg(x0):
                    x0 += 1; tl += 1
                while x1 > x0 + 24 and is_col_bg(x1 - 1):
                    x1 -= 1; tr += 1

            if self.bleed_trim > 0 and y1 - y0 > 32 and x1 - x0 > 32:
                sx = max(1, (x1 - x0) // 80)
                sy = max(1, (y1 - y0) // 80)
                if ry0 > 2:
                    ref_y = min(y1 - 1, y0 + self.bleed_trim + 2)
                    for _ in range(self.bleed_trim):
                        if float(np.abs(arr[y0, x0 + 8 : x1 - 8 : sx] - arr[ref_y, x0 + 8 : x1 - 8 : sx]).sum(axis=1).mean()) > 20.0:
                            y0 += 1; tt += 1
                        else:
                            break
                if ry1 < h - 2:
                    ref_y = max(y0, y1 - 1 - self.bleed_trim - 2)
                    for _ in range(self.bleed_trim):
                        if float(np.abs(arr[y1 - 1, x0 + 8 : x1 - 8 : sx] - arr[ref_y, x0 + 8 : x1 - 8 : sx]).sum(axis=1).mean()) > 20.0:
                            y1 -= 1; tb += 1
                        else:
                            break
                if rx0 > 2:
                    ref_x = min(x1 - 1, x0 + self.bleed_trim + 2)
                    for _ in range(self.bleed_trim):
                        if float(np.abs(arr[y0 + 8 : y1 - 8 : sy, x0] - arr[y0 + 8 : y1 - 8 : sy, ref_x]).sum(axis=1).mean()) > 20.0:
                            x0 += 1; tl += 1
                        else:
                            break
                if rx1 < w - 2:
                    ref_x = max(x0, x1 - 1 - self.bleed_trim - 2)
                    for _ in range(self.bleed_trim):
                        if float(np.abs(arr[y0 + 8 : y1 - 8 : sy, x1 - 1] - arr[y0 + 8 : y1 - 8 : sy, ref_x]).sum(axis=1).mean()) > 20.0:
                            x1 -= 1; tr += 1
                        else:
                            break

            out.append(
                SplitBox(
                    index=idx + 1,
                    row=cur_r,
                    col=cur_c,
                    x=x0,
                    y=y0,
                    width=max(1, x1 - x0),
                    height=max(1, y1 - y0),
                    confidence=min(1.0, conf / 150.0),
                    strategy="gutter" if is_g else "xycut",
                    trimmed_edges=(tt, tr, tb, tl),
                )
            )
        return out
