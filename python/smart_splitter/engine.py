"""
SmartSplitter 核心图像切割引擎模块。

集成原生 Rust（PyO3 + Rayon）零拷贝高性能加速内核、全特性对齐的向量化 OpenCV/NumPy 回退引擎，
以及可选的机器学习分割回调接口（如 SAM 2 / YOLO / ONNX）。

:author: tongliga
:date: 2026-11-18
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import cv2
import numpy as np

# 按包元数据探测原生 Rust 扩展模块是否可用，避免异常捕获式导入
if importlib.util.find_spec(f"{__package__}._core") is not None:
    from . import _core as _rust_core

    HAS_RUST_CORE = True
else:
    _rust_core = None
    HAS_RUST_CORE = False


SplitMode = Literal["auto", "xycut", "contour", "grid"]


@dataclass
class SplitBox:
    """
    单个检测子图区域的坐标与裁切元数据载体。

    :author: tongliga
    :date: 2026-11-18
    """

    index: int
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    confidence: float
    strategy: str
    trimmed_edges: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (上, 右, 下, 左) 已裁剪边缘像素数


@dataclass
class SplitResult:
    """
    整张图像的布局检测结果汇总对象。

    :author: tongliga
    :date: 2026-11-18
    """

    image_width: int
    image_height: int
    detected_mode: str
    backend: str
    elapsed_ms: float
    boxes: List[SplitBox] = field(default_factory=list)


class SmartSplitter:
    """
    生产级多策略复合图像智能切割引擎。

    - 当存在编译好的原生扩展 ``smart_splitter._core`` 时，优先走 Rust + Rayon 零拷贝多线程引擎；
    - 当环境仅安装纯 Python 依赖时，自动切换至算法全特性对齐的向量化 NumPy + OpenCV 内核；
    - 支持注入自定义机器学习分割回调函数（``ml_segmenter``）以应对前景重叠场景。

    :author: tongliga
    :date: 2026-11-18
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
        """
        初始化 ``SmartSplitter`` 切割配置与可选分割钩子。

        :param mode: 布局检测策略模式，可选值为 ``"auto"``、``"xycut"``、``"contour"`` 或 ``"grid"``。
        :type mode: Literal["auto", "xycut", "contour", "grid"]
        :param sensitivity: 边缘阶跃与前景阈值检测灵敏度，内部裁剪至 ``[0.1, 1.0]`` 区间。
        :type sensitivity: float
        :param min_region_ratio: 最小子图边长占图像短边的比例下限，内部裁剪至 ``[0.03, 0.35]``。
        :type min_region_ratio: float
        :param bleed_trim: 内部接缝抗锯齿去杂边的最大剥离像素数（px），最小为 0。
        :type bleed_trim: int
        :param contour_gap_px: 轮廓模式形态学闭运算连接相邻离散元素的像素间隙半径（px），最小为 1。
        :type contour_gap_px: int
        :param ml_segmenter: 可选的外部目标检测/分割回调函数，接收图像数组并返回 ``(x, y, w, h)`` 元组列表。
        :type ml_segmenter: Optional[Callable[[np.ndarray], List[Tuple[int, int, int, int]]]]
        :author: tongliga
        :date: 2026-11-18
        """
        # 第一步：保存模式配置与经过安全区间裁剪后的数值超参数
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
        """
        对输入内存图像数组执行布局分析，返回所有子图区域的精确包围盒集合。

        :param image_bgr_or_rgb: 形状为 ``(H, W, 3)`` 或 ``(H, W, 4)`` 的 3 维图像像素数组。
        :type image_bgr_or_rgb: numpy.ndarray
        :param rows: 当 ``mode == "grid"`` 时指定均匀切分的行数。
        :type rows: int
        :param cols: 当 ``mode == "grid"`` 时指定均匀切分的列数。
        :type cols: int
        :param gutter_px: 当 ``mode == "grid"`` 时指定行列网格间的固定留白间距（px）。
        :type gutter_px: int
        :return: 包含原图宽高、命中模式、执行后端名称、耗时（ms）及子图列表的 ``SplitResult`` 对象。
        :raises ValueError: 当输入图像数组维度不等于 3（非 ``HxWx3`` / ``HxWx4``）时抛出。
        :author: tongliga
        :date: 2026-11-18
        """
        t0 = time.perf_counter()

        # 第一步：校验输入像素数组维度，确保符合 HxWx3 或 HxWx4 契约
        if image_bgr_or_rgb.ndim != 3:
            raise ValueError("Expected HxWx3 or HxWx4 image array.")
        h, w = image_bgr_or_rgb.shape[:2]

        # 第二步：若已注入外部 ML 分割回调且成功返回候选框，则优先封装 ML 结果返回
        if self.ml_segmenter is not None:
            rects = self.ml_segmenter(image_bgr_or_rgb)
            if rects:
                boxes = [
                    SplitBox(i + 1, i, 0, x, y, bw, bh, 0.99, "ml-segmentation")
                    for i, (x, y, bw, bh) in enumerate(rects)
                ]
                return SplitResult(w, h, "ml", "ml-hook", (time.perf_counter() - t0) * 1000, boxes)

        # 第三步：优先调度原生 Rust（PyO3 + Rayon）核心（内部零拷贝借用连续缓冲区并释放 GIL）
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

        # 第四步：无原生 Rust 扩展时，按模式回退至全特性对齐的 NumPy / OpenCV 内核
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
                # 当背景高度一致且轮廓前景分散（总面积不足 65% 或 XY-Cut 无法切分出多块）时切换至轮廓模式
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
        """
        零磁盘落地内存级图像切割与编码，适用于流式服务端及深度学习数据流水线。

        :param image_bgr_or_rgb: 形状为 ``(H, W, 3)`` 或 ``(H, W, 4)`` 的输入图像像素数组。
        :type image_bgr_or_rgb: numpy.ndarray
        :param fmt: 输出图像编码格式，如 ``"png"``、``"jpg"`` 或 ``"webp"``。
        :type fmt: str
        :param rows: ``grid`` 模式下的均匀网格行数。
        :type rows: int
        :param cols: ``grid`` 模式下的均匀网格列数。
        :type cols: int
        :param gutter_px: ``grid`` 模式下的行列网格间隙像素数（px）。
        :type gutter_px: int
        :return: 二元组 ``(SplitResult, List[bytes])``，分别为布局分析摘要与各切片编码后的字节流列表。
        :raises ValueError: 当输入数组维度非 3 维或底层 Rust 编码遇到不支持的格式时抛出。
        :author: tongliga
        :date: 2026-11-18
        """
        # 第一步：执行布局检测，若未检出有效子图则直接返回空字节流列表
        res = self.analyze(image_bgr_or_rgb, rows=rows, cols=cols, gutter_px=gutter_px)
        if not res.boxes:
            return res, []

        # 第二步：若原生 Rust 核心可用，调用 Rayon 并发内存裁剪编码器
        if HAS_RUST_CORE:
            coords = [(b.x, b.y, b.width, b.height) for b in res.boxes]
            raw_list = _rust_core.batch_crop_to_memory_rs(
                np.ascontiguousarray(image_bgr_or_rgb), coords, fmt
            )
            return res, [bytes(item) for item in raw_list]

        # 第三步：OpenCV 回退路径逐块切片并内存编码
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
        """
        从磁盘读取输入图像并执行智能切割，将所有切片文件落盘保存至目标目录。

        :param input_path: 输入图像的文件路径。
        :type input_path: str | pathlib.Path
        :param output_dir: 输出切片文件的目标目录路径（不存在时自动创建）。
        :type output_dir: str | pathlib.Path
        :param prefix: 输出文件名前缀，最终文件名为 ``{prefix}_{index:02d}.{fmt}``。
        :type prefix: str
        :param fmt: 输出图像文件扩展格式，如 ``"png"``、``"jpg"`` 或 ``"webp"``。
        :type fmt: str
        :param rows: ``grid`` 模式下的划分行数。
        :type rows: int
        :param cols: ``grid`` 模式下的划分列数。
        :type cols: int
        :param gutter_px: ``grid`` 模式下的网格间隙像素数（px）。
        :type gutter_px: int
        :return: 二元组 ``(SplitResult, List[str])``，分别为分析结果摘要与所有输出切片文件的字符串路径列表。
        :raises FileNotFoundError: 当 ``input_path`` 不存在或无法被 OpenCV 解码时抛出。
        :raises ValueError: 当底层图像保存失败或通道格式不合法时抛出。
        :author: tongliga
        :date: 2026-11-18
        """
        # 第一步：使用 OpenCV 读取原图（保留 Alpha 透明通道），并将单通道灰度图转换为 BGR 三通道
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 第二步：分析子图布局并确保输出目录已就绪
        res = self.analyze(img, rows=rows, cols=cols, gutter_px=gutter_px)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 第三步：优先交由 Rust Rayon 并发读盘裁切落盘，否则走 OpenCV 逐块写入
        if HAS_RUST_CORE and res.boxes:
            coords = [(b.x, b.y, b.width, b.height) for b in res.boxes]
            saved_paths = _rust_core.batch_crop_and_save_rs(
                str(input_path), coords, str(out_dir), prefix, fmt
            )
            return res, saved_paths

        saved_paths: List[str] = []
        ext = fmt.lower().lstrip(".")
        for b in res.boxes:
            crop = img[b.y : b.y + b.height, b.x : b.x + b.width]
            out_file = out_dir / f"{prefix}_{b.index:02d}.{ext}"
            cv2.imwrite(str(out_file), crop)
            saved_paths.append(str(out_file))
        return res, saved_paths

    @staticmethod
    def _split_uniform_grid(w: int, h: int, rows: int, cols: int, gutter: int) -> List[SplitBox]:
        """
        按指定行列数与间距执行数学均匀网格切分。

        :param w: 图像总宽度（px）。
        :type w: int
        :param h: 图像总高度（px）。
        :type h: int
        :param rows: 目标划分行数（至少为 1）。
        :type rows: int
        :param cols: 目标划分列数（至少为 1）。
        :type cols: int
        :param gutter: 相邻单元格之间扣减的留白间隔（px）。
        :type gutter: int
        :return: 按行主序编号的 ``SplitBox`` 网格切片列表。
        :author: tongliga
        :date: 2026-11-18
        """
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
        """
        抽样计算图像四周外框像素的颜色一致性比例，用于辅助判断是否存在纯色大背景。

        :param img: 3 维输入图像像素数组（``HxWx3`` 或 ``HxWx4``）。
        :type img: numpy.ndarray
        :return: 外框采样点与中位色距离小于阈值的比例（范围 ``[0.0, 1.0]``）。
        :author: tongliga
        :date: 2026-11-18
        """
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
        """
        基于 OpenCV 形态学闭运算与 8-连通域统计的离散对象轮廓分割器。

        :param img: 3 维输入图像像素数组（``HxWx3`` 或 ``HxWx4``）。
        :type img: numpy.ndarray
        :return: 经过最小面积过滤与 1:1 像素级紧密收紧后的 ``SplitBox`` 列表。
        :author: tongliga
        :date: 2026-11-18
        """
        h, w = img.shape[:2]

        # 第一步：构建二值前景掩膜（含 25% 以上透明像素时优先采用 Alpha 通道）
        if img.shape[2] == 4 and (img[:, :, 3] < 24).mean() > 0.25:
            fg = (img[:, :, 3] > 24).astype(np.uint8) * 255
        else:
            border = np.concatenate([img[0, :, :3], img[-1, :, :3], img[:, 0, :3], img[:, -1, :3]], axis=0)
            bg_med = np.median(border, axis=0).astype(np.int16)
            diff = np.abs(img[:, :, :3].astype(np.int16) - bg_med).sum(axis=2)
            thresh = 32 - self.sensitivity * 18
            fg = (diff > thresh).astype(np.uint8) * 255

        # 第二步：执行矩形结构元闭运算，弥合组件内部微小空隙后提取 8-连通域
        k = self.contour_gap_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        # 第三步：过滤微噪点连通域，并回溯原始未膨胀掩膜 fg 将四边精确收紧到 1:1 像素边界
        min_area = (w * h) * (self.min_region_ratio ** 2) * 0.45
        # raw: 存放有效目标矩形的列表，元素契约为 (x, y, width, height)
        raw: List[Tuple[int, int, int, int]] = []
        for lbl in range(1, num_labels):
            x, y, bw, bh, area = stats[lbl]
            if area >= min_area:
                sub = fg[y : y + bh, x : x + bw]
                ys, xs = np.where(sub > 0)
                if len(xs) > 0:
                    tx0, tx1 = int(xs.min()), int(xs.max()) + 1
                    ty0, ty1 = int(ys.min()), int(ys.max()) + 1
                    raw.append((int(x) + tx0, int(y) + ty0, max(1, tx1 - tx0), max(1, ty1 - ty0)))
                else:
                    raw.append((int(x), int(y), int(bw), int(bh)))

        # 第四步：按近似行带与水平坐标排序并封装为 SplitBox
        raw.sort(key=lambda r: (r[1] // max(1, int(min(h, w) * 0.1)), r[0]))
        return [
            SplitBox(i + 1, i, 0, x, y, bw, bh, 0.98, "contour")
            for i, (x, y, bw, bh) in enumerate(raw)
        ]

    @staticmethod
    def _estimate_border_bg(img: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        采用 8x8x8 量化直方图从图像四边采样估计主背景颜色与一致性占比。

        :param img: 3 维输入图像像素数组（``HxWx3`` 或 ``HxWx4``）。
        :type img: numpy.ndarray
        :return: 三元组 ``(bg_rgb, is_transparent, uniformity_ratio)``，分别为背景均色向量、是否透明底及一致率。
        :author: tongliga
        :date: 2026-11-18
        """
        h, w = img.shape[:2]
        sx = max(1, w // 80)
        sy = max(1, h // 80)

        # 第一步：若四边过半像素处于低 Alpha 透明状态，直接判定为透明背景
        if img.shape[2] == 4:
            borders_a = np.concatenate([img[0, ::sx, 3], img[-1, ::sx, 3], img[::sy, 0, 3], img[::sy, -1, 3]])
            trans_ratio = float((borders_a < 32).mean())
            if trans_ratio > 0.5:
                return np.array([0, 0, 0], dtype=np.float32), True, trans_ratio

        # 第二步：统计四边采样像素在 5-bit（>>3）量化色桶中的众数直方图
        borders = np.concatenate(
            [img[0, ::sx, :3], img[-1, ::sx, :3], img[::sy, 0, :3], img[::sy, -1, :3]],
            axis=0,
        )
        # bins: 量化色桶直方图，结构为 {(r_bin, g_bin, b_bin): [count, sum_r, sum_g, sum_b]}
        bins: dict[tuple[int, int, int], list[int]] = {}
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

        # 第三步：计算命中频次最高色桶的平均颜色及覆盖比例
        best = max(bins.values(), key=lambda v: v[0])
        cnt = max(1, best[0])
        bg_rgb = np.array([best[1] / cnt, best[2] / cnt, best[3] / cnt], dtype=np.float32)
        return bg_rgb, False, float(best[0] / max(1, len(borders)))

    def _detect_xycut_opencv(self, img: np.ndarray) -> List[SplitBox]:
        """
        全特性对齐的向量化 NumPy/OpenCV 递归 XY-Cut BSP 切割算法（含同级网格线传播与去杂边）。

        :param img: 3 维输入图像像素数组（``HxWx3`` 或 ``HxWx4``）。
        :type img: numpy.ndarray
        :return: 经过递归剖分、同级微弱边界传播、空背景剔除与边缘精修后的 ``SplitBox`` 列表。
        :author: tongliga
        :date: 2026-11-18
        """
        h, w = img.shape[:2]
        arr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3] if img.shape[2] == 4 else None
        min_dim = max(24, int(round(min(w, h) * self.min_region_ratio)))
        threshold = 125.0 - self.sensitivity * 80.0
        bg_rgb, is_trans, bg_unif = self._estimate_border_bg(img)
        bg_uniform = bg_unif > 0.72

        def profile_axis(x0: int, y0: int, x1: int, y1: int, horizontal: bool) -> List[Tuple[int, float, bool]]:
            """
            计算子区域沿水平或垂直方向的一维接缝能量剖面，并执行谷值坍缩与 NMS 峰值提取。

            :param x0: 区域左上角 X 坐标。
            :type x0: int
            :param y0: 区域左上角 Y 坐标。
            :type y0: int
            :param x1: 区域右下角 X 坐标。
            :type x1: int
            :param y1: 区域右下角 Y 坐标。
            :type y1: int
            :param horizontal: 为 ``True`` 时按水平切线方向扫行，为 ``False`` 时按垂直方向扫列。
            :type horizontal: bool
            :return: 有效分割线列表，每项为 ``(分割坐标, 综合得分, 是否属于纯白/透明留白沟壑)``。
            :author: tongliga
            :date: 2026-11-18
            """
            # 第一步：沿主扫描轴以 1px 步长采样计算阶跃梯度 + 1px 细线脉冲能量及 ±2px 邻域覆盖率
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

            # 第二步：合成梯度接缝分与留白沟壑分
            # cands: 候选扫描线记录列表，元素契约为 [coord_pos, best_score, coverage_ratio, is_gutter_bool]
            cands: List[list] = []
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

            # 第三步：将宽留白沟壑的连续高分带坍缩为唯一几何中点，杜绝切片碎裂
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

            # 第四步：执行 1D 非极大值抑制（NMS）筛选满足最小间距的切割线并按坐标升序返回
            valid = [c for c in cands if c[1] >= threshold and (c[3] or c[2] >= 0.72)]
            valid.sort(key=lambda c: c[1], reverse=True)
            # picked: 经 NMS 去重后保留的分割线三元组列表 [(pos, score, is_gutter), ...]
            picked: List[Tuple[int, float, bool]] = []
            for pos, sc, _, is_g in valid:
                if all(abs(pos - p) >= min_dim for p, _, _ in picked):
                    picked.append((pos, sc, is_g))
            picked.sort(key=lambda c: c[0])
            return picked

        # 第一步：递归 XY-Cut BSP 主体分割，收集所有叶子子区域
        # leaves: BSP 递归分割后的叶子区域列表，元素契约为 (x0, y0, x1, y1, conf, is_gutter)
        leaves: List[Tuple[int, int, int, int, float, bool]] = []

        def recurse(x0: int, y0: int, x1: int, y1: int, depth: int, conf: float, is_g: bool) -> None:
            """
            递归评估子区域的水平与垂直主分割轴并向下一级剖分。

            :param x0: 当前区域左上角 X 坐标。
            :type x0: int
            :param y0: 当前区域左上角 Y 坐标。
            :type y0: int
            :param x1: 当前区域右下角 X 坐标。
            :type x1: int
            :param y1: 当前区域右下角 Y 坐标。
            :type y1: int
            :param depth: 当前递归深度（上限为 4）。
            :type depth: int
            :param conf: 累计父级边界置信度得分。
            :type conf: float
            :param is_g: 是否经由纯色/透明沟壑分割产生。
            :type is_g: bool
            :return: 无返回值；到达叶子条件时将区域写入外层 ``leaves`` 列表。
            :author: tongliga
            :date: 2026-11-18
            """
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

        # 第二步：同级网格线传播（±12px 窗口），挽救跨白底卡片之间的低对比度微弱边界
        for _ in range(3):
            propagated = False
            # h_lines / v_lines: 从现有叶子坐标集合中汇总出的全域内部水平/垂直参考线集合
            h_lines = sorted({y for leaf_box in leaves for y in (leaf_box[1], leaf_box[3]) if min_dim < y < h - min_dim})
            v_lines = sorted({x for leaf_box in leaves for x in (leaf_box[0], leaf_box[2]) if min_dim < x < w - min_dim})

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

        # 第三步：当背景高度统一时，过滤掉前景占比不足 3% 的纯空白叶子废图
        if bg_uniform:
            # kept: 经前景密度检验后保留下来的非空子图元组列表
            kept: List[Tuple[int, int, int, int, float, bool]] = []
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

        # 第四步：逐叶子执行 ±8px 局部边界吸附、外缘纯色收缩与抗锯齿去杂边（bleed_trim）
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
                        r0, r1, r2 = (
                            arr[y - 1, x0 + 4 : x1 - 4 : sx],
                            arr[y, x0 + 4 : x1 - 4 : sx],
                            arr[y + 1, x0 + 4 : x1 - 4 : sx],
                        )
                        e = float((np.abs(r1 - r2).sum(axis=1) + 0.5 * np.abs(2 * r1 - r0 - r2).sum(axis=1)).mean())
                        if e > be:
                            be, by = e, y
                    if be > 15.0:
                        y0 = by + 1
                if y1 < h - 2 and x1 - x0 > 8:
                    sx = max(1, (x1 - x0) // 120)
                    by, be = y1, -1.0
                    for y in range(max(y0 + 20, y1 - 8), min(h - 2, y1 + 8) + 1):
                        r0, r1, r2 = (
                            arr[y - 1, x0 + 4 : x1 - 4 : sx],
                            arr[y, x0 + 4 : x1 - 4 : sx],
                            arr[y + 1, x0 + 4 : x1 - 4 : sx],
                        )
                        e = float((np.abs(r0 - r1).sum(axis=1) + 0.5 * np.abs(2 * r1 - r0 - r2).sum(axis=1)).mean())
                        if e > be:
                            be, by = e, y
                    if be > 15.0:
                        y1 = by
                if x0 > 2 and y1 - y0 > 8:
                    sy = max(1, (y1 - y0) // 120)
                    bx, be = x0, -1.0
                    for x in range(max(1, x0 - 8), min(x1 - 20, x0 + 8) + 1):
                        c0, c1, c2 = (
                            arr[y0 + 4 : y1 - 4 : sy, x - 1],
                            arr[y0 + 4 : y1 - 4 : sy, x],
                            arr[y0 + 4 : y1 - 4 : sy, x + 1],
                        )
                        e = float((np.abs(c1 - c2).sum(axis=1) + 0.5 * np.abs(2 * c1 - c0 - c2).sum(axis=1)).mean())
                        if e > be:
                            be, bx = e, x
                    if be > 15.0:
                        x0 = bx + 1
                if x1 < w - 2 and y1 - y0 > 8:
                    sy = max(1, (y1 - y0) // 120)
                    bx, be = x1, -1.0
                    for x in range(max(x0 + 20, x1 - 8), min(w - 2, x1 + 8) + 1):
                        c0, c1, c2 = (
                            arr[y0 + 4 : y1 - 4 : sy, x - 1],
                            arr[y0 + 4 : y1 - 4 : sy, x],
                            arr[y0 + 4 : y1 - 4 : sy, x + 1],
                        )
                        e = float((np.abs(c0 - c1).sum(axis=1) + 0.5 * np.abs(2 * c1 - c0 - c2).sum(axis=1)).mean())
                        if e > be:
                            be, bx = e, x
                    if be > 15.0:
                        x1 = bx

            tt = tr = tb = tl = 0
            if bg_uniform:

                def is_row_bg(y_val: int) -> bool:
                    """
                    判定当前行在区域横向跨度内是否被背景色（或透明像素）覆盖 94% 以上。

                    :param y_val: 待检测行 Y 坐标。
                    :type y_val: int
                    :return: 为纯背景行返回 ``True``，否则返回 ``False``。
                    :author: tongliga
                    :date: 2026-11-18
                    """
                    sx = max(1, (x1 - x0) // 60)
                    if is_trans and alpha is not None:
                        return float((alpha[y_val, x0:x1:sx] < 24).mean()) > 0.94
                    return float((np.abs(arr[y_val, x0:x1:sx] - bg_rgb).sum(axis=1) < 22.0).mean()) > 0.94

                def is_col_bg(x_val: int) -> bool:
                    """
                    判定当前列在区域纵向跨度内是否被背景色（或透明像素）覆盖 94% 以上。

                    :param x_val: 待检测列 X 坐标。
                    :type x_val: int
                    :return: 为纯背景列返回 ``True``，否则返回 ``False``。
                    :author: tongliga
                    :date: 2026-11-18
                    """
                    sy = max(1, (y1 - y0) // 60)
                    if is_trans and alpha is not None:
                        return float((alpha[y0:y1:sy, x_val] < 24).mean()) > 0.94
                    return float((np.abs(arr[y0:y1:sy, x_val] - bg_rgb).sum(axis=1) < 22.0).mean()) > 0.94

                while y0 < y1 - 24 and is_row_bg(y0):
                    y0 += 1
                    tt += 1
                while y1 > y0 + 24 and is_row_bg(y1 - 1):
                    y1 -= 1
                    tb += 1
                while x0 < x1 - 24 and is_col_bg(x0):
                    x0 += 1
                    tl += 1
                while x1 > x0 + 24 and is_col_bg(x1 - 1):
                    x1 -= 1
                    tr += 1

            if self.bleed_trim > 0 and y1 - y0 > 32 and x1 - x0 > 32:
                sx = max(1, (x1 - x0) // 80)
                sy = max(1, (y1 - y0) // 80)
                if ry0 > 2:
                    ref_y = min(y1 - 1, y0 + self.bleed_trim + 2)
                    for _ in range(self.bleed_trim):
                        if (
                            float(np.abs(arr[y0, x0 + 8 : x1 - 8 : sx] - arr[ref_y, x0 + 8 : x1 - 8 : sx]).sum(axis=1).mean())
                            > 20.0
                        ):
                            y0 += 1
                            tt += 1
                        else:
                            break
                if ry1 < h - 2:
                    ref_y = max(y0, y1 - 1 - self.bleed_trim - 2)
                    for _ in range(self.bleed_trim):
                        if (
                            float(
                                np.abs(arr[y1 - 1, x0 + 8 : x1 - 8 : sx] - arr[ref_y, x0 + 8 : x1 - 8 : sx])
                                .sum(axis=1)
                                .mean()
                            )
                            > 20.0
                        ):
                            y1 -= 1
                            tb += 1
                        else:
                            break
                if rx0 > 2:
                    ref_x = min(x1 - 1, x0 + self.bleed_trim + 2)
                    for _ in range(self.bleed_trim):
                        if (
                            float(np.abs(arr[y0 + 8 : y1 - 8 : sy, x0] - arr[y0 + 8 : y1 - 8 : sy, ref_x]).sum(axis=1).mean())
                            > 20.0
                        ):
                            x0 += 1
                            tl += 1
                        else:
                            break
                if rx1 < w - 2:
                    ref_x = max(x0, x1 - 1 - self.bleed_trim - 2)
                    for _ in range(self.bleed_trim):
                        if (
                            float(
                                np.abs(arr[y0 + 8 : y1 - 8 : sy, x1 - 1] - arr[y0 + 8 : y1 - 8 : sy, ref_x])
                                .sum(axis=1)
                                .mean()
                            )
                            > 20.0
                        ):
                            x1 -= 1
                            tr += 1
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
