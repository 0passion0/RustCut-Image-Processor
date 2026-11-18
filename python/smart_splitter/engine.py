"""
SmartSplitter 核心图像切割引擎模块（纯 Rust 原生加速引擎）。

所有布局检测（XY-Cut BSP / 轮廓提取 / 均匀网格）、统一长宽规范化（pad / resize / crop）
与并行裁切编码均直接交由原生 Rust 扩展 ``smart_splitter._core``（PyO3 + Rayon）执行。

:author: tongliga
:date: 2026-11-18
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

from . import _core as _rust_core

HAS_RUST_CORE = True

SplitMode = Literal["auto", "xycut", "contour", "grid"]
UniformMode = Literal["pad", "resize", "crop"]
_VALID_UNIFORM_MODES = ("pad", "resize", "crop")


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
    基于原生 Rust（PyO3 + Rayon）的复合图像智能切割引擎。

    所有图像布局分析、统一长宽对齐、多线程内存编码与并发落盘均由 Rust 内核直接执行并释放 GIL；
    亦支持注入可选机器学习分割回调函数（``ml_segmenter``）以接入 SAM 2 / YOLO / ONNX 候选框。

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
        uniform_size: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        uniform_mode: UniformMode = "pad",
    ) -> None:
        """
        初始化 ``SmartSplitter`` 切割参数与可选分割钩子。

        :param mode: 布局检测策略模式，可选值为 ``"auto"``、``"xycut"``、``"contour"`` 或 ``"grid"``。
        :type mode: Literal["auto", "xycut", "contour", "grid"]
        :param sensitivity: 边缘阶跃与前景阈值检测灵敏度（建议范围 ``0.1..=1.0``）。
        :type sensitivity: float
        :param min_region_ratio: 最小子图边长占图像短边的比例下限（建议范围 ``0.03..=0.35``）。
        :type min_region_ratio: float
        :param bleed_trim: 内部接缝抗锯齿去杂边的最大剥离像素数（px）。
        :type bleed_trim: int
        :param contour_gap_px: 轮廓模式形态学闭运算连接相邻离散元素的像素间隙半径（px）。
        :type contour_gap_px: int
        :param ml_segmenter: 可选的外部目标检测/分割回调函数，接收图像数组并返回 ``(x, y, w, h)`` 元组列表。
        :type ml_segmenter: Optional[Callable[[np.ndarray], List[Tuple[int, int, int, int]]]]
        :param uniform_size: 是否强制将所有切片统一输出为相同宽高（默认自动对齐全部切片的最大宽高）。
        :type uniform_size: bool
        :param target_size: 显式指定所有切片的固定目标尺寸 ``(width, height)``；传入后自动启用尺寸统一输出。
        :type target_size: Optional[Tuple[int, int]]
        :param uniform_mode: 强制相同长宽时的尺寸对齐策略：``"pad"``（居中填充背景，默认）、``"resize"``（缩放）或 ``"crop"``（原图等宽框裁剪）。
        :type uniform_mode: Literal["pad", "resize", "crop"]
        :raises ValueError: 当 ``uniform_mode`` 非法或 ``target_size`` 宽高非正整数时抛出。
        :author: tongliga
        :date: 2026-11-18
        """
        if uniform_mode not in _VALID_UNIFORM_MODES:
            raise ValueError(f"uniform_mode must be one of {_VALID_UNIFORM_MODES}, got {uniform_mode!r}")
        if target_size is not None and (target_size[0] <= 0 or target_size[1] <= 0):
            raise ValueError(f"target_size width and height must be positive, got {target_size!r}")

        self.mode = mode
        self.sensitivity = sensitivity
        self.min_region_ratio = min_region_ratio
        self.bleed_trim = bleed_trim
        self.contour_gap_px = contour_gap_px
        self.ml_segmenter = ml_segmenter
        self.uniform_size = uniform_size or (target_size is not None)
        self.target_size = target_size
        self.uniform_mode: UniformMode = uniform_mode

    def analyze(
        self,
        image_bgr_or_rgb: np.ndarray,
        rows: int = 3,
        cols: int = 3,
        gutter_px: int = 0,
    ) -> SplitResult:
        """
        调用原生 Rust 内核执行图像布局分析，返回所有子图区域的精确包围盒集合。

        :param image_bgr_or_rgb: 形状为 ``(H, W, 3)`` 或 ``(H, W, 4)`` 的 3 维 ``uint8`` 图像数组。
        :type image_bgr_or_rgb: numpy.ndarray
        :param rows: 当 ``mode == "grid"`` 时指定均匀切分的行数。
        :type rows: int
        :param cols: 当 ``mode == "grid"`` 时指定均匀切分的列数。
        :type cols: int
        :param gutter_px: 当 ``mode == "grid"`` 时指定行列网格间的固定留白间距（px）。
        :type gutter_px: int
        :return: 包含原图宽高、命中模式、后端名称（``"rust-rayon"`` / ``"ml-hook"``）、耗时（ms）及子图列表的 ``SplitResult``。
        :raises ValueError: 当输入数组非 3 维 ``HxWx3`` / ``HxWx4`` 或 ``uniform_mode``/``target_size`` 非法时由底层抛出。
        :author: tongliga
        :date: 2026-11-18
        """
        t0 = time.perf_counter()
        if image_bgr_or_rgb.ndim != 3:
            raise ValueError("Expected HxWx3 or HxWx4 image array.")
        h, w = image_bgr_or_rgb.shape[:2]

        # 第一步：若注入了外部 ML 分割回调且返回候选框，则直接封装 ML 结果
        if self.ml_segmenter is not None:
            rects = self.ml_segmenter(image_bgr_or_rgb)
            if rects:
                boxes = [
                    SplitBox(i + 1, i, 0, x, y, bw, bh, 0.99, "ml-segmentation")
                    for i, (x, y, bw, bh) in enumerate(rects)
                ]
                return SplitResult(w, h, "ml", "ml-hook", (time.perf_counter() - t0) * 1000, boxes)

        # 第二步：调度原生 Rust（PyO3 + Rayon）引擎执行检测与可选 crop 同框对齐（内部释放 GIL）
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
            self.uniform_size,
            self.target_size,
            self.uniform_mode,
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

    def _resolve_target_size(self, boxes: List[SplitBox]) -> Optional[Tuple[int, int]]:
        """
        计算传给 Rust 裁剪编码器的统一目标宽高（仅在 ``pad`` / ``resize`` 模式下生效）。

        :param boxes: 当前检出的候选切片包围盒列表。
        :type boxes: List[SplitBox]
        :return: 目标宽高 ``(target_w, target_h)``，无需像素级缩放或填充时返回 ``None``。
        :author: tongliga
        :date: 2026-11-18
        """
        if not self.uniform_size or self.uniform_mode == "crop" or not boxes:
            return None
        if self.target_size is not None:
            return self.target_size
        return (max(b.width for b in boxes), max(b.height for b in boxes))

    def split_to_bytes(
        self,
        image_bgr_or_rgb: np.ndarray,
        fmt: str = "png",
        rows: int = 3,
        cols: int = 3,
        gutter_px: int = 0,
    ) -> Tuple[SplitResult, List[bytes]]:
        """
        调用 Rust Rayon 多线程在内存中直接完成切片、统一长宽规范化与图像字节流编码。

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
        :raises ValueError: 当输入数组非 3 维或编码格式不被支持时由 Rust 核心抛出。
        :author: tongliga
        :date: 2026-11-18
        """
        # 第一步：执行 Rust 布局分析
        res = self.analyze(image_bgr_or_rgb, rows=rows, cols=cols, gutter_px=gutter_px)
        if not res.boxes:
            return res, []

        # 第二步：调用 Rust Rayon 内存并行裁剪、统一宽高并编码为字节流
        coords = [(b.x, b.y, b.width, b.height) for b in res.boxes]
        raw_list = _rust_core.batch_crop_to_memory_rs(
            np.ascontiguousarray(image_bgr_or_rgb),
            coords,
            fmt,
            self._resolve_target_size(res.boxes),
            self.uniform_mode,
        )
        return res, [bytes(item) for item in raw_list]

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
        读取磁盘输入图像并交由 Rust Rayon 内核完成布局检测、统一长宽处理与并发落盘保存。

        :param input_path: 输入图像的文件路径。
        :type input_path: str | pathlib.Path
        :param output_dir: 输出切片文件的目标目录路径（不存在时由 Rust 端自动创建）。
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
        :return: 二元组 ``(SplitResult, List[str])``，分别为分析结果摘要与输出切片文件路径列表。
        :raises FileNotFoundError: 当 ``input_path`` 不存在时抛出。
        :raises ValueError: 当图像解码、尺寸对齐或磁盘写出失败时由底层抛出。
        :author: tongliga
        :date: 2026-11-18
        """
        # 第一步：加载输入图像并转换为 3/4 通道 uint8 数组送入 Rust 检测引擎
        with Image.open(input_path) as pil_img:
            mode_img = pil_img.convert("RGBA") if "A" in pil_img.getbands() else pil_img.convert("RGB")
            img = np.asarray(mode_img, dtype=np.uint8)

        res = self.analyze(img, rows=rows, cols=cols, gutter_px=gutter_px)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not res.boxes:
            return res, []

        # 第二步：调度 Rust Rayon 并行裁剪、规范化宽高并编码落盘
        coords = [(b.x, b.y, b.width, b.height) for b in res.boxes]
        saved_paths = _rust_core.batch_crop_and_save_rs(
            str(input_path),
            coords,
            str(out_dir),
            prefix,
            fmt,
            self._resolve_target_size(res.boxes),
            self.uniform_mode,
        )
        return res, saved_paths
