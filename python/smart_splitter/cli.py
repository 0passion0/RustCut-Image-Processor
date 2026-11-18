"""
SmartSplitter 命令行工具入口模块（``smart-split``）。

支持针对单张复合图像或整个图像目录执行自动布局检测（XY-Cut BSP / 轮廓提取 / 均匀网格）、
抗锯齿去杂边裁切，以及可选的强制统一输出长宽（背景补边 / 缩放 / 原图同框裁剪）。

:author: tongliga
:date: 2026-11-18
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple
from .engine import HAS_RUST_CORE, SmartSplitter


def _parse_target_size(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    解析命令行 ``--target-size`` 参数字符串为 ``(width, height)`` 整数二元组。

    :param spec: 形如 ``"512x512"`` 或 ``"800x600"`` 的尺寸字符串；未指定时为 ``None``。
    :type spec: Optional[str]
    :return: 解析后的 ``(width, height)`` 元组，未传入时返回 ``None``。
    :raises ValueError: 当尺寸字符串格式不符合 ``WIDTHxHEIGHT`` 或数值非正整数时抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    if spec is None:
        return None
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid --target-size '{spec}', expected WIDTHxHEIGHT such as 512x512")
    w = int(parts[0].strip())
    h = int(parts[1].strip())
    if w <= 0 or h <= 0:
        raise ValueError(f"--target-size dimensions must be positive, got {w}x{h}")
    return (w, h)


def main() -> None:
    """
    解析命令行参数并执行单文件或目录级批量图像智能切割任务。

    :return: 无返回值；直接将切片输出到磁盘并在标准输出打印执行统计摘要。
    :raises FileNotFoundError: 当指定的输入路径不存在或图像无法解码读取时由 ``split_file`` 抛出。
    :raises ValueError: 当输入图像通道维度不合法、目标尺寸格式错误或输出格式参数异常时抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：定义并解析 CLI 参数（包含输入路径、分割模式、灵敏度、去杂边像素、统一长宽配置及网格行列数）
    parser = argparse.ArgumentParser(
        prog="smart-split",
        description="基于 Rust (PyO3 + Rayon) 与 OpenCV 的高精度复合图像智能切割命令行工具。",
    )
    parser.add_argument("input", type=Path, help="待切割的单张图像文件路径或包含图像的目录路径")
    parser.add_argument("-o", "--output", type=Path, default=Path("./output_slices"), help="切片输出目录")
    parser.add_argument(
        "--mode",
        choices=["auto", "xycut", "contour", "grid"],
        default="auto",
        help="布局检测模式（默认 auto 自动在 xycut 与 contour 间择优）",
    )
    parser.add_argument("--sensitivity", type=float, default=0.65, help="接缝与前景检测灵敏度（0.1 - 1.0）")
    parser.add_argument("--bleed-trim", type=int, default=2, help="内部接缝抗锯齿去杂边最大裁剪像素（px）")
    parser.add_argument("--rows", type=int, default=3, help="grid 均匀网格模式下的划分行数")
    parser.add_argument("--cols", type=int, default=3, help="grid 均匀网格模式下的划分列数")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "webp"], help="导出切片图像格式")
    parser.add_argument(
        "--uniform-size",
        action="store_true",
        help="强制将所有输出切片统一为相同宽高（默认对齐到全部切片的最大宽高）",
    )
    parser.add_argument(
        "--target-size",
        default=None,
        help="显式指定所有输出切片的统一尺寸（格式 WIDTHxHEIGHT，如 512x512），传入后自动启用尺寸统一",
    )
    parser.add_argument(
        "--uniform-mode",
        default="pad",
        choices=["pad", "resize", "crop"],
        help="统一长宽处理方式：pad（居中填充背景）、resize（缩放）或 crop（原图同框裁剪）",
    )
    args = parser.parse_args()
    parsed_target_size = _parse_target_size(args.target_size)

    # 第二步：初始化复用的 SmartSplitter 切割引擎实例
    splitter = SmartSplitter(
        mode=args.mode,
        sensitivity=args.sensitivity,
        bleed_trim=args.bleed_trim,
        uniform_size=args.uniform_size,
        target_size=parsed_target_size,
        uniform_mode=args.uniform_mode,
    )

    # 第三步：收集待处理的图像文件列表（支持目录批处理或单文件直传）
    # inputs: 待遍历切割的输入图像 Path 列表（list[Path]），按文件名排序以保证批处理顺序稳定
    inputs = (
        sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if args.input.is_dir()
        else [args.input]
    )

    # 第四步：逐文件执行切割并在控制台打印当前加速后端与耗时摘要
    print(f"SmartSplitter v0.2.3 | Rust PyO3 Core: {'ACTIVE' if HAS_RUST_CORE else 'INACTIVE'}")
    for img_path in inputs:
        sub_out = args.output / img_path.stem if args.input.is_dir() else args.output
        result, saved = splitter.split_file(
            img_path, sub_out, prefix=img_path.stem, fmt=args.format, rows=args.rows, cols=args.cols
        )
        print(
            f"  [OK] {img_path.name} ({result.image_width}x{result.image_height}) -> "
            f"{len(saved)} slices [{result.detected_mode} / {result.backend}] in {result.elapsed_ms:.1f}ms"
        )


if __name__ == "__main__":
    main()
