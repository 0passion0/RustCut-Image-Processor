"""
SmartSplitter 命令行工具入口模块（``smart-split``）。

支持针对单张复合图像或整个图像目录执行自动布局检测（XY-Cut BSP / 轮廓提取 / 均匀网格）、
抗锯齿去杂边裁切，并将切片按指定格式保存至输出目录。

:author: tongliga
:date: 2026-11-18
"""

from __future__ import annotations

import argparse
from pathlib import Path
from .engine import HAS_RUST_CORE, SmartSplitter


def main() -> None:
    """
    解析命令行参数并执行单文件或目录级批量图像智能切割任务。

    :return: 无返回值；直接将切片输出到磁盘并在标准输出打印执行统计摘要。
    :raises FileNotFoundError: 当指定的输入路径不存在或图像无法解码读取时由 ``split_file`` 抛出。
    :raises ValueError: 当输入图像通道维度不合法或输出格式参数异常时由底层抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：定义并解析 CLI 参数（包含输入路径、分割模式、灵敏度、去杂边像素及网格行列数）
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
    args = parser.parse_args()

    # 第二步：初始化复用的 SmartSplitter 切割引擎实例
    splitter = SmartSplitter(
        mode=args.mode,
        sensitivity=args.sensitivity,
        bleed_trim=args.bleed_trim,
    )

    # 第三步：收集待处理的图像文件列表（支持目录批处理或单文件直传）
    # inputs: 待遍历切割的输入图像 Path 列表（list[Path]），按文件名排序以保证批处理顺序稳定
    inputs = (
        sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if args.input.is_dir()
        else [args.input]
    )

    # 第四步：逐文件执行切割并在控制台打印当前加速后端（Rust PyO3 或 OpenCV 回退）与耗时摘要
    print(f"SmartSplitter v0.2.2 | Rust PyO3 Core: {'ACTIVE' if HAS_RUST_CORE else 'OpenCV Fallback'}")
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
