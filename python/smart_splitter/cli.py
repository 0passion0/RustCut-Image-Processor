"""CLI entrypoint for SmartSplitter (`smart-split`)."""
from __future__ import annotations

import argparse
from pathlib import Path
from .engine import HAS_RUST_CORE, SmartSplitter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smart-split",
        description="High-performance Rust + OpenCV image splitter for multi-panel images.",
    )
    parser.add_argument("input", type=Path, help="Input image file or directory of images")
    parser.add_argument("-o", "--output", type=Path, default=Path("./output_slices"), help="Output directory")
    parser.add_argument("--mode", choices=["auto", "xycut", "contour", "grid"], default="auto")
    parser.add_argument("--sensitivity", type=float, default=0.65, help="Detection sensitivity (0.1 - 1.0)")
    parser.add_argument("--bleed-trim", type=int, default=2, help="Max seam anti-alias trim in px")
    parser.add_argument("--rows", type=int, default=3, help="Rows for uniform grid mode")
    parser.add_argument("--cols", type=int, default=3, help="Columns for uniform grid mode")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "webp"], help="Output format")
    args = parser.parse_args()

    splitter = SmartSplitter(
        mode=args.mode,
        sensitivity=args.sensitivity,
        bleed_trim=args.bleed_trim,
    )

    inputs = (
        sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if args.input.is_dir()
        else [args.input]
    )

    print(f"SmartSplitter v0.2.1 | Rust PyO3 Core: {'ACTIVE' if HAS_RUST_CORE else 'OpenCV Fallback'}")
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
