"""
smart-image-splitter Python 包入口模块。

提供高精度纯 Rust（PyO3 + Rayon）复合图像智能切割接口，
导出核心分割引擎 ``SmartSplitter``、边界框数据结构 ``SplitBox`` 与结果摘要 ``SplitResult``。

:author: tongliga
:date: 2026-11-18
"""

from .engine import SmartSplitter, SplitBox, SplitResult

__all__ = ["SmartSplitter", "SplitBox", "SplitResult"]
__version__ = "0.2.4"
