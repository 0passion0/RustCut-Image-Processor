# Contributing to smart-image-splitter / 参与贡献指南

[English](#english) | [中文](#中文)

---

<a id="english"></a>
## English

Thank you for contributing to **smart-image-splitter**! Both algorithm improvements and Rust/Python performance optimizations are warmly welcomed.

### 1. Development Environment Setup
- Rust stable (`rustup`), Python >= 3.9
- `pip install setuptools wheel setuptools-rust maturin pytest opencv-python-headless numpy pillow`
- Compile release Rust extension: `maturin develop --release` (or `pip install --no-build-isolation .`)
- Run unit tests: `pytest -v tests/`

### 2. Algorithm Parity Rule
To ensure identical output whether the Rust PyO3 core is compiled or the vectorized OpenCV/NumPy fallback is used:
- Any change to seam scoring weights, threshold formulas, sibling propagation tolerances, contour refinement, or bleed decontamination **must stay synchronized between `src/lib.rs` and `python/smart_splitter/engine.py`** and be covered by unit tests in `tests/test_splitter.py`.

### 3. Git Commit & Branch Conventions
- Branch prefixes: `feat/...`, `fix/...`, `perf/...`, `docs/...`, `refactor/...`
- Conventional Commits:
  - `feat(rust): add ...`
  - `perf(core): optimize ...`
  - `fix(bleed): resolve ...`
  - `docs(readme): update ...`

---

<a id="中文"></a>
## 中文

感谢参与 **smart-image-splitter** 的研发与优化！欢迎提交算法改进、极致性能优化或边界案例测试。

### 1. 本地开发环境配置
- Rust stable (`rustup`)、Python >= 3.9
- `pip install setuptools wheel setuptools-rust maturin pytest opencv-python-headless numpy pillow`
- 编译 Release 原生 Rust 扩展：`maturin develop --release`（或 `pip install --no-build-isolation .`）
- 运行单元测试：`pytest -v tests/`

### 2. 双后端算法一致性规范
为保证编译 Rust PyO3 扩展与纯 Python/OpenCV 回退内核在任何环境下效果完全一致：
- 任何对缝隙评分权重、阈值公式、同级网格线传播容差、轮廓 1:1 边缘收紧或防串色修边逻辑的修改，**须同步更新 `src/lib.rs` 与 `python/smart_splitter/engine.py`**，并在 `tests/test_splitter.py` 中补充对应像素级断言测试。

### 3. Git 分支与提交规范
- 分支命名：`feat/...`、`fix/...`、`perf/...`、`docs/...`、`refactor/...`
- 提交信息格式（Conventional Commits）：
  - `feat(rust): ...`
  - `perf(core): ...`
  - `fix(bleed): ...`
  - `docs(readme): ...`
