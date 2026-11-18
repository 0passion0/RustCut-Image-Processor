# Contributing to smart-image-splitter / 参与贡献指南

[English](#english) | [中文](#中文)

---

<a id="english"></a>
## English

Thank you for contributing to **smart-image-splitter**! Both algorithm improvements and Rust/Python performance optimizations are warmly welcomed.

### 1. Development Environment Setup
- Rust stable (`rustup`), Python >= 3.9
- `pip install setuptools wheel setuptools-rust maturin pytest numpy pillow`
- Compile release Rust extension: `maturin develop --release` (or `pip install --no-build-isolation .`)
- Run unit tests & lints: `pytest -v tests/` and `cargo clippy --all-targets -- -D warnings`

### 2. Core Implementation Rule
- Detection (`detect_auto_rs`), uniform dimension alignment (`pad` / `resize` / `crop`), and parallel image encoding (`batch_crop_and_save_rs`, `batch_crop_to_memory_rs`) all live inside `src/lib.rs`, with Python bindings exposed through `python/smart_splitter/engine.py`.
- Any algorithm or parameter change should include corresponding pixel-level assertions in `tests/test_splitter.py`.

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
- `pip install setuptools wheel setuptools-rust maturin pytest numpy pillow`
- 编译 Release 原生 Rust 扩展：`maturin develop --release`（或 `pip install --no-build-isolation .`）
- 运行单元测试与静态检查：`pytest -v tests/` 及 `cargo clippy --all-targets -- -D warnings`

### 2. 核心实现规范
- 全部检测算法（`detect_auto_rs`）、统一长宽规范化（`pad` / `resize` / `crop`）与并发裁剪编码（`batch_crop_and_save_rs`、`batch_crop_to_memory_rs`）均由 `src/lib.rs` 实现，并通过 `python/smart_splitter/engine.py` 暴露接口。
- 任何算法或参数改动须在 `tests/test_splitter.py` 中补充对应的像素级断言测试。

### 3. Git 分支与提交规范
- 分支命名：`feat/...`、`fix/...`、`perf/...`、`docs/...`、`refactor/...`
- 提交信息格式（Conventional Commits）：
  - `feat(rust): ...`
  - `perf(core): ...`
  - `fix(bleed): ...`
  - `docs(readme): ...`
