# Contributing to SliceCraft / 参与贡献指南

[English](#english) | [中文](#中文)

---

<a id="english"></a>
## English

Thank you for contributing to **SliceCraft**! Both algorithm improvements and performance optimizations are warmly welcomed.

### 1. Development Environment Setup
- **Web Studio & TypeScript Engine**:
  - Node.js >= 20
  - `npm install`
  - `npm run dev` (starts Express + Vite on `http://localhost:3000`)
  - Run unit tests: `npx tsx --test src/utils/splitterCore.test.ts`
  - Type check & build: `npm run lint && npm run build`
- **Rust (PyO3 + Rayon) & Python Engine**:
  - Rust stable (`rustup`), Python >= 3.9
  - `cd python_rust_splitter`
  - `pip install maturin pytest opencv-python-headless numpy pillow`
  - `maturin develop --release`
  - `pytest -v tests/`

### 2. Algorithm Parity Rule
To keep the interactive Web Studio (`src/utils/splitterCore.ts`) and the native Rust/Python engine (`python_rust_splitter/src/lib.rs`, `python/smart_splitter/engine.py`) consistent:
- Any change to seam scoring weights, threshold formulas, sibling propagation tolerances, contour refinement, or bleed decontamination **must be mirrored across both implementations** and covered by unit tests in both `src/utils/splitterCore.test.ts` and `python_rust_splitter/tests/test_splitter.py`.

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

感谢参与 **SliceCraft** 的研发与优化！欢迎提交算法改进、极致性能优化或边界案例测试。

### 1. 本地开发环境配置
- **Web 工作台与 TypeScript 引擎**：
  - Node.js >= 20
  - `npm install`
  - `npm run dev`（在 `http://localhost:3000` 启动 Express + Vite）
  - 运行单元测试：`npx tsx --test src/utils/splitterCore.test.ts`
  - 类型检查与构建：`npm run lint && npm run build`
- **Rust (PyO3 + Rayon) 与 Python 引擎**：
  - Rust stable (`rustup`)、Python >= 3.9
  - `cd python_rust_splitter`
  - `pip install maturin pytest opencv-python-headless numpy pillow`
  - `maturin develop --release`
  - `pytest -v tests/`

### 2. 双端算法一致性规范
为保证在线 Web 工作台（`src/utils/splitterCore.ts`）与离线 Rust/Python 引擎（`python_rust_splitter/src/lib.rs`、`python/smart_splitter/engine.py`）效果完全一致：
- 任何对缝隙评分权重、阈值公式、同级网格线传播容差、轮廓 1:1 边缘收紧或防串色修边逻辑的修改，**须同步更新两端代码**，并在 `src/utils/splitterCore.test.ts` 与 `python_rust_splitter/tests/test_splitter.py` 中补充对应像素级断言测试。

### 3. Git 分支与提交规范
- 分支命名：`feat/...`、`fix/...`、`perf/...`、`docs/...`、`refactor/...`
- 提交信息格式（Conventional Commits）：
  - `feat(rust): ...`
  - `perf(core): ...`
  - `fix(bleed): ...`
  - `docs(readme): ...`
