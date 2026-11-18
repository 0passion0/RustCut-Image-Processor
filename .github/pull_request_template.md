## Summary / 变更摘要
<!-- Describe what this PR changes and why. / 请简述本次变更的内容与动机。 -->

## Algorithm or Performance Impact / 算法与性能影响
- [ ] Tested on multi-panel / asymmetric grids (XY-Cut + Sibling Propagation) / 已在多宫格或非对称布局样本验证
- [ ] Tested on scattered contours / alpha sprites / 已在轮廓分割或透明通道样本验证
- [ ] Verified outer-edge pixel bleed decontamination / 已检查输出子图边缘无杂边串色

## Verification Checklist / 测试检查清单
- [ ] `cargo check --all-targets`
- [ ] `pip install --no-build-isolation . && pytest -v tests/`
- [ ] `maturin develop --release && pytest -v tests/`
