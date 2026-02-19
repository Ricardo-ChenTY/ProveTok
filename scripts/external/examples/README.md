# External baseline examples

这些文件仅用于演示 `scripts/external/to_pred_jsonl.py` 的输入形态。

它们的 `sample_id` 是虚构的，因此不能直接用于真实数据集评测。

想做真正的 smoke，请用：

```bash
python scripts/external/make_dummy_pred_jsonl_from_manifest.py --help
python -m provetok.experiments.eval_external_predictions --help
```
