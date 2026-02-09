from __future__ import annotations

import json
import sys
from pathlib import Path


def test_figx_llama2_contract_ablation_text_metrics_flag(tmp_path, monkeypatch) -> None:
    from provetok.experiments import figX_llama2_contract_ablation as mod

    def _fake_run_baselines(cfg):
        n = int(getattr(cfg, "n_samples", 3))
        methods = list(getattr(cfg, "methods", []) or ["provetok_lesionness"])
        method = str(methods[0])
        raw = {
            method: {
                "combined": [0.1] * n,
                "iou": [0.2] * n,
                "unsupported": [0.0] * n,
                "n_frames_pred_pos": [1.0] * n,
                "n_frames_with_citations": [1.0] * n,
            }
        }
        if bool(getattr(cfg, "compute_text_metrics", True)):
            raw[method].update(
                {
                    "bleu": [0.3] * n,
                    "rouge1": [0.4] * n,
                    "rouge2": [0.5] * n,
                    "rougeL": [0.6] * n,
                }
            )
        return {"raw": raw, "meta": {}, "config": {}, "budget_target": {}, "budgets": {}, "costs": {}, "summary": {}}

    monkeypatch.setattr(mod, "run_baselines", _fake_run_baselines)

    out_dir_with = tmp_path / "with_text"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dataset-type",
            "synthetic",
            "--manifest",
            "dummy",
            "--methods",
            "provetok_lesionness",
            "--budgets",
            "2",
            "--seeds",
            "0",
            "--contract-modes",
            "free_form",
            "full",
            "--n-samples",
            "3",
            "--n-bootstrap",
            "10",
            "--output-dir",
            str(out_dir_with),
        ],
    )
    mod.main()
    rep = json.loads(Path(out_dir_with / "figX_llama2_contract_ablation.json").read_text(encoding="utf-8"))
    assert rep["text_metrics_enabled"] is True
    assert "rougeL" in rep["metrics"]

    out_dir_no = tmp_path / "no_text"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dataset-type",
            "synthetic",
            "--manifest",
            "dummy",
            "--methods",
            "provetok_lesionness",
            "--budgets",
            "2",
            "--seeds",
            "0",
            "--contract-modes",
            "free_form",
            "full",
            "--n-samples",
            "3",
            "--n-bootstrap",
            "10",
            "--no-text-metrics",
            "--output-dir",
            str(out_dir_no),
        ],
    )
    mod.main()
    rep2 = json.loads(Path(out_dir_no / "figX_llama2_contract_ablation.json").read_text(encoding="utf-8"))
    assert rep2["text_metrics_enabled"] is False
    assert "rougeL" not in rep2["metrics"]

