from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path


def _load_module(module_path: Path):
    name = f"codex_dynamic_{module_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _minimal_curve_json(*, budgets: list[float], seeds: list[int], n_bootstrap: int, split: str, methods: list[str]) -> dict:
    return {
        "budgets": budgets,
        "seeds": seeds,
        "n_bootstrap": n_bootstrap,
        "methods": methods,
        "metrics": {},
        "meta": {"config": {"split": split}},
    }


def test_check_c0001_real_prefers_paper_grade_candidate_over_newer_single_budget(tmp_path: Path) -> None:
    proof_check = _load_module(Path("scripts/proof_check.py"))

    outputs = tmp_path / "outputs"
    # Newer, but non paper-grade (only one budget).
    fast = outputs / "E0164-b2e6-w4-fast" / "baselines_curve_multiseed.json"
    _write_json(
        fast,
        _minimal_curve_json(
            budgets=[2_000_000.0],
            seeds=[0, 1, 2, 3, 4],
            n_bootstrap=20_000,
            split="test",
            methods=["provetok_lesionness", "fixed_grid"],
        ),
    )
    # Older, but paper-grade.
    full_backup = outputs / "E0164-full_backup_20260206_144914" / "baselines_curve_multiseed.json"
    _write_json(
        full_backup,
        _minimal_curve_json(
            budgets=[2e6, 3e6, 4e6, 5e6, 6e6, 7e6],
            seeds=[0, 1, 2, 3, 4],
            n_bootstrap=20_000,
            split="test",
            methods=["provetok_lesionness", "fixed_grid"],
        ),
    )
    now = time.time()
    os.utime(full_backup, (now - 300, now - 300))
    os.utime(fast, (now, now))

    proof_check.ROOT = tmp_path
    proof_check.ACTIVE_PROFILE = "real"

    res = proof_check.check_c0001()
    assert res.details["baselines"] == str(full_backup)
    assert "need >=6 budgets" not in res.summary


def test_check_c0006_real_uses_paper_grade_curve_for_baseline_suite(tmp_path: Path) -> None:
    proof_check = _load_module(Path("scripts/proof_check.py"))

    outputs = tmp_path / "outputs"
    weights = outputs / "E0140-full" / "ct2rep_strong_locked.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"ok")

    fast = outputs / "E0164-b2e6-w4-fast" / "baselines_curve_multiseed.json"
    _write_json(
        fast,
        _minimal_curve_json(
            budgets=[2_000_000.0],
            seeds=[0, 1, 2, 3, 4],
            n_bootstrap=20_000,
            split="test",
            methods=["fixed_grid", "provetok_lesionness"],
        ),
    )

    full_backup = outputs / "E0164-full_backup_20260206_155111" / "baselines_curve_multiseed.json"
    _write_json(
        full_backup,
        {
            **_minimal_curve_json(
                budgets=[2e6, 3e6, 4e6, 5e6, 6e6, 7e6],
                seeds=[0, 1, 2, 3, 4],
                n_bootstrap=20_000,
                split="test",
                methods=[
                    "fixed_grid",
                    "slice_2d",
                    "slice_2p5d",
                    "roi_crop",
                    "roi_variance",
                    "ct2rep_strong",
                    "provetok_lesionness",
                ],
            ),
            "meta": {
                "config": {
                    "split": "test",
                    "costs_json": "outputs/compute_costs.json",
                    "ct2rep_strong_weights": "./outputs/E0140-full/ct2rep_strong_locked.pt",
                }
            },
            "budgets_by_method": {"fixed_grid": {"2000000.0": {"budget_tokens": 64}}},
            "metrics": {"frame_f1": {"ct2rep_strong": [{"mean": 0.7}]}},
        },
    )
    now = time.time()
    os.utime(full_backup, (now - 300, now - 300))
    os.utime(fast, (now, now))

    proof_check.ROOT = tmp_path
    proof_check.ACTIVE_PROFILE = "real"

    res = proof_check.check_c0006()
    assert res.proved is True
    assert res.details["path"] == str(full_backup)


def test_oral_audit_marks_real_c0001_non_paper_grade_as_invalid_artifact() -> None:
    oral_audit = _load_module(Path("scripts/oral_audit.py"))

    profile_report = {
        "profile": "real",
        "ok": True,
        "all_proved": False,
        "claims": {
            "C0001": {
                "proved": False,
                "summary": "not proved: need >=6 budgets for paper-grade C0001 (got 1).",
            }
        },
        "raw": {
            "checks": [
                {
                    "claim_id": "C0001",
                    "proved": False,
                    "summary": "not proved: need >=6 budgets for paper-grade C0001 (got 1).",
                    "details": {
                        "baselines": "outputs/E0164-b2e6-w4-fast/baselines_curve_multiseed.json",
                        "budgets": [2_000_000.0],
                    },
                }
            ]
        },
    }

    gaps, invalids = oral_audit._collect_claim_gaps(profile_report)
    assert len(gaps) == 1
    assert "artifact_invalid_for_gate" in gaps[0]
    assert len(invalids) == 1
    assert invalids[0]["profile"] == "real"
    assert invalids[0]["claim_id"] == "C0001"
    assert invalids[0]["reason"] == "insufficient_budgets"
