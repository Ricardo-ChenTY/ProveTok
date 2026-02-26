from __future__ import annotations

import subprocess
import sys


def _run_help_module(module: str) -> str:
    p = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    return (p.stdout + p.stderr).lower()


def _run_help_script(path: str) -> str:
    p = subprocess.run(
        [sys.executable, path, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    return (p.stdout + p.stderr).lower()


def test_train_split_policy_dpo_help() -> None:
    out = _run_help_module("provetok.experiments.train_split_policy_dpo")
    assert "dpo" in out


def test_figx_split_dpo_closed_loop_help() -> None:
    out = _run_help_module("provetok.experiments.figX_split_dpo_closed_loop")
    assert "split" in out


def test_compute_paper_metrics_help() -> None:
    _run_help_script("scripts/paper/compute_paper_metrics.py")


def test_compute_radeval_metrics_jsonl_help() -> None:
    out = _run_help_script("scripts/external/compute_radeval_metrics_jsonl.py")
    assert "radeval" in out


def test_to_pred_jsonl_help() -> None:
    out = _run_help_script("scripts/external/to_pred_jsonl.py")
    assert "pred_jsonl" in out


def test_eval_external_predictions_help() -> None:
    out = _run_help_module("provetok.experiments.eval_external_predictions")
    assert "external" in out


def test_fig4_agent_pareto_multiseed_help() -> None:
    out = _run_help_module("provetok.experiments.fig4_agent_pareto_multiseed")
    assert "pareto" in out


def test_preprocess_manifest_rrg_dpo_help() -> None:
    out = _run_help_script("scripts/data/preprocess_manifest_rrg_dpo.py")
    assert "rrg" in out


def test_build_ct_rate_manifest_help() -> None:
    out = _run_help_script("scripts/data/build_ct_rate_manifest.py")
    assert "ct-rate" in out or "manifest" in out


def test_validate_manifest_help() -> None:
    out = _run_help_script("scripts/data/validate_manifest.py")
    assert "manifest" in out


def test_fill_manifest_fields_help() -> None:
    out = _run_help_script("scripts/data/fill_manifest_fields.py")
    assert "manifest" in out and "split" in out
