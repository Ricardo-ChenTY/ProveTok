#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StageSpec:
    name: str
    manifest_key: str
    n_samples: Optional[int]


STAGES: Dict[str, StageSpec] = {
    "stageA": StageSpec(name="stageA", manifest_key="rex_mini", n_samples=20),
    "stageB": StageSpec(name="stageB", manifest_key="rex_mini", n_samples=57),
    "stageC": StageSpec(name="stageC", manifest_key="rex_100g", n_samples=100),
    "stageD": StageSpec(name="stageD", manifest_key="rex_100g", n_samples=100000000),
}


def _stage_order(start_from: str) -> List[StageSpec]:
    order = ["stageA", "stageB", "stageC", "stageD"]
    if start_from not in order:
        raise ValueError(f"Invalid start_from={start_from}")
    idx = order.index(start_from)
    return [STAGES[k] for k in order[idx:]]


def _resolve_manifest(spec: StageSpec, *, rex_mini_manifest: str, rex_100g_manifest: str) -> str:
    if spec.manifest_key == "rex_mini":
        return _expanduser_only(rex_mini_manifest)
    if spec.manifest_key == "rex_100g":
        return _expanduser_only(rex_100g_manifest)
    raise ValueError(f"Unknown manifest key: {spec.manifest_key}")


def _default_from_env(name: str, default: str) -> str:
    v = str(os.getenv(name, "")).strip()
    return v if v else default


def _expanduser_only(path: str) -> str:
    p = str(path or "").strip()
    if p.startswith("~"):
        return str(Path(p).expanduser())
    return p


def _write_dry_run_report(stage_dir: Path, spec: StageSpec, cmd: List[str]) -> None:
    report = {
        "stage_name": spec.name,
        "dry_run": True,
        "overall_pass": True,
        "command": cmd,
        "note": "Dry-run only. No training or inference executed.",
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage_check_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = "\n".join(
        [
            f"# Stage Check Report: {spec.name}",
            "",
            "- overall_pass: `true`",
            "- dry_run: `true`",
            "",
            "## Planned Command",
            "",
            "```bash",
            shlex.join(cmd),
            "```",
            "",
        ]
    )
    (stage_dir / "stage_check_report.md").write_text(md, encoding="utf-8")


def _run_cmd(cmd: List[str], *, cwd: Path, dry_run: bool) -> int:
    print(f"[CMD] {shlex.join(cmd)}")
    if dry_run:
        return 0
    return int(subprocess.call(cmd, cwd=str(cwd)))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser(description="Run staged ReX experiments with llama3-only interface and gate checks.")
    ap.add_argument("--start-from", type=str, default="stageA", choices=["stageA", "stageB", "stageC", "stageD"])
    ap.add_argument("--only-stage", type=str, default="", choices=["", "stageA", "stageB", "stageC", "stageD"])
    ap.add_argument("--python-bin", type=str, default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute.")
    ap.add_argument("--output-root", type=str, default="outputs/rex_llama3_staged")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    ap.add_argument("--llama3-path", type=str, default=_default_from_env("LLAMA3_PATH", "~/models/llama3"))
    ap.add_argument("--llama3-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument("--llama3-contract-mode", type=str, default="full", choices=["free_form", "schema_only", "schema_citations", "full", "inline_citation"])
    ap.add_argument("--llama3-citation-source", type=str, default="score_override", choices=["score_override", "llm"])
    ap.add_argument("--llama3-max-frames", type=int, default=1)
    ap.add_argument("--llama3-lora-adapter", type=str, default="")
    ap.add_argument("--llama3-lora-merge", action="store_true")

    ap.add_argument("--rex-mini-manifest", type=str, default=_default_from_env("REX_MINI_MANIFEST", "/data/provetok_datasets/rexgroundingct_mini/manifest.jsonl"))
    ap.add_argument("--rex-100g-manifest", type=str, default=_default_from_env("REX_100G_MANIFEST", "/data/provetok_datasets/rexgroundingct_100g/manifest.jsonl"))
    ap.add_argument("--methods", type=str, nargs="+", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk-citations", type=int, default=8)
    ap.add_argument("--extra-args", type=str, default="", help="Extra args forwarded to run_baselines.")
    ap.add_argument("--no-stop-on-fail", action="store_false", dest="stop_on_fail")
    ap.set_defaults(stop_on_fail=True)

    ap.add_argument("--max-parse-failure-rate", type=float, default=0.20)
    ap.add_argument("--min-citation-nonempty-rate", type=float, default=0.80)
    ap.add_argument("--max-abnormal-output-rate", type=float, default=0.75)
    args = ap.parse_args()

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    llama3_path = _expanduser_only(str(args.llama3_path))
    if str(args.only_stage).strip():
        stages = [STAGES[str(args.only_stage)]]
    else:
        stages = _stage_order(str(args.start_from))

    plan = {
        "repo_root": str(repo_root),
        "dry_run": bool(args.dry_run),
        "llama3_path": llama3_path,
        "stages": [{"name": s.name, "manifest_key": s.manifest_key, "n_samples": s.n_samples} for s in stages],
    }
    (output_root / "stage_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    for spec in stages:
        stage_dir = output_root / spec.name
        manifest = _resolve_manifest(
            spec,
            rex_mini_manifest=str(args.rex_mini_manifest),
            rex_100g_manifest=str(args.rex_100g_manifest),
        )

        cmd: List[str] = [
            str(args.python_bin),
            "-m",
            "provetok.experiments.run_baselines",
            "--dataset-type",
            "manifest",
            "--split",
            str(args.split),
            "--manifest",
            manifest,
            "--pcg",
            "llama3",
            "--llama3-path",
            llama3_path,
            "--llama3-quant",
            str(args.llama3_quant),
            "--llama3-contract-mode",
            str(args.llama3_contract_mode),
            "--llama3-citation-source",
            str(args.llama3_citation_source),
            "--llama3-max-frames",
            str(int(args.llama3_max_frames)),
            "--seed",
            str(int(args.seed)),
            "--topk-citations",
            str(int(args.topk_citations)),
            "--output-dir",
            str(stage_dir),
        ]
        if str(args.llama3_lora_adapter).strip():
            cmd += ["--llama3-lora-adapter", str(args.llama3_lora_adapter)]
        if bool(args.llama3_lora_merge):
            cmd += ["--llama3-lora-merge"]
        if spec.n_samples is not None:
            cmd += ["--n-samples", str(int(spec.n_samples))]
        if args.methods:
            cmd += ["--methods"] + list(args.methods)
        if str(args.extra_args).strip():
            cmd += shlex.split(str(args.extra_args))

        print(f"[STAGE] {spec.name} manifest={manifest} n_samples={spec.n_samples}")
        rc = _run_cmd(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
        if rc != 0:
            print(f"[FAIL] {spec.name} run failed with rc={rc}")
            if bool(args.stop_on_fail):
                raise SystemExit(rc)
            continue

        if bool(args.dry_run):
            _write_dry_run_report(stage_dir, spec, cmd)
            continue

        check_cmd = [
            str(args.python_bin),
            str(repo_root / "scripts/ops/stage_check_report.py"),
            "--stage-name",
            spec.name,
            "--stage-dir",
            str(stage_dir),
            "--max-parse-failure-rate",
            str(float(args.max_parse_failure_rate)),
            "--min-citation-nonempty-rate",
            str(float(args.min_citation_nonempty_rate)),
            "--max-abnormal-output-rate",
            str(float(args.max_abnormal_output_rate)),
        ]
        rc_check = _run_cmd(check_cmd, cwd=repo_root, dry_run=False)
        if rc_check != 0:
            print(f"[FAIL] {spec.name} check gate failed with rc={rc_check}")
            if bool(args.stop_on_fail):
                raise SystemExit(rc_check)
            continue

        print(f"[PASS] {spec.name} gate passed.")

    print(f"[OK] finished. plan={output_root / 'stage_plan.json'}")


if __name__ == "__main__":
    main()
