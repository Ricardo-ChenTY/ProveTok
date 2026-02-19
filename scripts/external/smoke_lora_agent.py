from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any, Dict

import torch

from provetok.agent.loop import AgentConfig, run_provetok_agent
from provetok.data import ManifestDataset
from provetok.eval.metrics_proof import compute_proof_metrics
from provetok.pcg.llama2_pcg import Llama2PCG, Llama2PCGConfig
from provetok.verifier.pp_v1_1 import create_pp_verifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke: load LoRA adapter and run 1-step ProveTok-Agent.")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--sample-id", type=str, default="", help="Optional scan_hash to select; default uses first record")

    ap.add_argument("--model-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--lora-adapter", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--quantization", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--max-tokens-in-prompt", type=int, default=16)
    ap.add_argument("--contract-mode", type=str, default="schema_only", choices=["free_form", "schema_only", "schema_citations", "full"])
    ap.add_argument("--max-frames", type=int, default=1)

    ap.add_argument("--budget-tokens", type=int, default=64)
    ap.add_argument("--init-level", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-steps-per-finding", type=int, default=1)
    ap.add_argument("--l-min", type=int, default=2)
    ap.add_argument("--k-max-citations", type=int, default=8)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default="outputs/E0225-lora_agent_smoke")

    args = ap.parse_args()

    ds = ManifestDataset(
        manifest_path=str(args.manifest),
        split=str(args.split),
        resize_shape=tuple(int(x) for x in args.resize_shape),
        seed=int(args.seed),
    )
    if len(ds) <= 0:
        raise SystemExit("Empty dataset")

    idx = 0
    if str(args.sample_id).strip():
        sid = str(args.sample_id).strip()
        found = None
        for i, r in enumerate(ds.records):
            if str(r.scan_hash) == sid:
                found = int(i)
                break
        if found is None:
            raise SystemExit(f"sample_id not found: {sid}")
        idx = int(found)

    row = ds[int(idx)]
    vol = row["volume"]
    affine_zyx = row.get("affine_zyx", None)

    pcg_cfg = Llama2PCGConfig(
        model_path=str(args.model_path),
        device=str(args.device),
        dtype=str(args.dtype),
        quantization=str(args.quantization),
        max_new_tokens=int(args.max_new_tokens),
        temperature=0.0,
        top_p=0.95,
        max_tokens_in_prompt=int(args.max_tokens_in_prompt),
        max_frames=int(args.max_frames),
        contract_mode=str(args.contract_mode),
        lora_adapter_path=str(args.lora_adapter),
        lora_merge=False,
    )

    verifier = create_pp_verifier(l_min=int(args.l_min))
    pcg = Llama2PCG(pcg_cfg)

    agent_cfg = AgentConfig(
        budget_tokens=int(args.budget_tokens),
        init_level=int(args.init_level),
        max_depth=int(args.max_depth),
        max_steps_per_finding=int(args.max_steps_per_finding),
        l_min=int(args.l_min),
        k_max_citations=int(args.k_max_citations),
        emb_dim=32,
    )

    res = run_provetok_agent(
        vol,
        generator_fn=pcg,
        verifier=verifier,
        cfg=agent_cfg,
        seed=int(args.seed),
        affine_zyx=affine_zyx,
        split_cell_fn=None,
    )

    m = compute_proof_metrics(res.generation, res.tokens, verifier=verifier, l_min=int(args.l_min))

    out_dir = Path(str(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lora_agent_smoke.json"

    out: Dict[str, Any] = {
        "sample_id": str(row.get("sample_id", "")),
        "pcg_cfg": asdict(pcg_cfg),
        "agent_cfg": asdict(agent_cfg),
        "proof": m,
        "findings_lines": list(res.findings_lines),
        "impression": str(res.impression),
        "issues": [
            {
                "frame_idx": int(getattr(x, "frame_idx", -1)),
                "rule_id": str(getattr(x, "rule_id", "")),
                "issue_type": str(getattr(x, "issue_type", "")),
                "severity": int(getattr(x, "severity", 0)),
                "message": str(getattr(x, "message", "")),
                "evidence_trace": getattr(x, "evidence_trace", {}) or {},
            }
            for x in (res.issues or [])
        ],
    }

    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(str(out_path))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
