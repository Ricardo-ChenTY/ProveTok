from __future__ import annotations
import json
import typer
from rich import print as rprint
from pathlib import Path
import torch
import numpy as np

from .data.io import load_volume
from .pcg import ToyPCG, Llama2PCG, Llama2PCGConfig
from .pcg.text_contract import enforce_pp_contract, render_findings, render_impression
from .verifier import verify
from .bet import run_refine_loop
from .agent import AgentConfig, run_provetok_agent
from .utils.artifact import build_artifact_meta
from .verifier.rules import RULE_SET_VERSION
from .verifier.taxonomy import TAXONOMY_VERSION
from .pcg.schema_version import SCHEMA_VERSION

app = typer.Typer(add_completion=False)

@app.command()
def main(
    loop: str = typer.Option("refine", help="Loop: refine | agent"),
    steps: int = typer.Option(3, help="Max refine iterations."),
    budget: int = typer.Option(64, help="Max tokens allowed (budget)."),
    seed: int = typer.Option(0, help="Random seed (deterministic demo)."),
    emb_dim: int = typer.Option(32, help="Token embedding dim (toy)."),
    topk: int = typer.Option(3, help="Citations per frame (toy)."),
    pcg: str = typer.Option("toy", help="PCG backend: toy | llama2"),
    llama2_path: str = typer.Option("/data/models/Llama-2-7b-chat-hf", help="Local path to LLaMA-2 model"),
    device: str = typer.Option("cuda", help="Device for LLaMA-2 (cuda|cpu)"),
    llama2_quant: str = typer.Option("fp16", help="LLaMA-2 quantization: fp16|8bit"),
):
    vol = load_volume(seed=seed)
    if pcg == "llama2":
        pcg_fn = Llama2PCG(
            Llama2PCGConfig(
                model_path=llama2_path,
                device=device,
                quantization=llama2_quant,
                max_new_tokens=512,
                temperature=0.0,
                topk_citations=topk,
            )
        )
    else:
        pcg_fn = ToyPCG(emb_dim=emb_dim, topk=topk, seed=seed)

    if loop == "agent":
        agent_res = run_provetok_agent(
            volume=vol,
            generator_fn=lambda toks: pcg_fn(toks),
            cfg=AgentConfig(
                budget_tokens=int(budget),
                emb_dim=int(emb_dim),
                max_steps_per_finding=int(steps),
                k_max_citations=max(1, int(topk)),
            ),
            seed=int(seed),
        )
        # Shim to reuse the artifact writer below.
        res = type(
            "AgentShim",
            (),
            {
                "tokens": agent_res.tokens,
                "gen": agent_res.generation,
                "issues": agent_res.issues,
                "trace": agent_res.trace,
                "final_cells": agent_res.final_cells,
                "stopped_reason": "agent",
            },
        )()
    else:
        res = run_refine_loop(
            volume=vol,
            budget_tokens=budget,
            steps=steps,
            generator_fn=lambda tokens: pcg_fn(tokens),
            verifier_fn=lambda gen, tokens: verify(gen, tokens),
            emb_dim=emb_dim,
            seed=seed,
            pcg_refresh_period=(steps if pcg == "llama2" else 1),
        )

    repo_root = Path(__file__).resolve().parents[1]
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=seed,
        config={"steps": steps, "budget": budget, "emb_dim": emb_dim, "topk": topk},
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision="synthetic",
        split_manifest_path="",
    )

    artifact = {
        "meta": meta.to_dict(),
        "loop": str(loop),
        "tokens": [
            {
                "token_id": t.token_id,
                "cell_id": t.cell_id,
                "level": t.level,
                "score": t.score,
                "uncertainty": t.uncertainty,
            } for t in res.tokens
        ],
        "frames": [f.__dict__ for f in res.gen.frames],
        "citations": res.gen.citations,
        "citations_ref": res.gen.citations_ref,
        "q": res.gen.q,
        "refusal": res.gen.refusal,
        "text": res.gen.text,
        "pp_findings": [],
        "pp_impression": "",
        "issues": [i.__dict__ for i in res.issues],
        "refine_trace": [getattr(t, "__dict__", {}) for t in res.trace],
    }

    gen_pp = enforce_pp_contract(res.gen, res.tokens, k_max=8)
    findings_lines = render_findings(gen_pp, k_max=8)
    artifact["pp_findings"] = findings_lines
    artifact["pp_impression"] = render_impression(findings_lines)

    rprint("[bold green]=== ProveTok v0 Artifact (JSON) ===[/bold green]")

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    print(json.dumps(convert(artifact), indent=2))

if __name__ == "__main__":
    app()
