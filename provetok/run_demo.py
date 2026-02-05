from __future__ import annotations
import json
import typer
from rich import print as rprint
from pathlib import Path
import torch
import numpy as np

from .data.io import load_volume
from .pcg import ToyPCG, Llama2PCG, Llama2PCGConfig
from .verifier import verify
from .bet import run_refine_loop
from .utils.artifact import build_artifact_meta
from .verifier.rules import RULE_SET_VERSION
from .verifier.taxonomy import TAXONOMY_VERSION
from .pcg.schema_version import SCHEMA_VERSION

app = typer.Typer(add_completion=False)

@app.command()
def main(
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
        "q": res.gen.q,
        "refusal": res.gen.refusal,
        "text": res.gen.text,
        "issues": [i.__dict__ for i in res.issues],
        "refine_trace": [t.__dict__ for t in res.trace],
    }

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
