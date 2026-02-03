from __future__ import annotations
import json
import typer
from rich import print as rprint
import torch

from .data.io import load_volume
from .pcg import ToyPCG
from .verifier import verify
from .bet import run_refine_loop

app = typer.Typer(add_completion=False)

@app.command()
def main(
    steps: int = typer.Option(3, help="Max refine iterations."),
    budget: int = typer.Option(64, help="Max tokens allowed (budget)."),
    seed: int = typer.Option(0, help="Random seed (deterministic demo)."),
    emb_dim: int = typer.Option(32, help="Token embedding dim (toy)."),
    topk: int = typer.Option(3, help="Citations per frame (toy)."),
):
    vol = load_volume(seed=seed)
    pcg = ToyPCG(emb_dim=emb_dim, topk=topk, seed=seed)

    res = run_refine_loop(
        volume=vol,
        budget_tokens=budget,
        steps=steps,
        generator_fn=lambda tokens: pcg(tokens),
        verifier_fn=lambda gen, tokens: verify(gen, tokens),
        emb_dim=emb_dim,
        seed=seed,
    )

    artifact = {
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
        "issues": [i.__dict__ for i in res.issues],
        "refine_trace": res.trace,
    }

    rprint("[bold green]=== ProveTok v0 Artifact (JSON) ===[/bold green]")
    print(json.dumps(artifact, indent=2))

if __name__ == "__main__":
    app()
