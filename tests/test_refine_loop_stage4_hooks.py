from __future__ import annotations

import numpy as np
import torch

from provetok.bet.refine_loop import run_refine_loop
from provetok.types import Frame, Generation, Issue
from provetok.verifier.rules import create_verifier


def test_refine_loop_r5_rerank_rewrites_citations() -> None:
    volume = torch.zeros((16, 16, 16), dtype=torch.float32)
    anatomy = np.zeros((16, 16, 16), dtype=np.int32)
    anatomy[:, :, :8] = 1   # disallowed for nodule
    anatomy[:, :, 8:] = 2   # allowed for nodule
    anatomy_label_map = {
        1: "liver",
        2: "left_lung_upper",
    }
    verifier = create_verifier(enable_r5=True, enable_r6=False)

    def gen_fn(tokens):
        bad_tid = next((int(t.token_id) for t in tokens if str(t.anatomy_label) == "liver"), int(tokens[0].token_id))
        return Generation(
            frames=[
                Frame(
                    finding="nodule",
                    polarity="present",
                    laterality="left",
                    confidence=0.9,
                )
            ],
            citations={0: [bad_tid]},
            q={0: 0.9},
            refusal={0: False},
            text="",
        )

    def ver_fn(gen, tokens):
        return verifier.verify(gen, tokens)

    res = run_refine_loop(
        volume=volume,
        budget_tokens=16,
        steps=0,
        generator_fn=gen_fn,
        verifier_fn=ver_fn,
        emb_dim=16,
        seed=0,
        init_level=1,
        max_depth=2,
        use_evidence_head=False,
        anatomy_labels=anatomy,
        anatomy_label_map=anatomy_label_map,
        semantic_rerank_on_violation=True,
        semantic_rerank_topk=2,
        semantic_rerank_rule_ids=["R5"],
    )

    token_map = {int(t.token_id): t for t in res.tokens}
    cited = [token_map[int(tid)] for tid in res.gen.citations.get(0, []) if int(tid) in token_map]
    cited_labels = [str(getattr(t, "anatomy_label", "") or "") for t in cited]

    assert cited_labels
    assert any(lbl == "left_lung_upper" for lbl in cited_labels)
    assert all(str(getattr(iss, "rule_id", "")) != "R5" for iss in res.issues)


def test_refine_loop_despecify_fallback_rewrites_frame_slots() -> None:
    volume = torch.zeros((8, 8, 8), dtype=torch.float32)

    def gen_fn(tokens):
        tid = int(tokens[0].token_id) if tokens else 0
        return Generation(
            frames=[
                Frame(
                    finding="nodule",
                    polarity="present",
                    laterality="right",
                    confidence=0.95,
                    location="RUL",
                    size_bin="9-20mm",
                    severity="severe",
                    uncertain=False,
                )
            ],
            citations={0: [tid]},
            q={0: 0.95},
            refusal={0: False},
            text="",
        )

    def ver_fn(_gen, _tokens):
        return [
            Issue(
                frame_idx=0,
                issue_type="I1_inconsistency",
                severity=2,
                rule_id="R5",
                message="forced violation",
                evidence_trace={},
            )
        ]

    res = run_refine_loop(
        volume=volume,
        budget_tokens=8,
        steps=0,
        generator_fn=gen_fn,
        verifier_fn=ver_fn,
        emb_dim=8,
        seed=0,
        use_evidence_head=False,
        despecify_on_remaining_issues=True,
        despecify_confidence_cap=0.6,
    )

    assert len(res.gen.frames) == 1
    fr = res.gen.frames[0]
    assert str(fr.laterality) == "unspecified"
    assert float(fr.confidence) <= 0.600001
    assert bool(fr.uncertain) is True

