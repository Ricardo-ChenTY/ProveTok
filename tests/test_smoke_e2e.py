"""End-to-end smoke tests for ProveTok (CPU, synthetic).

Goal: ensure the whole scaffolded pipeline runs end-to-end:
- data loading
- BET tokenization
- PCGHead forward/decode + backward
- rule-based verifier
- evidence head Δ(c)
- evidence graph builder
- refusal calibrator
- ProveTokSystem forward + inference
- Trainer loop (few steps)
"""

from __future__ import annotations

import torch
import pytest


def _divider(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


@pytest.fixture(scope="module")
def batch():
    _divider("1. Data Loading")
    from provetok.data import make_dataloader, CTRateDataset

    cfg = {"dataset_type": "synthetic", "num_samples": 8, "vol_shape": [32, 64, 64], "batch_size": 2, "seed": 0}
    loader = make_dataloader(cfg, split="train")
    batch = next(iter(loader))

    assert "volume" in batch and "frames" in batch and "sample_id" in batch
    assert batch["volume"].ndim == 4  # (B,D,H,W)
    assert isinstance(batch["frames"], list)

    # CT-RATE fallback (no real data, will use synthetic)
    ds = CTRateDataset(data_root="./nonexistent_data", split="train", max_samples=4, seed=0)
    assert len(ds) == 4
    item = ds[0]
    assert item["volume"].ndim == 3

    return batch


@pytest.fixture(scope="module")
def volume(batch):
    return batch["volume"][0]  # (D,H,W)


@pytest.fixture(scope="module")
def tokens(volume):
    _divider("2. BET Tokenization")
    from provetok.bet.tokenize import encode_tokens
    from provetok.grid.cells import root_cell, split

    cells = [root_cell()]
    tokens = encode_tokens(volume, cells, emb_dim=32, seed=0)
    assert len(tokens) == 1

    children = split(cells[0])
    tokens2 = encode_tokens(volume, children, emb_dim=32, seed=0)
    assert len(children) == 8
    assert len(tokens2) == 8

    return tokens


@pytest.fixture(scope="module")
def generation(tokens):
    _divider("3. PCG Head (decode)")
    from provetok.models.pcg_head import PCGHead

    head = PCGHead(emb_dim=32)
    token_embs = torch.stack([t.embedding for t in tokens])
    gen = head.decode(token_embs, tokens)
    assert len(gen.frames) > 0
    return gen


@pytest.fixture(scope="module")
def issues(generation, tokens):
    _divider("4. Verifier (Rule-Based)")
    from provetok.verifier.rules import create_verifier

    verifier = create_verifier()
    issues = verifier.verify(generation, tokens)
    assert isinstance(issues, list)
    return issues


def test_pcg_head_forward_backward(tokens):
    _divider("PCGHead forward/backward")
    from provetok.models.pcg_head import PCGHead
    from provetok.types import Frame

    head = PCGHead(emb_dim=32)
    token_embs = torch.stack([t.embedding for t in tokens])

    out = head(token_embs)
    assert "finding_logits" in out and "q_k" in out

    gt_frames = [Frame(finding="effusion", polarity="present", laterality="left", confidence=0.9)]
    losses = head.compute_loss(out, gt_frames)
    losses["total"].backward()
    assert any(p.grad is not None for p in head.parameters())


def test_evidence_head_delta(tokens, issues):
    _divider("Evidence Head Δ(c)")
    from provetok.bet.evidence_head import EvidenceHead, compute_delta
    from provetok.grid.cells import root_cell

    head = EvidenceHead(emb_dim=32)
    cell = root_cell()
    emb = tokens[0].embedding

    score = compute_delta(cell=cell, cell_embedding=emb, evidence_head=head, current_issues=issues)
    assert score.delta >= 0.0


def test_evidence_graph(tokens):
    _divider("Evidence Graph")
    from provetok.pcg.evidence_graph import EvidenceGraphBuilder, compute_support_score

    builder = EvidenceGraphBuilder(emb_dim=32)
    graph = builder.build_graph(tokens, top_k=3, min_conf=0.1)
    assert len(graph.entries) == len(tokens)

    if tokens:
        s = compute_support_score(graph, [tokens[0].token_id], "effusion")
        assert 0.0 <= s <= 1.0


def test_frame_extractor():
    _divider("Frame Extractor")
    from provetok.data.frame_extractor import FrameExtractor, frames_to_report

    extractor = FrameExtractor()
    report = (
        "There is a small left pleural effusion. "
        "A 5mm nodule is seen in the right upper lobe. "
        "No pneumothorax."
    )
    frames = extractor.extract_frames(report)
    assert frames
    text = frames_to_report(frames)
    assert isinstance(text, str) and len(text) > 0


def test_refusal_calibrator():
    _divider("Refusal Calibrator")
    from provetok.pcg.refusal import RefusalCalibrator
    from provetok.types import Frame, Generation

    calibrator = RefusalCalibrator(tau_refuse=0.5)
    gen = Generation(
        frames=[Frame(finding="effusion", polarity="present", laterality="left", confidence=0.8)],
        citations={0: [0, 1]},
        q={0: 0.7},
        refusal={0: False},
    )
    decisions = calibrator.decide_refusals(gen)
    assert len(decisions) == 1
    assert isinstance(decisions[0].should_refuse, bool)


def test_system_forward_inference(batch):
    _divider("ProveTokSystem")
    from provetok.models.system import ProveTokSystem

    system = ProveTokSystem(emb_dim=32, num_findings=3)
    out = system(batch)
    out.loss.backward()

    vol = batch["volume"][0]
    r0 = system.inference(vol, use_refinement=False, seed=0)
    assert "generation" in r0

    r1 = system.inference(vol, use_refinement=True, seed=0)
    assert "issues" in r1 and "trace" in r1


def test_llm_backend_dummy():
    _divider("LLM Backend (Dummy)")
    from provetok.models.llm_backend import create_llm_backend, DummyLLM

    llm = create_llm_backend("dummy")
    assert isinstance(llm, DummyLLM)
    out = llm.generate(
        prompt="Describe findings",
        constrained_vocab={"finding_type": {"nodule", "effusion"}, "laterality": {"left", "right"}},
    )
    assert isinstance(out.text, str)


def test_training_loop_smoke():
    _divider("Trainer loop (3 steps)")
    from provetok.training import Trainer, TrainerConfig

    cfg = TrainerConfig(
        stage="M1",
        device="cpu",
        emb_dim=32,
        dataset_cfg={
            "dataset_type": "synthetic",
            "num_samples": 8,
            "vol_shape": [32, 64, 64],
            "batch_size": 2,
        },
        overrides={
            "max_steps": 3,
            "log_every": 1,
            "eval_every": 2,
            "save_every": 100_000,
        },
    )
    trainer = Trainer(cfg)
    result = trainer.train()
    assert result["total_steps"] == 3


def test_refine_loop_require_full_budget_spends_budget(volume):
    _divider("BET refine loop (require_full_budget spends budget)")
    from provetok.bet.refine_loop import run_refine_loop
    from provetok.types import Generation

    def gen_fn(_tokens):
        return Generation(frames=[], citations={}, q={}, refusal={}, text="")

    def ver_fn(_gen, _tokens):
        return []

    res = run_refine_loop(
        volume=volume,
        budget_tokens=15,
        steps=1,  # intentionally too small; require_full_budget should override
        generator_fn=gen_fn,
        verifier_fn=ver_fn,
        emb_dim=32,
        seed=0,
        use_evidence_head=False,
        require_full_budget=True,
        max_depth=4,
    )
    assert len(res.tokens) >= 15
