"""ProveTok End-to-End Smoke Test

验证整个管线在没有 LLM 的情况下可以完整跑通:
1. 数据加载 (Synthetic)
2. BET tokenization + refinement
3. PCG Head forward + decode
4. Verifier (rule-based)
5. Evidence Head Δ(c) 计算
6. ProveTokSystem forward + inference
7. Training loop (几步)
"""
import sys
import torch
import numpy as np

def divider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_data_loading():
    divider("1. Data Loading")
    from provetok.data import make_dataloader, SyntheticCTReportDataset, CTRateDataset

    # Synthetic
    cfg = {"dataset_type": "synthetic", "num_samples": 8, "vol_shape": [32, 64, 64], "batch_size": 2}
    loader = make_dataloader(cfg, split="train")
    batch = next(iter(loader))
    print(f"  volume shape: {batch['volume'].shape}")
    print(f"  frames count: {[len(f) for f in batch['frames']]}")
    print(f"  sample_ids: {batch['sample_id']}")

    # CT-RATE fallback (no real data, will use synthetic)
    ds = CTRateDataset(data_root="./nonexistent_data", split="train", max_samples=4)
    print(f"  CTRateDataset (fallback): {len(ds)} samples")
    item = ds[0]
    print(f"  item keys: {list(item.keys())}")
    print(f"  volume shape: {item['volume'].shape}")

    print("  [PASS] Data loading")
    return batch


def test_bet_tokenization(batch):
    divider("2. BET Tokenization")
    from provetok.bet.tokenize import encode_tokens
    from provetok.grid.cells import root_cell, split

    vol = batch["volume"][0]  # (D, H, W)
    cells = [root_cell()]
    tokens = encode_tokens(vol, cells, emb_dim=32, seed=0)
    print(f"  Root cell tokens: {len(tokens)}")
    print(f"  Token[0]: id={tokens[0].token_id}, cell={tokens[0].cell_id}, "
          f"level={tokens[0].level}, score={tokens[0].score:.3f}")

    # Split root and re-encode
    children = split(cells[0])
    tokens2 = encode_tokens(vol, children, emb_dim=32, seed=0)
    print(f"  After split: {len(children)} cells, {len(tokens2)} tokens")

    print("  [PASS] BET tokenization")
    return vol, tokens


def test_pcg_head(tokens):
    divider("3. PCG Head")
    from provetok.models.pcg_head import PCGHead

    head = PCGHead(emb_dim=32)
    token_embs = torch.stack([t.embedding for t in tokens])
    print(f"  token_embs shape: {token_embs.shape}")

    # Forward
    out = head(token_embs)
    print(f"  finding_logits: {out['finding_logits'].shape}")
    print(f"  polarity_logits: {out['polarity_logits'].shape}")
    print(f"  laterality_logits: {out['laterality_logits'].shape}")
    print(f"  confidence: {out['confidence'].shape}")
    print(f"  attn_weights: {out['attn_weights'].shape}")
    print(f"  q_k: {out['q_k'].shape}")

    # Decode
    gen = head.decode(token_embs, tokens)
    print(f"  Generated {len(gen.frames)} frames:")
    for i, f in enumerate(gen.frames):
        print(f"    [{i}] {f.finding} / {f.polarity} / {f.laterality} "
              f"conf={f.confidence:.3f} q={gen.q[i]:.3f} refused={gen.refusal[i]}")

    # Loss
    from provetok.types import Frame
    gt_frames = [Frame(finding="effusion", polarity="present", laterality="left", confidence=0.9)]
    losses = head.compute_loss(out, gt_frames)
    print(f"  Loss: total={losses['total'].item():.4f} "
          f"finding={losses['finding'].item():.4f} "
          f"polarity={losses['polarity'].item():.4f}")

    # Backward test
    losses["total"].backward()
    grad_count = sum(1 for p in head.parameters() if p.grad is not None)
    print(f"  Backward OK, {grad_count} params have gradients")

    print("  [PASS] PCG Head")
    return gen


def test_verifier(gen, tokens):
    divider("4. Verifier (Rule-Based)")
    from provetok.verifier.rules import create_verifier

    verifier = create_verifier()
    issues = verifier.verify(gen, tokens)
    print(f"  Found {len(issues)} issues")
    for issue in issues[:5]:
        print(f"    [{issue.issue_type}] severity={issue.severity} rule={issue.rule_id}: {issue.message[:60]}")

    print("  [PASS] Verifier")
    return issues


def test_evidence_head(tokens, issues):
    divider("5. Evidence Head")
    from provetok.bet.evidence_head import EvidenceHead, compute_delta, rank_cells_by_delta
    from provetok.grid.cells import root_cell

    head = EvidenceHead(emb_dim=32)
    cell = root_cell()
    emb = tokens[0].embedding

    # compute_delta
    score = compute_delta(
        cell=cell,
        cell_embedding=emb,
        evidence_head=head,
        current_issues=issues,
    )
    print(f"  Delta(root): {score.delta:.4f}")
    print(f"    issue_reduction: {score.issue_reduction:.4f}")
    print(f"    uncertainty: {score.uncertainty:.4f}")
    print(f"    slot_probs: {list(score.slot_probs.keys())[:4]}...")

    print("  [PASS] Evidence Head")


def test_evidence_graph(tokens):
    divider("6. Evidence Graph")
    from provetok.pcg.evidence_graph import EvidenceGraphBuilder, compute_support_score

    builder = EvidenceGraphBuilder(emb_dim=32)
    graph = builder.build_graph(tokens, top_k=3, min_conf=0.1)
    print(f"  Graph entries: {len(graph.entries)}")

    valid_findings = graph.get_valid_domain("finding_type")
    print(f"  Valid finding_type domain: {valid_findings}")

    valid_locations = graph.get_valid_domain("location")
    print(f"  Valid location domain ({len(valid_locations)} values)")

    # Support score
    if tokens:
        score = compute_support_score(graph, [tokens[0].token_id], "effusion")
        print(f"  Support score for 'effusion': {score:.4f}")

    print("  [PASS] Evidence Graph")


def test_frame_extractor():
    divider("7. Frame Extractor")
    from provetok.data.frame_extractor import FrameExtractor, frame_to_text, frames_to_report

    extractor = FrameExtractor()

    report = (
        "There is a small left pleural effusion. "
        "A 5mm nodule is seen in the right upper lobe. "
        "No pneumothorax. No consolidation. "
        "The heart is normal in size."
    )
    findings = extractor.extract(report)
    frames = extractor.extract_frames(report)
    print(f"  Extracted {len(findings)} findings from report:")
    for f in findings:
        print(f"    {f.finding_type} / {f.polarity} / {f.laterality} / conf={f.confidence:.2f}")

    # Template output
    generated_report = frames_to_report(frames)
    print(f"  Generated report:\n    {generated_report[:120]}...")

    print("  [PASS] Frame Extractor")


def test_refusal_calibrator():
    divider("8. Refusal Calibrator")
    from provetok.pcg.refusal import RefusalCalibrator
    from provetok.types import Frame, Generation, Issue

    calibrator = RefusalCalibrator(tau_refuse=0.5)
    # Quick check
    gen = Generation(
        frames=[Frame(finding="effusion", polarity="present", laterality="left", confidence=0.8)],
        citations={0: [0, 1]},
        q={0: 0.7},
        refusal={0: False},
    )
    # should_refuse: simple threshold check on q_k
    refuse = calibrator.should_refuse(q_k=0.7, is_critical=False)
    print(f"  should_refuse(q_k=0.7, tau=0.5): {refuse}")  # False, 0.7 > 0.5

    # decide_refusals: batch decision over all frames in a Generation
    decisions = calibrator.decide_refusals(gen)
    print(f"  decide_refusals: {len(decisions)} decision(s), "
          f"refuse={decisions[0].refuse}, reason={decisions[0].reason}")

    print("  [PASS] Refusal Calibrator")


def test_system(batch):
    divider("9. ProveTokSystem")
    from provetok.models.system import ProveTokSystem

    system = ProveTokSystem(emb_dim=32, num_findings=3)
    print(f"  Params: {sum(p.numel() for p in system.parameters()):,}")

    # Forward (training)
    out = system(batch)
    print(f"  Forward loss: {out.loss.item():.4f}")
    print(f"  Logs: {out.logs}")

    # Backward
    out.loss.backward()
    print("  Backward OK")

    # Inference
    vol = batch["volume"][0]
    result = system.inference(vol, use_refinement=False, seed=0)
    print(f"  Inference (no refinement): {len(result['generation'].frames)} frames")

    result2 = system.inference(vol, use_refinement=True, seed=0)
    print(f"  Inference (with refinement): {len(result2['generation'].frames)} frames, "
          f"{len(result2['issues'])} issues, stopped={result2['stopped_reason']}")

    print("  [PASS] ProveTokSystem")


def test_llm_backend():
    divider("10. LLM Backend (Dummy)")
    from provetok.models.llm_backend import create_llm_backend, DummyLLM

    llm = create_llm_backend("dummy")
    assert isinstance(llm, DummyLLM)

    output = llm.generate(
        prompt="Describe findings",
        constrained_vocab={
            "finding_type": {"nodule", "effusion"},
            "laterality": {"left", "right"},
        },
    )
    print(f"  Generated text: {output.text}")
    print(f"  Slot values: {output.slot_values}")
    print(f"  Hidden dim: {llm.get_hidden_dim()}")

    print("  [PASS] LLM Backend")


def test_training_loop():
    divider("11. Training Loop (3 steps)")
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
            "save_every": 100,  # don't save during test
        },
    )
    trainer = Trainer(cfg)
    result = trainer.train()
    print(f"  Training result: {result}")

    print("  [PASS] Training Loop")


def main():
    print("\nProveTok Smoke Test")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    tests = [
        ("Data Loading", lambda: test_data_loading()),
        ("BET Tokenization", None),  # depends on data loading
        ("PCG Head", None),
        ("Verifier", None),
        ("Evidence Head", None),
        ("Evidence Graph", None),
        ("Frame Extractor", lambda: test_frame_extractor()),
        ("Refusal Calibrator", lambda: test_refusal_calibrator()),
        ("ProveTokSystem", None),
        ("LLM Backend", lambda: test_llm_backend()),
        ("Training Loop", lambda: test_training_loop()),
    ]

    # Run connected tests
    try:
        batch = test_data_loading()
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Data Loading: {e}")
        failed += 1
        errors.append(("Data Loading", str(e)))
        return

    try:
        vol, tokens = test_bet_tokenization(batch)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] BET Tokenization: {e}")
        failed += 1
        errors.append(("BET Tokenization", str(e)))
        tokens = []

    try:
        gen = test_pcg_head(tokens)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] PCG Head: {e}")
        failed += 1
        errors.append(("PCG Head", str(e)))
        gen = None

    try:
        issues = test_verifier(gen, tokens) if gen else []
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Verifier: {e}")
        failed += 1
        errors.append(("Verifier", str(e)))
        issues = []

    try:
        test_evidence_head(tokens, issues)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Evidence Head: {e}")
        failed += 1
        errors.append(("Evidence Head", str(e)))

    try:
        test_evidence_graph(tokens)
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Evidence Graph: {e}")
        failed += 1
        errors.append(("Evidence Graph", str(e)))

    # Independent tests
    for name, fn in [
        ("Frame Extractor", test_frame_extractor),
        ("Refusal Calibrator", test_refusal_calibrator),
        ("ProveTokSystem", lambda: test_system(batch)),
        ("LLM Backend", test_llm_backend),
        ("Training Loop", test_training_loop),
    ]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            errors.append((name, str(e)))

    # Summary
    divider("SUMMARY")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    if errors:
        print(f"\n  Failures:")
        for name, err in errors:
            print(f"    - {name}: {err}")

    if failed == 0:
        print("\n  ALL TESTS PASSED!")
    else:
        print(f"\n  {failed} test(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
