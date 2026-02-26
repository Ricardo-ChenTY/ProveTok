# Proposal Gap Audit (fix-r1-metrics)

Date: 2026-02-26  
Branch: `fix-r1-metrics`  
Sync status: local vs `origin/fix-r1-metrics` = `0/0` (up-to-date)

## Scope
This audit checks the current branch against the latest proposal revision (Stage 1-4), with the constraint:
- do not redesign core architecture/workflow/pipeline;
- only do minimal, compatible upgrades.

## Executive Summary
- M0 infra upgrade (dual A100 + 128^3 + real manifest data + Accelerator) is already in place.
- Stage 1 is mostly aligned for "variance-based BET default", but "frozen SwinUNETR as main encoder" is not hard-integrated yet.
- Stage 2 has the largest gap: current main path is still `Llama2PCG` + post-hoc citation override, not proposal-defined inline citation generation.
- Stage 3 has stable R1-R4 implementation for proof metrics, but R5 (atlas anatomy) and R6 (CT-CLIP semantic relevance) are not integrated.
- Stage 4 has working split/refine loop; de-specify fallback exists in agent loop, but not in `bet/refine_loop.py` path.

## Detailed Gap Matrix

### Stage 1: Multi-Scale BET
- Aligned:
  - 128^3 config for A100: `configs/m0_a100.yaml:9`
  - Real manifest passthrough + workers: `scripts/train_m0.py:62`, `scripts/train_m0.py:67`
  - Variance-based score default: `provetok/bet/tokenize.py:240`
  - Encoder-backed ROI pooling interface exists: `provetok/bet/tokenize.py:122`
- Partial:
  - Proposal says "SwinUNETR frozen as main encoder"; current code is generic `encoder` interface + toy fallback, with explicit TODO: `provetok/bet/tokenize.py:40`
  - Learned saliency as ablation is feasible, but not wired as a canonical Stage-1 switch in one place.

### Stage 2: Grounded Report Generation (PCG/LLM)
- Not aligned:
  - Main LLM path still Llama-2 naming/path:
    - `provetok/pcg/llama2_pcg.py`
    - `provetok/experiments/run_baselines.py:83`, `:84`, `:329`
    - `scripts/ops/launch_all_datasets_dual_a100.sh:125`
  - Inline citation token generation `[CIT_001]...[CIT_N]` is not implemented.
  - Current design remains structured JSON + citation repair/override:
    - prompt/schema path: `provetok/pcg/llama2_pcg.py:86`
    - post-hoc citation override: `provetok/pcg/llama2_pcg.py:533`
    - slot vocab only, no citation token vocab extension: `provetok/pcg/schema.py:16`
- Partial:
  - `L_cite` and `L_ground`-like terms exist in trainer for PCG head attention:
    - citation weak supervision: `provetok/training/trainer.py:310`
    - grounding consistency: `provetok/training/trainer.py:428`
  - But this is not end-to-end inline-citation LLM training.

### Stage 3: Multi-Level Verification
- Aligned (R1-R4):
  - Deterministic PP verifier (R1-R4): `provetok/verifier/pp_v1_1.py:135`
  - Proof metrics wired for R1-R4: `provetok/eval/metrics_proof.py:83`
- Partial / Not aligned:
  - Trainer currently uses `create_verifier()` from rules bundle (many extra U1/O1/I1/M1 rules), not PP-only verifier:
    - `provetok/training/trainer.py:110`
    - `provetok/verifier/rules.py:956`
  - R5 (TotalSegmentator atlas anatomy check) missing.
  - R6 (CT-CLIP shared alignment space relevance) missing.
  - Existing `I1_AnatomicalMismatch` is simplified laterality heuristic:
    - `provetok/verifier/rules.py:546`
  - Existing `U1_CitationRelevance` is toy-query attention proxy:
    - `provetok/verifier/rules.py:302`

### Stage 4: Closed-Loop Refinement
- Aligned:
  - BET refine loop and split policy path exist:
    - `provetok/bet/refine_loop.py:57`
    - `provetok/bet/split_policy_dpo.py`
  - Conservative de-specify fallback is implemented in agent loop:
    - frame rewrite: `provetok/agent/loop.py:143`
    - failure fallback path: `provetok/agent/loop.py:294`
- Partial:
  - `bet/refine_loop.py` itself does not apply de-specification to generation on stop.
  - DPO 5000+ preference pipeline is not packaged as a complete, reproducible data-prep+train entry in this branch.

## Minimal Completion Plan (No Re-architecture)

### P0 (must-do first, low risk)
1. Unify experiment entry to one "LLM backend path" (keep backward-compatible `llama2` args, add proposal-default Llama-3.1 config).
2. Freeze one canonical Stage-1/2/3 run profile (R1-first ECCV profile):
   - `128^3`, budget grid fixed, proof metrics mandatory, deterministic seeds.
3. Add manifest/data contract checks at startup:
   - fail fast when required fields are missing (`volume_path`, `report_text`, and if grounding enabled, `mask_path`).

### P1 (proposal-critical, medium risk)
1. Stage 3 R5:
   - add optional atlas-driven anatomy rule path (enabled only when atlas labels exist in data).
2. Stage 3 R6:
   - add CT-CLIP scorer interface + threshold calibration script on validation split.
3. Keep R1-R4 path unchanged to preserve comparability.

### P2 (proposal-critical, higher risk)
1. Stage 2 inline citation generation path:
   - add new mode parallel to current JSON/post-hoc mode (do not remove old path).
2. Add training/eval switch to compare:
   - `inline_citation_mode` vs `posthoc_mode`.

### P3 (ablation/paper completeness)
1. SwinUNETR frozen encoder integration as main path, keep toy/generic encoder as smoke fallback.
2. Learned saliency ablation switch + report template.

## Immediate Recommendation
Execute in this order for fastest paper-progress with least churn:
1. P0 profile lock + scripts;
2. P1 R5/R6 hooks;
3. P2 inline citation path;
4. P3 encoder/ablation polish.

