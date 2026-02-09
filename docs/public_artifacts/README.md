# Public Artifacts

This folder contains small, shareable, aggregated artifacts exported from `outputs/` for convenience.

Notes:
- Paths are sanitized (`<REPO_ROOT>`, `<DATA_ROOT>`, `<MODEL_ROOT>`).
- Sample-level arrays are removed from the counterfactual export (`counterfactual_E0162_public.json`).
- Full raw `outputs/` and `.rd_queue/` are intentionally not tracked (see `.gitignore`).

Sources:
- `oral_audit.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/oral_audit.json`
- `proof_report_default.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/_proof_report_default.json`
- `proof_report_real.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/_proof_report_real.json`
- `baselines_curve_multiseed_E0164.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/E0164-full/baselines_curve_multiseed.json`
- `fig3_regret_sweep_E0161.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/E0161-full/fig3_regret_sweep.json`
- `refusal_calibration_E0144.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/E0144-full/figX_refusal_calibration.json`
- `grounding_proof_E0156.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/E0156-grounding_proof_100g_saliency_seed20/figX_grounding_proof.json`
- `llama2_contract_ablation_E0183.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/E0183-full/figX_llama2_contract_ablation.json`
- `llama2_contract_ablation_E0186.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/E0186-full/figX_llama2_contract_ablation.json`
- `counterfactual_E0162_public.json` <- `/home/ubuntu/tiasha/ProveTok/outputs/E0162-full_retry3/figX_counterfactual_20260206_102521/figX_counterfactual.json`
