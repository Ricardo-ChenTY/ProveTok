# Table 1. Claim-Level Machine Verdict (real profile)

| Claim | Status | Summary | Primary Evidence |
|---|---|---|---|
| `C0001` | Pass | proved: provetok_lesionness beats fixed_grid on combined & iou at 6/6 and 6/6 budgets (Holm), with latency/unsupported constraints | `/home/ubuntu/tiasha/ProveTok/outputs/E0164-full/baselines_curve_multiseed.json` |
| `C0002` | Pass | proved: paper-grade dev→test regret has CI and beats naive policy (real pipeline) | `/home/ubuntu/tiasha/ProveTok/outputs/E0161-full/fig3_regret_sweep.json` |
| `C0003` | Pass | proved (no_cite breaks grounding + cite_swap breaks unsupported) | `/home/ubuntu/tiasha/ProveTok/outputs/E0162-full_retry3/figX_counterfactual_20260206_102521/figX_counterfactual.json` |
| `C0004` | Pass | proved: provetok_lesionness beats fixed_grid and roi_variance on iou_union (Holm) with >= 4 budgets each | `/home/ubuntu/tiasha/ProveTok/outputs/E0156-grounding_proof_100g_saliency_full/figX_grounding_proof.json` |
| `C0005` | Pass | proved: tau_refuse=0.002 meets miss/ECE/refusal constraints and reduces unsupported at 6/6 budgets | `/home/ubuntu/tiasha/ProveTok/outputs/E0144-full/figX_refusal_calibration.json` |
| `C0006` | Pass | proved: baseline suite present with cost accounting + reproducible strong baseline | `/home/ubuntu/tiasha/ProveTok/outputs/E0164-full/baselines_curve_multiseed.json` |
