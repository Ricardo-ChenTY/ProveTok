# Table 4. Oral Minimal Evidence Set (Paper-Grade)

| Item | Verdict | Key Numbers | Protocol | Evidence |
|---|---|---|---|---|
| `C0001` | Pass | combined_pass=6/6(need4); iou_pass=6/6(need4); lat_p95_pass=6/6; unsupported_pass=6/6 | budgets=6, seeds=5, n_boot=20000, paired bootstrap(H1>0)+Holm(budgets); lat_p95<=+5%, unsupported_delta<=+0.05 | `/home/ubuntu/tiasha/ProveTok/outputs/E0164-full/baselines_curve_multiseed.json` |
| `C0002` | Pass | n_boot=20000; CI_high=0.0000; naive_CI_low=0.4823 | dev->test, AIC/BIC model fit, bootstrap CI, requires CI_high<0.15 & beats naive | `/home/ubuntu/tiasha/ProveTok/outputs/E0161-full/fig3_regret_sweep.json` |
| `C0003` | Pass | no_cite: dIoU=0.0010, p_holm=0; cite_swap: dUnsup=0.0234, p_holm=0 | paired bootstrap + Holm (counterfactual family) | `/home/ubuntu/tiasha/ProveTok/outputs/E0162-full_retry3/figX_counterfactual_20260206_102521/figX_counterfactual.json` |
| `C0004` | Pass | fixed_grid_pass=6/6(need4); roi_variance_pass=6/6(need4) | one-sided (H1>0) + Holm(budgets), n_boot=20000 | `/home/ubuntu/tiasha/ProveTok/outputs/E0156-grounding_proof_100g_saliency_seed20/figX_grounding_proof.json` |
| `C0005` | Pass | tau=0.002; miss_max=0<= 0.05; ece_max=0.00183<= 0.15; rr_max=0.1<= 0.2; unsupported_improved=6/6 | hard gates per budget + need>=2/3 budgets improve unsupported | `/home/ubuntu/tiasha/ProveTok/outputs/E0144-full/figX_refusal_calibration.json` |
| `C0006` | Pass | budget_accounting=True; strong_weights=True; frame_f1_last=0.6967>= 0.05 | baseline coverage + audited cost accounting + strong baseline non-degenerate | `/home/ubuntu/tiasha/ProveTok/outputs/E0164-full/baselines_curve_multiseed.json` |
| `V0003/omega_perm` | Pass | mean_diff=0.0026; CI=[0.0013,0.0038]; p1=0.0001; p_holm=0.0006; positive=19/20 | pooled one-sided test + secondary Holm over counterfactual family | `outputs/E0167R2-ct_rate-tsseg-effusion-counterfactual-power-seed20/omega_perm_power_report.json` |
