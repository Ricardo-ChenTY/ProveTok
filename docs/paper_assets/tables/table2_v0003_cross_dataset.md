# Table 2. V0003 Cross-Dataset Grounding and Counterfactual Summary

| Item | Scope | Key Result | Verdict |
|---|---|---|---|
| E0166 grounding vs ROI | TS-Seg eval-only, budgets 2e6..7e6 | IoU_union positive 6/6, Holm significant 6/6 | Pass |
| E0166 grounding vs Fixed-Grid | TS-Seg eval-only, budgets 2e6..7e6 | IoU_union positive 5/6, Holm significant 4/6 | Partial pass |
| E0167 seed0..2 no_cite | counterfactual | mean_diff(avg)=0.0059, Holm significant 3/3 | Pass |
| E0167 seed0..2 omega_perm | counterfactual | mean_diff(avg)=0.0023, Holm significant 0/3 | Not significant |
| E0167R pooled | seeds 0..9 | mean_diff=0.0020, 95%CI=[0.0001,0.0037], p_one_sided=0.0187, p_holm=0.1122, positive=9/10 | Primary pass only |
| E0167R2 pooled | seeds 0..19 | mean_diff=0.0026, 95%CI=[0.0013,0.0038], p_one_sided=0.0001, p_holm=0.0006, positive=19/20 | Primary + Holm pass |
