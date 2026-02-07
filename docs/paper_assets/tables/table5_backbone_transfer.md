# Table 5. V0004 Backbone Transfer (Summary)

| Backbone | Setting | Seeds | N | n_boot | Combined (pos/sig) | IoU (pos/sig) | Unsupported (pos/sig) | Evidence |
|---|---|---:|---:|---:|---|---|---|---|
| `toy` | provetok_lesionness/fixed_grid | 0,1,2 | 57 | 20000 | 3/3 (sig 3/3, avg d=0.0006) | 3/3 (sig 3/3, avg d=0.0011) | 2/3 (sig 1/3, avg d=0.0637) | `outputs/E0173-backbone-toy-mini/baselines_curve_multiseed.json` |
| `llama2` | provetok_lesionness/fixed_grid | 0 | 30 | 20000 | 1/3 (sig 1/3, avg d=0.0280) | 1/3 (sig 0/3, avg d=0.0005) | 2/3 (sig 0/3, avg d=0.8667) | `outputs/E0172-backbone-llama2-mini/baselines_curve_multiseed.json` |

Notes: positive mean_diff indicates improvement; Holm family: `per-metric across budgets`; bootstrap: `avg over seeds per sample, bootstrap over samples`.
