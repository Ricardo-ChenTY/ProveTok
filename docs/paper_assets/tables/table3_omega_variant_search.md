# Table 3. Omega-Perm Variant Search (seed0)

| Variant | Setting | omega_perm mean_diff | omega_perm p_holm | no_cite mean_diff | no_cite p_holm |
|---|---|---:|---:|---:|---:|
| BASE | score + topk=3 (baseline) | 0.0015 | 1 | 0.0059 | 0 |
| RA | baseline + score_to_uncertainty | 0.0015 | 1 | 0.0059 | 0 |
| RD | baseline + score_level_power=1.0 | 0.0015 | 1 | 0.0059 | 0 |
| RC | score_interleave citations | 0.0006 | 1 | 0.0132 | 0 |
| RB | baseline with topk=1 | 0.0005 | 1 | 0.0017 | 0.0145 |

结论：RA/RB/RC/RD 在 seed0 上未超过 baseline 的 `omega_perm` 效果，因此后续使用 baseline 扩 seeds。
