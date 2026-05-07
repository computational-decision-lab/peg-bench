# Table 28

Model and method panel for PEG-Bench evaluation.

| Role | Model or method | Status | Why retained / status |
| --- | --- | --- | --- |
| anchor | no_trade | completed | All six 90d anchor rows exist under anchors_heuristics. |
| anchor | buy_and_hold | completed | All six 90d anchor rows exist under anchors_heuristics. |
| heuristic | ofi_threshold | completed | All six 90d heuristic rows exist under anchors_heuristics. |
| low_capacity_ml | ridge_tokens | partial | Current 90d runs exist but are not all protocol-clean seq96 / wrapper-matrix-complete full-paper rows. |
| low_capacity_ml | ElasticNet | target-only | No completed 90d run/result table found. |
| tabular_ml | histgbm_tokens | partial | Current 90d runs exist but full paper protocol boundary remains partial. |
| tabular_ml | lightgbm_lob | partial | Current 90d runs exist; wrapper audit is completed only for volume_h20. |
| lob_neural | deeplob_timebars | partial | Current 90d evidence exists for time panels only; not a six-panel completed row. |
| token_sequence | kairos_64code | partial | Current 90d evidence exists but remains single-level 64-code rather than a full benchmark-wide KAIROS family completion. |
| lob_neural | TABL | target-only | No completed 90d implementation/run output found. |
| financial_ts | Kronos | target-only | No completed benchmark integration or result table found. |
| financial_ts | iTransformer | target-only | No completed 90d implementation/run output found. |
| foundation_ts_appendix | TimesFM | target-only | No completed 90d implementation/run output found. |
| policy_learning | PPO | target-only | No completed 90d benchmark runner result found. |
| policy_learning | SAC | target-only | No completed 90d benchmark runner result found. |
| formal_method | OG-PA / DPO | diagnostic | 30d case-study evidence exists; current 90d rows are negative diagnostic slot artifacts and should not be framed as benchmark winners. |
| formal_method_ablation | ACI | diagnostic | Use as gating/ablation note, not as a standalone winning baseline. |
| audit-only | HftBacktest | audit-only | No completed integrated 90d leaderboard artifact; should remain outside ranked benchmark claims. |
