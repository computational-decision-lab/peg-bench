# Table 4

Execution diagnostics separated from the clean ranked R4 leaderboard.

| Diagnostic | Panel / method | Completed result | Interpretation |
| --- | --- | --- | --- |
| Wrapper audit | Volume h20 / LightGBM | Raw Sharpe -0.0723; wrapped Sharpe 0.6290 | A simple cost/turnover interface can flip the deployment verdict. |
| Wrapper audit | Volume h20 / Kairos-64 | Wrapped Sharpe -1.6269 | Execution-aware post-processing does not automatically rescue every strong predictor. |
| Formal slot | OG-PA / DPO rows | Main variants negative; conservative rescue remains negative and low-turnover rescue collapses to zero action | Reported as diagnostics, not as benchmark-winning evidence. |
| Stress audit | Volume h20 / LightGBM | Worst-regime Sharpe -0.8003; survival 1.0000 | Stress survival alone is insufficient without utility. |
| Stress audit | Volume h20 / Kairos-64 | Worst-regime Sharpe -3.5495; survival 1.0000 | Sequence-model strength at R0 can still be fragile under stress. |
