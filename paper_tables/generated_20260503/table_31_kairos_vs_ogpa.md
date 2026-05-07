# Table 31

Kairos versus OG-PA as method families.

| Dimension | Kairos | OG-PA / KAIROS-X | Expected benchmark effect |
| --- | --- | --- | --- |
| Input | LOB tensor or token sequence | Same benchmark state plus risk condition z | Preserves comparability under the shared contract |
| Output | Return score or ranking signal | Continuous-action policy or conditioned trajectory distribution | More direct control over execution and turnover |
| Training supervision | Pointwise surrogate losses such as MSE and RankIC | Winner/loser preferences from ex-post trajectories | Moves supervision closer to execution-aware metrics |
| Likely advantage zone | May still look better at R0 | Need not improve R0, but is designed for R4 | Should be judged by ranking shifts under realism |
