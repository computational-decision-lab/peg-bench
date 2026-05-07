# Table 23

Frozen benchmark contract versus completed evidence.

| Layer | Frozen in contract | Populated in this manuscript |
| --- | --- | --- |
| Data scope | 90 days; BTCUSDT and ETHUSDT; Binance and OKX; replayable benchmark views | 90D guard passes 360 time-bar and 360 volume-bar asset-days |
| Split and realism ladder | Strict 60/15/15 chronology; R0–R4 realism ladder; fixed 0/1/3/5/10 bps ladder | Core report card populated at R0 and R4 for the completed panels |
| Prediction-first panel | Shared feature contract, leakage controls, and evaluator | LightGBM, Kairos, DeepLOB, Ridge/HistGBM-style controls populated where implemented |
| Anchors and heuristics | No-trade, buy-and-hold, and deterministic microstructure rules | Reported separately from active ranked methods to avoid safety-floor leaderboard pollution |
| Gap and stress metrics | Metric Gap, Ranking Gap, Reversal Rate@k, Fragility Gap, stress survival | Gap metrics and stress diagnostics populated for the completed 90D result set |
| Deployment-aware slot | Preference-based family slot under the shared evaluator | OG-PA/DPO documented as diagnostic formal-method rows, not as benchmark-winning evidence |
| Artifacts | Processed results, manifests, execution specifications, metadata, and review subset | Anonymous submission artifact includes CSV/JSON evidence, paper tables, reproduction scripts, claim map, metadata, and SHA256 manifest |
