# Table 27

Implementation anchors for the PEG-Bench protocol.

| Anchor | Role |
| --- | --- |
| run_benchmark_evaluator.py | Aggregates per-day reports into benchmark report cards |
| src/eval/gap_metrics.py | Implements metric gap, ranking gap, reversal rate, and fragility-gap bundle |
| run_peg_bench_wrapper_audit.py | Runs validation-frozen wrapper sweeps for prediction-first methods |
| run_peg_bench_dpo_backtest.py | Runs OG-PA / DPO backtests and validation-only action-gating sweeps |
| generate_paper_artifacts_v3.py | This pass: regenerates paper tables, source maps, and release metadata from existing artifacts |
