# 90D Cost Sensitivity Summary

This file aggregates the existing six-panel 90D benchmark reports and does not launch any new training.

## Coverage
- Costs: `0 / 1 / 3 / 5 / 10 bps`
- Panels: `time_h5`, `time_h20`, `time_h50`, `volume_h5`, `volume_h20`, `volume_h50`
- Methods: all methods persisted in the existing benchmark reports for each panel

## Clean-active winner by cost (excerpt)

| panel | cost_bps | top_clean_active_method | top_clean_active_sharpe |
| --- | --- | --- | --- |
| time_h5 | 0.0 | lightgbm_lob | 1.3706 |
| time_h5 | 1.0 | ridge_tokens | 0.0256 |
| time_h5 | 3.0 | lightgbm_lob | -0.4318 |
| time_h5 | 5.0 | lightgbm_lob | -0.2827 |
| time_h5 | 10.0 | lightgbm_lob | -0.0730 |
| time_h20 | 0.0 | lightgbm_lob | 0.9262 |
| time_h20 | 1.0 | kairos_64code | 0.2536 |
| time_h20 | 3.0 | kairos_64code | 0.2047 |
| time_h20 | 5.0 | kairos_64code | 0.1251 |
| time_h20 | 10.0 | lightgbm_lob | -0.0870 |
| time_h50 | 0.0 | lightgbm_lob | 0.4539 |
| time_h50 | 1.0 | histgbm_tokens | 0.1408 |
| time_h50 | 3.0 | kairos_64code | -0.2072 |
| time_h50 | 5.0 | kairos_64code | -0.2168 |
| time_h50 | 10.0 | kairos_64code | -0.2409 |
| volume_h5 | 0.0 | lightgbm_lob | 3.3080 |
| volume_h5 | 1.0 | kairos_64code | 0.1026 |
| volume_h5 | 3.0 | kairos_64code | 0.0694 |
| volume_h5 | 5.0 | kairos_64code | 0.0363 |
| volume_h5 | 10.0 | kairos_64code | -0.0465 |
| volume_h20 | 0.0 | lightgbm_lob | 1.9110 |
| volume_h20 | 1.0 | kairos_64code | 0.0405 |
| volume_h20 | 3.0 | lightgbm_lob | -0.0723 |
| volume_h20 | 5.0 | lightgbm_lob | -0.0809 |
| volume_h20 | 10.0 | kairos_64code | -3.8184 |
| volume_h50 | 0.0 | lightgbm_lob | 2.2444 |
| volume_h50 | 1.0 | lightgbm_lob | 0.5186 |
| volume_h50 | 3.0 | kairos_64code | 0.0562 |
| volume_h50 | 5.0 | kairos_64code | -0.3574 |
| volume_h50 | 10.0 | histgbm_tokens | -0.2692 |

## Notes
- This is a normalized aggregation pass from current artifacts, not a rerun.
- `volume_h20` uses the current protocol-clean seq96 compare-root report.
- If the paper keeps 3 bps as headline, these tables should remain appendix/supporting evidence unless the main text is revised.
