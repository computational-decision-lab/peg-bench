# PEG-Bench Public Research Release

> Research site: [computational-decision-lab.github.io/peg-bench](https://computational-decision-lab.github.io/peg-bench/)

PEG-Bench audits predictive rankings against post-cost proxy-score rankings in cryptocurrency limit-order-book data. The evaluator applies mapped exposures to historical forward-return labels and subtracts fee/impact proxies. The [research site](https://computational-decision-lab.github.io/peg-bench/) presents the released summaries, protocol, uncertainty notes, and public release boundary; the arXiv revision uses the proxy interpretation described below.

This public research release provides the benchmark protocol, evaluator-facing artifacts, normalized evidence, and table/figure reproduction assets while excluding the underlying commercial market-data payloads and benchmark-ready processed arrays derived from them.

## Intended positioning
PEG-Bench provides an evaluation protocol, reference evaluator, and result-level audit evidence over licensed third-party market data. This release publishes:
- evaluator and code needed to inspect the benchmark contract
- frozen split/protocol/stress manifests
- normalized evidence files used by the paper
- historical evaluation scripts and table-source maps
- paper tables and auditable completion bundles

This release does **not** publish:
- raw vendor L2 market data
- full processed benchmark tensors/arrays reconstructed from licensed raw data
- matched historical validation/test forecasts and complete per-run configurations needed for path replay

## What the current scores measure
The post-cost proxy contribution is `position * h-step forward log-return - fee/impact proxy * absolute position change`. The implementation does not reconcile overlapping forward labels into a portfolio holding schedule. Its impact-volatility proxy also includes the current forward target, rather than using only information available at the decision time.

The historical `sharpe` field scales the mean-to-population-standard-deviation ratio by the square root of a capped sample count. It is reported as a **post-cost proxy score**, not an annualized portfolio Sharpe ratio. Historical code and CSV field names remain unchanged.

The main report card averages segment scores within each date and then across dates. The separate fixed-forecast mapping audit scores concatenated splits without those boundary resets, so its values cannot replace headline ranking scores. Ordering changes are descriptive: individual score intervals do not establish paired differences between models or statistically supported profitability.

## Why this release shape
This shape preserves result-level inspection without redistributing licensed market feeds. Access to the underlying market-data provider is necessary for raw-data reconstruction; historical replay also requires the absent forecast paths and complete run configurations.

## Directory map
- docs/: static GitHub Pages research site
- `artifact_snapshot/release/`: normalized 90D evidence snapshot used by the paper artifact
- `artifact_snapshot/scripts/`: archived table-source-map script
- `metadata/`: artifact/source manifests
- `paper_tables/generated_20260503/`: generated table CSV/TEX/MD files
- `review_bundle/`: supplemental evidence bundle
- `scripts/`, `configs/`, `src/`, `tests/`: code required to inspect and run the benchmark tooling

## Reproducibility boundary
Readers can inspect reported values in the normalized evidence and archived tables. The public archive supports result-level auditing, but lacks the matched forecast paths needed for historical replay, unified portfolio accounting, and common-day paired inference.

`python3 artifact_snapshot/scripts/reproduce_tables.py` prints the archived source map using original-machine paths; it does not regenerate the tables. Other historical evaluation scripts require inputs beyond this public archive. The arXiv source package includes a separate ancillary archive with pinned evidence and standalone table-derivation and accounting checks; its README specifies those commands and their scope.

## Quick links
- [Research site](https://computational-decision-lab.github.io/peg-bench/)
- [Technical report](https://computational-decision-lab.github.io/peg-bench/assets/peg-bench-technical-report.pdf)
- [Paper and evidence map](paper_tables/generated_20260503/)
- [Normalized 90D evidence](artifact_snapshot/release/)
- [Archived table-source map](artifact_snapshot/scripts/reproduce_tables.py)
- [Evidence mirror (access-gated)](https://huggingface.co/datasets/colin-wang-research/peg-bench): the dataset listing is public; file downloads require an approved, authenticated Hugging Face account.

GitHub Pages is enabled for this repository with the GitHub Actions source. A fork must enable Pages from repository settings before its first deployment; subsequent pushes to main use the Pages workflow.
