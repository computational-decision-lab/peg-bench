# PEG-Bench Safe Release Candidate

This release candidate is designed for the lowest-risk public distribution path for a licensed-data benchmark submission. It releases the benchmark protocol, evaluator-facing artifacts, normalized evidence, and table/figure reproduction assets, while excluding the underlying commercial market-data payloads and benchmark-ready processed arrays derived from them.

## Intended positioning
PEG-Bench is a benchmark protocol and executable evaluation artifact over licensed third-party market data. This release candidate publishes:
- evaluator and code needed to inspect the benchmark contract
- frozen split/protocol/stress manifests
- normalized evidence files used by the paper
- result and table reproduction scripts
- paper tables and auditable completion bundles

This release candidate does **not** publish:
- raw vendor L2 market data
- full processed benchmark tensors/arrays reconstructed from licensed raw data

## Why this release shape
This shape minimizes legal and licensing risk while preserving artifact auditability and paper-result reproducibility. Full raw-data reconstruction requires authorized access to the underlying market-data provider.

## Directory map
- `artifact_snapshot/release/`: normalized 90D evidence snapshot used by the paper artifact
- `artifact_snapshot/scripts/`: scripts for reproducing tables from normalized evidence
- `metadata/`: artifact/source manifests
- `paper_tables/generated_20260503/`: generated table CSV/TEX/MD files
- `review_bundle/`: reviewer-oriented supplemental evidence bundle
- `scripts/`, `configs/`, `src/`, `tests/`: code required to inspect and run the benchmark tooling

## Reproducibility boundary
Users can reproduce the paper tables/figures from the normalized evidence included here. Full end-to-end raw-data reconstruction is intentionally out of scope for this public release and requires separately authorized access to the underlying market-data provider.
