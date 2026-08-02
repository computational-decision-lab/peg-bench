# PEG-Bench Public Research Release

> Research site: [computational-decision-lab.github.io/peg-bench](https://computational-decision-lab.github.io/peg-bench/)

PEG-Bench is a benchmark protocol for testing when offline prediction rankings stop supporting execution-aware utility conclusions. Start with the [research site](https://computational-decision-lab.github.io/peg-bench/) for the structured technical report, six-panel evidence, uncertainty notes, and public release boundary.

This public research release provides the benchmark protocol, evaluator-facing artifacts, normalized evidence, and table/figure reproduction assets while excluding the underlying commercial market-data payloads and benchmark-ready processed arrays derived from them.

## Intended positioning
PEG-Bench is a benchmark protocol and executable evaluation artifact over licensed third-party market data. This release publishes:
- evaluator and code needed to inspect the benchmark contract
- frozen split/protocol/stress manifests
- normalized evidence files used by the paper
- result and table reproduction scripts
- paper tables and auditable completion bundles

This release does **not** publish:
- raw vendor L2 market data
- full processed benchmark tensors/arrays reconstructed from licensed raw data

## Why this release shape
This shape minimizes legal and licensing risk while preserving artifact auditability and paper-result reproducibility. Full raw-data reconstruction requires authorized access to the underlying market-data provider.

## Directory map
- docs/: static GitHub Pages research site
- `artifact_snapshot/release/`: normalized 90D evidence snapshot used by the paper artifact
- `artifact_snapshot/scripts/`: scripts for reproducing tables from normalized evidence
- `metadata/`: artifact/source manifests
- `paper_tables/generated_20260503/`: generated table CSV/TEX/MD files
- `review_bundle/`: supplemental evidence bundle
- `scripts/`, `configs/`, `src/`, `tests/`: code required to inspect and run the benchmark tooling

## Reproducibility boundary
Users can reproduce the paper tables/figures from the normalized evidence included here. Full end-to-end raw-data reconstruction is intentionally out of scope for this public release and requires separately authorized access to the underlying market-data provider.

## Quick links
- [Research site](https://computational-decision-lab.github.io/peg-bench/)
- [Technical report](https://computational-decision-lab.github.io/peg-bench/assets/peg-bench-technical-report.pdf)
- [Paper and evidence map](paper_tables/generated_20260503/)
- [Normalized 90D evidence](artifact_snapshot/release/)
- [Reproduction scripts](artifact_snapshot/scripts/)

GitHub Pages is enabled for this repository with the GitHub Actions source. A fork must enable Pages from repository settings before its first deployment; subsequent pushes to main use the Pages workflow.
