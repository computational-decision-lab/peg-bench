# Release Scope

## Public release shape
This package is the recommended public release shape for PEG-Bench under licensed third-party market-data constraints.

Released:
- benchmark protocol-facing code and configs
- normalized evidence used by the paper
- manifest and source metadata
- reproduction scripts for tables and figures
- reviewer-oriented supplemental artifact tables and audits

Not released:
- raw vendor market data
- full processed benchmark tensors reconstructed from licensed raw data
- any claim that this package alone can rebuild the raw L2 archive

## Intended paper wording
PEG-Bench is a benchmark protocol and executable evaluation artifact over licensed third-party market data. We release the evaluator, splits, schemas, execution contract, normalized evidence, result reproduction scripts, and review/sample artifacts. Full raw-data reconstruction requires authorized access to the underlying market-data provider.

## Practical implication
A reader can audit the benchmark contract and reproduce paper tables/figures from the normalized evidence in this package. A reader cannot reconstruct the underlying raw market data or the full processed training tensors from this public artifact alone.
