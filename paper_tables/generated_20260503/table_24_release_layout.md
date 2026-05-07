# Table 24

Canonical archival release layout.

| Release component | Location | Contents |
| --- | --- | --- |
| Root metadata | artifact root | README.md, VERSION, paper table source map, stage conclusion |
| Machine-readable metadata | metadata/ | artifact_manifest.json, source_index.json, checksums.sha256 |
| Split manifests | artifact_snapshot/release/ | copied manifest.json, panel summaries, and split CSVs |
| Reports and evidence | artifact_snapshot/release/ | report_card_90d.*, leaderboards, gap metrics, stress report, wrapper and OG-PA summaries |
| Generated paper tables | paper_tables/generated_20260503/ | markdown, CSV, and TeX snippets for tables generated in this pass |
| Reproduction scripts | artifact_snapshot/scripts/ | reproduce_tables.py and the source-map driven generation path |
