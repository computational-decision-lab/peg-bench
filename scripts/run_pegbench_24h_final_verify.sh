#!/usr/bin/env bash
set +e
OUT="outputs/pegbench_24h_unattended_v21"
missing=0
for f in final_24h_completion_report.md final_remaining_blockers.csv pdf_fill_values_after_24h.csv sha256_manifest.txt; do
  [ -s "$OUT/$f" ] || missing=$((missing+1))
done
exit $missing
