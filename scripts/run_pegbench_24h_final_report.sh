#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
OUT="outputs/pegbench_24h_unattended_v21"
mkdir -p "$OUT"
python scripts/pegbench_24h_final_report.py --output-root "$OUT" >> "$OUT/logs/final_report_py.log" 2>&1
find "$OUT" -type f ! -name sha256_manifest.txt -print0 | xargs -0 sha256sum > "$OUT/sha256_manifest.txt" 2>/dev/null
