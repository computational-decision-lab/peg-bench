#!/usr/bin/env bash
set +e
cd /root/prediction-execution-gap || exit 0
OUT="outputs/pegbench_24h_unattended_v21"
mkdir -p "$OUT"
find "$OUT" -type f ! -name sha256_manifest.txt -print0 | xargs -0 sha256sum > "$OUT/sha256_manifest.txt" 2>/dev/null
( cd outputs && tar -czf pegbench_24h_unattended_v21_latest.tar.gz pegbench_24h_unattended_v21 ) 2>/dev/null
