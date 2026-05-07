#!/usr/bin/env python3
"""Run non-GPU P1 dependency/source/audit sidecar tasks without waiting for DPO."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path("/root/prediction-execution-gap")
OUT = ROOT / "outputs/pegbench_24h_unattended_v21"
SCRIPT = ROOT / "scripts/pegbench_24h_p1_sidecar_worker.py"

spec = importlib.util.spec_from_file_location("p1_sidecar_worker", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

board = module.Board(OUT)
tasks = [
    module.task_timesfm_kronos_official,
    module.task_official_repo_resolution,
    module.task_hft_raw_l2,
    module.task_ppo_sac_smoke_and_adapter,
]

for task in tasks:
    try:
        task(board)
    except Exception as exc:
        board.blocker(
            task.__name__,
            "dependency_sidecar_exception",
            repr(exc),
            [("dependency sidecar wrapper", "exception caught, continuing")],
            "failed_after_resolution_attempt_exception",
            str(OUT / "p1_sidecar"),
        )

print(OUT / "p1_sidecar")
