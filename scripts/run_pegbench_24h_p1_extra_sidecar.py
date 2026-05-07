#!/usr/bin/env python3
from pathlib import Path
import importlib.util
ROOT=Path("/root/prediction-execution-gap")
OUT=ROOT/"outputs/pegbench_24h_unattended_v21"
SCRIPT=ROOT/"scripts/pegbench_24h_p1_sidecar_worker.py"
spec=importlib.util.spec_from_file_location("p1", SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
board=mod.Board(OUT)
for task in [mod.task_hft_raw_l2, mod.task_ppo_sac_smoke_and_adapter, mod.task_raw_reconstruction_seed_action_bootstrap]:
    try:
        task(board)
    except Exception as exc:
        board.blocker(task.__name__, "extra_sidecar_exception", repr(exc), [("extra sidecar wrapper", "exception caught, continuing")], "failed_after_resolution_attempt_exception", str(OUT/"p1_sidecar"))
print(OUT/"p1_sidecar")
