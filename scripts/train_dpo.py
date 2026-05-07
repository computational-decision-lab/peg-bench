#!/usr/bin/env python3
"""Train the continuous DPO policy."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.dpo_trainer import cli_main


if __name__ == "__main__":
    cli_main()
