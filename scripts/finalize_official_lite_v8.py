from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/prediction-execution-gap")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import rerun_motivating_slice_official_lite_v8 as lite
from src.eval.benchmark_baselines import _predict_kairos


def _install_markdown_fallback() -> None:
    def to_markdown_fallback(self: pd.DataFrame, *args, **kwargs) -> str:
        return self.to_csv(index=not bool(kwargs.get("index") is False))

    pd.DataFrame.to_markdown = to_markdown_fallback  # type: ignore[method-assign]


def main() -> None:
    _install_markdown_fallback()
    out = lite.OFF / "volume_h50_official_lite"
    method_root = out / "methods/kairos_64code"
    checkpoint = method_root / "checkpoints/best_baseline.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    specs = lite.build_specs("volume", 50, 50)
    encoded = {
        "train": out / "vqvae_64code/encoded_tokens.npz",
        "val": out / "vqvae_64code/encoded_val.npz",
        "test": out / "vqvae_64code/encoded_test.npz",
    }
    for split, path in encoded.items():
        if not path.exists():
            raise FileNotFoundError(path)
        specs[split].encoded_path = str(path)

    predictions = {}
    targets = {}
    for split in ["train", "val", "test"]:
        pred, y = _predict_kairos(
            checkpoint_path=checkpoint,
            encoded_path=specs[split].encoded_path,
            panel_path=specs[split].panel_path,
            seq_len=specs[split].seq_len,
            stride=specs[split].stride,
            target_key=specs[split].target_key,
            device="cuda",
            batch_size=2048,
        )
        predictions[split] = pred
        targets[split] = y

    method_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        method_root / "predictions.npz",
        train_predictions=predictions["train"],
        val_predictions=predictions["val"],
        test_predictions=predictions["test"],
        train_targets=targets["train"],
        val_targets=targets["val"],
        test_targets=targets["test"],
    )
    summary = {
        "method": "kairos_64code",
        "checkpoint_path": str(checkpoint.resolve()),
        "notes": [
            "Official Kairos trainer; epochs=1 lite v8 four-date rerun.",
            "Daily report generation skipped for stride-50 finalization because the upstream report helper indexes stride-50 predictions with stride-1 day masks.",
        ],
        "train_metrics": lite.metrics(predictions["train"], targets["train"]),
        "val_metrics": lite.metrics(predictions["val"], targets["val"]),
        "test_metrics": lite.metrics(predictions["test"], targets["test"]),
        "report_paths": [],
    }
    (method_root / "summary.json").write_text(json.dumps(summary, indent=2))

    lite.update_tables()
    print(json.dumps({"status": "finalized_h50_kairos_and_tables", "out": str(lite.OUT)}, indent=2))


if __name__ == "__main__":
    main()
