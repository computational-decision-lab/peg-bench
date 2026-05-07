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
from src.eval.metrics import directional_accuracy, rank_ic

OUT = ROOT / "outputs/final_motivating_slice_rerun_v9_time_official_lite"


def install_markdown_fallback() -> None:
    def to_markdown_fallback(self: pd.DataFrame, *args, **kwargs) -> str:
        return self.to_csv(index=not bool(kwargs.get("index") is False))

    pd.DataFrame.to_markdown = to_markdown_fallback  # type: ignore[method-assign]


def load_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["test_predictions"], z["test_targets"]


def finalize_kairos(root: Path, horizon: int, stride: int) -> dict:
    method_root = root / "methods/kairos_64code"
    pred_path = method_root / "predictions.npz"
    summary_path = method_root / "summary.json"
    if pred_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text())
    checkpoint = method_root / "checkpoints/best_baseline.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    specs = lite.build_specs("time", horizon, stride)
    encoded = {
        "train": root / "vqvae_64code/encoded_tokens.npz",
        "val": root / "vqvae_64code/encoded_val.npz",
        "test": root / "vqvae_64code/encoded_test.npz",
    }
    for split, path in encoded.items():
        specs[split].encoded_path = str(path)
    preds = {}
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
        preds[split] = pred
        targets[split] = y
    method_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_path,
        train_predictions=preds["train"],
        val_predictions=preds["val"],
        test_predictions=preds["test"],
        train_targets=targets["train"],
        val_targets=targets["val"],
        test_targets=targets["test"],
    )
    summary = {
        "method": "kairos_64code",
        "checkpoint_path": str(checkpoint.resolve()),
        "notes": [
            "v9 time official-lite Kairos.",
            "Daily reports skipped if upstream helper fails on stride/mask alignment; prediction arrays are direct dataset outputs.",
        ],
        "train_metrics": lite.metrics(preds["train"], targets["train"]),
        "val_metrics": lite.metrics(preds["val"], targets["val"]),
        "test_metrics": lite.metrics(preds["test"], targets["test"]),
        "report_paths": [],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_setting(name: str, horizon: int, stride: int) -> None:
    root = lite.OFF / f"{name}_official_lite"
    try:
        lite.run_setting(name, "time", horizon, stride)
    except Exception as exc:
        # The upstream helper can fail during daily report assembly after training.
        # Preserve completed method outputs and fill Kairos prediction arrays directly.
        (OUT / f"{name}_upstream_exception.txt").write_text(repr(exc))
        finalize_kairos(root, horizon, stride)
    rows = []
    for method in ["ridge_tokens", "histgbm_tokens", "lightgbm_lob", "kairos_64code"]:
        p, y = load_pred(root / "methods" / method / "predictions.npz")
        rows.append(
            {
                "setting": name,
                "method": method,
                "test_directional_accuracy": float(directional_accuracy(p, y)),
                "test_rank_ic": float(rank_ic(p, y)),
                "sample_count": int(len(y)),
                "source_file": str(root / "methods" / method / "predictions.npz"),
                "status": "completed_protocol_different",
            }
        )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}_results.csv", index=False)
    (OUT / f"{name}_results.md").write_text(df.to_csv(index=False))


def main() -> None:
    install_markdown_fallback()
    OUT.mkdir(parents=True, exist_ok=True)
    run_setting("time_h5", 5, 1)
    run_setting("time_h50", 50, 50)
    frames = [pd.read_csv(OUT / "time_h5_results.csv"), pd.read_csv(OUT / "time_h50_results.csv")]
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT / "time_official_lite_results.csv", index=False)
    (OUT / "time_official_lite_results.md").write_text(all_df.to_csv(index=False))
    (OUT / "final_status.md").write_text(
        "# Time Official-Lite v9 Status\n\n"
        "Completed supplementary time h5/h50 official-lite rows for Ridge, HistGBM, LightGBM, and Kairos. "
        "These are supplementary protocol-different rows, not exact historical reproductions.\n"
    )
    print(json.dumps({"status": "time_official_lite_completed", "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
