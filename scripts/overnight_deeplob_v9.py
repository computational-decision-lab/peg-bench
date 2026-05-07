from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/root/prediction-execution-gap")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.benchmark_baselines import PanelSpec, ensure_window_index
from src.eval.metrics import directional_accuracy, rank_ic
from src.models.deeplob import DeepLOBConfig
from src.train.deeplob_trainer import DeepLOBTrainConfig, predict_deeplob, train_deeplob

OUT = ROOT / "outputs/final_motivating_slice_rerun_v9"
V8 = ROOT / "outputs/final_motivating_slice_rerun_v8"
OFF = V8 / "official_four_date"


def panel_spec(mode: str, split: str, horizon: int, stride: int, seq_len: int) -> PanelSpec:
    panel = OFF / mode / "panels" / f"{mode}_{split}.npz"
    index = OUT / "manifests" / f"{mode}_{split}_seq{seq_len}_h{horizon}_stride{stride}.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    ensure_window_index(
        panel_path=panel,
        output_path=index,
        seq_len=seq_len,
        stride=stride,
        target_key=f"return_{horizon}",
    )
    return PanelSpec(
        panel_path=str(panel.resolve()),
        index_path=str(index.resolve()),
        split=split,
        seq_len=seq_len,
        stride=stride,
        target_key=f"return_{horizon}",
    )


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "directional_accuracy": float(directional_accuracy(pred, y)),
        "rank_ic": float(rank_ic(pred, y)),
        "mse": float(np.mean((pred - y) ** 2)) if len(y) else float("nan"),
        "mae": float(np.mean(np.abs(pred - y))) if len(y) else float("nan"),
    }


def run_deeplob_setting(name: str, horizon: int, stride: int, epochs: int = 3) -> dict[str, Any]:
    seq_len = 50
    root = OUT / "official_four_date" / name / "methods" / "deeplob_timebars"
    ckpt_dir = root / "checkpoints"
    specs = {split: panel_spec("time", split, horizon, stride, seq_len) for split in ["train", "val", "test"]}
    best = ckpt_dir / "best_deeplob.pt"
    if not best.exists():
        train_summary = train_deeplob(
            DeepLOBTrainConfig(
                train_dataset_path=specs["train"].panel_path,
                val_dataset_path=specs["val"].panel_path,
                test_dataset_path=specs["test"].panel_path,
                output_dir=str(ckpt_dir),
                seq_len=seq_len,
                stride=stride,
                target_key=f"return_{horizon}",
                batch_size=512,
                epochs=epochs,
                device="cuda",
                amp=True,
                seed=7,
                model=DeepLOBConfig(seq_len=seq_len),
            )
        )
        (root / "train_summary.json").write_text(json.dumps(train_summary, indent=2))
        ckpt = str(train_summary["best_checkpoint"])
    else:
        ckpt = str(best.resolve())

    preds: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        pred, y = predict_deeplob(
            checkpoint_path=ckpt,
            dataset_path=specs[split].panel_path,
            seq_len=seq_len,
            stride=stride,
            target_key=f"return_{horizon}",
            device="cuda",
            batch_size=4096,
        )
        preds[split] = pred
        targets[split] = y

    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "predictions.npz",
        train_predictions=preds["train"],
        val_predictions=preds["val"],
        test_predictions=preds["test"],
        train_targets=targets["train"],
        val_targets=targets["val"],
        test_targets=targets["test"],
    )
    summary = {
        "method": "deeplob_timebars",
        "checkpoint_path": ckpt,
        "setting": name,
        "mode": "time",
        "horizon": horizon,
        "stride": stride,
        "seq_len": seq_len,
        "notes": [
            "v9 overnight official DeepLOB rerun on four-date time panels.",
            "Daily report generation intentionally skipped; predictions and targets are aligned directly by the DeepLOB dataset iterator.",
        ],
        "train_metrics": metrics(preds["train"], targets["train"]),
        "val_metrics": metrics(preds["val"], targets["val"]),
        "test_metrics": metrics(preds["test"], targets["test"]),
        "report_paths": [],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["test_predictions"], z["test_targets"]


def pnl_metrics(pred: np.ndarray, y: np.ndarray, bps: float) -> tuple[float, float, float]:
    pos = np.sign(pred)
    turn = np.abs(np.diff(pos, prepend=0))
    pnl = pos * y - bps * 1e-4 * turn
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    bars_per_day = max(len(y), 1)
    sharpe = float(mean / (std + 1e-12) * math.sqrt(bars_per_day * 365))
    annual = float(mean * bars_per_day * 365)
    return mean, sharpe, annual


def update_v9_tables() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h5 = V8 / "official_four_date/volume_h5_official_lite"
    h50 = V8 / "official_four_date/volume_h50_official_lite"
    d5 = OUT / "official_four_date/time_h5_deeplob_official/methods/deeplob_timebars/predictions.npz"
    d50 = OUT / "official_four_date/time_h50_deeplob_official/methods/deeplob_timebars/predictions.npz"
    hist = [
        ("Overlap-allowing", "Ridge (tokens)", 0.624, 0.306, h5 / "methods/ridge_tokens/predictions.npz", "official-lite"),
        ("Overlap-allowing", "HistGBM (tokens)", 0.691, 0.586, h5 / "methods/histgbm_tokens/predictions.npz", "official-lite"),
        ("Overlap-allowing", "LightGBM (LOB)", 0.789, 0.696, h5 / "methods/lightgbm_lob/predictions.npz", "official-lite"),
        ("Overlap-allowing", "Kairos (64-code)", 0.594, 0.146, h5 / "methods/kairos_64code/predictions.npz", "official-lite"),
        ("Overlap-allowing", "DeepLOB (time bars)", 0.506, 0.027, d5, "official-deeplob"),
        ("Leakage-controlled", "Ridge (tokens)", 0.479, 0.048, h50 / "methods/ridge_tokens/predictions.npz", "official-lite"),
        ("Leakage-controlled", "HistGBM (tokens)", 0.090, 0.519, h50 / "methods/histgbm_tokens/predictions.npz", "official-lite"),
        ("Leakage-controlled", "LightGBM (LOB)", 0.648, 0.262, h50 / "methods/lightgbm_lob/predictions.npz", "official-lite"),
        ("Leakage-controlled", "Kairos (64-code)", 0.618, 0.328, h50 / "methods/kairos_64code/predictions.npz", "official-lite"),
        ("Leakage-controlled", "DeepLOB (time bars)", 0.506, 0.027, d50, "official-deeplob"),
    ]
    rows = []
    for setting, label, hda, hric, path, source_kind in hist:
        pred, y = load_pred(path)
        rows.append(
            {
                "audit_setting": setting,
                "model": label,
                "rerun_DA": float(directional_accuracy(pred, y)),
                "rerun_RIC": float(rank_ic(pred, y)),
                "historical_DA": hda,
                "historical_RIC": hric,
                "sample_count": int(len(y)),
                "source_file": str(path),
                "diff_status": "protocol_different",
                "protocol_caveat": f"v9_four_date_rerun; {source_kind}; original normalized motivating-slice artifact unavailable",
            }
        )
    t6 = pd.DataFrame(rows)
    t6.to_csv(OUT / "table6.csv", index=False)
    (OUT / "table6.md").write_text(t6.to_csv(index=False))
    (OUT / "table6.tex").write_text(t6.to_latex(index=False, escape=True))

    kairos_pred, kairos_y = load_pred(h50 / "methods/kairos_64code/predictions.npz")
    t7_rows = []
    for bps, hmean, hsh, hann in [(3, -5e-6, -98.6, -1.71), (1, -1e-6, -70.0, -0.39)]:
        mean, sh, ann = pnl_metrics(kairos_pred, kairos_y, bps)
        t7_rows.append(
            {
                "cost_bps": bps,
                "rerun_mean_pnl_per_bar": mean,
                "rerun_sharpe": sh,
                "rerun_annual_return": ann,
                "historical_mean_pnl_per_bar": hmean,
                "historical_sharpe": hsh,
                "historical_annual_return": hann,
                "source_file": str(h50 / "methods/kairos_64code/predictions.npz"),
                "diff_status": "protocol_different",
                "protocol_caveat": "v9 inherits official-lite Kairos action-sign diagnostic; original action path unavailable",
            }
        )
    t7 = pd.DataFrame(t7_rows)
    t7.to_csv(OUT / "table7.csv", index=False)
    (OUT / "table7.md").write_text(t7.to_csv(index=False))
    (OUT / "table7.tex").write_text(t7.to_latex(index=False, escape=True))

    p_lgbm1, y_lgbm1 = load_pred(h5 / "methods/lightgbm_lob/predictions.npz")
    p_lgbm50, y_lgbm50 = load_pred(h50 / "methods/lightgbm_lob/predictions.npz")
    p_kairos, y_kairos = load_pred(h50 / "methods/kairos_64code/predictions.npz")
    p_hist, y_hist = load_pred(h50 / "methods/histgbm_tokens/predictions.npz")
    t14 = pd.DataFrame(
        [
            ["Tokenizer", "512 codes", float(rank_ic(p_hist, y_hist)), 0.012, len(y_hist), str(h50 / "methods/histgbm_tokens/predictions.npz"), "protocol_different", "v9 inherits official-lite token-frequency proxy, not original 512-code artifact"],
            ["Tokenizer", "64 codes", float(rank_ic(p_kairos, y_kairos)), 0.146, len(y_kairos), str(h50 / "methods/kairos_64code/predictions.npz"), "protocol_different", "v9 inherits official-lite 64-code Kairos, reduced epoch"],
            ["LightGBM stride", "1", float(rank_ic(p_lgbm1, y_lgbm1)), 0.696, len(y_lgbm1), str(h5 / "methods/lightgbm_lob/predictions.npz"), "protocol_different", "v9 inherits official-lite LightGBM stride=1"],
            ["LightGBM stride", "50", float(rank_ic(p_lgbm50, y_lgbm50)), 0.262, len(y_lgbm50), str(h50 / "methods/lightgbm_lob/predictions.npz"), "protocol_different", "v9 inherits official-lite LightGBM stride=50"],
            ["Kairos", "stride=50", float(rank_ic(p_kairos, y_kairos)), 0.328, len(y_kairos), str(h50 / "methods/kairos_64code/predictions.npz"), "protocol_different", "v9 inherits official-lite Kairos, reduced epoch"],
        ],
        columns=["audit", "config", "rerun_RIC", "historical_RIC", "sample_count", "source_file", "diff_status", "protocol_caveat"],
    )
    t14.to_csv(OUT / "table14.csv", index=False)
    (OUT / "table14.md").write_text(t14.to_csv(index=False))
    (OUT / "table14.tex").write_text(t14.to_latex(index=False, escape=True))

    _, kairos_sh, _ = pnl_metrics(p_kairos, y_kairos, 3)
    _, lgbm_sh, _ = pnl_metrics(p_lgbm50, y_lgbm50, 3)
    t15 = pd.DataFrame(
        [
            ["Kairos", float(directional_accuracy(p_kairos, y_kairos) * 100), float(rank_ic(p_kairos, y_kairos)), kairos_sh, 61.8, 0.328, -98.6, len(y_kairos), str(h50 / "methods/kairos_64code/predictions.npz"), "protocol_different", "v9 inherits official-lite Kairos; original normalized motivating-slice artifact unavailable"],
            ["LightGBM", float(directional_accuracy(p_lgbm50, y_lgbm50) * 100), float(rank_ic(p_lgbm50, y_lgbm50)), lgbm_sh, 64.8, 0.262, -112.3, len(y_lgbm50), str(h50 / "methods/lightgbm_lob/predictions.npz"), "protocol_different", "v9 inherits official-lite LightGBM; original normalized motivating-slice artifact unavailable"],
        ],
        columns=["model", "rerun_DA_pct", "rerun_RIC", "rerun_sharpe_3bps", "historical_DA_pct", "historical_RIC", "historical_sharpe_3bps", "sample_count", "source_file", "diff_status", "protocol_caveat"],
    )
    t15.to_csv(OUT / "table15.csv", index=False)
    (OUT / "table15.md").write_text(t15.to_csv(index=False))
    (OUT / "table15.tex").write_text(t15.to_latex(index=False, escape=True))

    status = {
        "status": "completed",
        "what_changed_from_v8": "DeepLOB compact proxy rows replaced by official DeepLOB reruns on four-date time panels.",
        "rows": {"table6": len(t6), "table7": len(t7), "table14": len(t14), "table15": len(t15)},
        "caveat": "All rows remain protocol_different; original normalized motivating-slice artifact unavailable.",
    }
    (OUT / "final_rerun_status.md").write_text("# V9 Final Rerun Status\n\n" + json.dumps(status, indent=2) + "\n")
    (OUT / "motivating_slice_diff_summary.md").write_text("# V9 Motivating Slice Diff Summary\n\n- DeepLOB rows upgraded from compact proxy to official DeepLOB rerun.\n- All rows remain `protocol_different`.\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_deeplob_setting("time_h5_deeplob_official", horizon=5, stride=1, epochs=3),
        run_deeplob_setting("time_h50_deeplob_official", horizon=50, stride=50, epochs=3),
    ]
    (OUT / "deeplob_run_summaries.json").write_text(json.dumps(summaries, indent=2))
    update_v9_tables()
    print(json.dumps({"status": "v9_deeplob_completed", "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
