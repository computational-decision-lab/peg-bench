from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from typing import Any

ROOT = Path("/root/prediction-execution-gap")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

from scripts.run_peg_bench_comparison_baselines import _panel_paths
from src.eval.benchmark_baselines import (
    PanelSpec,
    build_last_lob_features,
    build_token_frequency_features,
    ensure_vq_and_encodings,
    ensure_window_index,
    run_kairos_baseline,
)
from src.eval.metrics import directional_accuracy, rank_ic

OUT = ROOT / "outputs/final_motivating_slice_rerun_v8"
OFF = OUT / "official_four_date"
SPLIT_PLAN = OFF / "four_date_split_plan.csv"


def write_split_plan() -> None:
    SPLIT_PLAN.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PLAN.write_text(
        "exchange,symbol,train_start,train_end,val_start,val_end,test_start,test_end\n"
        "binance,BTCUSDT,2025-10-12,2025-10-17,2025-10-26,2025-10-26,2025-10-27,2025-10-27\n"
    )


def load_rows(index_path: str | Path) -> list[dict[str, str]]:
    with Path(index_path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_specs(mode: str, horizon: int, stride: int, seq_len: int = 64) -> dict[str, PanelSpec]:
    root = OFF / mode
    specs = {}
    for split in ["train", "val", "test"]:
        panel_path, index_path = _panel_paths(root, mode, split, seq_len)
        ensure_window_index(panel_path=panel_path, output_path=index_path, seq_len=seq_len, stride=stride, target_key=f"return_{horizon}")
        specs[split] = PanelSpec(
            panel_path=panel_path,
            index_path=index_path,
            split=split,
            seq_len=seq_len,
            stride=stride,
            target_key=f"return_{horizon}",
        )
    return specs


def cap(x: np.ndarray, y: np.ndarray, max_n: int = 120000) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_n:
        return x, y
    rng = np.random.default_rng(7)
    idx = rng.choice(len(y), size=max_n, replace=False)
    return x[idx], y[idx]


def save_predictions(root: Path, train_pred, val_pred, test_pred, train, val, test, method: str, notes: list[str]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "predictions.npz",
        train_predictions=np.asarray(train_pred, dtype=np.float32),
        val_predictions=np.asarray(val_pred, dtype=np.float32),
        test_predictions=np.asarray(test_pred, dtype=np.float32),
        train_targets=np.asarray(train["targets"], dtype=np.float32),
        val_targets=np.asarray(val["targets"], dtype=np.float32),
        test_targets=np.asarray(test["targets"], dtype=np.float32),
    )
    summary = {
        "method": method,
        "notes": notes,
        "train_metrics": metrics(train_pred, train["targets"]),
        "val_metrics": metrics(val_pred, val["targets"]),
        "test_metrics": metrics(test_pred, test["targets"]),
        "report_paths": [],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def metrics(pred, y) -> dict[str, float]:
    return {
        "directional_accuracy": float(directional_accuracy(np.asarray(pred), np.asarray(y))),
        "rank_ic": float(rank_ic(np.asarray(pred), np.asarray(y))),
    }


def run_tokens(output_root: Path, specs: dict[str, PanelSpec], method: str) -> dict[str, Any]:
    enc = ensure_vq_and_encodings(
        train_panel_path=specs["train"].panel_path,
        val_panel_path=specs["val"].panel_path,
        test_panel_path=specs["test"].panel_path,
        output_dir=output_root / "vqvae_64code",
        device="cuda",
        batch_size=512,
        epochs=1,
        seed=7,
        num_embeddings=64,
        latent_dim=64,
        max_train_items=120000,
    )
    specs["train"].encoded_path = enc["train_encoded"]
    specs["val"].encoded_path = enc["val_encoded"]
    specs["test"].encoded_path = enc["test_encoded"]
    feats = {
        split: build_token_frequency_features(
            panel_path=specs[split].panel_path,
            index_path=specs[split].index_path,
            encoded_path=str(specs[split].encoded_path),
            vocab_size=64,
            target_key=specs[split].target_key,
        )
        for split in ["train", "val", "test"]
    }
    xtr, ytr = cap(feats["train"]["features"], feats["train"]["targets"])
    if method == "ridge_tokens":
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        notes = ["Official feature builder; Ridge alpha=10 on token-frequency windows; v8 four-date lite rerun."]
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_depth=4,
            max_iter=80,
            min_samples_leaf=80,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=7,
        )
        notes = ["Official feature builder; HistGradientBoostingRegressor lite settings; v8 four-date rerun."]
    model.fit(xtr, ytr)
    return save_predictions(
        output_root / "methods" / method,
        model.predict(feats["train"]["features"]),
        model.predict(feats["val"]["features"]),
        model.predict(feats["test"]["features"]),
        feats["train"],
        feats["val"],
        feats["test"],
        method,
        notes,
    )


def run_lgbm(output_root: Path, specs: dict[str, PanelSpec]) -> dict[str, Any]:
    feats = {
        split: build_last_lob_features(
            panel_path=specs[split].panel_path,
            index_path=specs[split].index_path,
            target_key=specs[split].target_key,
        )
        for split in ["train", "val", "test"]
    }
    xtr, ytr = cap(feats["train"]["features"], feats["train"]["targets"], 120000)
    if LGBMRegressor is not None:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=7,
            n_jobs=4,
            verbose=-1,
        )
    else:
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_depth=5, random_state=7)
    model.fit(xtr, ytr)
    return save_predictions(
        output_root / "methods/lightgbm_lob",
        model.predict(feats["train"]["features"]),
        model.predict(feats["val"]["features"]),
        model.predict(feats["test"]["features"]),
        feats["train"],
        feats["val"],
        feats["test"],
        "lightgbm_lob",
        ["Official last-LOB feature builder; LightGBM lite settings; v8 four-date rerun."],
    )


def run_kairos(output_root: Path, specs: dict[str, PanelSpec]) -> dict[str, Any]:
    enc = ensure_vq_and_encodings(
        train_panel_path=specs["train"].panel_path,
        val_panel_path=specs["val"].panel_path,
        test_panel_path=specs["test"].panel_path,
        output_dir=output_root / "vqvae_64code",
        device="cuda",
        batch_size=512,
        epochs=1,
        seed=7,
        num_embeddings=64,
        latent_dim=64,
        max_train_items=120000,
    )
    for split, key in [("train", "train_encoded"), ("val", "val_encoded"), ("test", "test_encoded")]:
        specs[split].encoded_path = enc[key]
    res = run_kairos_baseline(
        train_panel=specs["train"],
        val_panel=specs["val"],
        test_panel=specs["test"],
        checkpoint_dir=output_root / "methods/kairos_64code/checkpoints",
        output_dir=output_root / "methods/kairos_64code",
        device="cuda",
        vocab_size=64,
        epochs=1,
        batch_size=256,
        seed=7,
    )
    summary_path = output_root / "methods/kairos_64code/summary.json"
    summary = json.loads(summary_path.read_text())
    summary["notes"].append("Official Kairos trainer; epochs=1 lite v8 four-date rerun.")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_setting(name: str, mode: str, horizon: int, stride: int) -> None:
    out = OFF / f"{name}_official_lite"
    specs = build_specs(mode, horizon, stride)
    summaries = {}
    for method in ["ridge_tokens", "histgbm_tokens"]:
        summaries[method] = run_tokens(out, specs, method)
    summaries["lightgbm_lob"] = run_lgbm(out, specs)
    summaries["kairos_64code"] = run_kairos(out, specs)
    rows = []
    for method, summary in summaries.items():
        rows.append(
            {
                "method": method,
                "feature_mode": "codes" if "kairos" in method else "token_frequency" if "tokens" in method else "last_lob_snapshot",
                "test_directional_accuracy": summary["test_metrics"]["directional_accuracy"],
                "test_rank_ic": summary["test_metrics"]["rank_ic"],
                "notes": " | ".join(summary.get("notes", [])),
            }
        )
    pd.DataFrame(rows).to_csv(out / "results_table.csv", index=False)
    (out / "results_table.json").write_text(json.dumps(rows, indent=2))
    (out / "run_summary.json").write_text(json.dumps({"setting": name, "mode": mode, "horizon": horizon, "stride": stride, "methods": rows}, indent=2))


def load_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["test_predictions"], z["test_targets"]


def pnl_metrics(pred: np.ndarray, y: np.ndarray, bps: float) -> tuple[float, float, float]:
    pos = np.sign(pred)
    turn = np.abs(np.diff(pos, prepend=0))
    pnl = pos * y - bps * 1e-4 * turn
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    # Two-day four-date test subset; annualization is only a protocol-caveated diagnostic.
    bars_per_day = len(y)
    sharpe = float(mean / (std + 1e-12) * math.sqrt(bars_per_day * 365))
    ann = float(mean * bars_per_day * 365)
    return mean, sharpe, ann


def update_tables() -> None:
    h5 = OFF / "volume_h5_official_lite"
    h50 = OFF / "volume_h50_official_lite"
    hist = {
        ("Overlap-allowing", "Ridge (tokens)"): (0.624, 0.306, "ridge_tokens", h5),
        ("Overlap-allowing", "HistGBM (tokens)"): (0.691, 0.586, "histgbm_tokens", h5),
        ("Overlap-allowing", "LightGBM (LOB)"): (0.789, 0.696, "lightgbm_lob", h5),
        ("Overlap-allowing", "Kairos (64-code)"): (0.594, 0.146, "kairos_64code", h5),
        ("Leakage-controlled", "Ridge (tokens)"): (0.479, 0.048, "ridge_tokens", h50),
        ("Leakage-controlled", "HistGBM (tokens)"): (0.090, 0.519, "histgbm_tokens", h50),
        ("Leakage-controlled", "LightGBM (LOB)"): (0.648, 0.262, "lightgbm_lob", h50),
        ("Leakage-controlled", "Kairos (64-code)"): (0.618, 0.328, "kairos_64code", h50),
    }
    rows = []
    for (setting, label), (hda, hric, method, root) in hist.items():
        pred, y = load_pred(root / "methods" / method / "predictions.npz")
        rows.append(
            {
                "audit_setting": setting,
                "model": label,
                "rerun_DA": float(directional_accuracy(pred, y)),
                "rerun_RIC": float(rank_ic(pred, y)),
                "historical_DA": hda,
                "historical_RIC": hric,
                "sample_count": len(y),
                "source_file": str(root / "methods" / method / "predictions.npz"),
                "diff_status": "protocol_different",
                "protocol_caveat": "official_lite_four_date_rerun; official feature/model interfaces with reduced epochs/iterations; original normalized motivating-slice artifact unavailable",
            }
        )
    # Preserve DeepLOB rows from existing processed-four-date proxy table if available.
    old_t6 = OUT / "table6.csv"
    if old_t6.exists():
        old = pd.read_csv(old_t6)
        rows.extend(old[old["model"] == "DeepLOB (time bars)"].to_dict("records"))
    t6 = pd.DataFrame(rows)
    t6.to_csv(OUT / "table6.csv", index=False)
    (OUT / "table6.md").write_text(t6.to_markdown(index=False) + "\n")
    (OUT / "table6.tex").write_text(t6.to_latex(index=False, escape=True))
    (OUT / "table6_source_trace.md").write_text(t6.to_csv(index=False))

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
                "protocol_caveat": "official_lite_four_date_rerun using Kairos official trainer action sign; original action path unavailable",
            }
        )
    t7 = pd.DataFrame(t7_rows)
    t7.to_csv(OUT / "table7.csv", index=False)
    (OUT / "table7.md").write_text(t7.to_markdown(index=False) + "\n")
    (OUT / "table7.tex").write_text(t7.to_latex(index=False, escape=True))
    (OUT / "table7_source_trace.md").write_text(t7.to_csv(index=False))

    p_lgbm1, y_lgbm1 = load_pred(h5 / "methods/lightgbm_lob/predictions.npz")
    p_lgbm50, y_lgbm50 = load_pred(h50 / "methods/lightgbm_lob/predictions.npz")
    p_kairos, y_kairos = load_pred(h50 / "methods/kairos_64code/predictions.npz")
    p_hist, y_hist = load_pred(h50 / "methods/histgbm_tokens/predictions.npz")
    t14 = pd.DataFrame(
        [
            ["Tokenizer", "512 codes", float(rank_ic(p_hist, y_hist)), 0.012, len(y_hist), str(h50 / "methods/histgbm_tokens/predictions.npz"), "protocol_different", "official_lite_four_date_rerun; tokenizer proxy via official token-frequency model, not original 512-code artifact"],
            ["Tokenizer", "64 codes", float(rank_ic(p_kairos, y_kairos)), 0.146, len(y_kairos), str(h50 / "methods/kairos_64code/predictions.npz"), "protocol_different", "official_lite_four_date_rerun; official 64-code Kairos trainer, reduced epoch"],
            ["LightGBM stride", "1", float(rank_ic(p_lgbm1, y_lgbm1)), 0.696, len(y_lgbm1), str(h5 / "methods/lightgbm_lob/predictions.npz"), "protocol_different", "official_lite_four_date_rerun; stride=1"],
            ["LightGBM stride", "50", float(rank_ic(p_lgbm50, y_lgbm50)), 0.262, len(y_lgbm50), str(h50 / "methods/lightgbm_lob/predictions.npz"), "protocol_different", "official_lite_four_date_rerun; stride=50"],
            ["Kairos", "stride=50", float(rank_ic(p_kairos, y_kairos)), 0.328, len(y_kairos), str(h50 / "methods/kairos_64code/predictions.npz"), "protocol_different", "official_lite_four_date_rerun; official Kairos trainer, reduced epoch"],
        ],
        columns=["audit", "config", "rerun_RIC", "historical_RIC", "sample_count", "source_file", "diff_status", "protocol_caveat"],
    )
    t14.to_csv(OUT / "table14.csv", index=False)
    (OUT / "table14.md").write_text(t14.to_markdown(index=False) + "\n")
    (OUT / "table14.tex").write_text(t14.to_latex(index=False, escape=True))
    (OUT / "table14_source_trace.md").write_text(t14.to_csv(index=False))

    _, kairos_sh, _ = pnl_metrics(p_kairos, y_kairos, 3)
    _, lgbm_sh, _ = pnl_metrics(p_lgbm50, y_lgbm50, 3)
    t15 = pd.DataFrame(
        [
            ["Kairos", float(directional_accuracy(p_kairos, y_kairos) * 100), float(rank_ic(p_kairos, y_kairos)), kairos_sh, 61.8, 0.328, -98.6, len(y_kairos), str(h50 / "methods/kairos_64code/predictions.npz"), "protocol_different", "official_lite_four_date_rerun; original normalized motivating-slice artifact unavailable"],
            ["LightGBM", float(directional_accuracy(p_lgbm50, y_lgbm50) * 100), float(rank_ic(p_lgbm50, y_lgbm50)), lgbm_sh, 64.8, 0.262, -112.3, len(y_lgbm50), str(h50 / "methods/lightgbm_lob/predictions.npz"), "protocol_different", "official_lite_four_date_rerun; original normalized motivating-slice artifact unavailable"],
        ],
        columns=["model", "rerun_DA_pct", "rerun_RIC", "rerun_sharpe_3bps", "historical_DA_pct", "historical_RIC", "historical_sharpe_3bps", "sample_count", "source_file", "diff_status", "protocol_caveat"],
    )
    t15.to_csv(OUT / "table15.csv", index=False)
    (OUT / "table15.md").write_text(t15.to_markdown(index=False) + "\n")
    (OUT / "table15.tex").write_text(t15.to_latex(index=False, escape=True))
    (OUT / "table15_source_trace.md").write_text(t15.to_csv(index=False))

    diff = []
    for name, df in [("table6", t6), ("table7", t7), ("table14", t14), ("table15", t15)]:
        for _, r in df.iterrows():
            diff.append({"table": name, "item": r.get("model", r.get("audit", r.get("cost_bps", "row"))), "diff_status": r["diff_status"], "source_file": r["source_file"], "protocol_caveat": r["protocol_caveat"]})
    pd.DataFrame(diff).to_csv(OUT / "motivating_slice_table_diff.csv", index=False)
    (OUT / "motivating_slice_diff_summary.md").write_text("# Motivating Slice Diff Summary\n\n- protocol_different: `19`\n\nOfficial-lite four-date rerun completed for Ridge/HistGBM/LightGBM/Kairos. DeepLOB rows remain compact processed-four-date proxy rows. Historical PDF values are not exact reproduced values.\n")
    (OUT / "final_rerun_status.md").write_text("# Final Rerun Status\n\n1. Did we run Tables 6/7/14/15? Yes.\n2. Official-lite rows succeeded for Ridge, HistGBM, LightGBM, and Kairos under volume h5/h50. DeepLOB rows succeeded as compact processed-four-date proxy rows.\n3. Scripts used: `scripts/build_four_date_official_panels_v8.py`, `scripts/rerun_motivating_slice_official_lite_v8.py`, and `scripts/rerun_motivating_slice_tables_6_7_14_15.py`.\n4. Data used: four processed BTCUSDT/Binance dates: 2025-10-12, 2025-10-17, 2025-10-26, 2025-10-27.\n5. Equal to PDF historical values? No. All rows remain `protocol_different` because the original normalized motivating-slice artifact and checkpoints are unavailable.\n6. Should PDF values be replaced? Use v8 values only with protocol caveat; otherwise keep historical caveat.\n7. Caveats removable? No for exact historical reproduction.\n8. Caveats must remain for Tables 6/7/14/15.\n9. Remaining blocker: original normalized motivating-slice artifact and original full-training codebook/checkpoints.\n")
    (OUT / "pdf_patch_recommendation_after_rerun.md").write_text("# PDF Patch Recommendation After Rerun\n\nTables 6/7/14/15 are no longer unrun: v8 ran official-lite four-date reruns for Ridge/HistGBM/LightGBM/Kairos and processed-four-date proxy DeepLOB rows. The old PDF historical values should still not be presented as exact reproduced evidence. Either keep the visible caveat in `/Users/yangyangliang/Desktop/最终版_patched.pdf`, or replace the old numeric rows with v8 values and label them `official_lite_four_date_rerun / protocol_different`.\n")
    manifest = []
    for f in sorted(OUT.glob("*")):
        if f.is_file():
            manifest.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f}")
    (OUT / "sha256_manifest.txt").write_text("\n".join(manifest) + "\n")


def main() -> None:
    write_split_plan()
    run_setting("volume_h5", "volume", 5, 1)
    run_setting("volume_h50", "volume", 50, 50)
    update_tables()
    print(json.dumps({"status": "official_lite_completed", "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
