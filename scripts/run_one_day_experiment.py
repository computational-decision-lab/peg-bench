#!/usr/bin/env python3
"""Run the end-to-end one-day real-data experiment on the cloud repo."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random
import yaml
import numpy as np
import pandas as pd
import torch

from src.data.bars import BarConfig, build_bar_dataset, save_bar_dataset
from src.data.binance_public_depth import (
    BinancePublicDataConfig,
    build_time_bars_from_public_depth,
    build_volume_bars_from_public_depth,
    ensure_one_day_real_files,
    save_payload,
)
from src.eval.backtests import BacktestConfig, run_baseline_backtest
from src.eval.plots import (
    plot_aci_diagnostics,
    plot_code_usage,
    plot_cost_sensitivity,
    plot_embedding_tsne,
    plot_reconstruction_heatmap,
    plot_training_curves,
)
from src.eval.policy_backtest import PolicyBacktestConfig, run_policy_backtest
from src.eval.report import generate_markdown_report
from src.models.baseline import KairosBaselineConfig
from src.models.vqvae import ScaleInvariantVQVAE, VQVAEConfig
from src.preference.dataset import load_preference_dataset, save_preference_dataset
from src.preference.pair_builder import PreferenceBuilderConfig, build_preference_pairs
from src.train.aci_calibrator import ACICalibrationConfig, calibrate_aci
from src.train.baseline_trainer import BaselineTrainConfig, train_baseline
from src.train.dpo_trainer import ContinuousDPOTrainer, DPOTrainConfig
from src.train.vqvae_trainer import VQVAETrainConfig, train_vqvae
from src.utils.device import amp_enabled, resolve_torch_device


def _preferred_device() -> str:
    return str(resolve_torch_device("auto"))


def _build_event_l2_datasets(cfg: dict[str, object], output_root: Path, l2_path: Path) -> dict[str, Path]:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    requested_modes = list(cfg.get("bar_modes", [cfg.get("primary_bar", "time")]))
    result: dict[str, Path] = {}
    if "time" in requested_modes:
        time_path = data_root / "time_bars_real.npz"
        if not time_path.exists():
            payload = build_bar_dataset(
                str(l2_path),
                None,
                BarConfig(
                    mode="time",
                    interval_ms=1000,
                    depth=int(cfg["levels_per_side"]),
                    max_bars=cfg.get("time_max_bars"),
                ),
            )
            save_bar_dataset(payload, time_path)
        result["time"] = time_path
    if "volume" in requested_modes:
        volume_path = data_root / "volume_bars_real.npz"
        if not volume_path.exists():
            payload = build_bar_dataset(
                str(l2_path),
                None,
                BarConfig(
                    mode="volume",
                    volume_threshold=float(cfg["volume_threshold"]),
                    depth=int(cfg["levels_per_side"]),
                    max_bars=cfg.get("volume_max_bars"),
                ),
            )
            save_bar_dataset(payload, volume_path)
        result["volume"] = volume_path
    return result


def _compute_input_diagnostics(raw_paths: dict[str, Path]) -> dict[str, object]:
    if "l2_csv" in raw_paths:
        sample = pd.read_csv(raw_paths["l2_csv"], compression="infer", nrows=100000)
        return {
            "input_mode": "event_level_incremental_l2",
            "num_sample_rows": int(len(sample)),
            "sample_columns": list(sample.columns),
            "timestamp_min": int(sample["timestamp"].min()),
            "timestamp_max": int(sample["timestamp"].max()),
            "snapshot_rows_in_sample": int(sample["is_snapshot"].sum()),
            "bid_rows_in_sample": int((sample["side"] == "bid").sum()),
            "ask_rows_in_sample": int((sample["side"] == "ask").sum()),
        }
    depth = pd.read_csv(raw_paths["depth_csv"])
    depth["timestamp"] = pd.to_datetime(depth["timestamp"], utc=True)
    depth_ts = (depth["timestamp"].astype("int64") // 1_000_000).astype(np.int64)
    if depth_ts.max() < 10**12:
        depth_ts = depth_ts * 1000
    per_snapshot = depth.groupby(depth_ts).size()
    trades = pd.read_csv(raw_paths["trades_csv"], usecols=["time", "quote_qty", "price"])
    return {
        "num_depth_snapshots": int(len(per_snapshot)),
        "depth_rows_per_snapshot_mean": float(per_snapshot.mean()),
        "depth_rows_per_snapshot_std": float(per_snapshot.std()),
        "snapshot_interval_seconds_mean": float(np.diff(np.unique(depth_ts)).mean() / 1000.0),
        "snapshot_interval_seconds_std": float(np.diff(np.unique(depth_ts)).std() / 1000.0),
        "num_trade_rows": int(len(trades)),
        "trade_quote_volume_total": float(trades["quote_qty"].sum()),
        "trade_price_min": float(trades["price"].min()),
        "trade_price_max": float(trades["price"].max()),
        "input_mode": "binance_public_trades_plus_bookDepth_curve",
    }


def _maybe_build_real_datasets(cfg: dict[str, object], output_root: Path, raw_paths: dict[str, Path]) -> dict[str, Path]:
    data_root = output_root / "data"
    time_path = data_root / "time_bars_real.npz"
    volume_path = data_root / "volume_bars_real.npz"
    if not time_path.exists():
        payload = build_time_bars_from_public_depth(raw_paths["depth_csv"], raw_paths["trades_csv"], levels_per_side=int(cfg["levels_per_side"]))
        save_payload(payload, time_path)
    if not volume_path.exists():
        payload = build_volume_bars_from_public_depth(
            raw_paths["depth_csv"],
            raw_paths["trades_csv"],
            levels_per_side=int(cfg["levels_per_side"]),
            volume_threshold=float(cfg["volume_threshold"]),
        )
        save_payload(payload, volume_path)
    return {"time": time_path, "volume": volume_path}


def _train_or_load_vqvae(dataset_path: Path, run_root: Path, cfg: dict[str, object]) -> dict[str, object]:
    output_dir = run_root / "vqvae"
    encoded_path = output_dir / "encoded_tokens.npz"
    if encoded_path.exists():
        payload = {
            "checkpoint": str(sorted(output_dir.glob("vqvae_epoch_*.pt"))[-1]),
            "encoded_tokens": str(encoded_path),
            "history_path": str(output_dir / "vqvae_history.json"),
        }
        if (output_dir / "summary.json").exists():
            payload.update(json.loads((output_dir / "summary.json").read_text()))
        return payload
    result = train_vqvae(
        VQVAETrainConfig(
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            epochs=int(cfg["vqvae"]["epochs"]),
            batch_size=int(cfg["vqvae"]["batch_size"]),
            learning_rate=float(cfg["vqvae"]["learning_rate"]),
            device=_preferred_device(),
            amp=amp_enabled(True, resolve_torch_device("auto")),
            model=VQVAEConfig(num_embeddings=64, beta_sym=0.1, levels=int(cfg["levels_per_side"])),
        )
    )
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def _train_or_load_baseline(dataset_path: Path, encoded_path: str, run_root: Path, cfg: dict[str, object]) -> dict[str, object]:
    output_dir = run_root / "baseline"
    best_path = output_dir / "best_baseline.pt"
    if best_path.exists():
        return {
            "best_checkpoint": str(best_path),
            "history_path": str(output_dir / "baseline_history.json"),
        }
    baseline_cfg = cfg["baseline"]
    result = train_baseline(
        BaselineTrainConfig(
            encoded_path=encoded_path,
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            input_mode="codes",
            epochs=int(baseline_cfg["epochs"]),
            batch_size=int(baseline_cfg["batch_size"]),
            seq_len=int(baseline_cfg["seq_len"]),
            stride=int(baseline_cfg["stride"]),
            device=_preferred_device(),
            amp=amp_enabled(True, resolve_torch_device("auto")),
            model=KairosBaselineConfig(
                input_mode="codes",
                seq_len=int(baseline_cfg["seq_len"]),
                rankic_weight=float(baseline_cfg["rankic_weight"]),
            ),
        )
    )
    return result


def _cap_preference_dataset(path: Path, max_pairs: int) -> int:
    records = load_preference_dataset(path)
    if len(records) <= max_pairs:
        return len(records)
    rng = random.Random(7)
    sampled = rng.sample(records, max_pairs)
    save_preference_dataset(sampled, path)
    stats_path = path.with_suffix('.json')
    stats = json.loads(stats_path.read_text())
    stats['num_pairs'] = len(sampled)
    stats['capped_to'] = max_pairs
    stats['risk_mode_counts'] = {
        mode: sum(1 for record in sampled if record.risk_mode == mode)
        for mode in ['aggressive', 'balanced', 'conservative', 'low_turnover']
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return len(sampled)


def _build_preferences_with_fallback(dataset_path: Path, encoded_path: str, checkpoint: str, run_root: Path, cfg: dict[str, object]) -> dict[str, object]:
    pref_cfg = cfg["preference"]
    pref_path = run_root / "preferences" / "pairs.pkl"
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = [
        {"mode": pref_cfg["mode"], "template": pref_cfg.get("template", "auto"), "gap": 0.1, "aci": True},
        {"mode": pref_cfg["mode"], "template": pref_cfg.get("template", "auto"), "gap": 0.0, "aci": True},
        {"mode": pref_cfg["mode"], "template": pref_cfg.get("template", "auto"), "gap": 0.0, "aci": False},
        {"mode": "scalar", "template": pref_cfg.get("template", "auto"), "gap": 0.0, "aci": False},
    ]
    last_stats = None
    for attempt in attempts:
        stats = build_preference_pairs(
            PreferenceBuilderConfig(
                encoded_path=encoded_path,
                dataset_path=str(dataset_path),
                checkpoint_path=checkpoint,
                output_path=str(pref_path),
                mode=str(attempt["mode"]),
                lexicographic_template=str(attempt["template"]),
                min_utility_gap=float(attempt["gap"]),
                start_index=int(pref_cfg["seq_len"]),
                max_contexts=int(pref_cfg["max_contexts"]),
                seq_len=int(pref_cfg["seq_len"]),
                horizon=int(pref_cfg["horizon"]),
                stride=int(pref_cfg["stride"]),
                device=_preferred_device(),
                use_aci_filtering=bool(attempt["aci"]),
            )
        )
        stats["attempt"] = attempt
        last_stats = stats
        if stats["num_pairs"] >= int(pref_cfg["min_pairs_target"]):
            break
    if last_stats is None or last_stats["num_pairs"] == 0:
        raise RuntimeError("Preference construction produced zero pairs after all fallbacks.")
    _cap_preference_dataset(pref_path, int(pref_cfg["max_pairs"]))
    return json.loads(pref_path.with_suffix('.json').read_text())


def _train_or_load_dpo(preferences_path: Path, reference_checkpoint: str, run_root: Path, cfg: dict[str, object]) -> dict[str, object]:
    output_dir = run_root / "dpo"
    best_path = output_dir / "best_dpo.pt"
    if best_path.exists():
        return {
            "best_checkpoint": str(best_path),
            "history_path": str(output_dir / "dpo_history.json"),
        }
    dpo_cfg = cfg["dpo"]
    trainer = ContinuousDPOTrainer(
        DPOTrainConfig(
            preferences_path=str(preferences_path),
            reference_checkpoint=reference_checkpoint,
            output_dir=str(output_dir),
            epochs=int(dpo_cfg["epochs"]),
            batch_size=int(dpo_cfg["batch_size"]),
            beta=float(dpo_cfg["beta"]),
            entropy_coef=float(dpo_cfg["entropy_coef"]),
            device=_preferred_device(),
        )
    )
    return trainer.train()


def _maybe_generate_plots(dataset_path: Path, vq_summary: dict[str, object], baseline_summary: dict[str, object], baseline_bt: dict[str, object], dpo_bt: dict[str, object], run_root: Path) -> None:
    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    encoded = np.load(vq_summary["encoded_tokens"])
    plot_code_usage(encoded["codes"], plots_dir / "vq_code_usage.png")
    plot_embedding_tsne(encoded["latents"], encoded["codes"], plots_dir / "vq_tsne.png")
    dataset = np.load(dataset_path)
    vq_checkpoint = torch.load(vq_summary["checkpoint"], map_location="cpu")
    model = ScaleInvariantVQVAE(VQVAEConfig(**vq_checkpoint["config"]))
    model.load_state_dict(vq_checkpoint["model_state_dict"], strict=False)
    model.eval()
    sample = torch.from_numpy(dataset["lob"][:1]).float()
    with torch.no_grad():
        recon = model(sample)["reconstruction"][0].numpy()
    plot_reconstruction_heatmap(dataset["lob"][0], recon, plots_dir / "vq_reconstruction.png")
    plot_cost_sensitivity(baseline_bt["cost_scan"], plots_dir / "baseline_cost_scan.png")
    plot_cost_sensitivity(dpo_bt["cost_scan"], plots_dir / "dpo_cost_scan.png")
    baseline_history = json.loads(Path(baseline_summary["history_path"]).read_text())
    plot_training_curves(baseline_history, ["train_loss", "val_rank_ic", "test_rank_ic"], plots_dir / "baseline_training.png")
    dpo_history = json.loads(Path(run_root / "dpo" / "dpo_history.json").read_text())
    plot_training_curves(dpo_history, ["loss", "preference_accuracy", "validation_sharpe"], plots_dir / "dpo_training.png")


def _run_single_bar_mode(bar_mode: str, dataset_path: Path, output_root: Path, cfg: dict[str, object]) -> dict[str, object]:
    run_root = output_root / bar_mode
    run_root.mkdir(parents=True, exist_ok=True)
    vq_summary = _train_or_load_vqvae(dataset_path, run_root, cfg)
    baseline_summary = _train_or_load_baseline(dataset_path, vq_summary["encoded_tokens"], run_root, cfg)
    baseline_bt = run_baseline_backtest(
        BacktestConfig(
            checkpoint_path=baseline_summary["best_checkpoint"],
            encoded_path=vq_summary["encoded_tokens"],
            dataset_path=str(dataset_path),
            output_path=str(run_root / "eval" / "baseline_backtest.json"),
            input_mode="codes",
            seq_len=int(cfg["baseline"]["seq_len"]),
            stride=int(cfg["baseline"]["stride"]),
            device=_preferred_device(),
        )
    )
    aci_summary = calibrate_aci(
        ACICalibrationConfig(
            predictions_path=str((run_root / "eval" / "baseline_backtest.npz")),
            output_path=str(run_root / "aci" / "calibration.json"),
        )
    )
    controller_payload = json.loads((run_root / "aci" / "calibration.json").read_text())
    plot_aci_diagnostics(controller_payload["scores"], run_root / "plots" / "aci_scores.png")
    pref_summary = _build_preferences_with_fallback(dataset_path, vq_summary["encoded_tokens"], baseline_summary["best_checkpoint"], run_root, cfg)
    dpo_summary = _train_or_load_dpo(run_root / "preferences" / "pairs.pkl", baseline_summary["best_checkpoint"], run_root, cfg)
    dpo_bt = run_policy_backtest(
        PolicyBacktestConfig(
            dpo_checkpoint=dpo_summary["best_checkpoint"],
            reference_checkpoint=baseline_summary["best_checkpoint"],
            encoded_path=vq_summary["encoded_tokens"],
            dataset_path=str(dataset_path),
            output_path=str(run_root / "eval" / "dpo_backtest.json"),
            seq_len=int(cfg["baseline"]["seq_len"]),
            stride=int(cfg["baseline"]["stride"]),
            aci_path=str(run_root / "aci" / "calibration.json"),
            device=_preferred_device(),
        )
    )
    _maybe_generate_plots(dataset_path, vq_summary, baseline_summary, baseline_bt, dpo_bt, run_root)
    report = generate_markdown_report(
        run_root / "report.md",
        sections={
            "Dataset": {"bar_mode": bar_mode, "dataset": str(dataset_path)},
            "VQ-VAE": vq_summary,
            "Baseline": baseline_summary,
            "Baseline Backtest": baseline_bt,
            "ACI": aci_summary,
            "Preferences": pref_summary,
            "DPO": dpo_summary,
            "DPO Backtest": dpo_bt,
        },
        approximations=[
            "One-day real-data run uses Binance public bookDepth depth curves plus real trades instead of event-level incremental L2.",
            "Volume-clock bars are cut on real trade quote volume and sample the latest available depth curve snapshot.",
            "ACI is enabled by default for pair filtering, with automatic fallback to relaxed thresholds or no-filter mode if one-day sample size is too small.",
            "DPO backtest auto-selects risk mode and action post-processing (scale/deadzone) on the one-day sample to reduce over-trading.",
        ],
    )
    return {
        "run_root": str(run_root),
        "dataset": str(dataset_path),
        "vqvae": vq_summary,
        "baseline_backtest": baseline_bt,
        "aci": aci_summary,
        "preferences": pref_summary,
        "dpo": dpo_summary,
        "dpo_backtest": dpo_bt,
        "report": str(run_root / "report.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-day real-data KAIROS-X experiment.")
    parser.add_argument("--config", type=str, default="configs/one_day_real_data.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_mode = cfg.get("dataset_mode", "binance_public_depth")
    if dataset_mode == "event_l2":
        l2_path = Path(cfg["l2_path"])
        raw_paths = {"l2_csv": l2_path}
        datasets = _build_event_l2_datasets(cfg, output_root, l2_path)
    else:
        raw_paths = ensure_one_day_real_files(
            BinancePublicDataConfig(
                symbol=cfg["symbol"],
                date=cfg["date"],
                raw_dir=cfg["raw_dir"],
                levels_per_side=int(cfg["levels_per_side"]),
                volume_threshold=float(cfg["volume_threshold"]),
            )
        )
        datasets = _maybe_build_real_datasets(cfg, output_root, raw_paths)
    diagnostics = _compute_input_diagnostics(raw_paths)
    (output_root / "input_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    bar_modes = list(cfg.get("bar_modes", [cfg.get("primary_bar", "time")]))
    results = {}
    for bar_mode in bar_modes:
        results[bar_mode] = _run_single_bar_mode(bar_mode, datasets[bar_mode], output_root, cfg)

    comparison = {
        mode: {
            "baseline_3bps_pnl": results[mode]["baseline_backtest"]["cost_scan"]["3.0"]["net_pnl"],
            "baseline_3bps_turnover": results[mode]["baseline_backtest"]["cost_scan"]["3.0"]["turnover"],
            "dpo_3bps_pnl": results[mode]["dpo_backtest"]["cost_scan"]["3.0"]["net_pnl"],
            "dpo_3bps_turnover": results[mode]["dpo_backtest"]["cost_scan"]["3.0"]["turnover"],
            "dpo_selected_risk_mode": results[mode]["dpo_backtest"]["selected_risk_mode"],
            "dpo_selected_scale": results[mode]["dpo_backtest"]["selected_scale"],
            "dpo_selected_deadzone": results[mode]["dpo_backtest"]["selected_deadzone"],
            "vq_codes_source": results[mode]["vqvae"].get("codes_source", "unknown"),
            "vq_unique_codes": results[mode]["vqvae"].get("num_unique_codes", -1),
        }
        for mode in bar_modes
    }
    (output_root / "comparison.json").write_text(json.dumps(comparison, indent=2))
    comparison_report = generate_markdown_report(
        output_root / "comparison_report.md",
        sections={
            "Input Diagnostics": diagnostics,
            "Comparison": comparison,
        },
        approximations=[
            "Comparison uses the same one-day Binance public trades + bookDepth input for both bar modes.",
            "DPO comparison includes one-day sample action calibration (risk mode + scale + deadzone) to reduce turnover explosions.",
        ],
    )
    print(json.dumps({
        "output_root": str(output_root),
        "input_diagnostics": str(output_root / "input_diagnostics.json"),
        "comparison": str(output_root / "comparison.json"),
        "comparison_report": str(output_root / "comparison_report.md"),
        "runs": {mode: results[mode]["run_root"] for mode in bar_modes},
    }, indent=2))


if __name__ == "__main__":
    main()
