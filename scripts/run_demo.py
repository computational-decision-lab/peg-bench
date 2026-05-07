#!/usr/bin/env python3
"""Minimal end-to-end KAIROS-X demo on synthetic data."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path

from src.data.pipeline import DataPipelineConfig, run_data_pipeline
from src.data.synthetic import SyntheticMarketConfig, generate_synthetic_l2_and_trades
from src.eval.backtests import BacktestConfig, run_baseline_backtest
from src.eval.plots import plot_code_usage, plot_cost_sensitivity
from src.eval.report import generate_markdown_report
from src.models.baseline import KairosBaselineConfig
from src.models.vqvae import VQVAEConfig
from src.preference.pair_builder import PreferenceBuilderConfig, build_preference_pairs
from src.train.baseline_trainer import BaselineTrainConfig, train_baseline
from src.train.dpo_trainer import ContinuousDPOTrainer, DPOTrainConfig
from src.train.vqvae_trainer import VQVAETrainConfig, train_vqvae


def main() -> None:
    root = Path("outputs/demo")
    raw_dir = root / "raw"
    l2_path, trades_path = generate_synthetic_l2_and_trades(raw_dir, SyntheticMarketConfig(num_steps=240))
    data_path = root / "volume_bars.npz"
    run_data_pipeline(
        DataPipelineConfig(
            l2_path=str(l2_path),
            trades_path=str(trades_path),
            output_path=str(data_path),
            bar_mode="volume",
            volume_threshold=3_000.0,
            depth=10,
            stride=10,
            max_bars=96,
        )
    )
    vq_summary = train_vqvae(
        VQVAETrainConfig(
            dataset_path=str(data_path),
            output_dir=str(root / "vqvae"),
            epochs=1,
            batch_size=32,
            device="cpu",
            model=VQVAEConfig(num_embeddings=64, beta_sym=0.05),
        )
    )
    base_summary = train_baseline(
        BaselineTrainConfig(
            encoded_path=vq_summary["encoded_tokens"],
            dataset_path=str(data_path),
            output_dir=str(root / "baseline"),
            input_mode="codes",
            epochs=1,
            batch_size=32,
            seq_len=32,
            stride=8,
            device="cpu",
            model=KairosBaselineConfig(input_mode="codes", seq_len=32),
        )
    )
    pref_summary = build_preference_pairs(
        PreferenceBuilderConfig(
            encoded_path=vq_summary["encoded_tokens"],
            dataset_path=str(data_path),
            checkpoint_path=base_summary["best_checkpoint"],
            output_path=str(root / "preferences.pkl"),
            mode="scalar",
            min_utility_gap=0.0,
            start_index=32,
            max_contexts=2,
            seq_len=32,
            horizon=16,
            stride=32,
            device="cpu",
            use_aci_filtering=False,
        )
    )
    dpo_summary = ContinuousDPOTrainer(
        DPOTrainConfig(
            preferences_path=str(root / "preferences.pkl"),
            reference_checkpoint=base_summary["best_checkpoint"],
            output_dir=str(root / "dpo"),
            epochs=1,
            batch_size=8,
            device="cpu",
        )
    ).train()
    backtest_summary = run_baseline_backtest(
        BacktestConfig(
            checkpoint_path=base_summary["best_checkpoint"],
            encoded_path=vq_summary["encoded_tokens"],
            dataset_path=str(data_path),
            output_path=str(root / "backtest.json"),
            input_mode="codes",
            seq_len=32,
            stride=8,
            device="cpu",
        )
    )
    codes = __import__("numpy").load(vq_summary["encoded_tokens"])["codes"]
    plot_code_usage(codes, root / "code_usage.png")
    plot_cost_sensitivity(backtest_summary["cost_scan"], root / "cost_scan.png")
    generate_markdown_report(
        root / "report.md",
        sections={
            "Data": {"dataset": str(data_path)},
            "VQ-VAE": vq_summary,
            "Baseline": base_summary,
            "Preferences": pref_summary,
            "DPO": dpo_summary,
            "Backtest": backtest_summary,
        },
        approximations=[
            "Synthetic data is used for the demo instead of Binance BTCUSDT raw data.",
            "Execution cost and fills are capacity-aware approximations rather than full matching-engine simulation.",
            "Preference filtering uses residual-based ACI with lightweight calibration.",
        ],
    )
    print("Demo complete. Artifacts written to", root)


if __name__ == "__main__":
    main()
