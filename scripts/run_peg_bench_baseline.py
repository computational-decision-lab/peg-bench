#!/usr/bin/env python3
"""Run a PEG-Bench baseline pipeline from a processed multi-day bundle."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import Any

import numpy as np
import yaml

from peg_bench.data.panel_builder import SPLITS, build_split_panel
from src.models.baseline import KairosBaselineConfig
from src.models.vqvae import VQVAEConfig
from src.train.baseline_trainer import BaselineTrainConfig, train_baseline
from src.train.vqvae_trainer import VQVAETrainConfig, export_vqvae_encodings, train_vqvae


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _prepare_vq_config(template_path: str | Path, dataset_path: str, output_dir: str, device: str) -> VQVAETrainConfig:
    payload = _load_yaml(template_path)
    model_cfg = VQVAEConfig(**payload.pop("model"))
    payload["dataset_path"] = dataset_path
    payload["output_dir"] = output_dir
    payload["device"] = device
    return VQVAETrainConfig(model=model_cfg, **payload)


def _prepare_baseline_config(
    template_path: str | Path,
    train_dataset_path: str,
    train_encoded_path: str,
    val_dataset_path: str,
    val_encoded_path: str,
    test_dataset_path: str,
    test_encoded_path: str,
    output_dir: str,
    device: str,
) -> BaselineTrainConfig:
    payload = _load_yaml(template_path)
    model_cfg = KairosBaselineConfig(**payload.pop("model"))
    payload["dataset_path"] = train_dataset_path
    payload["encoded_path"] = train_encoded_path
    payload["val_dataset_path"] = val_dataset_path
    payload["val_encoded_path"] = val_encoded_path
    payload["test_dataset_path"] = test_dataset_path
    payload["test_encoded_path"] = test_encoded_path
    payload["output_dir"] = output_dir
    payload["device"] = device
    return BaselineTrainConfig(model=model_cfg, **payload)


def run_peg_bench_baseline(
    manifest_path: str | Path,
    mode: str,
    output_root: str | Path,
    vq_config_path: str | Path,
    baseline_config_path: str | Path,
    device: str,
    split_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build split panels, train VQ-VAE, export codes, then train baseline."""
    output_root = Path(output_root)
    panels_root = output_root / "panels"
    vq_root = output_root / "vqvae"
    baseline_root = output_root / "baseline"

    panel_paths: dict[str, Path] = {}
    panel_summaries: dict[str, Any] = {}
    for split in SPLITS:
        panel_path = panels_root / f"{mode}_{split}.npz"
        panel_paths[split] = panel_path
        if panel_path.exists():
            summary_path = panel_path.with_suffix(".json")
            panel_summaries[split] = json.loads(summary_path.read_text()) if summary_path.exists() else {"output_path": str(panel_path.resolve())}
        else:
            panel_summaries[split] = build_split_panel(
                manifest_path,
                split=split,
                mode=mode,
                output_path=panel_path,
                split_plan_path=split_plan_path,
            )

    vq_summary_path = vq_root / "summary.json"
    if vq_summary_path.exists():
        vq_summary = json.loads(vq_summary_path.read_text())
    else:
        vq_cfg = _prepare_vq_config(vq_config_path, str(panel_paths["train"]), str(vq_root), device)
        vq_summary = train_vqvae(vq_cfg)
        vq_root.mkdir(parents=True, exist_ok=True)
        vq_summary_path.write_text(json.dumps(vq_summary, indent=2))

    fallback_codebook = vq_summary.get("fallback_codebook")
    encoded_paths = {
        "train": Path(vq_summary["encoded_tokens"]),
        "val": vq_root / "encoded_val.npz",
        "test": vq_root / "encoded_test.npz",
    }
    encoding_summaries: dict[str, Any] = {"train": {"encoded_tokens": str(encoded_paths["train"]), "codes_source": vq_summary["codes_source"], "num_unique_codes": vq_summary["num_unique_codes"]}}
    for split in ("val", "test"):
        if encoded_paths[split].exists():
            payload = np.load(encoded_paths[split])
            encoding_summaries[split] = {
                "encoded_tokens": str(encoded_paths[split].resolve()),
                "num_unique_codes": int(len(np.unique(payload["codes"]))),
                "codes_source": "kmeans_fallback" if fallback_codebook else "vq_codebook",
            }
            payload.close()
            continue
        encoding_summaries[split] = export_vqvae_encodings(
            checkpoint_path=vq_summary["checkpoint"],
            dataset_path=str(panel_paths[split]),
            output_path=str(encoded_paths[split]),
            batch_size=int(_load_yaml(vq_config_path).get("batch_size", 128)),
            device=device,
            fallback_codebook_path=fallback_codebook,
        )

    baseline_summary_path = baseline_root / "summary.json"
    if baseline_summary_path.exists():
        baseline_summary = json.loads(baseline_summary_path.read_text())
    else:
        baseline_cfg = _prepare_baseline_config(
            baseline_config_path,
            train_dataset_path=str(panel_paths["train"]),
            train_encoded_path=str(encoded_paths["train"]),
            val_dataset_path=str(panel_paths["val"]),
            val_encoded_path=str(encoded_paths["val"]),
            test_dataset_path=str(panel_paths["test"]),
            test_encoded_path=str(encoded_paths["test"]),
            output_dir=str(baseline_root),
            device=device,
        )
        baseline_summary = train_baseline(baseline_cfg)
        baseline_root.mkdir(parents=True, exist_ok=True)
        baseline_summary_path.write_text(json.dumps(baseline_summary, indent=2))

    result = {
        "mode": mode,
        "manifest_path": str(Path(manifest_path).resolve()),
        "output_root": str(output_root.resolve()),
        "panels": panel_summaries,
        "vqvae": vq_summary,
        "encodings": encoding_summaries,
        "baseline": baseline_summary,
    }
    (output_root / "run_summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEG-Bench baseline on a processed bundle.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--mode", type=str, default="time", choices=["time", "volume"])
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--vq-config", type=str, default="configs/vqvae.yaml")
    parser.add_argument("--baseline-config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split-plan", type=str, default=None)
    args = parser.parse_args()

    result = run_peg_bench_baseline(
        manifest_path=args.manifest,
        mode=args.mode,
        output_root=args.output_root,
        vq_config_path=args.vq_config,
        baseline_config_path=args.baseline_config,
        device=args.device,
        split_plan_path=args.split_plan,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
