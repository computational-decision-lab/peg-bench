#!/usr/bin/env python3
"""Run paper-style comparison baselines for PEG-Bench."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json

from src.eval.benchmark_baselines import (
    PanelSpec,
    build_last_lob_features,
    build_token_frequency_features,
    ensure_vq_and_encodings,
    ensure_window_index,
    run_anchor_baseline,
    run_deeplob_baseline,
    run_kairos_baseline,
    run_sklearn_baseline,
)
from src.eval.benchmark_evaluator import (
    BenchmarkEvaluatorConfig,
    _load_split_lookup_from_split_plan,
    evaluate_benchmark_methods,
)
from src.eval.benchmark_evaluator import BenchmarkMethodSpec
from src.eval.report_coverage import summarize_report_coverage, write_report_coverage_outputs


def _panel_paths(standardized_root: Path, mode: str, split: str, seq_len: int) -> tuple[str, str]:
    panel = standardized_root / "panels" / f"{mode}_{split}.npz"
    index = standardized_root / "manifests" / f"{mode}_{split}_window_index_seq{seq_len}.csv"
    return str(panel.resolve()), str(index.resolve())


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return str(output.resolve())
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(output.resolve())


def _resolve_split_lookup(
    *,
    bundle_manifest: str | Path,
    standardized_root: Path,
    mode: str,
    split_plan_path: str | Path | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Resolve evaluation split lookup, preferring panel-level split overrides when present."""
    resolved_split_plan: Path | None = None
    if split_plan_path is not None:
        candidate = Path(split_plan_path)
        if candidate.exists():
            resolved_split_plan = candidate
    if resolved_split_plan is None:
        panel_summary_path = standardized_root / "panels" / f"{mode}_test.json"
        if panel_summary_path.exists():
            payload = json.loads(panel_summary_path.read_text())
            if payload.get("split_source") == "split_plan_override" and payload.get("split_plan_path"):
                candidate = Path(str(payload["split_plan_path"]))
                if candidate.exists():
                    resolved_split_plan = candidate
    if resolved_split_plan is not None:
        return _load_split_lookup_from_split_plan(resolved_split_plan), {
            "split_source": "split_plan_override",
            "split_plan_path": str(resolved_split_plan.resolve()),
        }
    bundle_payload = json.loads(Path(bundle_manifest).read_text())
    split_lookup = {day: str(meta["split"]) for day, meta in bundle_payload.get("split_index", {}).items()}
    return split_lookup, {
        "split_source": "bundle_manifest",
        "bundle_manifest": str(Path(bundle_manifest).resolve()),
    }


def _result_row(result) -> dict[str, object]:
    return {
        "method": result.name,
        "feature_mode": result.feature_mode,
        "test_directional_accuracy": result.test_metrics["directional_accuracy"],
        "test_rank_ic": result.test_metrics["rank_ic"],
        "test_mse": result.test_metrics["mse"],
        "test_mae": result.test_metrics["mae"],
        "report_dir": result.report_dir,
        "notes": " | ".join(result.notes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEG-Bench comparison baselines under the leakage-controlled setting.")
    parser.add_argument("--standardized-root", type=str, required=True)
    parser.add_argument("--bundle-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--mode", type=str, default="time", choices=["time", "volume"])
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--target-key", type=str, default="return_20")
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--methods", type=str, default="no_trade,buy_and_hold,ofi_threshold,ridge_tokens,histgbm_tokens,lightgbm_lob,kairos_64code,deeplob_timebars")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vq-epochs", type=int, default=5)
    parser.add_argument("--kairos-epochs", type=int, default=5)
    parser.add_argument("--vq-batch-size", type=int, default=256)
    parser.add_argument("--kairos-batch-size", type=int, default=128)
    parser.add_argument("--deeplob-seq-len", type=int, default=50)
    parser.add_argument("--deeplob-epochs", type=int, default=3)
    parser.add_argument("--deeplob-batch-size", type=int, default=256)
    parser.add_argument("--vq-max-train-items", type=int, default=1000000)
    parser.add_argument("--split-plan", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    standardized_root = Path(args.standardized_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}

    panel_specs: dict[str, PanelSpec] = {}
    for split in ("train", "val", "test"):
        panel_path, index_path = _panel_paths(standardized_root, args.mode, split, args.seq_len)
        ensured_index = ensure_window_index(
            panel_path=panel_path,
            output_path=index_path,
            seq_len=args.seq_len,
            stride=args.stride,
            target_key=args.target_key,
        )
        panel_specs[split] = PanelSpec(
            panel_path=panel_path,
            index_path=ensured_index,
            split=split,
            seq_len=args.seq_len,
            stride=args.stride,
            target_key=args.target_key,
        )

    deeplob_specs: dict[str, PanelSpec] = {}
    if "deeplob_timebars" in methods:
        for split in ("train", "val", "test"):
            panel_path, index_path = _panel_paths(standardized_root, args.mode, split, args.deeplob_seq_len)
            ensured_index = ensure_window_index(
                panel_path=panel_path,
                output_path=index_path,
                seq_len=args.deeplob_seq_len,
                stride=args.stride,
                target_key=args.target_key,
            )
            deeplob_specs[split] = PanelSpec(
                panel_path=panel_path,
                index_path=ensured_index,
                split=split,
                seq_len=args.deeplob_seq_len,
                stride=args.stride,
                target_key=args.target_key,
            )

    token_methods = {"ridge_tokens", "histgbm_tokens", "kairos_64code"} & methods
    if token_methods:
        encodings = ensure_vq_and_encodings(
            train_panel_path=panel_specs["train"].panel_path,
            val_panel_path=panel_specs["val"].panel_path,
            test_panel_path=panel_specs["test"].panel_path,
            output_dir=output_root / "vqvae_64code",
            device=args.device,
            batch_size=args.vq_batch_size,
            epochs=args.vq_epochs,
            seed=args.seed,
            num_embeddings=args.vocab_size,
            latent_dim=64,
            max_train_items=args.vq_max_train_items,
        )
        panel_specs["train"].encoded_path = encodings["train_encoded"]
        panel_specs["val"].encoded_path = encodings["val_encoded"]
        panel_specs["test"].encoded_path = encodings["test_encoded"]

    results = []
    report_specs: list[BenchmarkMethodSpec] = []
    token_cache: dict[str, dict[str, object]] = {}
    lob_cache: dict[str, dict[str, object]] = {}

    anchor_methods = {"no_trade", "buy_and_hold", "always_short", "ofi_threshold"} & methods
    if anchor_methods:
        if not lob_cache:
            lob_cache["train"] = build_last_lob_features(
                panel_path=panel_specs["train"].panel_path,
                index_path=panel_specs["train"].index_path,
                target_key=args.target_key,
            )
            lob_cache["val"] = build_last_lob_features(
                panel_path=panel_specs["val"].panel_path,
                index_path=panel_specs["val"].index_path,
                target_key=args.target_key,
            )
            lob_cache["test"] = build_last_lob_features(
                panel_path=panel_specs["test"].panel_path,
                index_path=panel_specs["test"].index_path,
                target_key=args.target_key,
            )
        for method_name in ("no_trade", "buy_and_hold", "always_short", "ofi_threshold"):
            if method_name not in methods:
                continue
            result = run_anchor_baseline(
                method=method_name,
                train=lob_cache["train"],
                val=lob_cache["val"],
                test=lob_cache["test"],
                output_dir=output_root / "methods" / method_name,
            )
            results.append(result)
            report_specs.append(
                BenchmarkMethodSpec(
                    name=result.name,
                    reports=tuple(str(path.resolve()) for path in (Path(result.report_dir)).glob("*.json")),
                )
            )

    if "ridge_tokens" in methods:
        if not token_cache:
            token_cache["train"] = build_token_frequency_features(
                panel_path=panel_specs["train"].panel_path,
                index_path=panel_specs["train"].index_path,
                encoded_path=str(panel_specs["train"].encoded_path),
                vocab_size=args.vocab_size,
                target_key=args.target_key,
            )
            token_cache["val"] = build_token_frequency_features(
                panel_path=panel_specs["val"].panel_path,
                index_path=panel_specs["val"].index_path,
                encoded_path=str(panel_specs["val"].encoded_path),
                vocab_size=args.vocab_size,
                target_key=args.target_key,
            )
            token_cache["test"] = build_token_frequency_features(
                panel_path=panel_specs["test"].panel_path,
                index_path=panel_specs["test"].index_path,
                encoded_path=str(panel_specs["test"].encoded_path),
                vocab_size=args.vocab_size,
                target_key=args.target_key,
            )
        result = run_sklearn_baseline(
            method="ridge_tokens",
            train=token_cache["train"],
            val=token_cache["val"],
            test=token_cache["test"],
            output_dir=output_root / "methods" / "ridge_tokens",
            seed=args.seed,
        )
        results.append(result)
        report_specs.append(
            BenchmarkMethodSpec(
                name=result.name,
                reports=tuple(str(path.resolve()) for path in (Path(result.report_dir)).glob("*.json")),
            )
        )

    if "histgbm_tokens" in methods:
        if not token_cache:
            token_cache["train"] = build_token_frequency_features(
                panel_path=panel_specs["train"].panel_path,
                index_path=panel_specs["train"].index_path,
                encoded_path=str(panel_specs["train"].encoded_path),
                vocab_size=args.vocab_size,
                target_key=args.target_key,
            )
            token_cache["val"] = build_token_frequency_features(
                panel_path=panel_specs["val"].panel_path,
                index_path=panel_specs["val"].index_path,
                encoded_path=str(panel_specs["val"].encoded_path),
                vocab_size=args.vocab_size,
                target_key=args.target_key,
            )
            token_cache["test"] = build_token_frequency_features(
                panel_path=panel_specs["test"].panel_path,
                index_path=panel_specs["test"].index_path,
                encoded_path=str(panel_specs["test"].encoded_path),
                vocab_size=args.vocab_size,
                target_key=args.target_key,
            )
        result = run_sklearn_baseline(
            method="histgbm_tokens",
            train=token_cache["train"],
            val=token_cache["val"],
            test=token_cache["test"],
            output_dir=output_root / "methods" / "histgbm_tokens",
            seed=args.seed,
        )
        results.append(result)
        report_specs.append(
            BenchmarkMethodSpec(
                name=result.name,
                reports=tuple(str(path.resolve()) for path in (Path(result.report_dir)).glob("*.json")),
            )
        )

    if "lightgbm_lob" in methods:
        if not lob_cache:
            lob_cache["train"] = build_last_lob_features(
                panel_path=panel_specs["train"].panel_path,
                index_path=panel_specs["train"].index_path,
                target_key=args.target_key,
            )
            lob_cache["val"] = build_last_lob_features(
                panel_path=panel_specs["val"].panel_path,
                index_path=panel_specs["val"].index_path,
                target_key=args.target_key,
            )
            lob_cache["test"] = build_last_lob_features(
                panel_path=panel_specs["test"].panel_path,
                index_path=panel_specs["test"].index_path,
                target_key=args.target_key,
            )
        result = run_sklearn_baseline(
            method="lightgbm_lob",
            train=lob_cache["train"],
            val=lob_cache["val"],
            test=lob_cache["test"],
            output_dir=output_root / "methods" / "lightgbm_lob",
            seed=args.seed,
        )
        results.append(result)
        report_specs.append(
            BenchmarkMethodSpec(
                name=result.name,
                reports=tuple(str(path.resolve()) for path in (Path(result.report_dir)).glob("*.json")),
            )
        )

    if "kairos_64code" in methods:
        result = run_kairos_baseline(
            train_panel=panel_specs["train"],
            val_panel=panel_specs["val"],
            test_panel=panel_specs["test"],
            checkpoint_dir=output_root / "methods" / "kairos_64code" / "checkpoints",
            output_dir=output_root / "methods" / "kairos_64code",
            device=args.device,
            vocab_size=args.vocab_size,
            epochs=args.kairos_epochs,
            batch_size=args.kairos_batch_size,
            seed=args.seed,
        )
        results.append(result)
        report_specs.append(
            BenchmarkMethodSpec(
                name=result.name,
                reports=tuple(str(path.resolve()) for path in (Path(result.report_dir)).glob("*.json")),
            )
        )

    if "deeplob_timebars" in methods:
        result = run_deeplob_baseline(
            train_panel=deeplob_specs["train"],
            val_panel=deeplob_specs["val"],
            test_panel=deeplob_specs["test"],
            checkpoint_dir=output_root / "methods" / "deeplob_timebars" / "checkpoints",
            output_dir=output_root / "methods" / "deeplob_timebars",
            device=args.device,
            epochs=args.deeplob_epochs,
            batch_size=args.deeplob_batch_size,
            seed=args.seed,
        )
        results.append(result)
        report_specs.append(
            BenchmarkMethodSpec(
                name=result.name,
                reports=tuple(str(path.resolve()) for path in (Path(result.report_dir)).glob("*.json")),
            )
        )

    rows = [_result_row(result) for result in results]
    results_json = output_root / "results_table.json"
    results_csv = output_root / "results_table.csv"
    results_json.write_text(json.dumps(rows, indent=2))
    _write_csv(results_csv, rows)

    split_lookup, split_meta = _resolve_split_lookup(
        bundle_manifest=args.bundle_manifest,
        standardized_root=standardized_root,
        mode=args.mode,
        split_plan_path=args.split_plan,
    )
    benchmark_report = evaluate_benchmark_methods(
        method_specs=report_specs,
        split_lookup=split_lookup,
        config=BenchmarkEvaluatorConfig(
            output_path=str(output_root / "benchmark_report.json"),
            markdown_path=str(output_root / "benchmark_report.md"),
            evaluation_split="test",
        ),
    )
    coverage = summarize_report_coverage(
        method_specs=report_specs,
        split_lookup=split_lookup,
        evaluation_split="test",
    )
    (output_root / "benchmark_report.json").write_text(json.dumps(benchmark_report, indent=2))
    write_report_coverage_outputs(
        coverage,
        csv_path=output_root / "benchmark_report_coverage.csv",
        json_path=output_root / "benchmark_report_coverage.json",
        markdown_path=output_root / "benchmark_report_coverage.md",
    )

    summary = {
        "results_table_json": str(results_json.resolve()),
        "results_table_csv": str(results_csv.resolve()),
        "benchmark_report_json": str((output_root / "benchmark_report.json").resolve()),
        "benchmark_report_coverage_json": str((output_root / "benchmark_report_coverage.json").resolve()),
        "benchmark_split_meta": split_meta,
        "methods": rows,
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
