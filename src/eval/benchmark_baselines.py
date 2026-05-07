"""Paper-style comparison baselines for PEG-Bench."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from peg_bench.data.window_index import build_panel_window_index
from src.data.datasets import DatasetSpec, LOBTokenSequenceDataset
from src.eval.metrics import directional_accuracy, mean_absolute_error, mean_squared_error, rank_ic
from src.eval.peg_bench import run_realism_ladder
from src.models.action_interface import build_action_interface_result, serialize_action_interface_result
from src.models.baseline import KairosBaselineConfig, KairosBaselineModel
from src.models.deeplob import DeepLOBConfig
from src.train.baseline_trainer import BaselineTrainConfig, train_baseline
from src.train.deeplob_trainer import DeepLOBTrainConfig, predict_deeplob, train_deeplob
from src.train.vqvae_trainer import VQVAETrainConfig, export_vqvae_encodings, train_vqvae
from src.models.vqvae import VQVAEConfig
from src.utils.device import resolve_torch_device


@dataclass(slots=True)
class PanelSpec:
    """Paths and slicing metadata for one split panel."""

    panel_path: str
    index_path: str
    split: str
    seq_len: int
    stride: int
    target_key: str
    encoded_path: str | None = None


@dataclass(slots=True)
class ComparisonMethodResult:
    """Summary row for one fitted comparison method."""

    name: str
    feature_mode: str
    notes: list[str]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    report_dir: str


def ensure_window_index(
    *,
    panel_path: str | Path,
    output_path: str | Path,
    seq_len: int,
    stride: int,
    target_key: str,
) -> str:
    """Build a segmented window index if it does not already exist."""
    output = Path(output_path)
    if output.exists():
        return str(output.resolve())
    build_panel_window_index(
        panel_path=panel_path,
        output_path=output,
        seq_len=seq_len,
        stride=stride,
        target_key=target_key,
    )
    return str(output.resolve())


def _load_rows(index_path: str | Path) -> list[dict[str, str]]:
    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _window_core_arrays(panel_path: str | Path, rows: list[dict[str, str]], target_key: str) -> dict[str, Any]:
    payload = np.load(panel_path, allow_pickle=True)
    target_indices = np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64)
    dates = np.asarray([str(row["date"]) for row in rows])
    asset_keys = np.asarray([f'{row["exchange"]}:{row["symbol"]}' for row in rows])
    values = payload[target_key][target_indices].astype(np.float32)
    timestamps = payload["timestamps"][target_indices].astype(np.int64)
    lob_states = payload["lob"][target_indices].astype(np.float32)
    return {
        "payload": payload,
        "target_indices": target_indices,
        "dates": dates,
        "asset_keys": asset_keys,
        "targets": values,
        "timestamps": timestamps,
        "lob_states": lob_states,
    }


def build_token_frequency_features(
    *,
    panel_path: str | Path,
    index_path: str | Path,
    encoded_path: str | Path,
    vocab_size: int,
    target_key: str,
) -> dict[str, Any]:
    """Convert each token window into a normalized frequency vector."""
    rows = _load_rows(index_path)
    arrays = _window_core_arrays(panel_path, rows, target_key)
    codes = np.load(encoded_path)["codes"].astype(np.int64)
    features = np.zeros((len(rows), vocab_size), dtype=np.float32)
    for idx, row in enumerate(rows):
        start = int(row["window_start_index"])
        end = int(row["window_end_index_exclusive"])
        window = codes[start:end]
        counts = np.bincount(window, minlength=vocab_size).astype(np.float32)
        denom = float(max(end - start, 1))
        features[idx] = counts / denom
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    arrays.update({"rows": rows, "features": features})
    return arrays


def build_last_lob_features(
    *,
    panel_path: str | Path,
    index_path: str | Path,
    target_key: str,
) -> dict[str, Any]:
    """Use the last order-book snapshot inside each context window."""
    rows = _load_rows(index_path)
    arrays = _window_core_arrays(panel_path, rows, target_key)
    payload = arrays["payload"]
    lob = payload["lob"].astype(np.float32)
    feature_dim = int(np.prod(lob.shape[1:]))
    features = np.zeros((len(rows), feature_dim), dtype=np.float32)
    for idx, row in enumerate(rows):
        last_index = int(row["window_end_index_exclusive"]) - 1
        features[idx] = lob[last_index].reshape(-1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    arrays.update({"rows": rows, "features": features})
    return arrays


def _predict_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return {
        "directional_accuracy": directional_accuracy(predictions, targets),
        "rank_ic": rank_ic(predictions, targets),
        "mse": mean_squared_error(predictions, targets),
        "mae": mean_absolute_error(predictions, targets),
    }


def _fit_sklearn_model(
    *,
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[Any, list[str]]:
    if method == "ridge_tokens":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        )
        model.fit(x_train, y_train)
        return model, ["Ridge alpha=10 on normalized token-frequency windows."]

    if method == "histgbm_tokens":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=50,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model, ["HistGradientBoostingRegressor used as the tree baseline on token-frequency windows."]

    if method == "lightgbm_lob":
        try:
            from lightgbm import LGBMRegressor
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised on remote when lightgbm is absent
            raise RuntimeError("lightgbm is not installed for the LightGBM(LOB) baseline.") from exc

        model = LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        )
        fit_kwargs: dict[str, Any] = {}
        if len(x_val) > 0:
            fit_kwargs["eval_set"] = [(x_val, y_val)]
            fit_kwargs["callbacks"] = []
        model.fit(x_train, y_train, **fit_kwargs)
        return model, ["LightGBM regressor on flattened last-snapshot LOB windows."]

    raise ValueError(f"Unsupported method: {method}")


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return str(output.resolve())


def generate_daily_reports(
    *,
    method_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    asset_keys: np.ndarray,
    timestamps: np.ndarray,
    lob_states: np.ndarray,
    market_returns: np.ndarray,
    output_dir: str | Path,
    notes: list[str],
    cost_bps_scan: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 10.0),
    anchor_cost_bps: float = 3.0,
    action_delta: float = 0.0,
    action_rho: float = 0.9,
) -> list[str]:
    """Write one PEG-Bench-style report per calendar date."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_paths: list[str] = []
    def _mean(values: list[float | None]) -> float | None:
        present = [float(item) for item in values if item is not None]
        if not present:
            return None
        return float(np.mean(np.asarray(present, dtype=np.float64)))

    def _aggregate_report_cards(cards: list[dict[str, Any]], day_predictions: np.ndarray, day_targets: np.ndarray) -> dict[str, Any]:
        report_cards = [card["report_card"] for card in cards]
        execution_cards = [card["execution"] for card in report_cards]
        robustness_cards = [card["robustness"] for card in report_cards]
        realism_cards = [card.get("realism_ladder", {}) for card in cards]
        cost_keys = sorted({str(float(cost)) for cost in cost_bps_scan}, key=float)
        aggregated_cost_ladder: dict[str, dict[str, float | None]] = {}
        for cost in cost_keys:
            aggregated_cost_ladder[cost] = {
                metric: _mean(
                    [
                        card.get("cost_ladder", {}).get(cost, {}).get(metric)
                        for card in execution_cards
                        if isinstance(card.get("cost_ladder"), dict)
                    ]
                )
                for metric in ("sharpe", "max_drawdown", "turnover", "calmar", "net_pnl")
            }

        def _aggregate_stage(stage: str) -> dict[str, dict[str, float | None]]:
            stage_summary: dict[str, dict[str, float | None]] = {}
            for cost in cost_keys:
                stage_summary[cost] = {
                    metric: _mean(
                        [
                            card.get(stage, {}).get("cost_scan", {}).get(cost, {}).get(metric)
                            for card in realism_cards
                            if isinstance(card.get(stage, {}), dict)
                        ]
                    )
                    for metric in ("sharpe", "max_drawdown", "turnover", "calmar", "net_pnl")
                }
            return stage_summary

        regime_metric_lists: dict[str, dict[str, list[float]]] = {}
        for card in robustness_cards:
            regime_metrics = card.get("regime_metrics", {})
            if not isinstance(regime_metrics, dict):
                continue
            for regime_name, metrics in regime_metrics.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name, value in metrics.items():
                    if value is None:
                        continue
                    regime_metric_lists.setdefault(str(regime_name), {}).setdefault(str(metric_name), []).append(float(value))

        aggregated_regime_metrics: dict[str, dict[str, float | None]] = {}
        for regime_name, metrics in sorted(regime_metric_lists.items()):
            aggregated_regime_metrics[regime_name] = {
                metric_name: _mean(values) for metric_name, values in sorted(metrics.items())
            }

        action_audit: dict[str, dict[str, float | None]] = {}
        for cost in cost_keys:
            entries = [
                card.get("action_interface", {}).get(cost, {}).get("summary", {})
                for card in cards
                if isinstance(card.get("action_interface"), dict)
            ]
            if not entries:
                continue
            metrics = sorted({key for entry in entries if isinstance(entry, dict) for key in entry.keys()})
            action_audit[cost] = {
                metric_name: _mean(
                    [
                        entry.get(metric_name)
                        for entry in entries
                        if isinstance(entry, dict) and entry.get(metric_name) is not None
                    ]
                )
                for metric_name in metrics
            }

        return {
            "prediction": _predict_metrics(day_predictions, day_targets),
            "execution": {
                "anchor_level": execution_cards[0].get("anchor_level"),
                "anchor_cost_bps": execution_cards[0].get("anchor_cost_bps"),
                "post_cost_sharpe": _mean([card.get("post_cost_sharpe") for card in execution_cards]),
                "max_drawdown": _mean([card.get("max_drawdown") for card in execution_cards]),
                "turnover": _mean([card.get("turnover") for card in execution_cards]),
                "calmar": _mean([card.get("calmar") for card in execution_cards]),
                "net_return": _mean([card.get("net_return") for card in execution_cards]),
                "cost_ladder": aggregated_cost_ladder,
                "r1_cost_ladder": _aggregate_stage("R1"),
                "r2_cost_ladder": _aggregate_stage("R2"),
                "r3_cost_ladder": _aggregate_stage("R3"),
                "r4_cost_ladder": _aggregate_stage("R4"),
                "cvar95": _mean([card.get("cvar95") for card in execution_cards]),
                "average_holding_period": _mean([card.get("average_holding_period") for card in execution_cards]),
            },
            "gap": report_cards[0].get("gap", {}),
            "robustness": {
                "worst_regime_sharpe": _mean([card.get("worst_regime_sharpe") for card in robustness_cards]),
                "stress_day_survival": _mean([card.get("stress_day_survival") for card in robustness_cards]),
                "regime_variance": _mean([card.get("regime_variance") for card in robustness_cards]),
                "regime_metrics": aggregated_regime_metrics,
                "regime_counts": {},
                "regime_utilities": [
                    float(value)
                    for card in robustness_cards
                    for value in card.get("regime_utilities", [])
                        if value is not None
                ],
                "stress_intervals": [
                    item
                    for card in robustness_cards
                    for item in card.get("stress_intervals", [])
                    if isinstance(item, dict)
                ],
                "stress_days": [
                    item
                    for card in robustness_cards
                    for item in card.get("stress_days", [])
                    if isinstance(item, dict)
                ],
                "drawdown_violation_rate": _mean([card.get("drawdown_violation_rate") for card in robustness_cards]),
            },
            "auditability": {
                **report_cards[0].get("auditability", {}),
                "action_interface": action_audit.get(str(float(anchor_cost_bps))),
                "action_interface_by_cost": action_audit,
            },
            "realism_ladder": {
                "R1": {"cost_scan": _aggregate_stage("R1")},
                "R2": {"cost_scan": _aggregate_stage("R2")},
                "R3": {"cost_scan": _aggregate_stage("R3")},
                "R4": {"cost_scan": _aggregate_stage("R4")},
            },
        }

    for date in sorted({str(item) for item in dates.tolist()}):
        mask = dates == date
        if int(mask.sum()) == 0:
            continue
        day_predictions = predictions[mask]
        day_targets = targets[mask]
        day_asset_keys = asset_keys[mask]
        day_timestamps = timestamps[mask]
        day_lob = lob_states[mask]
        day_returns = market_returns[mask]
        asset_cards: list[dict[str, Any]] = []
        for asset_key in sorted({str(item) for item in day_asset_keys.tolist()}):
            asset_mask = day_asset_keys == asset_key
            order = np.argsort(day_timestamps[asset_mask])
            asset_predictions = day_predictions[asset_mask][order]
            asset_targets = day_targets[asset_mask][order]
            asset_timestamps = day_timestamps[asset_mask][order]
            asset_lob = day_lob[asset_mask][order]
            asset_returns = day_returns[asset_mask][order]
            actions_by_cost: dict[str, np.ndarray] = {}
            action_interface: dict[str, dict[str, Any]] = {}
            for cost in cost_bps_scan:
                key = str(float(cost))
                action_result = build_action_interface_result(
                    asset_predictions,
                    cost_bps=float(cost),
                    delta=action_delta,
                    rho=action_rho,
                )
                actions_by_cost[key] = action_result.wrapped_deltas
                action_payload = serialize_action_interface_result(action_result)
                action_payload["cost_bps"] = float(cost)
                action_payload["delta"] = float(action_delta)
                action_payload["rho"] = float(action_rho)
                action_interface[key] = action_payload
            ladder = run_realism_ladder(
                predictions=asset_predictions,
                targets=asset_targets,
                state_tokens=np.zeros(len(asset_predictions), dtype=np.int64),
                lob_states=asset_lob,
                market_returns=asset_returns,
                timestamps=asset_timestamps,
                actions_by_cost=actions_by_cost,
                anchor_cost_bps=anchor_cost_bps,
                cost_bps_scan=cost_bps_scan,
                action_metadata=action_interface,
            )
            asset_cards.append(
                {
                    "report_card": ladder["report_card"],
                    "realism_ladder": ladder["realism_ladder"],
                    "action_interface": action_interface,
                }
            )
        report_card = _aggregate_report_cards(asset_cards, day_predictions=day_predictions, day_targets=day_targets)
        payload = {
            "method": method_name,
            "date": date,
            "split": "test",
            "num_windows": int(mask.sum()),
            "notes": notes,
            "prediction_metrics": _predict_metrics(day_predictions, day_targets),
            "report_card": report_card,
            "realism_ladder": report_card["realism_ladder"],
            "action_interface": report_card["auditability"].get("action_interface_by_cost", {}),
        }
        report_path = output_root / f"{date}.json"
        report_path.write_text(json.dumps(payload, indent=2))
        report_paths.append(str(report_path.resolve()))
    return report_paths


def run_sklearn_baseline(
    *,
    method: str,
    train: dict[str, Any],
    val: dict[str, Any],
    test: dict[str, Any],
    output_dir: str | Path,
    seed: int = 7,
) -> ComparisonMethodResult:
    """Fit one sklearn/LightGBM baseline and emit per-day reports."""
    model, notes = _fit_sklearn_model(
        method=method,
        x_train=train["features"],
        y_train=train["targets"],
        x_val=val["features"],
        y_val=val["targets"],
        seed=seed,
    )
    train_pred = np.asarray(model.predict(train["features"]), dtype=np.float32)
    val_pred = np.asarray(model.predict(val["features"]), dtype=np.float32)
    test_pred = np.asarray(model.predict(test["features"]), dtype=np.float32)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "predictions.npz",
        train_predictions=train_pred,
        val_predictions=val_pred,
        test_predictions=test_pred,
        train_targets=train["targets"],
        val_targets=val["targets"],
        test_targets=test["targets"],
    )
    report_paths = generate_daily_reports(
        method_name=method,
        predictions=test_pred,
        targets=test["targets"],
        dates=test["dates"],
        asset_keys=test["asset_keys"],
        timestamps=test["timestamps"],
        lob_states=test["lob_states"],
        market_returns=test["targets"],
        output_dir=root / "reports",
        notes=notes,
    )
    summary = {
        "method": method,
        "notes": notes,
        "train_metrics": _predict_metrics(train_pred, train["targets"]),
        "val_metrics": _predict_metrics(val_pred, val["targets"]),
        "test_metrics": _predict_metrics(test_pred, test["targets"]),
        "report_paths": report_paths,
    }
    _write_json(root / "summary.json", summary)
    return ComparisonMethodResult(
        name=method,
        feature_mode="token_frequency" if "tokens" in method else "last_lob_snapshot",
        notes=notes,
        train_metrics=summary["train_metrics"],
        val_metrics=summary["val_metrics"],
        test_metrics=summary["test_metrics"],
        report_dir=str((root / "reports").resolve()),
    )


def _anchor_predictions(method: str, panel: dict[str, Any]) -> np.ndarray:
    targets = np.asarray(panel["targets"], dtype=np.float32)
    if method == "no_trade":
        return np.zeros_like(targets, dtype=np.float32)
    if method == "buy_and_hold":
        return np.ones_like(targets, dtype=np.float32)
    if method == "always_short":
        return -np.ones_like(targets, dtype=np.float32)
    if method == "ofi_threshold":
        lob_states = np.asarray(panel["lob_states"], dtype=np.float32)
        return lob_states[:, 2].sum(axis=(1, 2)).astype(np.float32)
    raise ValueError(f"Unsupported anchor baseline: {method}")


def run_anchor_baseline(
    *,
    method: str,
    train: dict[str, Any],
    val: dict[str, Any],
    test: dict[str, Any],
    output_dir: str | Path,
) -> ComparisonMethodResult:
    """Emit simple audit anchors and heuristics under the shared evaluator."""
    notes_map = {
        "no_trade": ["Constant-zero forecast anchor; unified action wrapper produces flat positions."],
        "buy_and_hold": ["Constant-positive forecast anchor; unified action wrapper produces persistent long exposure."],
        "always_short": ["Constant-negative forecast anchor; unified action wrapper produces persistent short exposure."],
        "ofi_threshold": ["Simple OFI heuristic using the summed OFI channel from the last LOB snapshot."],
    }
    train_pred = _anchor_predictions(method, train)
    val_pred = _anchor_predictions(method, val)
    test_pred = _anchor_predictions(method, test)
    notes = notes_map[method]

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "predictions.npz",
        train_predictions=train_pred,
        val_predictions=val_pred,
        test_predictions=test_pred,
        train_targets=train["targets"],
        val_targets=val["targets"],
        test_targets=test["targets"],
    )
    report_paths = generate_daily_reports(
        method_name=method,
        predictions=test_pred,
        targets=test["targets"],
        dates=test["dates"],
        asset_keys=test["asset_keys"],
        timestamps=test["timestamps"],
        lob_states=test["lob_states"],
        market_returns=test["targets"],
        output_dir=root / "reports",
        notes=notes,
    )
    summary = {
        "method": method,
        "notes": notes,
        "train_metrics": _predict_metrics(train_pred, train["targets"]),
        "val_metrics": _predict_metrics(val_pred, val["targets"]),
        "test_metrics": _predict_metrics(test_pred, test["targets"]),
        "report_paths": report_paths,
    }
    _write_json(root / "summary.json", summary)
    return ComparisonMethodResult(
        name=method,
        feature_mode="anchor" if method != "ofi_threshold" else "heuristic",
        notes=notes,
        train_metrics=summary["train_metrics"],
        val_metrics=summary["val_metrics"],
        test_metrics=summary["test_metrics"],
        report_dir=str((root / "reports").resolve()),
    )


def ensure_vq_and_encodings(
    *,
    train_panel_path: str | Path,
    val_panel_path: str | Path,
    test_panel_path: str | Path,
    output_dir: str | Path,
    device: str,
    batch_size: int = 256,
    epochs: int = 5,
    seed: int = 7,
    num_embeddings: int = 64,
    latent_dim: int = 64,
    max_train_items: int | None = None,
) -> dict[str, str]:
    """Train one 64-code VQ-VAE and export train/val/test encodings."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        return {
            "train_encoded": str(payload["train_encoded"]),
            "val_encoded": str(payload["val_encoded"]),
            "test_encoded": str(payload["test_encoded"]),
            "checkpoint": str(payload["checkpoint"]),
        }

    train_cfg = VQVAETrainConfig(
        dataset_path=str(train_panel_path),
        output_dir=str(root),
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        amp=device.startswith("cuda"),
        seed=seed,
        max_train_items=max_train_items,
        model=VQVAEConfig(
            num_embeddings=num_embeddings,
            latent_dim=latent_dim,
        ),
    )
    train_summary = train_vqvae(train_cfg)
    fallback = train_summary.get("fallback_codebook")
    val_encoded = root / "encoded_val.npz"
    test_encoded = root / "encoded_test.npz"
    export_vqvae_encodings(
        checkpoint_path=train_summary["checkpoint"],
        dataset_path=str(val_panel_path),
        output_path=str(val_encoded),
        batch_size=batch_size,
        device=device,
        fallback_codebook_path=fallback,
    )
    export_vqvae_encodings(
        checkpoint_path=train_summary["checkpoint"],
        dataset_path=str(test_panel_path),
        output_path=str(test_encoded),
        batch_size=batch_size,
        device=device,
        fallback_codebook_path=fallback,
    )
    payload = {
        "checkpoint": train_summary["checkpoint"],
        "train_encoded": train_summary["encoded_tokens"],
        "val_encoded": str(val_encoded.resolve()),
        "test_encoded": str(test_encoded.resolve()),
        "codes_source": train_summary.get("codes_source"),
        "num_unique_codes": train_summary.get("num_unique_codes"),
        "fallback_codebook": fallback,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    return {
        "train_encoded": str(payload["train_encoded"]),
        "val_encoded": str(payload["val_encoded"]),
        "test_encoded": str(payload["test_encoded"]),
        "checkpoint": str(payload["checkpoint"]),
    }


def _load_kairos_model(checkpoint_path: str | Path, device: torch.device) -> KairosBaselineModel:
    payload = torch.load(checkpoint_path, map_location=device)
    config = KairosBaselineConfig(**payload["config"])
    model = KairosBaselineModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model


def _predict_kairos(
    *,
    checkpoint_path: str | Path,
    encoded_path: str | Path,
    panel_path: str | Path,
    seq_len: int,
    stride: int,
    target_key: str,
    device: str,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    spec = DatasetSpec(seq_len=seq_len, stride=stride, target_key=target_key)
    dataset = LOBTokenSequenceDataset(encoded_path, panel_path, spec)
    inputs: list[np.ndarray] = []
    targets: list[float] = []
    for index in range(len(dataset)):
        sample, target = dataset[index]
        inputs.append(sample.numpy())
        targets.append(float(target.item()))
    if not inputs:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    features = np.stack(inputs).astype(np.int64)
    y_true = np.asarray(targets, dtype=np.float32)
    torch_device = resolve_torch_device(device)
    model = _load_kairos_model(checkpoint_path, torch_device)
    predictions_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            tensor = torch.from_numpy(features[start:stop]).long().to(torch_device)
            batch_pred = model(input_ids=tensor)["return_pred"].cpu().numpy().astype(np.float32)
            predictions_batches.append(batch_pred)
    predictions = np.concatenate(predictions_batches, axis=0) if predictions_batches else np.zeros(0, dtype=np.float32)
    return predictions, y_true


def run_kairos_baseline(
    *,
    train_panel: PanelSpec,
    val_panel: PanelSpec,
    test_panel: PanelSpec,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    device: str,
    vocab_size: int = 64,
    epochs: int = 5,
    batch_size: int = 128,
    seed: int = 7,
) -> ComparisonMethodResult:
    """Train the Transformer baseline on exported 64-code windows."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"

    if not summary_path.exists():
        existing_best = Path(checkpoint_dir) / "best_baseline.pt"
        if existing_best.exists():
            checkpoint_path = str(existing_best.resolve())
        else:
            config = BaselineTrainConfig(
                encoded_path=str(train_panel.encoded_path),
                dataset_path=train_panel.panel_path,
                val_encoded_path=str(val_panel.encoded_path),
                val_dataset_path=val_panel.panel_path,
                test_encoded_path=str(test_panel.encoded_path),
                test_dataset_path=test_panel.panel_path,
                output_dir=str(checkpoint_dir),
                input_mode="codes",
                seq_len=train_panel.seq_len,
                stride=train_panel.stride,
                target_key=train_panel.target_key,
                batch_size=batch_size,
                epochs=epochs,
                device=device,
                amp=device.startswith("cuda"),
                seed=seed,
                model=KairosBaselineConfig(
                    input_mode="codes",
                    vocab_size=vocab_size,
                    seq_len=train_panel.seq_len,
                ),
            )
            train_summary = train_baseline(config)
            _write_json(root / "train_summary.json", train_summary)
            checkpoint_path = str(train_summary["best_checkpoint"])
    else:
        checkpoint_path = json.loads(summary_path.read_text())["checkpoint_path"]

    train_pred, train_targets = _predict_kairos(
        checkpoint_path=checkpoint_path,
        encoded_path=str(train_panel.encoded_path),
        panel_path=train_panel.panel_path,
        seq_len=train_panel.seq_len,
        stride=train_panel.stride,
        target_key=train_panel.target_key,
        device=device,
        batch_size=2048,
    )
    val_pred, val_targets = _predict_kairos(
        checkpoint_path=checkpoint_path,
        encoded_path=str(val_panel.encoded_path),
        panel_path=val_panel.panel_path,
        seq_len=val_panel.seq_len,
        stride=val_panel.stride,
        target_key=val_panel.target_key,
        device=device,
        batch_size=2048,
    )
    test_pred, test_targets = _predict_kairos(
        checkpoint_path=checkpoint_path,
        encoded_path=str(test_panel.encoded_path),
        panel_path=test_panel.panel_path,
        seq_len=test_panel.seq_len,
        stride=test_panel.stride,
        target_key=test_panel.target_key,
        device=device,
        batch_size=2048,
    )
    test_rows = _load_rows(test_panel.index_path)
    test_arrays = _window_core_arrays(test_panel.panel_path, test_rows, test_panel.target_key)
    notes = ["Kairos Transformer on 64-code windows; leakage-controlled seq=64, stride=50."]
    report_paths = generate_daily_reports(
        method_name="kairos_64code",
        predictions=test_pred,
        targets=test_targets,
        dates=test_arrays["dates"],
        asset_keys=test_arrays["asset_keys"],
        timestamps=test_arrays["timestamps"],
        lob_states=test_arrays["lob_states"],
        market_returns=test_arrays["targets"],
        output_dir=root / "reports",
        notes=notes,
    )
    summary = {
        "method": "kairos_64code",
        "checkpoint_path": checkpoint_path,
        "notes": notes,
        "train_metrics": _predict_metrics(train_pred, train_targets),
        "val_metrics": _predict_metrics(val_pred, val_targets),
        "test_metrics": _predict_metrics(test_pred, test_targets),
        "report_paths": report_paths,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        root / "predictions.npz",
        train_predictions=train_pred,
        val_predictions=val_pred,
        test_predictions=test_pred,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
    )
    return ComparisonMethodResult(
        name="kairos_64code",
        feature_mode="codes",
        notes=notes,
        train_metrics=summary["train_metrics"],
        val_metrics=summary["val_metrics"],
        test_metrics=summary["test_metrics"],
        report_dir=str((root / "reports").resolve()),
    )


def run_deeplob_baseline(
    *,
    train_panel: PanelSpec,
    val_panel: PanelSpec,
    test_panel: PanelSpec,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    device: str,
    epochs: int = 3,
    batch_size: int = 256,
    seed: int = 7,
) -> ComparisonMethodResult:
    """Train a DeepLOB-style time-bar baseline and emit daily reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"

    if not summary_path.exists():
        existing_best = Path(checkpoint_dir) / "best_deeplob.pt"
        if existing_best.exists():
            checkpoint_path = str(existing_best.resolve())
        else:
            train_summary = train_deeplob(
                DeepLOBTrainConfig(
                    train_dataset_path=train_panel.panel_path,
                    val_dataset_path=val_panel.panel_path,
                    test_dataset_path=test_panel.panel_path,
                    output_dir=str(checkpoint_dir),
                    seq_len=train_panel.seq_len,
                    stride=train_panel.stride,
                    target_key=train_panel.target_key,
                    batch_size=batch_size,
                    epochs=epochs,
                    device=device,
                    amp=device.startswith("cuda"),
                    seed=seed,
                    model=DeepLOBConfig(seq_len=train_panel.seq_len),
                )
            )
            _write_json(root / "train_summary.json", train_summary)
            checkpoint_path = str(train_summary["best_checkpoint"])
    else:
        checkpoint_path = json.loads(summary_path.read_text())["checkpoint_path"]

    train_pred, train_targets = predict_deeplob(
        checkpoint_path=checkpoint_path,
        dataset_path=train_panel.panel_path,
        seq_len=train_panel.seq_len,
        stride=train_panel.stride,
        target_key=train_panel.target_key,
        device=device,
    )
    val_pred, val_targets = predict_deeplob(
        checkpoint_path=checkpoint_path,
        dataset_path=val_panel.panel_path,
        seq_len=val_panel.seq_len,
        stride=val_panel.stride,
        target_key=val_panel.target_key,
        device=device,
    )
    test_pred, test_targets = predict_deeplob(
        checkpoint_path=checkpoint_path,
        dataset_path=test_panel.panel_path,
        seq_len=test_panel.seq_len,
        stride=test_panel.stride,
        target_key=test_panel.target_key,
        device=device,
    )
    test_rows = _load_rows(test_panel.index_path)
    test_arrays = _window_core_arrays(test_panel.panel_path, test_rows, test_panel.target_key)
    notes = ["DeepLOB-inspired CNN+LSTM on raw time-bar LOB windows; seq=50, 3 epochs."]
    report_paths = generate_daily_reports(
        method_name="deeplob_timebars",
        predictions=test_pred,
        targets=test_targets,
        dates=test_arrays["dates"],
        asset_keys=test_arrays["asset_keys"],
        timestamps=test_arrays["timestamps"],
        lob_states=test_arrays["lob_states"],
        market_returns=test_arrays["targets"],
        output_dir=root / "reports",
        notes=notes,
    )
    summary = {
        "method": "deeplob_timebars",
        "checkpoint_path": checkpoint_path,
        "notes": notes,
        "train_metrics": _predict_metrics(train_pred, train_targets),
        "val_metrics": _predict_metrics(val_pred, val_targets),
        "test_metrics": _predict_metrics(test_pred, test_targets),
        "report_paths": report_paths,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        root / "predictions.npz",
        train_predictions=train_pred,
        val_predictions=val_pred,
        test_predictions=test_pred,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
    )
    return ComparisonMethodResult(
        name="deeplob_timebars",
        feature_mode="raw_time_bars",
        notes=notes,
        train_metrics=summary["train_metrics"],
        val_metrics=summary["val_metrics"],
        test_metrics=summary["test_metrics"],
        report_dir=str((root / "reports").resolve()),
    )
