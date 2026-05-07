#!/usr/bin/env python3
"""Run reviewer-prioritized target baselines for the frozen 90D panels.

The methods here are benchmark-native implementations intended to close the
"target-only" evidence gap without mutating the original baseline outputs:

* elasticnet_tokens: scikit-learn ElasticNet on VQ token-frequency features.
* tabl_lite: compact temporal-attention bilinear-style LOB network.
* itransformer_lite: inverted-transformer-style network on benchmark features.

The names deliberately include ``_lite`` where the implementation is not a
verbatim external repository reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.benchmark_baselines import (
    _load_rows,
    _predict_metrics,
    _window_core_arrays,
    build_token_frequency_features,
    generate_daily_reports,
)
from src.eval.benchmark_evaluator import (
    BenchmarkEvaluatorConfig,
    BenchmarkMethodSpec,
    _load_split_lookup_from_split_plan,
    evaluate_benchmark_methods,
)


PANEL_SPECS: dict[str, dict[str, str]] = {
    "time_h5": {
        "mode": "time",
        "target_key": "return_5",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_h5_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs",
        "bundle_manifest": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_bundle/manifest.json",
    },
    "time_h20": {
        "mode": "time",
        "target_key": "return_20",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs",
        "bundle_manifest": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_bundle/manifest.json",
    },
    "time_h50": {
        "mode": "time",
        "target_key": "return_50",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_h50_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_standardized_inputs",
        "bundle_manifest": "/root/autodl-tmp/peg_bench_runs/tardis_90d_time_bundle/manifest.json",
    },
    "volume_h5": {
        "mode": "volume",
        "target_key": "return_5",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h5_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs",
        "bundle_manifest": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_bundle/manifest.json",
    },
    "volume_h20": {
        "mode": "volume",
        "target_key": "return_20",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs",
        "bundle_manifest": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_bundle/manifest.json",
    },
    "volume_h50": {
        "mode": "volume",
        "target_key": "return_50",
        "compare_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_h50_table2_compare_v1",
        "standardized_root": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_standardized_inputs",
        "bundle_manifest": "/root/autodl-tmp/peg_bench_runs/tardis_90d_volume_bundle/manifest.json",
    },
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def panel_path(spec: dict[str, str], split: str) -> Path:
    return Path(spec["standardized_root"]) / "panels" / f"{spec['mode']}_{split}.npz"


def index_path(spec: dict[str, str], split: str, seq_len: int) -> Path:
    return Path(spec["standardized_root"]) / "manifests" / f"{spec['mode']}_{split}_window_index_seq{seq_len}.csv"


def encoded_path(spec: dict[str, str], split: str) -> Path:
    return Path(spec["compare_root"]) / "vqvae_64code" / ("encoded_tokens.npz" if split == "train" else f"encoded_{split}.npz")


def load_window_arrays(spec: dict[str, str], split: str, seq_len: int) -> dict[str, Any]:
    rows = _load_rows(index_path(spec, split, seq_len))
    arrays = _window_core_arrays(panel_path(spec, split), rows, spec["target_key"])
    arrays["rows"] = rows
    return arrays


def save_predictions(
    root: Path,
    *,
    train_pred: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "predictions.npz",
        train_predictions=train_pred.astype(np.float32),
        val_predictions=val_pred.astype(np.float32),
        test_predictions=test_pred.astype(np.float32),
        train_targets=train_targets.astype(np.float32),
        val_targets=val_targets.astype(np.float32),
        test_targets=test_targets.astype(np.float32),
    )


def emit_method_outputs(
    *,
    method: str,
    feature_mode: str,
    notes: list[str],
    output_root: Path,
    spec: dict[str, str],
    train_pred: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    test_arrays: dict[str, Any],
) -> dict[str, Any]:
    method_root = output_root / "methods" / method
    save_predictions(
        method_root,
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
    )
    report_paths = generate_daily_reports(
        method_name=method,
        predictions=test_pred,
        targets=test_targets,
        dates=test_arrays["dates"],
        asset_keys=test_arrays["asset_keys"],
        timestamps=test_arrays["timestamps"],
        lob_states=test_arrays["lob_states"],
        market_returns=test_arrays["targets"],
        output_dir=method_root / "reports",
        notes=notes,
    )
    summary = {
        "method": method,
        "feature_mode": feature_mode,
        "notes": notes,
        "train_metrics": _predict_metrics(train_pred, train_targets),
        "val_metrics": _predict_metrics(val_pred, val_targets),
        "test_metrics": _predict_metrics(test_pred, test_targets),
        "report_paths": report_paths,
    }
    (method_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return {
        "method": method,
        "feature_mode": feature_mode,
        "test_directional_accuracy": summary["test_metrics"]["directional_accuracy"],
        "test_rank_ic": summary["test_metrics"]["rank_ic"],
        "test_mse": summary["test_metrics"]["mse"],
        "test_mae": summary["test_metrics"]["mae"],
        "report_dir": str((method_root / "reports").resolve()),
        "notes": " | ".join(notes),
    }


def run_elasticnet(spec: dict[str, str], output_root: Path, seq_len: int, seed: int) -> dict[str, Any]:
    train = build_token_frequency_features(
        panel_path=panel_path(spec, "train"),
        index_path=index_path(spec, "train", seq_len),
        encoded_path=encoded_path(spec, "train"),
        vocab_size=64,
        target_key=spec["target_key"],
    )
    val = build_token_frequency_features(
        panel_path=panel_path(spec, "val"),
        index_path=index_path(spec, "val", seq_len),
        encoded_path=encoded_path(spec, "val"),
        vocab_size=64,
        target_key=spec["target_key"],
    )
    test = build_token_frequency_features(
        panel_path=panel_path(spec, "test"),
        index_path=index_path(spec, "test", seq_len),
        encoded_path=encoded_path(spec, "test"),
        vocab_size=64,
        target_key=spec["target_key"],
    )
    best_model = None
    best_cfg = None
    best_mse = float("inf")
    for alpha in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
        for l1_ratio in (0.15, 0.5, 0.85):
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "elasticnet",
                        ElasticNet(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            model.fit(train["features"], train["targets"])
            val_pred = np.asarray(model.predict(val["features"]), dtype=np.float32)
            val_mse = float(np.mean((val_pred - val["targets"]) ** 2))
            if val_mse < best_mse:
                best_mse = val_mse
                best_model = model
                best_cfg = {"alpha": alpha, "l1_ratio": l1_ratio, "val_mse": val_mse}
    assert best_model is not None and best_cfg is not None
    method_root = output_root / "methods" / "elasticnet_tokens"
    method_root.mkdir(parents=True, exist_ok=True)
    (method_root / "selected_config.json").write_text(json.dumps(best_cfg, indent=2))
    train_pred = np.asarray(best_model.predict(train["features"]), dtype=np.float32)
    val_pred = np.asarray(best_model.predict(val["features"]), dtype=np.float32)
    test_pred = np.asarray(best_model.predict(test["features"]), dtype=np.float32)
    notes = [f"ElasticNet token-frequency baseline; selected {best_cfg} on validation MSE."]
    return emit_method_outputs(
        method="elasticnet_tokens",
        feature_mode="token_frequency",
        notes=notes,
        output_root=output_root,
        spec=spec,
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
        train_targets=train["targets"],
        val_targets=val["targets"],
        test_targets=test["targets"],
        test_arrays=test,
    )


class WindowTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, panel_file: Path, index_file: Path, target_key: str, *, input_key: str) -> None:
        self.payload = np.load(panel_file)
        self.rows = _load_rows(index_file)
        self.target_key = target_key
        self.input_key = input_key
        self.data = self.payload[input_key].astype(np.float32)
        self.targets = np.nan_to_num(self.payload[target_key].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        start = int(row["window_start_index"])
        stop = int(row["window_end_index_exclusive"])
        target_index = int(row["target_index"])
        x = self.data[start:stop]
        if self.input_key == "lob":
            x = x.reshape(x.shape[0], -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.targets[target_index]
        return torch.from_numpy(x).float(), torch.tensor(float(y), dtype=torch.float32)


class TABLLite(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.time_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.context_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        h = torch.tanh(self.feature_proj(x))
        weights = torch.softmax(self.time_gate(h), dim=1)
        context = (weights * h).sum(dim=1)
        gated = context * self.context_gate(context)
        return self.head(gated).squeeze(-1)


class ITransformerLite(nn.Module):
    def __init__(self, seq_len: int, num_vars: int, d_model: int = 96, nhead: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.time_projection = nn.Linear(seq_len, d_model)
        self.var_embedding = nn.Parameter(torch.zeros(1, num_vars, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inverted TS representation: variables become tokens, time is embedded.
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        tokens = self.time_projection(x.transpose(1, 2)) + self.var_embedding
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def select_subset(dataset: Dataset, max_items: int | None, seed: int) -> Dataset:
    if max_items is None or len(dataset) <= max_items:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_items, replace=False))
    return Subset(dataset, indices.tolist())


def collect_predictions(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int, y_mean: float, y_std: float) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).detach().cpu().numpy().astype(np.float32) * y_std + y_mean
            preds.append(pred)
            targets.append(y.numpy().astype(np.float32))
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32), np.concatenate(targets) if targets else np.zeros(0, dtype=np.float32)


def train_neural(
    *,
    method: str,
    spec: dict[str, str],
    output_root: Path,
    seq_len: int,
    device_name: str,
    epochs: int,
    batch_size: int,
    max_train_windows: int,
    seed: int,
) -> dict[str, Any]:
    input_key = "lob" if method == "tabl_lite" else "benchmark_features"
    train_ds_full = WindowTensorDataset(panel_path(spec, "train"), index_path(spec, "train", seq_len), spec["target_key"], input_key=input_key)
    val_ds = WindowTensorDataset(panel_path(spec, "val"), index_path(spec, "val", seq_len), spec["target_key"], input_key=input_key)
    test_ds = WindowTensorDataset(panel_path(spec, "test"), index_path(spec, "test", seq_len), spec["target_key"], input_key=input_key)
    train_ds = select_subset(train_ds_full, max_train_windows, seed)

    sample_x, _ = train_ds[0]
    if method == "tabl_lite":
        model = TABLLite(input_dim=int(sample_x.shape[-1]))
        feature_mode = "raw_lob_temporal_attention"
        notes = ["TABL-lite benchmark-native temporal-attention bilinear-style network on raw LOB windows."]
    elif method == "itransformer_lite":
        model = ITransformerLite(seq_len=seq_len, num_vars=int(sample_x.shape[-1]))
        feature_mode = "benchmark_feature_inverted_transformer"
        notes = ["iTransformer-lite benchmark-native inverted Transformer on standardized benchmark-feature windows."]
    else:
        raise ValueError(method)

    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    model.to(device)
    y_values = []
    for _, y in DataLoader(train_ds, batch_size=8192, shuffle=False, num_workers=0):
        y_values.append(y.numpy())
    y_all = np.concatenate(y_values).astype(np.float32)
    y_mean = float(np.mean(y_all))
    y_std = float(np.std(y_all) + 1e-8)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_state = None
    best_val_mse = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_scaled = ((y.to(device, non_blocking=True) - y_mean) / y_std).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(x)
                loss = torch.mean((pred - y_scaled) ** 2)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
        val_pred, val_targets = collect_predictions(model, val_ds, device, batch_size=2048, y_mean=y_mean, y_std=y_std)
        val_mse = float(np.mean((val_pred - val_targets) ** 2))
        val_rankic = _predict_metrics(val_pred, val_targets)["rank_ic"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mse": val_mse, "val_rank_ic": val_rankic})
        print(f"[target] {method} epoch={epoch} train_loss={np.mean(losses):.6f} val_mse={val_mse:.8g} val_rankic={val_rankic:.4f}", flush=True)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    method_root = output_root / "methods" / method
    method_root.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "method": method, "seq_len": seq_len, "y_mean": y_mean, "y_std": y_std}, method_root / "best_model.pt")
    (method_root / "train_history.json").write_text(json.dumps(history, indent=2))

    train_pred, train_targets = collect_predictions(model, train_ds_full, device, batch_size=2048, y_mean=y_mean, y_std=y_std)
    val_pred, val_targets = collect_predictions(model, val_ds, device, batch_size=2048, y_mean=y_mean, y_std=y_std)
    test_pred, test_targets = collect_predictions(model, test_ds, device, batch_size=2048, y_mean=y_mean, y_std=y_std)
    test_arrays = load_window_arrays(spec, "test", seq_len)
    return emit_method_outputs(
        method=method,
        feature_mode=feature_mode,
        notes=notes,
        output_root=output_root,
        spec=spec,
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
        test_arrays=test_arrays,
    )


def aggregate_panel(output_root: Path, rows: list[dict[str, Any]], spec: dict[str, str]) -> None:
    write_csv(output_root / "results_table.csv", rows)
    (output_root / "results_table.json").write_text(json.dumps(rows, indent=2))
    method_specs = []
    for row in rows:
        report_dir = Path(str(row["report_dir"]))
        method_specs.append(
            BenchmarkMethodSpec(
                name=str(row["method"]),
                reports=tuple(str(path.resolve()) for path in sorted(report_dir.glob("*.json"))),
            )
        )
    bundle = json.loads(Path(spec["bundle_manifest"]).read_text())
    split_lookup = {day: str(meta["split"]) for day, meta in bundle.get("split_index", {}).items()}
    report = evaluate_benchmark_methods(
        method_specs=method_specs,
        split_lookup=split_lookup,
        config=BenchmarkEvaluatorConfig(
            output_path=str(output_root / "benchmark_report.json"),
            markdown_path=str(output_root / "benchmark_report.md"),
            evaluation_split="test",
        ),
    )
    (output_root / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    summary = {
        "results_table_csv": str((output_root / "results_table.csv").resolve()),
        "benchmark_report_json": str((output_root / "benchmark_report.json").resolve()),
        "methods": rows,
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True, choices=sorted(PANEL_SPECS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", default="elasticnet_tokens,tabl_lite,itransformer_lite")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-train-windows", type=int, default=160000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = PANEL_SPECS[args.panel]
    output_root = Path(args.output_root) / args.panel
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    for method in methods:
        print(f"[target] panel={args.panel} method={method}", flush=True)
        if method == "elasticnet_tokens":
            rows.append(run_elasticnet(spec, output_root, args.seq_len, args.seed))
        elif method in {"tabl_lite", "itransformer_lite"}:
            rows.append(
                train_neural(
                    method=method,
                    spec=spec,
                    output_root=output_root,
                    seq_len=args.seq_len,
                    device_name=args.device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_train_windows=args.max_train_windows,
                    seed=args.seed,
                )
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        aggregate_panel(output_root, rows, spec)
    print(f"[target] completed panel={args.panel} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
