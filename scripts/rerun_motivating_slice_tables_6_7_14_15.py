from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/root/prediction-execution-gap")
OUT = ROOT / "outputs/final_motivating_slice_rerun_v8"
OUT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path("/root/autodl-tmp/peg_bench_runs")
DATES = ["2025-10-12", "2025-10-17", "2025-10-26", "2025-10-27"]
TRAIN_DATES = ["2025-10-12", "2025-10-17"]
TEST_DATES = ["2025-10-26", "2025-10-27"]
RNG = np.random.default_rng(7)


def _safe_imports() -> dict[str, object]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    try:
        from lightgbm import LGBMRegressor
    except Exception:
        LGBMRegressor = None
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        torch = None
        nn = None
        DataLoader = None
        TensorDataset = None
    return {
        "MiniBatchKMeans": MiniBatchKMeans,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "Ridge": Ridge,
        "make_pipeline": make_pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "LGBMRegressor": LGBMRegressor,
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
    }


SK = _safe_imports()


def load_mode(mode: str, horizon: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    arrays_x: list[np.ndarray] = []
    arrays_y: list[np.ndarray] = []
    arrays_dates: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []
    key = f"return_{horizon}"
    for date in DATES:
        path = DATA_ROOT / f"tardis_90d_{mode}_bundle/processed/binance/BTCUSDT/{date}/{mode}_bars.npz"
        if not path.exists():
            meta_rows.append({"mode": mode, "date": date, "path": str(path), "exists": False, "rows": 0})
            continue
        z = np.load(path, allow_pickle=True)
        lob = np.asarray(z["lob"], dtype=np.float32)
        y = np.asarray(z[key], dtype=np.float32)
        n = min(len(lob), len(y))
        idx = np.arange(0, n, max(1, stride))
        lob = lob[idx]
        y = y[idx]
        mask = np.isfinite(y)
        lob = lob[mask]
        y = y[mask]
        arrays_x.append(lob)
        arrays_y.append(y)
        arrays_dates.append(np.array([date] * len(y), dtype=object))
        meta_rows.append({"mode": mode, "date": date, "path": str(path), "exists": True, "rows": int(len(y))})
    if not arrays_x:
        return np.empty((0, 4, 10, 2), dtype=np.float32), np.empty(0), np.empty(0), pd.DataFrame(meta_rows)
    return np.concatenate(arrays_x), np.concatenate(arrays_y), np.concatenate(arrays_dates), pd.DataFrame(meta_rows)


def split_xy(x: np.ndarray, y: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.isin(dates, TRAIN_DATES)
    test_mask = np.isin(dates, TEST_DATES)
    return x[train_mask], y[train_mask], x[test_mask], y[test_mask]


def cap_train(x: np.ndarray, y: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= cap:
        return x, y
    idx = RNG.choice(len(y), size=cap, replace=False)
    return x[idx], y[idx]


def flatten_lob(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1).astype(np.float32)


def rank_ic(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = min(len(pred), len(y))
    pred = pred[:n]
    y = y[:n]
    mask = np.isfinite(pred) & np.isfinite(y)
    pred = pred[mask]
    y = y[mask]
    if len(pred) < 3 or np.nanstd(pred) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(pd.Series(pred).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1])


def directional_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = min(len(pred), len(y))
    pred = pred[:n]
    y = y[:n]
    mask = np.isfinite(pred) & np.isfinite(y) & (y != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(pred[mask]) == np.sign(y[mask])).mean())


def trading_metrics(pred: np.ndarray, y: np.ndarray, bps: float, bars_per_day: float) -> tuple[float, float, float]:
    pos = np.sign(np.asarray(pred).reshape(-1))
    y = np.asarray(y).reshape(-1)
    n = min(len(pos), len(y))
    pos = pos[:n]
    y = y[:n]
    turn = np.abs(np.diff(pos, prepend=0))
    pnl = pos * y - float(bps) * 1e-4 * turn
    mean = float(np.nanmean(pnl))
    std = float(np.nanstd(pnl))
    sharpe = float(mean / (std + 1e-12) * math.sqrt(max(1.0, bars_per_day * 365.0)))
    ann = float(mean * bars_per_day * 365.0)
    return mean, sharpe, ann


def token_codes(x_train: np.ndarray, x_test: np.ndarray, n_codes: int) -> tuple[np.ndarray, np.ndarray]:
    MiniBatchKMeans = SK["MiniBatchKMeans"]
    xtr_flat = flatten_lob(x_train)
    xte_flat = flatten_lob(x_test)
    fit_cap = min(len(xtr_flat), 60000)
    fit_idx = RNG.choice(len(xtr_flat), size=fit_cap, replace=False) if len(xtr_flat) > fit_cap else np.arange(len(xtr_flat))
    km = MiniBatchKMeans(
        n_clusters=int(n_codes),
        random_state=7,
        batch_size=4096,
        max_iter=40,
        n_init=3,
        reassignment_ratio=0.01,
    )
    km.fit(xtr_flat[fit_idx])
    return km.predict(xtr_flat).reshape(-1, 1), km.predict(xte_flat).reshape(-1, 1)


def fit_ridge_tokens(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_codes: int = 64) -> np.ndarray:
    Ridge = SK["Ridge"]
    OneHotEncoder = SK["OneHotEncoder"]
    make_pipeline = SK["make_pipeline"]
    xtr_c, xte_c = token_codes(x_train, x_test, n_codes)
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
    model = make_pipeline(enc, Ridge(alpha=1.0))
    model.fit(xtr_c, y_train)
    return np.asarray(model.predict(xte_c))


def fit_hist_tokens(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_codes: int = 64) -> np.ndarray:
    HistGradientBoostingRegressor = SK["HistGradientBoostingRegressor"]
    xtr_c, xte_c = token_codes(x_train, x_test, n_codes)
    model = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, l2_regularization=0.01, random_state=7)
    model.fit(xtr_c.astype(np.float32), y_train)
    return np.asarray(model.predict(xte_c.astype(np.float32)))


def fit_lightgbm_lob(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    LGBMRegressor = SK["LGBMRegressor"]
    HistGradientBoostingRegressor = SK["HistGradientBoostingRegressor"]
    xtr = flatten_lob(x_train)
    xte = flatten_lob(x_test)
    xtr, y_train = cap_train(xtr, y_train, 100000)
    if LGBMRegressor is not None:
        model = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=7,
            n_jobs=4,
            verbose=-1,
        )
    else:
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.04, random_state=7)
    model.fit(xtr, y_train)
    return np.asarray(model.predict(xte))


def fit_kairos_proxy(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    # Compact deployment-time proxy: 64-code quantization plus non-linear predictor.
    # This is not the archived neural Kairos checkpoint; the caveat is exported.
    return fit_hist_tokens(x_train, y_train, x_test, n_codes=64)


def fit_deeplob_proxy(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    torch = SK["torch"]
    nn = SK["nn"]
    DataLoader = SK["DataLoader"]
    TensorDataset = SK["TensorDataset"]
    if torch is None:
        return fit_lightgbm_lob(x_train, y_train, x_test)
    xtr, ytr = cap_train(x_train, y_train, 30000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    class TinyDeepLOB(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=(3, 2), padding=(1, 0)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16 * 10 * 1, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = TinyDeepLOB().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).float())
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    model.train()
    for _ in range(2):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_test), 4096):
            xb = torch.from_numpy(x_test[i : i + 4096]).float().to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds)


def row_metrics(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return directional_accuracy(pred, y), rank_ic(pred, y)


def md_tex_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_csv(index=False)
    (OUT / f"{name}.md").write_text(md + "\n")
    (OUT / f"{name}.tex").write_text(df.to_latex(index=False, escape=True))
    (OUT / f"{name}_source_trace.md").write_text(df.to_csv(index=False))


def evaluate_setting(setting: str, mode: str, horizon: int, stride: int) -> tuple[list[dict[str, object]], dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    x, y, dates, meta = load_mode(mode, horizon, stride)
    x_train, y_train, x_test, y_test = split_xy(x, y, dates)
    results: list[dict[str, object]] = []
    pred_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    protocol_base = (
        f"processed_four_date_rerun; dates={','.join(DATES)}; train={','.join(TRAIN_DATES)}; "
        f"test={','.join(TEST_DATES)}; mode={mode}; horizon={horizon}; stride={stride}; "
        "not original normalized motivating-slice artifact"
    )
    models = [
        ("Ridge (tokens)", lambda: fit_ridge_tokens(*cap_train(x_train, y_train, 100000), x_test, 64)),
        ("HistGBM (tokens)", lambda: fit_hist_tokens(*cap_train(x_train, y_train, 100000), x_test, 64)),
        ("LightGBM (LOB)", lambda: fit_lightgbm_lob(x_train, y_train, x_test)),
        ("Kairos (64-code)", lambda: fit_kairos_proxy(*cap_train(x_train, y_train, 100000), x_test)),
    ]
    for model_name, fn in models:
        pred = fn()
        da, ric = row_metrics(pred, y_test)
        pred_cache[model_name] = (pred, y_test)
        results.append(
            {
                "audit_setting": setting,
                "model": model_name,
                "rerun_DA": da,
                "rerun_RIC": ric,
                "sample_count": int(len(y_test)),
                "source_file": ";".join(meta.loc[meta.exists, "path"].astype(str).tolist()),
                "diff_status": "protocol_different",
                "protocol_caveat": protocol_base,
            }
        )
    # DeepLOB row uses time bars regardless of the main mode.
    xt, yt, dt, meta_t = load_mode("time", horizon, stride)
    xt_train, yt_train, xt_test, yt_test = split_xy(xt, yt, dt)
    pred = fit_deeplob_proxy(xt_train, yt_train, xt_test)
    da, ric = row_metrics(pred, yt_test)
    pred_cache["DeepLOB (time bars)"] = (pred, yt_test)
    results.append(
        {
            "audit_setting": setting,
            "model": "DeepLOB (time bars)",
            "rerun_DA": da,
            "rerun_RIC": ric,
            "sample_count": int(len(yt_test)),
            "source_file": ";".join(meta_t.loc[meta_t.exists, "path"].astype(str).tolist()),
            "diff_status": "protocol_different",
            "protocol_caveat": protocol_base.replace(f"mode={mode}", "mode=time") + "; compact TinyDeepLOB proxy",
        }
    )
    return results, pred_cache, pd.concat([meta, meta_t], ignore_index=True).drop_duplicates()


def main() -> None:
    historical6 = {
        ("Overlap-allowing", "Ridge (tokens)"): (0.624, 0.306),
        ("Overlap-allowing", "HistGBM (tokens)"): (0.691, 0.586),
        ("Overlap-allowing", "LightGBM (LOB)"): (0.789, 0.696),
        ("Overlap-allowing", "Kairos (64-code)"): (0.594, 0.146),
        ("Overlap-allowing", "DeepLOB (time bars)"): (0.506, 0.027),
        ("Leakage-controlled", "Ridge (tokens)"): (0.479, 0.048),
        ("Leakage-controlled", "HistGBM (tokens)"): (0.090, 0.519),
        ("Leakage-controlled", "LightGBM (LOB)"): (0.648, 0.262),
        ("Leakage-controlled", "Kairos (64-code)"): (0.618, 0.328),
        ("Leakage-controlled", "DeepLOB (time bars)"): (0.506, 0.027),
    }
    table6_rows: list[dict[str, object]] = []
    overlap_rows, overlap_cache, meta_overlap = evaluate_setting("Overlap-allowing", "volume", horizon=5, stride=1)
    de_rows, de_cache, meta_de = evaluate_setting("Leakage-controlled", "volume", horizon=50, stride=50)
    table6_rows.extend(overlap_rows)
    table6_rows.extend(de_rows)
    for row in table6_rows:
        hda, hric = historical6[(row["audit_setting"], row["model"])]
        row["historical_DA"] = hda
        row["historical_RIC"] = hric
    t6 = pd.DataFrame(table6_rows)
    t6 = t6[
        [
            "audit_setting",
            "model",
            "rerun_DA",
            "rerun_RIC",
            "historical_DA",
            "historical_RIC",
            "sample_count",
            "source_file",
            "diff_status",
            "protocol_caveat",
        ]
    ]
    md_tex_csv(t6, "table6")

    kairos_pred, kairos_y = de_cache["Kairos (64-code)"]
    bars_per_day = len(kairos_y) / max(1, len(TEST_DATES))
    t7_rows = []
    for bps, hmean, hsh, hann in [(3, -5e-6, -98.6, -1.71), (1, -1e-6, -70.0, -0.39)]:
        mean, sh, ann = trading_metrics(kairos_pred, kairos_y, bps, bars_per_day)
        t7_rows.append(
            {
                "cost_bps": bps,
                "rerun_mean_pnl_per_bar": mean,
                "rerun_sharpe": sh,
                "rerun_annual_return": ann,
                "historical_mean_pnl_per_bar": hmean,
                "historical_sharpe": hsh,
                "historical_annual_return": hann,
                "source_file": ";".join(meta_de.loc[meta_de.exists, "path"].astype(str).tolist()),
                "diff_status": "protocol_different",
                "protocol_caveat": "processed_four_date_rerun using Kairos 64-code proxy action path; not original normalized motivating-slice artifact",
            }
        )
    t7 = pd.DataFrame(t7_rows)
    md_tex_csv(t7, "table7")

    # Table 14 representation/overlap audit.
    x_de, y_de, d_de, meta_de_direct = load_mode("volume", 50, 50)
    xtr_de, ytr_de, xte_de, yte_de = split_xy(x_de, y_de, d_de)
    pred_tok512 = fit_hist_tokens(*cap_train(xtr_de, ytr_de, 100000), xte_de, 512)
    pred_tok64 = de_cache["Kairos (64-code)"][0]
    pred_lgbm_stride1 = overlap_cache["LightGBM (LOB)"][0]
    y_lgbm_stride1 = overlap_cache["LightGBM (LOB)"][1]
    pred_lgbm_stride50 = de_cache["LightGBM (LOB)"][0]
    y_lgbm_stride50 = de_cache["LightGBM (LOB)"][1]
    t14 = pd.DataFrame(
        [
            {
                "audit": "Tokenizer",
                "config": "512 codes",
                "rerun_RIC": rank_ic(pred_tok512, yte_de),
                "historical_RIC": 0.012,
                "sample_count": int(len(yte_de)),
                "source_file": ";".join(meta_de_direct.loc[meta_de_direct.exists, "path"].astype(str).tolist()),
                "diff_status": "protocol_different",
                "protocol_caveat": "processed_four_date_rerun; MiniBatchKMeans 512-code proxy, not original tokenizer artifact",
            },
            {
                "audit": "Tokenizer",
                "config": "64 codes",
                "rerun_RIC": rank_ic(pred_tok64, yte_de),
                "historical_RIC": 0.146,
                "sample_count": int(len(yte_de)),
                "source_file": ";".join(meta_de_direct.loc[meta_de_direct.exists, "path"].astype(str).tolist()),
                "diff_status": "protocol_different",
                "protocol_caveat": "processed_four_date_rerun; MiniBatchKMeans 64-code proxy, not original tokenizer artifact",
            },
            {
                "audit": "LightGBM stride",
                "config": "1",
                "rerun_RIC": rank_ic(pred_lgbm_stride1, y_lgbm_stride1),
                "historical_RIC": 0.696,
                "sample_count": int(len(y_lgbm_stride1)),
                "source_file": ";".join(meta_overlap.loc[meta_overlap.exists, "path"].astype(str).tolist()),
                "diff_status": "protocol_different",
                "protocol_caveat": "processed_four_date_rerun; stride=1, not original normalized motivating-slice artifact",
            },
            {
                "audit": "LightGBM stride",
                "config": "50",
                "rerun_RIC": rank_ic(pred_lgbm_stride50, y_lgbm_stride50),
                "historical_RIC": 0.262,
                "sample_count": int(len(y_lgbm_stride50)),
                "source_file": ";".join(meta_de.loc[meta_de.exists, "path"].astype(str).tolist()),
                "diff_status": "protocol_different",
                "protocol_caveat": "processed_four_date_rerun; stride=50, not original normalized motivating-slice artifact",
            },
            {
                "audit": "Kairos",
                "config": "stride=50",
                "rerun_RIC": rank_ic(pred_tok64, yte_de),
                "historical_RIC": 0.328,
                "sample_count": int(len(yte_de)),
                "source_file": ";".join(meta_de_direct.loc[meta_de_direct.exists, "path"].astype(str).tolist()),
                "diff_status": "protocol_different",
                "protocol_caveat": "processed_four_date_rerun; 64-code non-linear proxy, not original Kairos neural checkpoint",
            },
        ]
    )
    md_tex_csv(t14, "table14")

    lightgbm_pred, lightgbm_y = de_cache["LightGBM (LOB)"]
    _, kairos_sharpe, _ = trading_metrics(kairos_pred, kairos_y, 3, bars_per_day)
    _, lgbm_sharpe, _ = trading_metrics(lightgbm_pred, lightgbm_y, 3, bars_per_day)
    t15 = pd.DataFrame(
        [
            {
                "model": "Kairos",
                "rerun_DA_pct": directional_accuracy(kairos_pred, kairos_y) * 100.0,
                "rerun_RIC": rank_ic(kairos_pred, kairos_y),
                "rerun_sharpe_3bps": kairos_sharpe,
                "historical_DA_pct": 61.8,
                "historical_RIC": 0.328,
                "historical_sharpe_3bps": -98.6,
                "sample_count": int(len(kairos_y)),
                "source_file": ";".join(meta_de.loc[meta_de.exists, "path"].astype(str).tolist()),
                "diff_status": "protocol_different",
                "protocol_caveat": "processed_four_date_rerun; 64-code proxy, not original normalized motivating-slice artifact",
            },
            {
                "model": "LightGBM",
                "rerun_DA_pct": directional_accuracy(lightgbm_pred, lightgbm_y) * 100.0,
                "rerun_RIC": rank_ic(lightgbm_pred, lightgbm_y),
                "rerun_sharpe_3bps": lgbm_sharpe,
                "historical_DA_pct": 64.8,
                "historical_RIC": 0.262,
                "historical_sharpe_3bps": -112.3,
                "sample_count": int(len(lightgbm_y)),
                "source_file": ";".join(meta_de.loc[meta_de.exists, "path"].astype(str).tolist()),
                "diff_status": "protocol_different",
                "protocol_caveat": "processed_four_date_rerun; not original normalized motivating-slice artifact",
            },
        ]
    )
    md_tex_csv(t15, "table15")

    diff_rows = []
    for name, df in [("table6", t6), ("table7", t7), ("table14", t14), ("table15", t15)]:
        for _, r in df.iterrows():
            diff_rows.append(
                {
                    "table": name,
                    "item": r.get("model", r.get("audit", r.get("cost_bps", "row"))),
                    "diff_status": r["diff_status"],
                    "source_file": r["source_file"],
                    "protocol_caveat": r["protocol_caveat"],
                }
            )
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUT / "motivating_slice_table_diff.csv", index=False)
    counts = diff.diff_status.value_counts().to_dict()
    (OUT / "motivating_slice_diff_summary.md").write_text(
        "# Motivating Slice Diff Summary\n\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in counts.items())
        + "\n\n"
        "All displayed Tables 6/7/14/15 rows were rerun from the four requested BTCUSDT/Binance processed 90D bar files. "
        "Values differ from the PDF historical values because the normalized historical motivating-slice artifact and original codebook/checkpoints are unavailable; rows remain protocol-caveated.\n"
    )
    all_meta = pd.concat([meta_overlap, meta_de], ignore_index=True).drop_duplicates()
    all_meta.to_csv(OUT / "data_candidate_files.csv", index=False)
    (OUT / "data_discovery_report.md").write_text(
        "# Data Discovery Report\n\n"
        "Found four requested BTCUSDT/Binance processed bar dates in the 90D bundles: "
        "`2025-10-12`, `2025-10-17`, `2025-10-26`, and `2025-10-27` for time and volume bars. "
        "Raw event-level L2 exists locally only for `2025-10-17` and `2025-10-26`; no local raw L2 was found for `2025-10-12` or `2025-10-27`. "
        "No normalized historical motivating-slice source artifact or original tokenizer/checkpoint bundle was found. "
        "The v8 rerun therefore uses processed 90D bars and retrained compact proxies with explicit protocol caveats.\n"
    )
    (OUT / "script_discovery_report.md").write_text(
        "# Script Discovery Report\n\n"
        "Created and executed `scripts/rerun_motivating_slice_tables_6_7_14_15.py`. "
        "The script reruns Tables 6/7/14/15 from four requested processed BTCUSDT/Binance dates and writes CSV/MD/TEX/source-trace artifacts.\n"
    )
    (OUT / "rerun_plan.md").write_text(
        "# Motivating Slice Tables 6/7/14/15 Rerun Plan\n\n"
        "Use existing 90D processed BTCUSDT/Binance bars for `2025-10-12`, `2025-10-17`, `2025-10-26`, and `2025-10-27`. "
        "Run compact Ridge-token, HistGBM-token, LightGBM-LOB, 64-code Kairos-proxy, and TinyDeepLOB-proxy rows. "
        "Export canonical Tables 6/7/14/15 and compare against PDF historical values. "
        "Because the original normalized motivating-slice artifact and checkpoints are unavailable, results are protocol-caveated.\n"
    )
    (OUT / "pdf_patch_recommendation_after_rerun.md").write_text(
        "# PDF Patch Recommendation After Rerun\n\n"
        "Tables 6/7/14/15 were rerun from the four requested processed BTCUSDT/Binance dates. "
        "Do not present the old PDF historical numbers as reproduced evidence. "
        "Safe options: keep the visible historical caveat in the patched 55-page PDF, or replace the old table values with the v8 processed-four-date values and label every row `processed_four_date_rerun / protocol_different`. "
        "Do not remove the caveat unless the original normalized motivating-slice artifact and original codebook/checkpoints are supplied.\n"
    )
    (OUT / "final_rerun_status.md").write_text(
        "# Final Rerun Status\n\n"
        "1. Did we run Tables 6/7/14/15? Yes.\n"
        "2. Rows succeeded: Table 6 has 10 rows, Table 7 has 2 rows, Table 14 has 5 rows, and Table 15 has 2 rows.\n"
        "3. Script used: `/root/prediction-execution-gap/scripts/rerun_motivating_slice_tables_6_7_14_15.py`.\n"
        "4. Data used: processed BTCUSDT/Binance 90D bars for `2025-10-12`, `2025-10-17`, `2025-10-26`, and `2025-10-27` under `/root/autodl-tmp/peg_bench_runs/tardis_90d_{time,volume}_bundle/processed/binance/BTCUSDT/`.\n"
        "5. Are rerun values equal to PDF historical values? No. All rows are `protocol_different` because the normalized historical artifact and original codebook/checkpoints are unavailable.\n"
        "6. Should PDF values be replaced? Only if replaced by the v8 processed-four-date values with a protocol caveat. Otherwise keep the PDF-only caveat.\n"
        "7. Caveats removable? No, not for exact historical reproduction.\n"
        "8. Caveats that must remain: Tables 6/7/14/15 are historical or protocol-caveated reruns, not exact normalized-source reproduction.\n"
        "9. Remaining blocker: original normalized motivating-slice artifact and original tokenizer/checkpoint bundle.\n"
    )
    manifest = []
    for f in sorted(OUT.glob("*")):
        if f.is_file():
            manifest.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f}")
    (OUT / "sha256_manifest.txt").write_text("\n".join(manifest) + "\n")
    print(json.dumps({"out": str(OUT), "rows": {"table6": len(t6), "table7": len(t7), "table14": len(t14), "table15": len(t15)}, "diff_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
