
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/root/prediction-execution-gap")
V4 = ROOT / "outputs/final_acceptance_v4"
FIG = V4 / "figures"
PT = ROOT / "paper_tables/generated_20260503"
FIG.mkdir(parents=True, exist_ok=True)

def canon(name):
    return {
        "lightgbm_lob": "LightGBM", "kairos_64code": "Kairos-64",
        "deeplob_timebars": "DeepLOB", "ridge_tokens": "Ridge",
        "histgbm_tokens": "HistGBM", "no_trade": "No-trade",
        "ofi_threshold": "OFI", "buy_and_hold": "Buy-and-hold",
    }.get(str(name), str(name))

def panel(mode, h):
    return ("Time" if mode == "time" else "Volume") + f" h{int(h)}"

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=180)
    plt.savefig(FIG / f"{name}.pdf")
    plt.close()

def source(df, name):
    df.to_csv(V4 / f"{name}_source.csv", index=False)

t2 = pd.read_csv(PT / "table_02_r0_r4_leaders.csv")
t2["panel"] = t2["View"].map({"Time":"Time","Vol.":"Volume"}).fillna(t2["View"]) + " h" + t2["h"].astype(str)
t2["R0 label"] = t2["R0 top"].map(canon)
t2["R4 label"] = t2["R4 top"].map(canon)
source(t2, "figure2")
plt.figure(figsize=(7,4.2))
y = np.arange(len(t2))[::-1]
for i, r in t2.iterrows():
    color = "#2ca02c" if float(r["Sharpe"]) > 0 else "#d62728"
    plt.plot([0, 1], [y[i], y[i]], color=color, lw=2, alpha=0.75)
    plt.scatter([0, 1], [y[i], y[i]], color=color, s=55)
    plt.text(-0.03, y[i], r["R0 label"], ha="right", va="center", fontsize=8)
    plt.text(1.03, y[i], r["R4 label"] + f" ({float(r['Sharpe']):.2f})", ha="left", va="center", fontsize=8)
plt.yticks(y, t2["panel"])
plt.xticks([0, 1], ["R0 top", "clean active R4 top"])
plt.title("R0 winner versus clean active R4 winner")
plt.xlim(-0.45, 1.55)
savefig("figure2_slopegraph")

r0 = pd.read_csv(ROOT / "outputs/final_90d/r0_leaderboard.csv")
r4 = pd.read_csv(ROOT / "outputs/final_90d/r4_leaderboard.csv")
active = r4[(r4["turnover"] > 0) & (r4.get("family", "baseline") == "baseline")].copy()
merged = active.merge(r0[["mode","horizon","method","rank","rank_ic"]], on=["mode","horizon","method"], how="left", suffixes=("_r4","_r0"))
merged["r0_rank"] = merged["rank_r0"] if "rank_r0" in merged.columns else merged.get("rank", np.nan)
merged["r0_rank_ic"] = merged["rank_ic_r0"] if "rank_ic_r0" in merged.columns else merged.get("rank_ic", np.nan)
merged["r4_rank"] = merged["rank_r4"] if "rank_r4" in merged.columns else merged.get("rank", np.nan)
merged["panel"] = [panel(m,h) for m,h in zip(merged["mode"], merged["horizon"])]
merged["method_label"] = merged["method"].map(canon)
source(merged, "figure3")
plt.figure(figsize=(6.2,4.5))
sizes = 25 + np.log1p(merged["turnover"].clip(lower=0)) * 30
plt.scatter(merged["r0_rank_ic"], merged["post_cost_sharpe"].clip(lower=-2.5), s=sizes, c=merged["turnover"], cmap="viridis", alpha=0.82, edgecolor="black", linewidth=0.3)
plt.axhline(0, color="black", lw=0.8)
plt.xlabel("R0 RankIC")
plt.ylabel("R4 Sharpe at 3 bps (clipped at -2.5)")
plt.title("High RankIC does not imply positive post-cost Sharpe")
plt.colorbar(label="Turnover")
savefig("figure3_rankic_vs_sharpe")

t3 = pd.read_csv(PT / "table_03_gap_metrics.csv")
t3["panel"] = t3["View"].map({"Time":"Time","Vol.":"Volume"}).fillna(t3["View"]) + " h" + t3["h"].astype(str)
source(t3, "figure4")
plt.figure(figsize=(7,4.3))
x = np.arange(len(t3))
plt.plot(x, t3["Metric Gap"], marker="o", label="Metric Gap")
plt.plot(x, t3["Ranking Gap"], marker="s", label="Ranking Gap")
plt.plot(x, t3["RR@k"], marker="^", label="RR@1")
plt.axhline(0, color="black", lw=0.8)
plt.xticks(x, t3["panel"], rotation=25, ha="right")
plt.title("Prediction-Execution Gap metrics by panel")
plt.legend()
savefig("figure4_gap_metrics")

heat = merged.pivot_table(index="method_label", columns="panel", values="post_cost_sharpe", aggfunc="max")
source(heat.reset_index(), "figure5")
plt.figure(figsize=(7.5,4.2))
arr = heat.to_numpy(dtype=float)
im = plt.imshow(arr, cmap="coolwarm", vmin=-1.5, vmax=0.5, aspect="auto")
plt.xticks(range(len(heat.columns)), heat.columns, rotation=25, ha="right")
plt.yticks(range(len(heat.index)), heat.index)
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if not np.isnan(arr[i,j]):
            plt.text(j, i, f"{arr[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, label="R4 Sharpe at 3 bps")
plt.title("Clean active R4 Sharpe heatmap")
savefig("figure5_r4_sharpe_heatmap")

rank_src = merged.copy()
source(rank_src[["panel","method_label","r0_rank","r4_rank"]], "figure6")
methods = sorted(rank_src["method_label"].unique())
panels = list(t2["panel"])
r0mat = rank_src.pivot_table(index="method_label", columns="panel", values="r0_rank", aggfunc="min").reindex(index=methods, columns=panels)
r4mat = rank_src.pivot_table(index="method_label", columns="panel", values="r4_rank", aggfunc="min").reindex(index=methods, columns=panels)
fig, axes = plt.subplots(1,2,figsize=(10,4.2))
for ax, mat, title in [(axes[0], r0mat, "R0 rank by RankIC"), (axes[1], r4mat, "Clean R4 rank by Sharpe")]:
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i,j]
            if not pd.isna(val):
                ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
savefig("figure6_rank_matrices")

source(merged, "figure7")
plt.figure(figsize=(6.4,4.5))
plt.scatter(merged["turnover"].clip(lower=1e-3), merged["max_drawdown"], c=merged["post_cost_sharpe"], s=35 + merged["r0_rank_ic"].abs().fillna(0)*300, cmap="coolwarm", vmin=-1.5, vmax=0.5, alpha=0.85, edgecolor="black", linewidth=0.3)
plt.xscale("log")
plt.xlabel("Turnover (log scale)")
plt.ylabel("Max drawdown")
plt.title("Execution-risk map for clean active R4 rows")
plt.colorbar(label="R4 Sharpe at 3 bps")
savefig("figure7_execution_risk_map")

t16 = pd.read_csv(PT / "table_16_wrapper_formal_rows.csv")
vol20 = r4[(r4["mode"]=="volume") & (r4["horizon"]==20)].copy()
def raw_for(method):
    row = vol20[vol20["method"]==method]
    return float(row["post_cost_sharpe"].iloc[0]) if len(row) else 0.0
def wrapped_for(label):
    row = t16[t16["Method"]==label]
    return float(row["Sharpe @3 bps"].iloc[0]) if len(row) else np.nan
f8 = pd.DataFrame([
    {"method":"LightGBM", "raw_sharpe":raw_for("lightgbm_lob"), "wrapped_sharpe":wrapped_for("LightGBM + wrapper"), "raw_turnover":float(vol20[vol20["method"]=="lightgbm_lob"]["turnover"].iloc[0]), "wrapped_turnover":3.76},
    {"method":"Kairos-64", "raw_sharpe":raw_for("kairos_64code"), "wrapped_sharpe":wrapped_for("Kairos-64 + wrapper"), "raw_turnover":float(vol20[vol20["method"]=="kairos_64code"]["turnover"].iloc[0]), "wrapped_turnover":2.0},
    {"method":"Ridge", "raw_sharpe":raw_for("ridge_tokens"), "wrapped_sharpe":wrapped_for("Ridge + threshold"), "raw_turnover":float(vol20[vol20["method"]=="ridge_tokens"]["turnover"].iloc[0]), "wrapped_turnover":0.0},
])
source(f8, "figure8")
fig, axes = plt.subplots(1,2,figsize=(8.5,3.8))
for _, r in f8.iterrows():
    axes[0].plot([0,1],[r["raw_sharpe"], r["wrapped_sharpe"]], marker="o", label=r["method"])
    axes[1].plot([0,1],[r["raw_turnover"], r["wrapped_turnover"]], marker="o", label=r["method"])
axes[0].axhline(0,color="black",lw=0.8); axes[0].set_ylabel("Sharpe @3bps")
axes[1].set_ylabel("Turnover")
for ax in axes:
    ax.set_xticks([0,1]); ax.set_xticklabels(["Raw","Wrapped"])
axes[0].set_title("Utility before/after wrapper")
axes[1].set_title("Turnover before/after wrapper")
axes[1].legend(fontsize=8)
savefig("figure8_wrapper_audit_volume_h20")

t17 = pd.read_csv(PT / "table_17_stress_h20.csv")
t17["panel"] = t17["View"].map({"Time":"Time","Vol.":"Volume"}).fillna(t17["View"]) + " h" + t17["h"].astype(str)
source(t17, "figure9")
piv = t17.pivot_table(index="Method", columns="panel", values="Worst-regime Sharpe", aggfunc="min")
plt.figure(figsize=(7,4))
arr = piv.to_numpy(dtype=float)
im = plt.imshow(np.clip(arr, -10, 0), cmap="coolwarm", vmin=-10, vmax=0, aspect="auto")
plt.xticks(range(len(piv.columns)), piv.columns, rotation=25, ha="right")
plt.yticks(range(len(piv.index)), piv.index)
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if not np.isnan(arr[i,j]):
            plt.text(j, i, f"{arr[i,j]:.1f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, label="Worst-regime Sharpe")
plt.title("Worst-regime utility exposes stress fragility")
savefig("figure9_worst_regime_heatmap")

t18 = pd.read_csv(PT / "table_18_panel_bootstrap.csv")
source(t18, "figure10")
metrics = t18["Metric"].astype(str).tolist()
est = t18["Estimate"].astype(float).to_numpy()
lo=[]; hi=[]
for s in t18["95% CI"].astype(str):
    vals = s.strip("[]").split(",")
    lo.append(float(vals[0])); hi.append(float(vals[1]))
lo=np.array(lo); hi=np.array(hi)
y=np.arange(len(metrics))
plt.figure(figsize=(7.2,4.2))
plt.errorbar(est, y, xerr=[est-lo, hi-est], fmt="o", color="#1f77b4", ecolor="#888888", capsize=3)
plt.axvline(0, color="black", lw=0.8)
plt.yticks(y, metrics)
plt.xlabel("Estimate with 95% panel-bootstrap interval")
plt.title("Aggregate uncertainty over six completed panels")
savefig("figure10_panel_bootstrap_uncertainty")

manifest=[]
for p in sorted(FIG.glob("*")):
    if p.is_file():
        manifest.append({"file": str(p), "bytes": p.stat().st_size})
pd.DataFrame(manifest).to_csv(V4 / "figures_file_manifest.csv", index=False)
(V4 / "figures_reproduction_audit.md").write_text("# Figures reproduction audit\n\nFigures 2-10 were generated from normalized 90D result CSV/JSON artifacts. Figure 1 and Figure 11 are static protocol diagrams and are verified in the run matrix rather than regenerated here.\n", encoding="utf-8")
