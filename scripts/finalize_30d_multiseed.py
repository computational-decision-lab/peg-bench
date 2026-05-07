#!/usr/bin/env python3
"""Freeze 30d final artifacts plus DPO multiseed outputs into one clear bundle."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        mean_value = value.get("mean")
        if isinstance(mean_value, (int, float)):
            return float(mean_value)
    return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _best_execution_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(
        entries,
        key=lambda item: float(item.get("post_cost_sharpe", float("-inf"))),
    )


def _best_prediction_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(
        entries,
        key=lambda item: float(item.get("rank_ic", float("-inf"))),
    )


def _build_result_index(
    *,
    output_dir: Path,
    base_final_dir: Path,
    report_card: dict[str, Any],
    leaderboards: dict[str, Any],
    gap_metrics: dict[str, Any],
    time_summary: dict[str, Any],
    volume_summary: dict[str, Any],
) -> str:
    time_pred = _best_prediction_entry(leaderboards.get("prediction", {}).get("time_h20", []))
    volume_pred = _best_prediction_entry(leaderboards.get("prediction", {}).get("volume_h20", []))
    time_exec = _best_execution_entry(leaderboards.get("execution", {}).get("time_h20", []))
    volume_exec = _best_execution_entry(leaderboards.get("execution", {}).get("volume_h20", []))
    dpo_time = report_card.get("execution", {}).get("DPO_execution_first", {}).get("time", {})
    dpo_volume = report_card.get("execution", {}).get("DPO_execution_first", {}).get("volume", {})
    time_gap = gap_metrics.get("time", {}).get("h20", {})
    volume_gap = gap_metrics.get("volume", {}).get("h20", {})

    lines = [
        "# 30d 结果索引",
        "",
        "## 先看哪里",
        "",
        f"- 总结果目录：`{output_dir}`",
        f"- baseline / evaluator 总报告：`{base_final_dir / 'report_card.md'}`",
        f"- baseline 机器可读总表：`{base_final_dir / 'report_card.json'}`",
        f"- gap 指标：`{base_final_dir / 'gap_metrics.json'}`",
        f"- cost ladder：`{base_final_dir / 'cost_ladder_results.json'}`",
        f"- regime / stress：`{base_final_dir / 'regime_stress_results.json'}`",
        f"- leaderboard：`{base_final_dir / 'model_leaderboards.json'}`",
        f"- time multiseed：`{output_dir / 'multiseed' / 'time' / 'multiseed_summary.json'}`",
        f"- volume multiseed：`{output_dir / 'multiseed' / 'volume' / 'multiseed_summary.json'}`",
        "",
        "## baseline 快速结论",
        "",
        f"- time h20 预测最好：`{time_pred.get('method') if time_pred else 'n/a'}`，RankIC={_fmt(time_pred.get('rank_ic') if time_pred else None)}",
        f"- time h20 执行最好：`{time_exec.get('method') if time_exec else 'n/a'}`，Sharpe@3bps={_fmt(time_exec.get('post_cost_sharpe') if time_exec else None)}",
        f"- volume h20 预测最好：`{volume_pred.get('method') if volume_pred else 'n/a'}`，RankIC={_fmt(volume_pred.get('rank_ic') if volume_pred else None)}",
        f"- volume h20 执行最好：`{volume_exec.get('method') if volume_exec else 'n/a'}`，Sharpe@3bps={_fmt(volume_exec.get('post_cost_sharpe') if volume_exec else None)}",
        "",
        "## DPO multiseed 快速结论",
        "",
        f"- time multiseed Sharpe@3bps 均值：{_fmt(time_summary.get('anchor_sharpe_3bps'))}",
        f"- time multiseed RankIC proxy 均值：{_fmt(time_summary.get('r0_proxy_rank_ic'))}",
        f"- volume multiseed Sharpe@3bps 均值：{_fmt(volume_summary.get('anchor_sharpe_3bps'))}",
        f"- volume multiseed RankIC proxy 均值：{_fmt(volume_summary.get('r0_proxy_rank_ic'))}",
        f"- 单次 time DPO 当前主结果 Sharpe@3bps：{_fmt(dpo_time.get('post_cost_sharpe'))}",
        f"- 单次 volume DPO 当前主结果 Sharpe@3bps：{_fmt(dpo_volume.get('post_cost_sharpe'))}",
        "",
        "## Prediction-Execution Gap",
        "",
        f"- time h20 MetricGap：{_fmt(time_gap.get('metric_gap'))}",
        f"- time h20 RankingGap：{_fmt(time_gap.get('ranking_gap'))}",
        f"- volume h20 MetricGap：{_fmt(volume_gap.get('metric_gap'))}",
        f"- volume h20 RankingGap：{_fmt(volume_gap.get('ranking_gap'))}",
        "",
        "## 文件说明",
        "",
        "- `snapshot_final_30d/`：当前 accepted baseline / evaluator 全量快照。",
        "- `multiseed/time/`：time 路径 3 个 seed 的 DPO 结果与汇总。",
        "- `multiseed/volume/`：volume 路径 3 个 seed 的 DPO 结果与汇总。",
        "- `freeze_manifest.json`：这次冻结时的来源路径、时间戳和文件列表。",
        "",
        "## 注意",
        "",
        "- 这是 30 天版本，不是 90 天完整版。",
        "- 当前 multiseed 是在公平性收紧后重跑的 DPO 扩展，不是重新跑 baseline。",
    ]
    return "\n".join(lines) + "\n"


def _build_cn_readme(*, output_dir: Path, base_final_dir: Path) -> str:
    lines = [
        "# 30d PEG-Bench 代码说明（简版）",
        "",
        "这个仓库现在主要分四块：",
        "",
        "- `src/data/`：L2/trades 读取、盘口重建、time bar / volume bar、特征和标签。",
        "- `src/models/`：baseline 模型、KAIROS 风格模型、策略网络。",
        "- `src/eval/`：统一 action、cost ladder、回测、gap 指标、regime / stress 评估。",
        "- `scripts/`：可直接运行的入口脚本。",
        "",
        "最常用脚本：",
        "",
        "- `scripts/run_peg_bench_comparison_baselines.py`：跑 baseline 对比。",
        "- `scripts/run_peg_bench_dpo_stage.py`：跑单个 DPO / OG-PA 风格阶段。",
        "- `scripts/run_peg_bench_dpo_multiseed.py`：同一套 DPO 配置跑多个 seed。",
        "- `scripts/audit_pegbench_readiness.py`：检查当前 30d 是否到 R4、还缺什么。",
        "- `scripts/final_acceptance_check.py`：做最终验收。",
        "- `scripts/finalize_30d_multiseed.py`：把 baseline + multiseed 结果冻结成一套清晰目录。",
        "",
        "结果主要看两个地方：",
        "",
        f"- accepted baseline / evaluator：`{base_final_dir}`",
        f"- 本次 multiseed 冻结结果：`{output_dir}`",
        "",
        "建议打开顺序：",
        "",
        "- 先看 `result_index_cn.md`",
        "- 再看 `snapshot_final_30d/report_card.md`",
        "- 然后看 `multiseed/time/multiseed_summary.json` 和 `multiseed/volume/multiseed_summary.json`",
        "",
        "说明：",
        "",
        "- baseline 是 prediction-first，对比重点是 RankIC / DA 和执行层 Sharpe 的落差。",
        "- DPO / OG-PA 风格方法是 execution-first，重点看 cost 后 Sharpe、MaxDD、turnover 和 stress 表现。",
        "- 当前交付是 30d 版本，不要当成 90d 全量版本引用。",
    ]
    return "\n".join(lines) + "\n"


def finalize(
    *,
    base_final_dir: Path,
    time_multiseed_root: Path,
    volume_multiseed_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    required_final = [
        "report_card.json",
        "report_card.md",
        "report_card.csv",
        "gap_metrics.json",
        "cost_ladder_results.json",
        "regime_stress_results.json",
        "model_leaderboards.json",
        "readiness_audit.json",
        "stage_conclusion.md",
    ]
    for name in required_final:
        path = base_final_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing base final artifact: {path}")

    required_multiseed = [
        time_multiseed_root / "multiseed_summary.json",
        time_multiseed_root / "multiseed_runs.json",
        volume_multiseed_root / "multiseed_summary.json",
        volume_multiseed_root / "multiseed_runs.json",
    ]
    for path in required_multiseed:
        if not path.exists():
            raise FileNotFoundError(f"missing multiseed artifact: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "snapshot_final_30d"
    multiseed_time_dir = output_dir / "multiseed" / "time"
    multiseed_volume_dir = output_dir / "multiseed" / "volume"

    _copy_tree(base_final_dir, snapshot_dir)
    for src in [time_multiseed_root / "multiseed_summary.json", time_multiseed_root / "multiseed_runs.json"]:
        _copy_file(src, multiseed_time_dir / src.name)
    for src in [volume_multiseed_root / "multiseed_summary.json", volume_multiseed_root / "multiseed_runs.json"]:
        _copy_file(src, multiseed_volume_dir / src.name)

    time_summary = _load_json(time_multiseed_root / "multiseed_summary.json")
    volume_summary = _load_json(volume_multiseed_root / "multiseed_summary.json")
    report_card = _load_json(base_final_dir / "report_card.json")
    leaderboards = _load_json(base_final_dir / "model_leaderboards.json")
    gap_metrics = _load_json(base_final_dir / "gap_metrics.json")

    result_index_path = output_dir / "result_index_cn.md"
    readme_path = output_dir / "README_30D_RESULTS_CN.md"
    root_readme_path = ROOT / "README_30D_RESULTS_CN.md"
    freeze_manifest_path = output_dir / "freeze_manifest.json"

    _write_text(
        result_index_path,
        _build_result_index(
            output_dir=output_dir,
            base_final_dir=base_final_dir,
            report_card=report_card,
            leaderboards=leaderboards,
            gap_metrics=gap_metrics,
            time_summary=time_summary,
            volume_summary=volume_summary,
        ),
    )
    readme_text = _build_cn_readme(output_dir=output_dir, base_final_dir=base_final_dir)
    _write_text(readme_path, readme_text)
    _write_text(root_readme_path, readme_text)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_final_dir": str(base_final_dir.resolve()),
        "time_multiseed_root": str(time_multiseed_root.resolve()),
        "volume_multiseed_root": str(volume_multiseed_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "copied_files": [
            str(result_index_path.resolve()),
            str(readme_path.resolve()),
            str(root_readme_path.resolve()),
            str(snapshot_dir.resolve()),
            str((multiseed_time_dir / "multiseed_summary.json").resolve()),
            str((multiseed_time_dir / "multiseed_runs.json").resolve()),
            str((multiseed_volume_dir / "multiseed_summary.json").resolve()),
            str((multiseed_volume_dir / "multiseed_runs.json").resolve()),
        ],
    }
    _write_json(freeze_manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze 30d baseline + multiseed DPO results into one bundle.")
    parser.add_argument(
        "--base-final-dir",
        type=Path,
        default=ROOT / "outputs" / "final_30d",
        help="Existing accepted 30d final result directory.",
    )
    parser.add_argument("--time-multiseed-root", type=Path, required=True, help="Time multiseed run directory.")
    parser.add_argument("--volume-multiseed-root", type=Path, required=True, help="Volume multiseed run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "final_30d_multiseed_fair",
        help="Destination directory for the frozen result bundle.",
    )
    args = parser.parse_args()

    manifest = finalize(
        base_final_dir=args.base_final_dir,
        time_multiseed_root=args.time_multiseed_root,
        volume_multiseed_root=args.volume_multiseed_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
