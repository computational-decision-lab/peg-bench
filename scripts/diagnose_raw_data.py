#!/usr/bin/env python3
"""Run the full PEG-Bench raw-data diagnosis pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from peg_bench.data.build_manifest import build_data_manifest
from peg_bench.data.diagnose_time import diagnose_time_continuity
from peg_bench.data.quality_check import write_data_quality_report
from peg_bench.data.scan_files import scan_tensor_files


def configure_logging(log_path: str | Path) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PEG-Bench data diagnosis pipeline.")
    parser.add_argument("--config", type=str, default="configs/data_diagnosis.yaml")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--diagnosis_output_dir", type=str, default="")
    parser.add_argument("--manifest_output", type=str, default="")
    parser.add_argument("--split_output", type=str, default="")
    parser.add_argument("--report_output", type=str, default="")
    parser.add_argument("--log_file", type=str, default="")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    input_dir = args.input_dir or config.get("input_dir", "data/raw")
    diagnosis_output_dir = args.diagnosis_output_dir or config.get("diagnosis_output_dir", "outputs/diagnosis")
    manifest_output = args.manifest_output or config.get("manifest_output", "outputs/manifests/data_manifest.csv")
    split_output = args.split_output or config.get("split_output", "outputs/manifests/split_plan.csv")
    if args.report_output:
        report_output = args.report_output
    elif args.diagnosis_output_dir:
        report_output = str(Path(diagnosis_output_dir) / "data_quality_report.md")
    else:
        report_output = config.get("report_output", "outputs/diagnosis/data_quality_report.md")
    log_file = args.log_file or config.get("log_file", "outputs/logs/diagnose_raw_data.log")
    expected_days = int(config.get("expected_days", 30))
    tolerance_days = int(config.get("tolerance_days", 3))
    train_ratio = float(config.get("train_ratio", 0.7))
    val_ratio = float(config.get("val_ratio", 0.15))
    test_ratio = float(config.get("test_ratio", 0.15))

    configure_logging(log_file)
    logging.info("Step 1/4: scanning %s", input_dir)
    scan_result = scan_tensor_files(input_dir, diagnosis_output_dir)
    logging.info("Step 2/4: building manifest")
    manifest_result = build_data_manifest(
        input_dir=input_dir,
        manifest_output=manifest_output,
        diagnosis_output_dir=diagnosis_output_dir,
        inventory_path=scan_result["inventory_path"],
    )
    logging.info("Step 3/4: diagnosing time continuity")
    time_result = diagnose_time_continuity(
        manifest_path=manifest_result["manifest_path"],
        output_dir=diagnosis_output_dir,
        expected_days=expected_days,
        tolerance_days=tolerance_days,
    )
    logging.info("Step 4/4: writing quality report and split plan")
    quality_result = write_data_quality_report(
        manifest_path=manifest_result["manifest_path"],
        inventory_path=scan_result["inventory_path"],
        schema_path=manifest_result["schema_path"],
        continuity_path=time_result["continuity_path"],
        missing_path=time_result["missing_path"],
        duplicate_path=time_result["duplicate_path"],
        anomaly_path=time_result["anomaly_path"],
        key_summary_path=manifest_result["key_summary_path"],
        report_path=report_output,
        split_plan_path=split_output,
        expected_days=expected_days,
        tolerance_days=tolerance_days,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    logging.info("Report written to %s", quality_result["report_path"])
    logging.info("Split plan written to %s", split_output)
    logging.info("Ratios used: train=%.2f val=%.2f test=%.2f", train_ratio, val_ratio, test_ratio)


if __name__ == "__main__":
    main()
