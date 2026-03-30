#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
append_results.py

Purpose
-------
Append one scored trial into results.tsv.

This file acts like the campaign index / experiment ledger.
It reads:
- trial_record.json
- score.json
- used_config.json

Then it:
1. builds one flat TSV row,
2. compares with previous valid rows,
3. assigns a simple decision: keep / discard / crash,
4. appends the row to results.tsv,
5. optionally writes the decision back into score.json.

Modified in this version
------------------------
- add campaign_name column
- detect campaign_name from trial_root path
- leave campaign_name empty for non-campaign rows, e.g. baseline rows outside
  runs/campaign_xxx/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


RESULTS_HEADER = [
    "timestamp_start",
    "timestamp_end",
    "campaign_name",
    "run_tag",
    "trial_id",
    "decision",
    "trial_status",
    "score",
    "score_version",
    "score_formula",
    "baseline_path",
    "trial_config_path",
    "used_config_path",
    "trial_root",
    "level",
    "num_crops",
    "crop_ids",
    "cell_count_mean",
    "cell_count_std",
    "cell_count_max",
    "cell_count_min",
    "hollow_instances_total",
    "hollow_instance_ratio_mean",
    "hollow_slice_ratio_mean",
    "hollow_severity_mean",
    "artifact_penalty",
    "runtime_total_sec",
    "runtime_mean_sec",
    "peak_mem_gb_max",
    "success_count",
    "fail_count",
    "input_path",
    "image_shape",
    "dtype",
    "model_path",
    "model_config_path",
    "step1_params_json",
    "step2_params_json",
    "step3_params_json",
    "changed_keys",
    "hypothesis",
    "expected_risk_json",
    "decision_reason",
    "notes",
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def compact_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writeheader()


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def best_existing_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    best_row = None
    best_score = None
    for row in rows:
        if row.get("decision") not in {"keep", "baseline_keep"}:
            continue
        score = safe_float(row.get("score"))
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_row = row
    return best_row


def detect_campaign_name_from_trial_root(trial_root: str | Path) -> str:
    """
    Detect campaign name from a path like:
      .../runs/campaign_001/trials/trial_0002

    If the path does not live under runs/campaign_xxx/, return empty string.
    This is the intended behavior for baseline rows.
    """
    parts = Path(trial_root).parts
    for i, part in enumerate(parts):
        if part == "runs" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt.startswith("campaign_"):
                return nxt
    return ""


def decide_row(
    row: dict[str, Any],
    previous_best: dict[str, str] | None,
    *,
    min_improvement: float,
    tie_tol: float,
) -> tuple[str, str]:
    trial_status = row["trial_status"]
    score = safe_float(row["score"])
    hollow_ratio = safe_float(row["hollow_instance_ratio_mean"])
    runtime_mean = safe_float(row["runtime_mean_sec"])

    if trial_status == "crash" or score is None:
        return "crash", "No valid score was produced for this trial."

    if previous_best is None:
        return "baseline_keep", "First valid row in results ledger; keep as initial baseline/reference."

    best_score = safe_float(previous_best.get("score"))
    best_hollow = safe_float(previous_best.get("hollow_instance_ratio_mean"))
    best_runtime = safe_float(previous_best.get("runtime_mean_sec"))

    if best_score is None:
        return "keep", "Previous best score is unreadable; keeping current valid row."

    if score > best_score + min_improvement:
        return "keep", f"Score improved from {best_score:.6f} to {score:.6f}."

    if abs(score - best_score) <= tie_tol:
        hollow_better = (
            hollow_ratio is not None and best_hollow is not None and hollow_ratio < best_hollow
        )
        runtime_better = (
            runtime_mean is not None and best_runtime is not None and runtime_mean < best_runtime
        )
        if hollow_better and runtime_better:
            return "keep", "Score is tied, but hollow ratio and runtime are both better than the current best."
        if hollow_better:
            return "keep", "Score is tied, but hollow ratio is better than the current best."
        if runtime_better:
            return "keep", "Score is tied, but runtime is better than the current best."

    return "discard", f"Did not beat current best score {best_score:.6f}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one scored trial into results.tsv.")
    parser.add_argument("--trial-root", required=True, help="Path to the trial root directory.")
    parser.add_argument("--results", required=True, help="Path to results.tsv")
    parser.add_argument("--used-config", default=None, help="Optional explicit used_config.json path.")
    parser.add_argument("--score-json", default=None, help="Optional explicit score.json path.")
    parser.add_argument("--trial-record", default=None, help="Optional explicit trial_record.json path.")
    parser.add_argument("--min-improvement", type=float, default=1e-9, help="Minimum score improvement to auto-keep.")
    parser.add_argument("--tie-tol", type=float, default=1e-9, help="Score tie tolerance.")
    parser.add_argument(
        "--writeback-score-json",
        action="store_true",
        help="Write decision and decision_reason back into score.json after appending.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial_root = Path(args.trial_root).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    used_config_path = Path(args.used_config).expanduser().resolve() if args.used_config else (trial_root / "used_config.json")
    score_json_path = Path(args.score_json).expanduser().resolve() if args.score_json else (trial_root / "score.json")
    trial_record_path = Path(args.trial_record).expanduser().resolve() if args.trial_record else (trial_root / "trial_record.json")

    if not used_config_path.exists():
        raise FileNotFoundError(f"Missing used_config.json: {used_config_path}")
    if not score_json_path.exists():
        raise FileNotFoundError(f"Missing score.json: {score_json_path}")
    if not trial_record_path.exists():
        raise FileNotFoundError(f"Missing trial_record.json: {trial_record_path}")

    used_cfg = load_json(used_config_path)
    score_json = load_json(score_json_path)
    trial_record = load_json(trial_record_path)

    ensure_results_file(results_path)
    existing_rows = read_existing_rows(results_path)
    previous_best = best_existing_row(existing_rows)

    agg = score_json.get("aggregate_metrics", {}) or {}
    score_block = score_json.get("score", {}) or {}
    materialization = used_cfg.get("materialization", {}) or {}
    search_edit = used_cfg.get("search_edit_summary", {}) or {}

    campaign_name = detect_campaign_name_from_trial_root(trial_root)

    row: dict[str, Any] = {
        "timestamp_start": trial_record.get("timestamp_start"),
        "timestamp_end": trial_record.get("timestamp_end"),
        "campaign_name": campaign_name,
        "run_tag": trial_record.get("run_tag") or used_cfg.get("run_tag"),
        "trial_id": trial_record.get("trial_id") or used_cfg.get("trial_id"),
        "trial_status": score_json.get("status"),
        "score": score_block.get("value"),
        "score_version": score_block.get("score_version"),
        "score_formula": score_block.get("formula"),
        "baseline_path": materialization.get("baseline_path"),
        "trial_config_path": materialization.get("trial_path"),
        "used_config_path": str(used_config_path),
        "trial_root": str(trial_root),
        "level": (used_cfg.get("evaluation_request", {}) or {}).get("level", "crop"),
        "num_crops": agg.get("num_crops"),
        "crop_ids": compact_json(trial_record.get("selected_crop_ids", [])),
        "cell_count_mean": agg.get("cell_count_mean"),
        "cell_count_std": agg.get("cell_count_std"),
        "cell_count_max": agg.get("cell_count_max"),
        "cell_count_min": agg.get("cell_count_min"),
        "hollow_instances_total": agg.get("hollow_instances_total"),
        "hollow_instance_ratio_mean": agg.get("hollow_instance_ratio_mean"),
        "hollow_slice_ratio_mean": agg.get("hollow_slice_ratio_mean"),
        "hollow_severity_mean": agg.get("hollow_severity_mean"),
        "artifact_penalty": (score_block.get("components", {}) or {}).get("artifact_penalty"),
        "runtime_total_sec": agg.get("runtime_total_sec"),
        "runtime_mean_sec": agg.get("runtime_mean_sec"),
        "peak_mem_gb_max": agg.get("peak_mem_gb_max"),
        "success_count": agg.get("success_count"),
        "fail_count": agg.get("fail_count"),
        "input_path": (used_cfg.get("input", {}) or {}).get("input_path"),
        "image_shape": compact_json((trial_record.get("input_source", {}) or {}).get("shape")),
        "dtype": (trial_record.get("input_source", {}) or {}).get("dtype"),
        "model_path": (used_cfg.get("model", {}) or {}).get("model_path"),
        "model_config_path": (used_cfg.get("model", {}) or {}).get("config_path"),
        "step1_params_json": compact_json(used_cfg.get("step1_sparse_sim", {})),
        "step2_params_json": compact_json(used_cfg.get("step2_local_normalization", {})),
        "step3_params_json": compact_json(used_cfg.get("step3_cellpose_3d", {})),
        "changed_keys": compact_json(search_edit.get("changed_keys", [])),
        "hypothesis": search_edit.get("hypothesis"),
        "expected_risk_json": compact_json(used_cfg.get("expected_risk", {})),
        "notes": compact_json(score_json.get("notes", {})),
    }

    decision, decision_reason = decide_row(
        row,
        previous_best,
        min_improvement=args.min_improvement,
        tie_tol=args.tie_tol,
    )
    row["decision"] = decision
    row["decision_reason"] = decision_reason

    with results_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in RESULTS_HEADER})

    if args.writeback_score_json:
        score_json["decision"] = decision
        score_json["decision_reason"] = decision_reason
        score_json["results_tsv_path"] = str(results_path)
        score_json["campaign_name"] = campaign_name
        save_json(score_json_path, score_json)

    print("=" * 90)
    print("Append results finished")
    print(f"results.tsv      : {results_path}")
    print(f"campaign_name    : {campaign_name or '<empty>'}")
    print(f"trial_id         : {row['trial_id']}")
    print(f"decision         : {decision}")
    print(f"decision_reason  : {decision_reason}")
    print(f"score            : {row['score']}")
    print("=" * 90)


if __name__ == "__main__":
    main()