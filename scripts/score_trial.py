#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_trial.py

Purpose
-------
Score one trial after `run_trial.py` finishes.

This script reads:
- trial_record.json
- used_config.json

It then:
1. finds the label-mask TIFF for each successful crop,
2. computes a simple hollow / donut proxy metric directly from the masks,
3. aggregates cell-count / hollow / runtime / memory statistics,
4. computes a single score,
5. writes score.json.

Important notes
---------------
- This is intentionally a *simple, explainable* first scoring script.
- The hollow metric is a proxy, not a perfect biological truth metric.
- It is designed for crop-level autoresearch screening, not final publication.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Mask discovery
# -----------------------------------------------------------------------------

EXCLUDE_NAME_TOKENS = (
    "flow",
    "flows",
    "outline",
    "outlines",
    "overlay",
    "prob",
    "style",
    "rgb",
    "image",
)

PREFERRED_NAME_TOKENS = (
    "mask",
    "masks",
    "label",
    "labels",
)



def _candidate_score(path: Path) -> tuple[int, int]:
    """
    Lower is better.
    First component: filename preference.
    Second component: shorter names get a slight preference.
    """
    name = path.name.lower()
    if any(tok in name for tok in EXCLUDE_NAME_TOKENS):
        return (99, len(name))
    if any(tok in name for tok in PREFERRED_NAME_TOKENS):
        return (0, len(name))
    return (10, len(name))



def find_label_mask_path(step3_output_dir: Path) -> Path | None:
    candidates = sorted(
        list(step3_output_dir.glob("*.tif")) + list(step3_output_dir.glob("*.tiff")),
        key=_candidate_score,
    )
    if not candidates:
        return None

    # Fast path: use the best-looking filename first.
    for path in candidates:
        if _candidate_score(path)[0] == 0:
            return path

    # Fallback: open files until we find something that looks like a 3D label volume.
    for path in candidates:
        try:
            arr = tifffile.imread(path)
        except Exception:
            continue
        if arr.ndim != 3:
            continue
        if not np.issubdtype(arr.dtype, np.integer):
            continue
        unique_vals = np.unique(arr)
        if unique_vals.size >= 3:
            return path

    return candidates[0]


# -----------------------------------------------------------------------------
# Hollow / donut metric
# -----------------------------------------------------------------------------


def fill_holes_2d(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill holes in a 2D binary image using border-connected background flood fill.

    Returns
    -------
    filled : np.ndarray[bool]
        Binary region with enclosed holes filled.
    holes : np.ndarray[bool]
        Only the enclosed holes.
    """
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D binary image, got shape={binary.shape}")

    binary = np.asarray(binary, dtype=bool)
    h, w = binary.shape
    background = ~binary
    visited = np.zeros_like(background, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(y: int, x: int) -> None:
        if 0 <= y < h and 0 <= x < w and background[y, x] and not visited[y, x]:
            visited[y, x] = True
            q.append((y, x))

    # Seed the flood fill from border background pixels.
    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    while q:
        y, x = q.popleft()
        if y > 0:
            push(y - 1, x)
        if y + 1 < h:
            push(y + 1, x)
        if x > 0:
            push(y, x - 1)
        if x + 1 < w:
            push(y, x + 1)

    holes = background & (~visited)
    filled = binary | holes
    return filled, holes



def compute_hollow_metrics(
    labels: np.ndarray,
    *,
    min_hole_area: int = 4,
    min_hole_ratio: float = 0.02,
    min_instance_pixels_in_slice: int = 12,
    bbox_pad: int = 1,
) -> dict[str, Any]:
    """
    Compute a simple hollow-artifact proxy from a 3D labeled volume.

    A label instance is marked as hollow if any 2D slice of that instance contains
    a sufficiently large enclosed hole.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected 3D label volume, got shape={labels.shape}")

    instance_ids = np.unique(labels)
    instance_ids = instance_ids[instance_ids > 0]

    total_instances = int(instance_ids.size)
    hollow_instances = 0
    inspected_slices = 0
    hollow_slices = 0
    severities: list[float] = []

    zdim = labels.shape[0]

    for inst_id in instance_ids:
        inst_hollow = False
        # Iterate only the z slices that contain this instance.
        z_indices = np.where(np.any(labels == inst_id, axis=(1, 2)))[0]
        for z in z_indices:
            mask = labels[z] == inst_id
            area = int(mask.sum())
            if area < min_instance_pixels_in_slice:
                continue

            ys, xs = np.where(mask)
            y0 = max(0, int(ys.min()) - bbox_pad)
            y1 = min(mask.shape[0], int(ys.max()) + 1 + bbox_pad)
            x0 = max(0, int(xs.min()) - bbox_pad)
            x1 = min(mask.shape[1], int(xs.max()) + 1 + bbox_pad)
            region = mask[y0:y1, x0:x1]
            filled, holes = fill_holes_2d(region)
            filled_area = int(filled.sum())
            hole_area = int(holes.sum())
            if filled_area <= 0:
                continue

            inspected_slices += 1
            severity = float(hole_area / filled_area)
            if hole_area >= min_hole_area and severity >= min_hole_ratio:
                hollow_slices += 1
                severities.append(severity)
                inst_hollow = True

        if inst_hollow:
            hollow_instances += 1

    return {
        "total_instances": total_instances,
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instances / total_instances) if total_instances > 0 else None,
        "inspected_slices": int(inspected_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slices / inspected_slices) if inspected_slices > 0 else None,
        "mean_hollow_severity": float(np.mean(severities)) if severities else 0.0,
        "num_severity_events": int(len(severities)),
        "thresholds": {
            "min_hole_area": int(min_hole_area),
            "min_hole_ratio": float(min_hole_ratio),
            "min_instance_pixels_in_slice": int(min_instance_pixels_in_slice),
            "bbox_pad": int(bbox_pad),
        },
    }


# -----------------------------------------------------------------------------
# Scoring helpers
# -----------------------------------------------------------------------------


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



def collect_peak_mem_gb(crop_record: dict[str, Any]) -> float | None:
    """Try a few common places where memory might be recorded."""
    candidates = []
    step3_meta = crop_record.get("step3_meta", {}) or {}
    step12_meta = crop_record.get("step12_meta", {}) or {}
    metrics = crop_record.get("metrics", {}) or {}
    for src in (step3_meta, step12_meta, metrics):
        for key in ("peak_mem_gb", "peak_memory_gb", "max_mem_gb", "gpu_mem_gb"):
            candidates.append(src.get(key))
    vals = [safe_float(v) for v in candidates]
    vals = [v for v in vals if v is not None]
    return float(max(vals)) if vals else None



def build_default_scoring_weights() -> dict[str, float]:
    return {
        "cell_count": 1.0,
        "hollow_penalty": 1.0,
        "failure_penalty": 100000.0,
        "memory_penalty": 0.1,
        "runtime_penalty": 0.01,
    }



def compute_score(used_cfg: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    scoring_cfg = used_cfg.get("scoring", {}) or {}
    weights = build_default_scoring_weights()
    weights.update(scoring_cfg.get("weights", {}) or {})

    cell_count_mean = safe_float(aggregate.get("cell_count_mean")) or 0.0
    hollow_instances_total = safe_float(aggregate.get("hollow_instances_total")) or 0.0
    hollow_instance_ratio_mean = safe_float(aggregate.get("hollow_instance_ratio_mean")) or 0.0
    runtime_mean_sec = safe_float(aggregate.get("runtime_mean_sec")) or 0.0
    peak_mem_gb_max = safe_float(aggregate.get("peak_mem_gb_max")) or 0.0
    fail_count = int(aggregate.get("fail_count", 0))

    artifact_penalty = weights["hollow_penalty"] * (
        hollow_instances_total + 1000.0 * hollow_instance_ratio_mean
    )
    failure_penalty = weights["failure_penalty"] * fail_count
    memory_penalty = weights["memory_penalty"] * peak_mem_gb_max
    runtime_penalty = weights["runtime_penalty"] * runtime_mean_sec
    count_reward = weights["cell_count"] * cell_count_mean

    total_score = count_reward - artifact_penalty - failure_penalty - memory_penalty - runtime_penalty

    formula = (
        "score = cell_count_weight * cell_count_mean "
        "- hollow_penalty * (hollow_instances_total + 1000 * hollow_instance_ratio_mean) "
        "- failure_penalty * fail_count "
        "- memory_penalty * peak_mem_gb_max "
        "- runtime_penalty * runtime_mean_sec"
    )

    return {
        "score_version": scoring_cfg.get("score_formula_version", "v1"),
        "formula": formula,
        "weights": weights,
        "components": {
            "count_reward": float(count_reward),
            "artifact_penalty": float(artifact_penalty),
            "failure_penalty": float(failure_penalty),
            "memory_penalty": float(memory_penalty),
            "runtime_penalty": float(runtime_penalty),
        },
        "value": float(total_score),
    }


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one trial using cell-count and hollow-artifact proxy.")
    parser.add_argument("--trial-root", required=True, help="Path to the trial root directory.")
    parser.add_argument("--used-config", default=None, help="Optional explicit used_config.json path.")
    parser.add_argument("--output", default=None, help="Optional explicit score.json output path.")
    parser.add_argument("--min-hole-area", type=int, default=4, help="Minimum hole pixels to count as a hollow event.")
    parser.add_argument("--min-hole-ratio", type=float, default=0.02, help="Minimum hole_area / filled_area ratio.")
    parser.add_argument(
        "--min-instance-pixels-in-slice",
        type=int,
        default=12,
        help="Ignore tiny slice fragments below this area during hollow analysis.",
    )
    parser.add_argument("--bbox-pad", type=int, default=1, help="Padding for per-instance slice bounding box.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    trial_root = Path(args.trial_root).expanduser().resolve()
    used_config_path = Path(args.used_config).expanduser().resolve() if args.used_config else (trial_root / "used_config.json")
    output_path = Path(args.output).expanduser().resolve() if args.output else (trial_root / "score.json")
    trial_record_path = trial_root / "trial_record.json"

    if not trial_record_path.exists():
        raise FileNotFoundError(f"Missing trial_record.json: {trial_record_path}")
    if not used_config_path.exists():
        raise FileNotFoundError(f"Missing used_config.json: {used_config_path}")

    trial_record = load_json(trial_record_path)
    used_cfg = load_json(used_config_path)

    scored_crops: list[dict[str, Any]] = []
    cell_counts: list[float] = []
    runtimes: list[float] = []
    peak_mems: list[float] = []
    hollow_instances_total = 0
    hollow_instance_ratios: list[float] = []
    hollow_slice_ratios: list[float] = []
    hollow_severities: list[float] = []
    success_count = 0
    fail_count = 0

    for crop_record in trial_record.get("per_crop", []):
        crop_out: dict[str, Any] = {
            "crop_id": crop_record.get("crop_id"),
            "role": crop_record.get("role"),
            "input_status": crop_record.get("status"),
        }

        if crop_record.get("status") != "success":
            crop_out["status"] = "failed"
            crop_out["error_message"] = crop_record.get("error_message")
            fail_count += 1
            scored_crops.append(crop_out)
            continue

        step3_output_dir = Path(crop_record.get("artifacts", {}).get("step3_output_dir", ""))
        mask_path = find_label_mask_path(step3_output_dir) if step3_output_dir.exists() else None
        if mask_path is None:
            crop_out["status"] = "failed"
            crop_out["error_message"] = f"Could not find a label-mask TIFF in {step3_output_dir}"
            fail_count += 1
            scored_crops.append(crop_out)
            continue

        labels = tifffile.imread(mask_path)
        if labels.ndim != 3:
            crop_out["status"] = "failed"
            crop_out["error_message"] = f"Mask is not 3D: {mask_path} shape={labels.shape}"
            fail_count += 1
            scored_crops.append(crop_out)
            continue

        hollow = compute_hollow_metrics(
            labels,
            min_hole_area=args.min_hole_area,
            min_hole_ratio=args.min_hole_ratio,
            min_instance_pixels_in_slice=args.min_instance_pixels_in_slice,
            bbox_pad=args.bbox_pad,
        )

        num_instances = crop_record.get("metrics", {}).get("num_instances")
        if num_instances is None:
            unique_ids = np.unique(labels)
            num_instances = int(np.sum(unique_ids > 0))

        pipeline_elapsed = safe_float(crop_record.get("metrics", {}).get("pipeline_elapsed_sec"))
        peak_mem_gb = collect_peak_mem_gb(crop_record)

        crop_out.update(
            {
                "status": "success",
                "mask_path": str(mask_path),
                "num_instances": int(num_instances),
                "pipeline_elapsed_sec": pipeline_elapsed,
                "peak_mem_gb": peak_mem_gb,
                "hollow_metrics": hollow,
            }
        )
        scored_crops.append(crop_out)
        success_count += 1
        cell_counts.append(float(num_instances))
        hollow_instances_total += int(hollow["hollow_instances"])
        if hollow["hollow_instance_ratio"] is not None:
            hollow_instance_ratios.append(float(hollow["hollow_instance_ratio"]))
        if hollow["hollow_slice_ratio"] is not None:
            hollow_slice_ratios.append(float(hollow["hollow_slice_ratio"]))
        hollow_severities.append(float(hollow["mean_hollow_severity"]))
        if pipeline_elapsed is not None:
            runtimes.append(float(pipeline_elapsed))
        if peak_mem_gb is not None:
            peak_mems.append(float(peak_mem_gb))

    aggregate = {
        "num_crops": len(scored_crops),
        "success_count": int(success_count),
        "fail_count": int(fail_count),
        "cell_count_mean": float(np.mean(cell_counts)) if cell_counts else None,
        "cell_count_std": float(np.std(cell_counts)) if cell_counts else None,
        "cell_count_max": float(np.max(cell_counts)) if cell_counts else None,
        "cell_count_min": float(np.min(cell_counts)) if cell_counts else None,
        "hollow_instances_total": int(hollow_instances_total),
        "hollow_instance_ratio_mean": float(np.mean(hollow_instance_ratios)) if hollow_instance_ratios else None,
        "hollow_slice_ratio_mean": float(np.mean(hollow_slice_ratios)) if hollow_slice_ratios else None,
        "hollow_severity_mean": float(np.mean(hollow_severities)) if hollow_severities else 0.0,
        "runtime_total_sec": float(np.sum(runtimes)) if runtimes else None,
        "runtime_mean_sec": float(np.mean(runtimes)) if runtimes else None,
        "peak_mem_gb_max": float(np.max(peak_mems)) if peak_mems else None,
    }

    score = compute_score(used_cfg, aggregate)
    status = "scored" if success_count > 0 else "crash"

    out = {
        "schema_version": "1.0",
        "trial_id": trial_record.get("trial_id"),
        "run_tag": trial_record.get("run_tag"),
        "trial_root": str(trial_root),
        "used_config_path": str(used_config_path),
        "trial_record_path": str(trial_record_path),
        "status": status,
        "aggregate_metrics": aggregate,
        "per_crop": scored_crops,
        "score": score,
        "notes": {
            "hollow_metric_comment": (
                "This is an explainable proxy metric based on enclosed 2D holes inside labeled-instance slices. "
                "It is intended for screening, not final biological truth evaluation."
            )
        },
    }

    save_json(output_path, out)
    print("=" * 90)
    print("Trial scoring finished")
    print(f"trial_root      : {trial_root}")
    print(f"score_json      : {output_path}")
    print(f"status          : {status}")
    print(f"success/fail    : {success_count}/{fail_count}")
    print(f"cell_count_mean : {aggregate['cell_count_mean']}")
    print(f"hollow_ratio    : {aggregate['hollow_instance_ratio_mean']}")
    print(f"score           : {score['value']:.6f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
