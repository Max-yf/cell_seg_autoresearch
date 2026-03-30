#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
materialize_config.py

Purpose
-------
Expand a trial config into a full used_config.json by merging:
- baseline_config.json
- trial_config.json (overrides)

Also performs safety validation so later trial scripts do not silently run with
forbidden settings.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


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
# Merge logic
# -----------------------------------------------------------------------------

def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for k, v in override.items():
            if k in merged:
                merged[k] = deep_merge(merged[k], v)
            else:
                merged[k] = copy.deepcopy(v)
        return merged
    return copy.deepcopy(override)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_required_sections(cfg: dict[str, Any]) -> None:
    required_top = [
        "input",
        "model",
        "hard_constraints",
        "step1_sparse_sim",
        "step2_local_normalization",
        "step3_cellpose_3d",
    ]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Missing required top-level sections: {missing}")


def validate_hard_constraints(cfg: dict[str, Any]) -> None:
    hard = cfg.get("hard_constraints", {})
    step3 = cfg.get("step3_cellpose_3d", {})
    inp = cfg.get("input", {})

    if hard.get("diameter_must_be_null", False):
        if step3.get("diameter", None) is not None:
            raise ValueError(
                f"Hard constraint violated: diameter must be null/None, got {step3.get('diameter')!r}"
            )

    if hard.get("require_do_3d", False):
        if bool(step3.get("do_3D", False)) is not True:
            raise ValueError("Hard constraint violated: do_3D must be true.")

    if hard.get("require_z_axis_0", False):
        if int(step3.get("z_axis", -1)) != 0:
            raise ValueError(f"Hard constraint violated: z_axis must be 0, got {step3.get('z_axis')!r}")

    if hard.get("allow_only_single_3d_tiff", False):
        suffixes = inp.get("expected_suffix", [".tif", ".tiff"])
        if not isinstance(suffixes, list) or not suffixes:
            raise ValueError("input.expected_suffix must be a non-empty list when TIFF-only mode is enabled.")


def validate_config_shape(cfg: dict[str, Any]) -> None:
    step2 = cfg["step2_local_normalization"]
    step3 = cfg["step3_cellpose_3d"]

    if int(step2.get("radius", 0)) <= 0:
        raise ValueError("step2_local_normalization.radius must be > 0")
    if float(step2.get("bias", 0.0)) <= 0.0:
        raise ValueError("step2_local_normalization.bias must be > 0")
    if int(step3.get("batch_size_3d", 0)) <= 0:
        raise ValueError("step3_cellpose_3d.batch_size_3d must be > 0")
    if int(step3.get("bsize", 0)) <= 0:
        raise ValueError("step3_cellpose_3d.bsize must be > 0")


# -----------------------------------------------------------------------------
# Materialization
# -----------------------------------------------------------------------------

def materialize_config(baseline_cfg: dict[str, Any], trial_cfg: dict[str, Any]) -> dict[str, Any]:
    used = copy.deepcopy(baseline_cfg)
    used["materialization"] = {
        "baseline_run_tag": baseline_cfg.get("run_tag"),
        "baseline_description": baseline_cfg.get("description"),
        "trial_id": trial_cfg.get("trial_id"),
        "trial_run_tag": trial_cfg.get("run_tag"),
        "trial_description": trial_cfg.get("description"),
    }

    overrides = trial_cfg.get("overrides", {})
    used = deep_merge(used, overrides)

    # Carry over a few useful bookkeeping fields.
    used["config_type"] = "used_config"
    if "trial_id" in trial_cfg:
        used["trial_id"] = trial_cfg["trial_id"]
    if "run_tag" in trial_cfg:
        used["run_tag"] = trial_cfg["run_tag"]
    if "search_edit_summary" in trial_cfg:
        used["search_edit_summary"] = copy.deepcopy(trial_cfg["search_edit_summary"])
    if "evaluation_request" in trial_cfg:
        used["evaluation_request"] = copy.deepcopy(trial_cfg["evaluation_request"])
    if "expected_risk" in trial_cfg:
        used["expected_risk"] = copy.deepcopy(trial_cfg["expected_risk"])
    return used


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge baseline config and trial config into used_config.json")
    parser.add_argument("--baseline", required=True, help="Path to baseline_config.json")
    parser.add_argument("--trial", required=True, help="Path to trial_config.json")
    parser.add_argument("--output", required=True, help="Output used_config.json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline).expanduser().resolve()
    trial_path = Path(args.trial).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    baseline_cfg = load_json(baseline_path)
    trial_cfg = load_json(trial_path)

    validate_required_sections(baseline_cfg)
    used_cfg = materialize_config(baseline_cfg, trial_cfg)
    validate_required_sections(used_cfg)
    validate_hard_constraints(used_cfg)
    validate_config_shape(used_cfg)

    used_cfg["materialization"]["baseline_path"] = str(baseline_path)
    used_cfg["materialization"]["trial_path"] = str(trial_path)
    used_cfg["materialization"]["output_path"] = str(output_path)

    save_json(output_path, used_cfg)
    print("=" * 90)
    print("Config materialization finished")
    print(f"Baseline : {baseline_path}")
    print(f"Trial    : {trial_path}")
    print(f"Output   : {output_path}")
    print(f"trial_id : {used_cfg.get('trial_id', '<missing>')}")
    print(f"run_tag  : {used_cfg.get('run_tag', '<missing>')}")
    print("=" * 90)


if __name__ == "__main__":
    main()
