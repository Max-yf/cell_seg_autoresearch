#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_trial.py

Purpose
-------
Create a new trial_config.json proposal for linked Step1/Step2/Step3 search.

First version strategy
----------------------
- If results.tsv has a current best kept row, start from that row's used_config.
- Otherwise start from baseline_config.json.
- Mutate a small number of parameters.
- Convert the mutated full config back into `overrides` relative to baseline.
- Save a trial_config.json that can be passed to materialize_config.py.

This gives you a simple autoresearch-style loop without changing the pipeline
core code.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
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



def compact_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


# -----------------------------------------------------------------------------
# Search-space defaults
# -----------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "step1_sparse_sim.sparse_iter": [80, 120, 160],
    "step1_sparse_sim.fidelity": [100.0, 150.0, 180.0, 220.0],
    "step1_sparse_sim.z_continuity": [0.5, 1.0, 1.5],
    "step1_sparse_sim.sparsity": [4.0, 6.0, 8.0],
    "step1_sparse_sim.deconv_iter": [6, 8, 10],
    "step2_local_normalization.radius": [20, 30, 40, 50],
    "step2_local_normalization.bias": [0.0003, 0.0005, 0.0008, 0.001],
    "step3_cellpose_3d.cellprob_threshold": [-1.0, 0.0, 0.5, 1.0, 2.0, 2.5],
    "step3_cellpose_3d.min_size": [30, 50, 80, 100],
    "step3_cellpose_3d.anisotropy": [1.0, 1.5, 2.0],
    "step3_cellpose_3d.rescale": [0.8, 1.0, 1.1, 1.25, 1.5],
    "step3_cellpose_3d.flow_threshold": [0.2, 0.4, 0.6],
    "step3_cellpose_3d.tile_overlap": [0.1, 0.2, 0.25],
    "step3_cellpose_3d.batch_size_3d": [2, 4],
    "step3_cellpose_3d.bsize": [224, 256],
    "step3_cellpose_3d.augment": [False, True],
}

FORBIDDEN_PATHS = {
    "step3_cellpose_3d.diameter",
    "step3_cellpose_3d.do_3D",
    "step3_cellpose_3d.z_axis",
    "model.model_path",
    "model.config_path",
}


# -----------------------------------------------------------------------------
# Dict path helpers
# -----------------------------------------------------------------------------


def get_by_path(cfg: dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        cur = cur[key]
    return cur



def set_by_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur: Any = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value



def diff_dict(base: Any, new: Any) -> Any:
    if isinstance(base, dict) and isinstance(new, dict):
        out: dict[str, Any] = {}
        for key in new.keys():
            if key not in base:
                out[key] = copy.deepcopy(new[key])
            else:
                child = diff_dict(base[key], new[key])
                if child not in ({}, None):
                    out[key] = child
        return out
    if base != new:
        return copy.deepcopy(new)
    return None


# -----------------------------------------------------------------------------
# Results parsing / parent selection
# -----------------------------------------------------------------------------


def read_results_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))



def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None



def choose_seed_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    best = None
    best_score = None
    for row in rows:
        if row.get("decision") not in {"keep", "baseline_keep"}:
            continue
        score = safe_float(row.get("score"))
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best = row
    return best



def next_trial_id(rows: list[dict[str, str]], prefix: str = "trial") -> str:
    max_num = 0
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for row in rows:
        tid = row.get("trial_id", "")
        m = pat.match(tid)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"{prefix}_{max_num + 1:04d}"


# -----------------------------------------------------------------------------
# Mutation logic
# -----------------------------------------------------------------------------


def choose_search_space(baseline_cfg: dict[str, Any]) -> dict[str, list[Any]]:
    cfg_space = baseline_cfg.get("search_space")
    if isinstance(cfg_space, dict) and cfg_space:
        return cfg_space
    return copy.deepcopy(DEFAULT_SEARCH_SPACE)



def nearest_index(options: list[Any], current: Any) -> int | None:
    if not options:
        return None
    try:
        return options.index(current)
    except ValueError:
        pass

    # For numeric options, use nearest value.
    if isinstance(current, (int, float)) and all(isinstance(x, (int, float)) for x in options):
        dists = [abs(float(x) - float(current)) for x in options]
        return int(min(range(len(options)), key=lambda i: dists[i]))
    return None



def choose_new_value(path: str, current: Any, options: list[Any], strategy: str, rng: random.Random) -> Any:
    if not options:
        raise ValueError(f"Search-space options are empty for {path}")

    # Never mutate forbidden paths.
    if path in FORBIDDEN_PATHS:
        return current

    # Prefer local neighbors if possible.
    if strategy == "local":
        idx = nearest_index(options, current)
        if idx is not None and len(options) > 1:
            neighbors = []
            if idx - 1 >= 0:
                neighbors.append(options[idx - 1])
            if idx + 1 < len(options):
                neighbors.append(options[idx + 1])
            if neighbors:
                return rng.choice(neighbors)

    # Otherwise choose any different option.
    different = [opt for opt in options if opt != current]
    if different:
        return rng.choice(different)
    return current



def mutate_seed_config(
    seed_cfg: dict[str, Any],
    search_space: dict[str, list[Any]],
    *,
    n_changes: int,
    strategy: str,
    rng: random.Random,
) -> tuple[dict[str, Any], list[str]]:
    mutated = copy.deepcopy(seed_cfg)
    paths = [p for p in search_space.keys() if p not in FORBIDDEN_PATHS]
    if not paths:
        raise ValueError("No mutable paths are available in the search space.")

    n_changes = max(1, min(int(n_changes), len(paths)))
    chosen_paths = rng.sample(paths, n_changes)

    changed_paths: list[str] = []
    for path in chosen_paths:
        current = get_by_path(mutated, path)
        new_value = choose_new_value(path, current, list(search_space[path]), strategy, rng)
        if new_value != current:
            set_by_path(mutated, path, new_value)
            changed_paths.append(path)

    # Safety hard-fix, just in case.
    set_by_path(mutated, "step3_cellpose_3d.diameter", None)
    set_by_path(mutated, "step3_cellpose_3d.do_3D", True)
    set_by_path(mutated, "step3_cellpose_3d.z_axis", 0)

    return mutated, changed_paths


# -----------------------------------------------------------------------------
# Trial-config building
# -----------------------------------------------------------------------------


def summarize_changed_sections(paths: list[str]) -> tuple[bool, bool, bool]:
    s1 = any(p.startswith("step1_sparse_sim.") for p in paths)
    s2 = any(p.startswith("step2_local_normalization.") for p in paths)
    s3 = any(p.startswith("step3_cellpose_3d.") for p in paths)
    return s1, s2, s3



def build_hypothesis(paths: list[str]) -> str:
    if not paths:
        return "No parameter changes were made; this proposal is likely invalid and should be regenerated."
    return (
        "Test whether changing "
        + ", ".join(paths)
        + " improves the linked Step1/Step2/Step3 trade-off between cell count and hollow artifacts."
    )



def risk_summary(paths: list[str]) -> dict[str, str]:
    artifact_keys = {"step3_cellpose_3d.cellprob_threshold", "step3_cellpose_3d.min_size", "step3_cellpose_3d.rescale"}
    oom_keys = {"step3_cellpose_3d.batch_size_3d", "step3_cellpose_3d.bsize", "step3_cellpose_3d.rescale"}
    runtime_keys = {"step1_sparse_sim.sparse_iter", "step1_sparse_sim.deconv_iter", "step3_cellpose_3d.batch_size_3d"}
    return {
        "oom_risk": "medium" if any(p in oom_keys for p in paths) else "low",
        "artifact_risk": "medium" if any(p in artifact_keys for p in paths) else "low",
        "runtime_risk": "medium" if any(p in runtime_keys for p in paths) else "low",
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propose a new trial_config.json from baseline and prior results.")
    parser.add_argument("--baseline", required=True, help="Path to baseline_config.json")
    parser.add_argument("--output", required=True, help="Output path for the new trial_config.json")
    parser.add_argument("--results", default=None, help="Optional results.tsv path; if given, seed from current best kept row.")
    parser.add_argument("--run-tag", default=None, help="Optional run_tag override for the proposed trial.")
    parser.add_argument("--strategy", choices=["local", "random"], default="local", help="How to mutate parameters.")
    parser.add_argument("--n-changes", type=int, default=3, help="How many parameter paths to mutate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible proposal generation.")
    parser.add_argument(
        "--force-seed-used-config",
        default=None,
        help="Optional explicit used_config.json path to seed from. Overrides results.tsv selection.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    baseline_cfg = load_json(baseline_path)
    rng = random.Random(args.seed)

    rows: list[dict[str, str]] = []
    if args.results:
        rows = read_results_tsv(Path(args.results).expanduser().resolve())

    seed_cfg = copy.deepcopy(baseline_cfg)
    proposed_from: dict[str, Any] = {
        "source": "baseline",
        "baseline_path": str(baseline_path),
    }
    parent_trial_id = None

    if args.force_seed_used_config:
        forced_path = Path(args.force_seed_used_config).expanduser().resolve()
        seed_cfg = load_json(forced_path)
        proposed_from = {
            "source": "forced_used_config",
            "used_config_path": str(forced_path),
        }
    else:
        seed_row = choose_seed_row(rows)
        if seed_row is not None:
            seed_used_cfg_path = Path(seed_row["used_config_path"]).expanduser().resolve()
            if seed_used_cfg_path.exists():
                seed_cfg = load_json(seed_used_cfg_path)
                parent_trial_id = seed_row.get("trial_id")
                proposed_from = {
                    "source": "results_best",
                    "used_config_path": str(seed_used_cfg_path),
                    "parent_trial_id": parent_trial_id,
                    "parent_score": safe_float(seed_row.get("score")),
                }

    search_space = choose_search_space(baseline_cfg)
    mutated_cfg, changed_paths = mutate_seed_config(
        seed_cfg,
        search_space,
        n_changes=args.n_changes,
        strategy=args.strategy,
        rng=rng,
    )

    # Convert back into baseline-relative overrides so materialize_config.py can consume it.
    overrides = diff_dict(baseline_cfg, mutated_cfg) or {}

    s1_changed, s2_changed, s3_changed = summarize_changed_sections(changed_paths)
    run_tag = args.run_tag or baseline_cfg.get("run_tag", "linked_search")
    trial_id = next_trial_id(rows)

    trial_cfg = {
        "schema_version": "1.0",
        "config_type": "trial",
        "trial_id": trial_id,
        "parent_trial_id": parent_trial_id,
        "parent_config": str(baseline_path),
        "run_tag": run_tag,
        "description": f"Autogenerated proposal from {proposed_from['source']} using {args.strategy} search.",
        "inherit_from_parent": True,
        "proposed_from": proposed_from,
        "search_edit_summary": {
            "step1_changed": s1_changed,
            "step2_changed": s2_changed,
            "step3_changed": s3_changed,
            "changed_keys": changed_paths,
            "hypothesis": build_hypothesis(changed_paths),
        },
        "overrides": overrides,
        "expected_risk": risk_summary(changed_paths),
        "evaluation_request": {
            "level": "crop"
        },
        "proposal_debug": {
            "strategy": args.strategy,
            "n_changes": int(args.n_changes),
            "random_seed": int(args.seed),
            "search_space_keys": sorted(search_space.keys()),
        },
    }

    save_json(output_path, trial_cfg)
    print("=" * 90)
    print("Trial proposal generated")
    print(f"output         : {output_path}")
    print(f"trial_id       : {trial_id}")
    print(f"proposed_from  : {compact_json(proposed_from)}")
    print(f"changed_keys   : {compact_json(changed_paths)}")
    print("=" * 90)


if __name__ == "__main__":
    main()
