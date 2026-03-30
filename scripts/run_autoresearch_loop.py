#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_autoresearch_loop.py

Prepare a small batch of autoresearch trials locally for later execution on a
remote Slurm cluster.

This script intentionally stays on the orchestration side only. It does not run
the scientific pipeline itself. Instead, it:

1. reads the baseline config,
2. proposes a small number of trial configs,
3. materializes runnable used_config.json files,
4. creates per-trial metadata,
5. generates Slurm job scripts that execute:
   - run_trial.py
   - score_trial.py
   - append_results.py

The first version is deliberately simple and reproducible.
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from materialize_config import (
    load_json as load_json_file,
    materialize_config,
    save_json,
    validate_config_shape,
    validate_hard_constraints,
    validate_required_sections,
)
from propose_trial import (
    build_hypothesis,
    choose_search_space,
    choose_seed_row,
    compact_json,
    mutate_seed_config,
    next_trial_id,
    read_results_tsv,
    risk_summary,
    summarize_changed_sections,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local autoresearch batch for later Slurm execution."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline_config.json",
    )
    parser.add_argument(
        "--crop-manifest",
        required=True,
        help="Path to crop_manifest.json used by the prepared trials.",
    )
    parser.add_argument(
        "--results",
        default="results.tsv",
        help="Path to results.tsv used for seeding and later result append.",
    )
    parser.add_argument(
        "--output-root",
        default="runs",
        help="Root directory where prepared campaign artifacts will be written.",
    )
    parser.add_argument(
        "--campaign-name",
        default=None,
        help="Optional campaign directory name. Default is a timestamped prepared_* name.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run_tag override for the prepared trial configs.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="How many candidate trials to prepare.",
    )
    parser.add_argument(
        "--strategy",
        choices=["local", "random"],
        default="local",
        help="Mutation strategy for candidate proposals.",
    )
    parser.add_argument(
        "--n-changes",
        type=int,
        default=3,
        help="How many parameter paths to change per candidate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for deterministic proposal generation.",
    )
    parser.add_argument(
        "--cluster-repo-root",
        required=True,
        help="Absolute repository path on the Slurm cluster.",
    )
    parser.add_argument(
        "--cluster-python",
        default="python",
        help="Python executable to use inside the Slurm job.",
    )
    parser.add_argument(
        "--cluster-step12-script",
        required=True,
        help="Path to run_step12_pipeline.py on the cluster.",
    )
    parser.add_argument(
        "--cluster-step3-script",
        required=True,
        help="Path to run_infer_3d.py on the cluster.",
    )
    parser.add_argument(
        "--cluster-results-path",
        default=None,
        help="Optional explicit results.tsv path on the cluster. Defaults to <cluster_repo_root>/results.tsv.",
    )
    parser.add_argument(
        "--cluster-setup-line",
        action="append",
        default=[],
        help="Optional shell line to include in each Slurm script, e.g. 'module load cuda'. Repeat as needed.",
    )
    parser.add_argument("--slurm-partition", default="gpu", help="Slurm partition.")
    parser.add_argument("--slurm-account", default=None, help="Optional Slurm account.")
    parser.add_argument("--slurm-time", default="04:00:00", help="Slurm wall time.")
    parser.add_argument("--slurm-gpus", type=int, default=1, help="GPUs per job.")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=4, help="CPUs per task.")
    parser.add_argument("--slurm-mem", default="32G", help="Memory request.")
    return parser.parse_args()


def ensure_crop_manifest_has_crops(path: Path) -> dict[str, Any]:
    manifest = load_json_file(path)
    crops = manifest.get("crops", [])
    if not isinstance(crops, list) or not crops:
        raise ValueError(f"Crop manifest has no crops: {path}")
    return manifest


def choose_seed_config(
    baseline_cfg: dict[str, Any],
    baseline_path: Path,
    existing_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    seed_cfg = copy.deepcopy(baseline_cfg)
    proposed_from: dict[str, Any] = {
        "source": "baseline",
        "baseline_path": str(baseline_path),
    }
    parent_trial_id = None

    seed_row = choose_seed_row(existing_rows)
    if seed_row is None:
        return seed_cfg, proposed_from, parent_trial_id

    used_config_path = seed_row.get("used_config_path")
    if not used_config_path:
        return seed_cfg, proposed_from, parent_trial_id

    seed_path = Path(used_config_path)
    if not seed_path.is_absolute():
        seed_path = (REPO_ROOT / seed_path).resolve()
    if not seed_path.exists():
        return seed_cfg, proposed_from, parent_trial_id

    seed_cfg = load_json_file(seed_path)
    parent_trial_id = seed_row.get("trial_id")
    proposed_from = {
        "source": "results_best",
        "used_config_path": str(seed_path),
        "parent_trial_id": parent_trial_id,
        "parent_score": seed_row.get("score"),
    }
    return seed_cfg, proposed_from, parent_trial_id


def build_trial_config(
    baseline_cfg: dict[str, Any],
    baseline_path: Path,
    existing_rows: list[dict[str, str]],
    *,
    run_tag: str,
    strategy: str,
    n_changes: int,
    seed: int,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    search_space = choose_search_space(baseline_cfg)
    trial_id = next_trial_id(existing_rows)
    seed_cfg, proposed_from, parent_trial_id = choose_seed_config(
        baseline_cfg,
        baseline_path,
        existing_rows,
    )

    mutated_cfg = None
    changed_paths: list[str] = []
    for _ in range(12):
        candidate_cfg, candidate_changed_paths = mutate_seed_config(
            seed_cfg,
            search_space,
            n_changes=n_changes,
            strategy=strategy,
            rng=rng,
        )
        if candidate_changed_paths:
            mutated_cfg = candidate_cfg
            changed_paths = candidate_changed_paths
            break
    if mutated_cfg is None:
        raise RuntimeError("Failed to generate a non-empty mutation set after repeated attempts.")

    overrides = diff_dict(baseline_cfg, mutated_cfg) or {}
    step1_changed, step2_changed, step3_changed = summarize_changed_sections(changed_paths)

    return {
        "schema_version": "1.0",
        "config_type": "trial",
        "trial_id": trial_id,
        "parent_trial_id": parent_trial_id,
        "parent_config": str(baseline_path),
        "run_tag": run_tag,
        "description": f"Prepared locally for Slurm execution using {strategy} search.",
        "inherit_from_parent": True,
        "proposed_from": proposed_from,
        "search_edit_summary": {
            "step1_changed": step1_changed,
            "step2_changed": step2_changed,
            "step3_changed": step3_changed,
            "changed_keys": changed_paths,
            "hypothesis": build_hypothesis(changed_paths),
        },
        "overrides": overrides,
        "expected_risk": risk_summary(changed_paths),
        "evaluation_request": {
            "level": "crop",
        },
        "proposal_debug": {
            "strategy": strategy,
            "n_changes": int(n_changes),
            "random_seed": int(seed),
            "search_space_keys": sorted(search_space.keys()),
        },
    }


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


def shell_quote(path_or_text: str) -> str:
    return shlex.quote(str(path_or_text))


def render_slurm_script(
    *,
    job_name: str,
    cluster_repo_root: str,
    cluster_python: str,
    cluster_step12_script: str,
    cluster_step3_script: str,
    cluster_results_path: str,
    trial_root_rel: Path,
    used_config_rel: Path,
    crop_manifest_rel: Path,
    setup_lines: list[str],
    slurm_partition: str,
    slurm_account: str | None,
    slurm_time: str,
    slurm_gpus: int,
    slurm_cpus_per_task: int,
    slurm_mem: str,
) -> str:
    cluster_repo_root_posix = PurePosixPath(cluster_repo_root)
    trial_root_cluster = cluster_repo_root_posix / trial_root_rel.as_posix()
    used_config_cluster = cluster_repo_root_posix / used_config_rel.as_posix()
    crop_manifest_cluster = cluster_repo_root_posix / crop_manifest_rel.as_posix()
    run_trial_cluster = cluster_repo_root_posix / "scripts" / "run_trial.py"
    score_trial_cluster = cluster_repo_root_posix / "scripts" / "score_trial.py"
    append_results_cluster = cluster_repo_root_posix / "scripts" / "append_results.py"

    header_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={slurm_partition}",
        f"#SBATCH --time={slurm_time}",
        f"#SBATCH --gpus={slurm_gpus}",
        f"#SBATCH --cpus-per-task={slurm_cpus_per_task}",
        f"#SBATCH --mem={slurm_mem}",
        "#SBATCH --output=%x-%j.out",
        "#SBATCH --error=%x-%j.err",
    ]
    if slurm_account:
        header_lines.append(f"#SBATCH --account={slurm_account}")

    body_lines = [
        "set -eo pipefail",
        f"CLUSTER_REPO_ROOT={shell_quote(cluster_repo_root)}",
        f"CLUSTER_PYTHON={shell_quote(cluster_python)}",
        f"TRIAL_ROOT={shell_quote(str(trial_root_cluster))}",
        f"USED_CONFIG={shell_quote(str(used_config_cluster))}",
        f"CROP_MANIFEST={shell_quote(str(crop_manifest_cluster))}",
        f"RESULTS_TSV={shell_quote(cluster_results_path)}",
        f"STEP12_SCRIPT={shell_quote(cluster_step12_script)}",
        f"STEP3_SCRIPT={shell_quote(cluster_step3_script)}",
        f"RUN_TRIAL_SCRIPT={shell_quote(str(run_trial_cluster))}",
        f"SCORE_TRIAL_SCRIPT={shell_quote(str(score_trial_cluster))}",
        f"APPEND_RESULTS_SCRIPT={shell_quote(str(append_results_cluster))}",
    ]

    if setup_lines:
        body_lines.append("set +u")
        body_lines.extend(setup_lines)
        body_lines.append("set -u")

    body_lines.extend(
        [
            "",
            'echo "===== ENV CHECK ====="',
            'echo "HOSTNAME=$(hostname)"',
            'echo "PWD=$(pwd)"',
            'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"',
            'which python || true',
            'python -V || true',
            '"$CLUSTER_PYTHON" -V',
            'nvidia-smi || true',
            'echo "====================="',
            "",
            '"$CLUSTER_PYTHON" "$RUN_TRIAL_SCRIPT" \\',
            '  --used-config "$USED_CONFIG" \\',
            '  --crop-manifest "$CROP_MANIFEST" \\',
            '  --trial-root "$TRIAL_ROOT" \\',
            '  --step12-script "$STEP12_SCRIPT" \\',
            '  --step3-script "$STEP3_SCRIPT" \\',
            "  --copy-used-config",
            "",
            '"$CLUSTER_PYTHON" "$SCORE_TRIAL_SCRIPT" \\',
            '  --trial-root "$TRIAL_ROOT"',
            "",
            'if command -v flock >/dev/null 2>&1; then',
            '  flock "${RESULTS_TSV}.lock" "$CLUSTER_PYTHON" "$APPEND_RESULTS_SCRIPT" \\',
            '    --trial-root "$TRIAL_ROOT" \\',
            '    --results "$RESULTS_TSV" \\',
            "    --writeback-score-json",
            "else",
            '  "$CLUSTER_PYTHON" "$APPEND_RESULTS_SCRIPT" \\',
            '    --trial-root "$TRIAL_ROOT" \\',
            '    --results "$RESULTS_TSV" \\',
            "    --writeback-score-json",
            "fi",
            "",
        ]
    )
    return "\n".join(header_lines + [""] + body_lines)


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline).expanduser().resolve()
    crop_manifest_path = Path(args.crop_manifest).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    try:
        crop_manifest_rel = crop_manifest_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"crop_manifest must live inside the repository so cluster-relative paths stay stable: {crop_manifest_path}"
        ) from exc

    baseline_cfg = load_json_file(baseline_path)
    validate_required_sections(baseline_cfg)
    ensure_crop_manifest_has_crops(crop_manifest_path)

    run_tag = args.run_tag or baseline_cfg.get("run_tag", "autoresearch")
    campaign_name = args.campaign_name or f"prepared_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    campaign_root = output_root / campaign_name
    trials_root = campaign_root / "trials"
    campaign_root.mkdir(parents=True, exist_ok=True)
    trials_root.mkdir(parents=True, exist_ok=True)

    existing_rows = read_results_tsv(results_path)
    cluster_repo_root_posix = PurePosixPath(args.cluster_repo_root)
    cluster_results_path = args.cluster_results_path or str(cluster_repo_root_posix / "results.tsv")
    prepared_trials: list[dict[str, Any]] = []

    for idx in range(args.num_candidates):
        trial_seed = args.seed + idx
        trial_cfg = build_trial_config(
            baseline_cfg,
            baseline_path,
            existing_rows,
            run_tag=run_tag,
            strategy=args.strategy,
            n_changes=args.n_changes,
            seed=trial_seed,
        )
        trial_id = trial_cfg["trial_id"]
        trial_root = trials_root / trial_id
        slurm_dir = trial_root / "slurm"
        trial_root.mkdir(parents=True, exist_ok=True)
        slurm_dir.mkdir(parents=True, exist_ok=True)

        trial_config_path = trial_root / "trial_config.json"
        used_config_path = trial_root / "used_config.json"
        metadata_path = trial_root / "trial_metadata.json"
        slurm_script_path = slurm_dir / f"{trial_id}.sbatch"

        used_cfg = materialize_config(baseline_cfg, trial_cfg)
        validate_required_sections(used_cfg)
        validate_hard_constraints(used_cfg)
        validate_config_shape(used_cfg)
        used_cfg["materialization"]["baseline_path"] = str(baseline_path)
        used_cfg["materialization"]["trial_path"] = str(trial_config_path)
        used_cfg["materialization"]["output_path"] = str(used_config_path)

        save_json(trial_config_path, trial_cfg)
        save_json(used_config_path, used_cfg)

        trial_root_rel = trial_root.relative_to(REPO_ROOT)
        used_config_rel = used_config_path.relative_to(REPO_ROOT)
        slurm_script = render_slurm_script(
            job_name=trial_id,
            cluster_repo_root=args.cluster_repo_root,
            cluster_python=args.cluster_python,
            cluster_step12_script=args.cluster_step12_script,
            cluster_step3_script=args.cluster_step3_script,
            cluster_results_path=cluster_results_path,
            trial_root_rel=trial_root_rel,
            used_config_rel=used_config_rel,
            crop_manifest_rel=crop_manifest_rel,
            setup_lines=args.cluster_setup_line,
            slurm_partition=args.slurm_partition,
            slurm_account=args.slurm_account,
            slurm_time=args.slurm_time,
            slurm_gpus=args.slurm_gpus,
            slurm_cpus_per_task=args.slurm_cpus_per_task,
            slurm_mem=args.slurm_mem,
        )
        slurm_script_path.write_text(slurm_script, encoding="utf-8")

        metadata = {
            "schema_version": "1.0",
            "status": "prepared",
            "prepared_at": datetime.now().isoformat(timespec="seconds"),
            "campaign_name": campaign_name,
            "run_tag": run_tag,
            "trial_id": trial_id,
            "baseline_path": str(baseline_path),
            "results_path_local": str(results_path),
            "results_path_cluster": cluster_results_path,
            "crop_manifest_local": str(crop_manifest_path),
            "crop_manifest_cluster": str(cluster_repo_root_posix / crop_manifest_rel.as_posix()),
            "trial_config_path": str(trial_config_path),
            "used_config_path": str(used_config_path),
            "trial_root": str(trial_root),
            "slurm_script_path": str(slurm_script_path),
            "cluster_execution": {
                "cluster_repo_root": args.cluster_repo_root,
                "cluster_python": args.cluster_python,
                "cluster_step12_script": args.cluster_step12_script,
                "cluster_step3_script": args.cluster_step3_script,
                "slurm_partition": args.slurm_partition,
                "slurm_account": args.slurm_account,
                "slurm_time": args.slurm_time,
                "slurm_gpus": args.slurm_gpus,
                "slurm_cpus_per_task": args.slurm_cpus_per_task,
                "slurm_mem": args.slurm_mem,
                "setup_lines": args.cluster_setup_line,
            },
            "trial_config_summary": {
                "changed_keys": trial_cfg["search_edit_summary"]["changed_keys"],
                "hypothesis": trial_cfg["search_edit_summary"]["hypothesis"],
                "expected_risk": trial_cfg["expected_risk"],
                "proposal_debug": trial_cfg["proposal_debug"],
                "proposed_from": trial_cfg["proposed_from"],
            },
        }
        save_json(metadata_path, metadata)

        prepared_trials.append(
            {
                "trial_id": trial_id,
                "trial_root": str(trial_root),
                "trial_config_path": str(trial_config_path),
                "used_config_path": str(used_config_path),
                "metadata_path": str(metadata_path),
                "slurm_script_path": str(slurm_script_path),
                "changed_keys": trial_cfg["search_edit_summary"]["changed_keys"],
            }
        )

        existing_rows.append(
            {
                "trial_id": trial_id,
                "used_config_path": str(used_config_path),
            }
        )

    submit_all_path = campaign_root / "submit_all.sh"
    submit_all_lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'cd "$SCRIPT_DIR"',
        "",
    ]
    for trial in prepared_trials:
        trial_id = trial["trial_id"]
        submit_all_lines.append(
            f"sbatch {shell_quote(f'trials/{trial_id}/slurm/{trial_id}.sbatch')}"
        )
    submit_all_path.write_text("\n".join(submit_all_lines) + "\n", encoding="utf-8")

    campaign_manifest = {
        "schema_version": "1.0",
        "campaign_name": campaign_name,
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "baseline_path": str(baseline_path),
        "crop_manifest_path": str(crop_manifest_path),
        "results_path": str(results_path),
        "run_tag": run_tag,
        "num_candidates": len(prepared_trials),
        "strategy": args.strategy,
        "n_changes": args.n_changes,
        "base_seed": args.seed,
        "cluster_execution": {
            "cluster_repo_root": args.cluster_repo_root,
            "cluster_python": args.cluster_python,
            "cluster_step12_script": args.cluster_step12_script,
            "cluster_step3_script": args.cluster_step3_script,
            "cluster_results_path": cluster_results_path,
            "setup_lines": args.cluster_setup_line,
        },
        "trials": prepared_trials,
    }
    save_json(campaign_root / "campaign_manifest.json", campaign_manifest)

    print("=" * 90)
    print("Prepared autoresearch campaign for Slurm execution")
    print(f"campaign_root  : {campaign_root}")
    print(f"baseline       : {baseline_path}")
    print(f"crop_manifest  : {crop_manifest_path}")
    print(f"results.tsv    : {results_path}")
    print(f"num_candidates : {len(prepared_trials)}")
    for trial in prepared_trials:
        print(f" - {trial['trial_id']}: {compact_json(trial['changed_keys'])}")
    print("=" * 90)


if __name__ == "__main__":
    main()