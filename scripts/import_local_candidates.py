#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
import_local_candidates.py

Import a local next_round_upload/ package into a new campaign directory and
prepare Slurm execution artifacts.

Expected local upload package
-----------------------------
next_round_upload/
├── next_round_plan.json
├── trial_configs/
│   ├── *.json
├── README_UPLOAD.md              # optional
└── optional_notes/               # optional

This script does NOT invent new candidates.
It only imports, validates, materializes, and prepares execution artifacts.
"""

from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import local next_round_upload/ into a new campaign."
    )
    parser.add_argument(
        "--upload-dir",
        required=True,
        help="Path to next_round_upload/ directory.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline_config.json",
    )
    parser.add_argument(
        "--crop-manifest",
        required=True,
        help="Path to crop_manifest.json used by the imported trials.",
    )
    parser.add_argument(
        "--results",
        default="results.tsv",
        help="Path to global results.tsv",
    )
    parser.add_argument(
        "--output-root",
        default="runs",
        help="Root directory where campaign artifacts will be written.",
    )
    parser.add_argument(
        "--campaign-name",
        default=None,
        help="Optional explicit campaign name. If omitted, auto-increment campaign_xxx.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run_tag override for all imported trials.",
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
        help="Optional explicit cluster results.tsv path. Defaults to <cluster_repo_root>/results.tsv",
    )
    parser.add_argument(
        "--cluster-setup-line",
        action="append",
        default=[],
        help="Optional shell line to include in each Slurm script. Repeat as needed.",
    )
    parser.add_argument("--slurm-partition", default="gpu", help="Slurm partition.")
    parser.add_argument("--slurm-account", default=None, help="Optional Slurm account.")
    parser.add_argument("--slurm-time", default="04:00:00", help="Slurm wall time.")
    parser.add_argument("--slurm-gpus", type=int, default=1, help="GPUs per job.")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=4, help="CPUs per task.")
    parser.add_argument("--slurm-mem", default="32G", help="Memory request.")
    return parser.parse_args()


def shell_quote(path_or_text: str) -> str:
    return shlex.quote(str(path_or_text))


def detect_next_campaign_name(runs_root: Path, prefix: str = "campaign_") -> str:
    max_num = 0
    if runs_root.exists():
        for item in runs_root.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                if suffix.isdigit():
                    max_num = max(max_num, int(suffix))
    return f"{prefix}{max_num + 1:03d}"


def ensure_crop_manifest_has_crops(path: Path) -> dict[str, Any]:
    manifest = load_json_file(path)
    crops = manifest.get("crops", [])
    if not isinstance(crops, list) or not crops:
        raise ValueError(f"Crop manifest has no crops: {path}")
    return manifest


def normalize_trial_config(
    trial_cfg: dict[str, Any],
    *,
    baseline_path: Path,
    run_tag: str | None,
    fallback_trial_id: str,
) -> dict[str, Any]:
    cfg = dict(trial_cfg)

    # Preserve user's provided fields when present.
    cfg.setdefault("schema_version", "1.0")
    cfg.setdefault("config_type", "trial")
    cfg.setdefault("inherit_from_parent", True)
    cfg.setdefault("parent_config", str(baseline_path))
    cfg.setdefault("trial_id", fallback_trial_id)
    cfg.setdefault("description", "Imported from local next_round_upload.")
    cfg.setdefault("evaluation_request", {"level": "crop"})
    cfg.setdefault("proposed_from", {"source": "local_codex_upload"})
    cfg.setdefault("search_edit_summary", {
        "step1_changed": False,
        "step2_changed": False,
        "step3_changed": False,
        "changed_keys": [],
        "hypothesis": "Imported local candidate trial.",
    })
    cfg.setdefault("expected_risk", {})

    if run_tag is not None:
        cfg["run_tag"] = run_tag
    else:
        cfg.setdefault("run_tag", "linked_search")

    return cfg


def render_trial_sbatch(
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


def render_finalize_sbatch(
    *,
    campaign_name: str,
    cluster_repo_root: str,
    cluster_python: str,
    cluster_results_path: str,
    setup_lines: list[str],
    slurm_partition: str,
    slurm_account: str | None,
    slurm_time: str,
    slurm_cpus_per_task: int,
    slurm_mem: str,
) -> str:
    cluster_repo_root_posix = PurePosixPath(cluster_repo_root)
    finalize_script_cluster = cluster_repo_root_posix / "scripts" / "finalize_campaign.py"

    header_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={campaign_name}_finalize",
        f"#SBATCH --partition={slurm_partition}",
        f"#SBATCH --time={slurm_time}",
        f"#SBATCH --cpus-per-task={max(1, slurm_cpus_per_task)}",
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
        f"FINALIZE_SCRIPT={shell_quote(str(finalize_script_cluster))}",
        f"RESULTS_TSV={shell_quote(cluster_results_path)}",
    ]

    if setup_lines:
        body_lines.append("set +u")
        body_lines.extend(setup_lines)
        body_lines.append("set -u")

    body_lines.extend(
        [
            "",
            '"$CLUSTER_PYTHON" "$FINALIZE_SCRIPT" \\',
            f"  --campaign-name {shell_quote(campaign_name)} \\",
            '  --repo-root "$CLUSTER_REPO_ROOT" \\',
            '  --results "$RESULTS_TSV"',
            "",
        ]
    )
    return "\n".join(header_lines + [""] + body_lines)


def render_submit_all(
    *,
    prepared_trials: list[dict[str, Any]],
    campaign_name: str,
) -> str:
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'cd "$SCRIPT_DIR"',
        "",
        "declare -a JOB_IDS=()",
        "",
    ]

    for trial in prepared_trials:
        trial_id = trial["trial_id"]
        rel_sbatch = f"trials/{trial_id}/slurm/{trial_id}.sbatch"
        lines.extend(
            [
                f'jid=$(sbatch {shell_quote(rel_sbatch)} | awk \'{{print $4}}\')',
                'echo "submitted trial job: $jid"',
                'JOB_IDS+=("$jid")',
                "",
            ]
        )

    lines.extend(
        [
            'if [ "${#JOB_IDS[@]}" -eq 0 ]; then',
            '  echo "No trial jobs were prepared."',
            "  exit 1",
            "fi",
            "",
            'dep=$(IFS=:; echo "${JOB_IDS[*]}")',
            f'finalize_jid=$(sbatch --dependency=afterany:$dep slurm/finalize_{campaign_name}.sbatch | awk \'{{print $4}}\')',
            'echo "submitted finalize job: $finalize_jid"',
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    upload_dir = Path(args.upload_dir).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    crop_manifest_path = Path(args.crop_manifest).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not upload_dir.exists():
        raise FileNotFoundError(f"Upload directory not found: {upload_dir}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline config not found: {baseline_path}")
    if not crop_manifest_path.exists():
        raise FileNotFoundError(f"Crop manifest not found: {crop_manifest_path}")

    ensure_crop_manifest_has_crops(crop_manifest_path)

    trial_configs_dir = upload_dir / "trial_configs"
    if not trial_configs_dir.exists():
        raise FileNotFoundError(f"Missing trial_configs/ directory in upload: {trial_configs_dir}")

    trial_config_files = sorted(trial_configs_dir.glob("*.json"))
    if not trial_config_files:
        raise ValueError(f"No *.json files found under {trial_configs_dir}")

    try:
        crop_manifest_rel = crop_manifest_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"crop_manifest must live inside repository for stable cluster-relative paths: {crop_manifest_path}"
        ) from exc

    runs_root = output_root
    campaign_name = args.campaign_name or detect_next_campaign_name(runs_root)
    campaign_root = runs_root / campaign_name
    trials_root = campaign_root / "trials"
    slurm_root = campaign_root / "slurm"

    campaign_root.mkdir(parents=True, exist_ok=True)
    trials_root.mkdir(parents=True, exist_ok=True)
    slurm_root.mkdir(parents=True, exist_ok=True)

    baseline_cfg = load_json_file(baseline_path)
    validate_required_sections(baseline_cfg)

    cluster_repo_root_posix = PurePosixPath(args.cluster_repo_root)
    cluster_results_path = args.cluster_results_path or str(cluster_repo_root_posix / "results.tsv")

    prepared_trials: list[dict[str, Any]] = []

    for idx, cfg_path in enumerate(trial_config_files, start=1):
        raw_trial_cfg = load_json_file(cfg_path)
        fallback_trial_id = f"trial_{idx:04d}"
        trial_cfg = normalize_trial_config(
            raw_trial_cfg,
            baseline_path=baseline_path,
            run_tag=args.run_tag,
            fallback_trial_id=fallback_trial_id,
        )

        trial_id = str(trial_cfg["trial_id"])
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

        used_cfg.setdefault("materialization", {})
        used_cfg["materialization"]["baseline_path"] = str(baseline_path)
        used_cfg["materialization"]["trial_path"] = str(trial_config_path)
        used_cfg["materialization"]["output_path"] = str(used_config_path)

        save_json(trial_config_path, trial_cfg)
        save_json(used_config_path, used_cfg)

        trial_root_rel = trial_root.relative_to(REPO_ROOT)
        used_config_rel = used_config_path.relative_to(REPO_ROOT)

        slurm_script = render_trial_sbatch(
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
            "run_tag": trial_cfg.get("run_tag"),
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
                "changed_keys": (trial_cfg.get("search_edit_summary", {}) or {}).get("changed_keys", []),
                "hypothesis": (trial_cfg.get("search_edit_summary", {}) or {}).get("hypothesis"),
                "expected_risk": trial_cfg.get("expected_risk", {}),
                "proposed_from": trial_cfg.get("proposed_from", {}),
            },
            "source_upload_trial_config": str(cfg_path),
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
                "changed_keys": (trial_cfg.get("search_edit_summary", {}) or {}).get("changed_keys", []),
            }
        )

    finalize_sbatch_path = slurm_root / f"finalize_{campaign_name}.sbatch"
    finalize_sbatch = render_finalize_sbatch(
        campaign_name=campaign_name,
        cluster_repo_root=args.cluster_repo_root,
        cluster_python=args.cluster_python,
        cluster_results_path=cluster_results_path,
        setup_lines=args.cluster_setup_line,
        slurm_partition=args.slurm_partition,
        slurm_account=args.slurm_account,
        slurm_time=args.slurm_time,
        slurm_cpus_per_task=max(1, args.slurm_cpus_per_task),
        slurm_mem=args.slurm_mem,
    )
    finalize_sbatch_path.write_text(finalize_sbatch, encoding="utf-8")

    submit_all_path = campaign_root / "submit_all.sh"
    submit_all_path.write_text(
        render_submit_all(prepared_trials=prepared_trials, campaign_name=campaign_name),
        encoding="utf-8",
    )

    upload_manifest = {
        "schema_version": "1.0",
        "imported_at": datetime.now().isoformat(timespec="seconds"),
        "upload_dir": str(upload_dir),
        "trial_configs_count": len(trial_config_files),
        "trial_config_files": [str(p) for p in trial_config_files],
        "next_round_plan_json": str(upload_dir / "next_round_plan.json") if (upload_dir / "next_round_plan.json").exists() else None,
        "readme_upload_md": str(upload_dir / "README_UPLOAD.md") if (upload_dir / "README_UPLOAD.md").exists() else None,
    }
    save_json(campaign_root / "import_manifest.json", upload_manifest)

    campaign_manifest = {
        "schema_version": "1.0",
        "campaign_name": campaign_name,
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "baseline_path": str(baseline_path),
        "crop_manifest_path": str(crop_manifest_path),
        "results_path": str(results_path),
        "run_tag": args.run_tag,
        "num_candidates": len(prepared_trials),
        "strategy": "imported_local_candidates",
        "n_changes": None,
        "base_seed": None,
        "cluster_execution": {
            "cluster_repo_root": args.cluster_repo_root,
            "cluster_python": args.cluster_python,
            "cluster_step12_script": args.cluster_step12_script,
            "cluster_step3_script": args.cluster_step3_script,
            "cluster_results_path": cluster_results_path,
            "setup_lines": args.cluster_setup_line,
        },
        "trials": prepared_trials,
        "finalize_sbatch_path": str(finalize_sbatch_path),
        "import_manifest_path": str(campaign_root / "import_manifest.json"),
    }
    save_json(campaign_root / "campaign_manifest.json", campaign_manifest)

    print("=" * 90)
    print("Imported local candidates into new campaign")
    print(f"upload_dir       : {upload_dir}")
    print(f"campaign_root    : {campaign_root}")
    print(f"campaign_name    : {campaign_name}")
    print(f"num_trials       : {len(prepared_trials)}")
    print(f"submit_all.sh    : {submit_all_path}")
    print(f"finalize_sbatch  : {finalize_sbatch_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()