#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finalize_campaign.py

Campaign finalization orchestrator.

Steps
-----
1. summarize_campaign.py
2. render_trial_previews.py
3. export_codex_bundle.py
4. make_codex_bundle_zip.py

Design requirement
------------------
Preview rendering must not block the entire finalization. If preview generation
fails, finalization should continue and still produce the bundle + zip.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(cmd: list[str], *, step_name: str, cwd: Path | None = None, allow_failure: bool = False) -> int:
    print("=" * 90)
    print(f"[finalize] step: {step_name}")
    print(f"[finalize] cmd : {' '.join(cmd)}")
    print("=" * 90)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None)
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(f"{step_name} failed with returncode={proc.returncode}")
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize one campaign into codex_bundle.zip")
    parser.add_argument(
        "--campaign-name",
        required=True,
        help="Campaign directory name, e.g. campaign_001",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root. Defaults to current project root.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Optional explicit results.tsv path. Defaults to <repo_root>/results.tsv",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for subprocess calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    scripts_root = repo_root / "scripts"
    results_path = Path(args.results).expanduser().resolve() if args.results else (repo_root / "results.tsv")

    summarize_script = scripts_root / "summarize_campaign.py"
    preview_script = scripts_root / "render_trial_previews.py"
    export_script = scripts_root / "export_codex_bundle.py"
    zip_script = scripts_root / "make_codex_bundle_zip.py"

    if not summarize_script.exists():
        raise FileNotFoundError(f"Missing script: {summarize_script}")
    if not preview_script.exists():
        raise FileNotFoundError(f"Missing script: {preview_script}")
    if not export_script.exists():
        raise FileNotFoundError(f"Missing script: {export_script}")
    if not zip_script.exists():
        raise FileNotFoundError(f"Missing script: {zip_script}")

    common = [
        "--campaign-name",
        args.campaign_name,
        "--repo-root",
        str(repo_root),
    ]

    common_with_results = common + ["--results", str(results_path)]

    # 1) summary: required
    run_step(
        [args.python, str(summarize_script), *common_with_results],
        step_name="summarize_campaign",
        cwd=repo_root,
        allow_failure=False,
    )

    # 2) previews: best effort only
    preview_rc = run_step(
        [args.python, str(preview_script), *common],
        step_name="render_trial_previews",
        cwd=repo_root,
        allow_failure=True,
    )

    # 3) export bundle: required
    run_step(
        [args.python, str(export_script), *common_with_results],
        step_name="export_codex_bundle",
        cwd=repo_root,
        allow_failure=False,
    )

    # 4) zip bundle: required
    run_step(
        [args.python, str(zip_script), *common],
        step_name="make_codex_bundle_zip",
        cwd=repo_root,
        allow_failure=False,
    )

    print("=" * 90)
    print("Campaign finalization finished")
    print(f"campaign_name   : {args.campaign_name}")
    print(f"repo_root       : {repo_root}")
    print(f"results_path    : {results_path}")
    print(f"preview_rc      : {preview_rc}")
    print("=" * 90)


if __name__ == "__main__":
    main()