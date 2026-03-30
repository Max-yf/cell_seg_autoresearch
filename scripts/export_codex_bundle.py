#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_codex_bundle.py

Create a lightweight codex_bundle/ directory for one campaign.

Bundle principles
-----------------
- Include only small files genuinely useful for local review / Codex analysis.
- Do NOT copy large TIFF volumes or bulky intermediate outputs.
- Do copy summary files, manifest, filtered results rows, preview PNGs, and
  per-trial JSON metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_results_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def detect_campaign_name_from_trial_root(trial_root: str) -> str | None:
    parts = Path(trial_root).parts
    for i, part in enumerate(parts):
        if part == "runs" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt.startswith("campaign_"):
                return nxt
    return None


def filter_rows_for_campaign(rows: list[dict[str, str]], campaign_name: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        trial_root = row.get("trial_root", "") or ""
        detected = detect_campaign_name_from_trial_root(trial_root)
        if detected == campaign_name:
            out.append(row)
    return out


def write_results_tsv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_prompt_for_local_codex(summary: dict[str, Any]) -> str:
    best = summary.get("best_trial")
    best_trial_id = best.get("trial_id") if isinstance(best, dict) else None

    return f"""You are reading one lightweight campaign bundle from a Slurm-based autoresearch project.

Your tasks:
1. Read campaign_summary.json first.
2. Read campaign_summary.md for the human-readable overview.
3. Read results_campaign.tsv for flat comparison rows.
4. Inspect best_trial/ and trials/*/ JSON files.
5. Focus on:
   - which parameter changes improved score,
   - which changes reduced hollow artifacts,
   - which changes increased cell count,
   - which changes look risky or unstable,
   - what should be fixed vs. searched next.
6. Produce a local upload package for the next round:
   - next_round_plan.json
   - next_round_plan.md
   - trial_configs/*.json
   - README_UPLOAD.md

Important constraints:
- Treat score as the primary ranking metric.
- Higher score is better.
- Do not assume TIFF volumes are available in this bundle.
- Use only the files included here.

Current best trial from summary:
- best_trial_id: {best_trial_id}
"""


def build_readme_for_codex(
    campaign_name: str,
    bundle_root: Path,
    summary: dict[str, Any],
    copied: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# Codex bundle for {campaign_name}")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Bundle root: `{bundle_root}`")
    lines.append("")
    lines.append("## Included files")
    lines.append("")
    lines.append("- campaign_summary.json")
    lines.append("- campaign_summary.md")
    lines.append("- campaign_manifest.json")
    lines.append("- results_campaign.tsv")
    lines.append("- prompts/prompt_for_local_codex.txt")
    lines.append("- best_trial/*")
    lines.append("- trials/*/{trial_config.json, used_config.json, trial_record.json, score.json, trial_metadata.json}")
    lines.append("- previews/* (if available)")
    lines.append("")
    lines.append("## Excluded on purpose")
    lines.append("")
    lines.append("- raw TIFF volumes")
    lines.append("- step12_output/")
    lines.append("- step3_output/")
    lines.append("- large run logs")
    lines.append("")
    lines.append("## Best trial")
    lines.append("")
    best = summary.get("best_trial")
    if best:
        lines.append(f"- trial_id: **{best.get('trial_id')}**")
        lines.append(f"- score: **{best.get('score')}**")
        lines.append(f"- changed_keys: `{best.get('changed_keys', [])}`")
    else:
        lines.append("_No scored best trial found._")
    lines.append("")
    lines.append("## Copy report")
    lines.append("")
    lines.append(f"- campaign rows copied: **{copied.get('campaign_results_rows', 0)}**")
    lines.append(f"- trial JSON groups copied: **{copied.get('trial_json_groups', 0)}**")
    lines.append(f"- preview files copied: **{copied.get('preview_files', 0)}**")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a lightweight codex_bundle/ directory.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    campaign_name = args.campaign_name
    campaign_root = repo_root / "runs" / campaign_name
    bundle_root = campaign_root / "codex_bundle"
    results_path = Path(args.results).expanduser().resolve() if args.results else (repo_root / "results.tsv")

    if not campaign_root.exists():
        raise FileNotFoundError(f"Campaign root not found: {campaign_root}")

    summary_json_path = campaign_root / "campaign_summary.json"
    summary_md_path = campaign_root / "campaign_summary.md"
    manifest_path = campaign_root / "campaign_manifest.json"
    previews_root = campaign_root / "previews"

    if not summary_json_path.exists():
        raise FileNotFoundError(f"Missing campaign_summary.json: {summary_json_path}")
    if not summary_md_path.exists():
        raise FileNotFoundError(f"Missing campaign_summary.md: {summary_md_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing campaign_manifest.json: {manifest_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.tsv: {results_path}")

    ensure_clean_dir(bundle_root)

    summary = load_json(summary_json_path)
    manifest = load_json(manifest_path)

    copied = {
        "campaign_results_rows": 0,
        "trial_json_groups": 0,
        "preview_files": 0,
    }

    # Core summary files
    copy_if_exists(summary_json_path, bundle_root / "campaign_summary.json")
    copy_if_exists(summary_md_path, bundle_root / "campaign_summary.md")
    copy_if_exists(manifest_path, bundle_root / "campaign_manifest.json")

    # Filter results rows to this campaign only
    header, all_rows = read_results_tsv(results_path)
    campaign_rows = filter_rows_for_campaign(all_rows, campaign_name)
    copied["campaign_results_rows"] = len(campaign_rows)
    write_results_tsv(bundle_root / "results_campaign.tsv", header, campaign_rows)

    # Copy previews if any
    if previews_root.exists():
        for path in sorted(previews_root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(previews_root)
                copy_if_exists(path, bundle_root / "previews" / rel)
                copied["preview_files"] += 1

    # Copy per-trial JSONs
    trials_root = campaign_root / "trials"
    trial_dirs = sorted([p for p in trials_root.iterdir() if p.is_dir()]) if trials_root.exists() else []
    trial_json_names = [
        "trial_config.json",
        "used_config.json",
        "trial_record.json",
        "score.json",
        "trial_metadata.json",
    ]

    for trial_dir in trial_dirs:
        out_dir = bundle_root / "trials" / trial_dir.name
        any_copied = False
        for name in trial_json_names:
            src = trial_dir / name
            if copy_if_exists(src, out_dir / name):
                any_copied = True
        if any_copied:
            copied["trial_json_groups"] += 1

    # Copy best trial into best_trial/
    best = summary.get("best_trial")
    if isinstance(best, dict) and best.get("trial_id"):
        best_trial_id = best["trial_id"]
        best_src_dir = trials_root / best_trial_id
        best_out_dir = bundle_root / "best_trial"
        for name in trial_json_names:
            copy_if_exists(best_src_dir / name, best_out_dir / name)

    # Prompt file
    prompt_text = build_prompt_for_local_codex(summary)
    prompt_path = bundle_root / "prompts" / "prompt_for_local_codex.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")

    # Bundle manifest
    bundle_manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "campaign_name": campaign_name,
        "campaign_root": str(campaign_root),
        "bundle_root": str(bundle_root),
        "summary_json": str(bundle_root / "campaign_summary.json"),
        "summary_md": str(bundle_root / "campaign_summary.md"),
        "results_campaign_tsv": str(bundle_root / "results_campaign.tsv"),
        "prompt_path": str(prompt_path),
        "copy_report": copied,
        "notes": [
            "This bundle intentionally excludes large TIFF outputs and bulky intermediate directories.",
            "results_campaign.tsv is filtered from the global results.tsv to include only this campaign.",
        ],
    }
    save_json(bundle_root / "bundle_manifest.json", bundle_manifest)

    # Human-readable README
    readme_text = build_readme_for_codex(campaign_name, bundle_root, summary, copied)
    (bundle_root / "README_FOR_CODEX.md").write_text(readme_text, encoding="utf-8")

    print("=" * 90)
    print("Codex bundle exported")
    print(f"campaign_root          : {campaign_root}")
    print(f"bundle_root            : {bundle_root}")
    print(f"campaign_rows_copied   : {copied['campaign_results_rows']}")
    print(f"trial_json_groups      : {copied['trial_json_groups']}")
    print(f"preview_files_copied   : {copied['preview_files']}")
    print("=" * 90)


if __name__ == "__main__":
    main()