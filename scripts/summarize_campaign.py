#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_campaign.py

Generate campaign-level summary files from one prepared/executed campaign.

Inputs
------
- runs/<campaign_name>/campaign_manifest.json
- results.tsv
- runs/<campaign_name>/trials/*/{trial_config.json, used_config.json, trial_record.json, score.json}

Outputs
-------
- runs/<campaign_name>/campaign_summary.json
- runs/<campaign_name>/campaign_summary.md

Design notes
------------
- Ranking is based on score descending (higher is better), consistent with the
  current project logic.
- This script is intentionally robust to partially missing trial outputs.
- It does not require every trial to have succeeded.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def compact_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def parse_changed_keys(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
    return []


def read_results_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def detect_campaign_name_from_trial_root(trial_root: str) -> str | None:
    parts = Path(trial_root).parts
    # expect ... / runs / campaign_xxx / trials / trial_xxxx
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


def find_ledger_first_valid_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        decision = row.get("decision", "")
        score = safe_float(row.get("score"))
        if decision in {"keep", "baseline_keep"} and score is not None:
            return row
    return None


def score_sort_key(item: dict[str, Any]) -> tuple[float, str]:
    score = safe_float(item.get("score"))
    return (score if score is not None else float("-inf"), str(item.get("trial_id", "")))


def summarize_decision_reason(score_json: dict[str, Any]) -> str:
    reason = score_json.get("decision_reason")
    return str(reason) if reason is not None else ""


def build_trial_entry(
    trial_dir: Path,
    campaign_trial_decl: dict[str, Any] | None,
    campaign_name: str,
) -> dict[str, Any]:
    trial_id = trial_dir.name
    trial_config_path = trial_dir / "trial_config.json"
    used_config_path = trial_dir / "used_config.json"
    trial_record_path = trial_dir / "trial_record.json"
    score_path = trial_dir / "score.json"
    trial_metadata_path = trial_dir / "trial_metadata.json"

    entry: dict[str, Any] = {
        "trial_id": trial_id,
        "campaign_name": campaign_name,
        "trial_root": str(trial_dir),
        "trial_config_path": str(trial_config_path),
        "used_config_path": str(used_config_path),
        "trial_record_path": str(trial_record_path),
        "score_json_path": str(score_path),
        "trial_metadata_path": str(trial_metadata_path),
        "declared_in_manifest": campaign_trial_decl is not None,
        "files_present": {
            "trial_config": trial_config_path.exists(),
            "used_config": used_config_path.exists(),
            "trial_record": trial_record_path.exists(),
            "score_json": score_path.exists(),
            "trial_metadata": trial_metadata_path.exists(),
        },
        "status": "unknown",
        "trial_status": None,
        "decision": None,
        "decision_reason": None,
        "score": None,
        "run_tag": None,
        "changed_keys": [],
        "hypothesis": None,
        "expected_risk": {},
        "metrics": {},
        "notes": [],
    }

    trial_cfg = load_json(trial_config_path) if trial_config_path.exists() else None
    used_cfg = load_json(used_config_path) if used_config_path.exists() else None
    trial_record = load_json(trial_record_path) if trial_record_path.exists() else None
    score_json = load_json(score_path) if score_path.exists() else None
    trial_meta = load_json(trial_metadata_path) if trial_metadata_path.exists() else None

    if trial_cfg:
        entry["run_tag"] = trial_cfg.get("run_tag")
    if used_cfg and entry["run_tag"] is None:
        entry["run_tag"] = used_cfg.get("run_tag")

    if used_cfg:
        search_edit = used_cfg.get("search_edit_summary", {}) or {}
        entry["changed_keys"] = list(search_edit.get("changed_keys", []) or [])
        entry["hypothesis"] = search_edit.get("hypothesis")
        entry["expected_risk"] = used_cfg.get("expected_risk", {}) or {}
    elif trial_cfg:
        search_edit = trial_cfg.get("search_edit_summary", {}) or {}
        entry["changed_keys"] = list(search_edit.get("changed_keys", []) or [])
        entry["hypothesis"] = search_edit.get("hypothesis")
        entry["expected_risk"] = trial_cfg.get("expected_risk", {}) or {}

    if trial_record:
        entry["trial_status"] = trial_record.get("status")
        entry["metrics"]["trial_record_aggregate"] = trial_record.get("aggregate_metrics", {}) or {}

    if score_json:
        entry["status"] = "scored"
        entry["decision"] = score_json.get("decision")
        entry["decision_reason"] = summarize_decision_reason(score_json)
        entry["score"] = ((score_json.get("score", {}) or {}).get("value"))
        entry["score_version"] = ((score_json.get("score", {}) or {}).get("score_version"))
        entry["score_formula"] = ((score_json.get("score", {}) or {}).get("formula"))
        entry["trial_status"] = score_json.get("status", entry["trial_status"])
        entry["metrics"]["score_aggregate"] = score_json.get("aggregate_metrics", {}) or {}
    else:
        if trial_record:
            status = trial_record.get("status")
            if status == "success":
                entry["status"] = "executed_no_score"
            elif status in {"failed", "partial"}:
                entry["status"] = status
            else:
                entry["status"] = "executed_unknown"
        elif trial_meta:
            if trial_meta.get("status") == "prepared":
                entry["status"] = "prepared_only"
            else:
                entry["status"] = str(trial_meta.get("status", "unknown"))
        else:
            entry["status"] = "missing_outputs"

    if not score_path.exists():
        entry["notes"].append("score.json missing")
    if not trial_record_path.exists():
        entry["notes"].append("trial_record.json missing")

    return entry


def build_parameter_change_frequency(trial_entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    all_trials = Counter()
    kept_trials = Counter()
    discarded_trials = Counter()
    top_trials = Counter()

    ranked = sorted(
        [t for t in trial_entries if safe_float(t.get("score")) is not None],
        key=score_sort_key,
        reverse=True,
    )
    top_n = ranked[: min(3, len(ranked))]

    for entry in trial_entries:
        keys = entry.get("changed_keys", []) or []
        all_trials.update(keys)
        if entry.get("decision") in {"keep", "baseline_keep"}:
            kept_trials.update(keys)
        elif entry.get("decision") == "discard":
            discarded_trials.update(keys)

    for entry in top_n:
        top_trials.update(entry.get("changed_keys", []) or [])

    return {
        "all_trials": dict(all_trials),
        "kept_trials": dict(kept_trials),
        "discarded_trials": dict(discarded_trials),
        "top_trials": dict(top_trials),
    }


def build_comparison_to_best(best_entry: dict[str, Any] | None, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if best_entry is None:
        return []

    best_score = safe_float(best_entry.get("score"))
    out: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda x: str(x.get("trial_id", ""))):
        score = safe_float(entry.get("score"))
        delta = None
        if score is not None and best_score is not None:
            delta = score - best_score
        out.append(
            {
                "trial_id": entry.get("trial_id"),
                "score": score,
                "delta_vs_best": delta,
                "decision": entry.get("decision"),
                "changed_keys": entry.get("changed_keys", []),
            }
        )
    return out


def render_summary_md(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {summary['campaign_name']} summary")
    lines.append("")
    lines.append(f"- Generated at: `{summary['generated_at']}`")
    lines.append(f"- Campaign root: `{summary['campaign_root']}`")
    lines.append(f"- Trials declared: **{summary['num_trials_declared']}**")
    lines.append(f"- Trials found: **{summary['num_trials_found']}**")
    lines.append(f"- Scored: **{summary['num_scored']}**")
    lines.append(f"- Success: **{summary['num_success']}**")
    lines.append(f"- Partial: **{summary['num_partial']}**")
    lines.append(f"- Failed: **{summary['num_failed']}**")
    lines.append("")

    baseline_ref = summary.get("baseline_reference")
    if baseline_ref:
        lines.append("## Baseline / reference")
        lines.append("")
        lines.append(f"- Type: `{baseline_ref.get('type')}`")
        lines.append(f"- Used config: `{baseline_ref.get('used_config_path')}`")
        lines.append(f"- Score: `{baseline_ref.get('score')}`")
        lines.append("")

    best = summary.get("best_trial")
    if best:
        lines.append("## Best trial")
        lines.append("")
        lines.append(f"- Trial id: **{best.get('trial_id')}**")
        lines.append(f"- Score: **{best.get('score')}**")
        lines.append(f"- Decision: `{best.get('decision')}`")
        lines.append(f"- Used config: `{best.get('used_config_path')}`")
        changed = best.get("changed_keys", []) or []
        lines.append(f"- Changed keys: `{changed}`")
        if best.get("decision_reason"):
            lines.append(f"- Decision reason: {best.get('decision_reason')}")
        lines.append("")

    lines.append("## Ranking")
    lines.append("")
    ranking = summary.get("ranking", [])
    if not ranking:
        lines.append("_No scored trials found._")
        lines.append("")
    else:
        for idx, item in enumerate(ranking, start=1):
            lines.append(
                f"{idx}. `{item.get('trial_id')}` | score={item.get('score')} | "
                f"decision={item.get('decision')} | changed_keys={item.get('changed_keys', [])}"
            )
        lines.append("")

    lines.append("## Parameter change frequency")
    lines.append("")
    param_freq = summary.get("parameter_change_frequency", {}) or {}
    for bucket in ["all_trials", "kept_trials", "discarded_trials", "top_trials"]:
        lines.append(f"### {bucket}")
        stats = param_freq.get(bucket, {}) or {}
        if not stats:
            lines.append("")
            lines.append("_No data._")
            lines.append("")
            continue
        for key, count in sorted(stats.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{key}`: {count}")
        lines.append("")

    lines.append("## Trial details")
    lines.append("")
    for item in summary.get("trial_entries", []):
        lines.append(f"### {item.get('trial_id')}")
        lines.append("")
        lines.append(f"- status: `{item.get('status')}`")
        lines.append(f"- trial_status: `{item.get('trial_status')}`")
        lines.append(f"- decision: `{item.get('decision')}`")
        lines.append(f"- score: `{item.get('score')}`")
        lines.append(f"- changed_keys: `{item.get('changed_keys', [])}`")
        if item.get("decision_reason"):
            lines.append(f"- decision_reason: {item.get('decision_reason')}")
        notes = item.get("notes", []) or []
        if notes:
            lines.append(f"- notes: `{notes}`")
        lines.append("")

    rec = summary.get("recommended_seed_trial")
    if rec:
        lines.append("## Recommended seed")
        lines.append("")
        lines.append(f"- Trial id: **{rec.get('trial_id')}**")
        lines.append(f"- Reason: {rec.get('reason')}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate campaign-level summary files.")
    parser.add_argument(
        "--campaign-name",
        required=True,
        help="Campaign directory name, e.g. campaign_001",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root. Defaults to the current project root.",
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
    manifest_path = campaign_root / "campaign_manifest.json"
    results_path = Path(args.results).expanduser().resolve() if args.results else (repo_root / "results.tsv")

    if not campaign_root.exists():
        raise FileNotFoundError(f"Campaign root not found: {campaign_root}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing campaign_manifest.json: {manifest_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.tsv: {results_path}")

    manifest = load_json(manifest_path)
    results_rows = read_results_tsv(results_path)
    campaign_rows = filter_rows_for_campaign(results_rows, campaign_name)
    ledger_first_valid = find_ledger_first_valid_row(results_rows)

    trials_root = campaign_root / "trials"
    trial_dirs = sorted([p for p in trials_root.iterdir() if p.is_dir()]) if trials_root.exists() else []
    manifest_trials = {t.get("trial_id"): t for t in (manifest.get("trials", []) or []) if t.get("trial_id")}

    trial_entries: list[dict[str, Any]] = []
    for trial_dir in trial_dirs:
        decl = manifest_trials.get(trial_dir.name)
        entry = build_trial_entry(trial_dir, decl, campaign_name)
        trial_entries.append(entry)

    scored_entries = [t for t in trial_entries if safe_float(t.get("score")) is not None]
    ranked = sorted(scored_entries, key=score_sort_key, reverse=True)
    best_entry = ranked[0] if ranked else None

    num_success = sum(1 for t in trial_entries if t.get("trial_status") in {"success", "scored"})
    num_partial = sum(1 for t in trial_entries if t.get("trial_status") == "partial")
    num_failed = sum(1 for t in trial_entries if t.get("trial_status") in {"failed", "crash"})
    num_scored = sum(1 for t in trial_entries if t.get("status") == "scored")

    ranking = [
        {
            "trial_id": t.get("trial_id"),
            "score": safe_float(t.get("score")),
            "decision": t.get("decision"),
            "decision_reason": t.get("decision_reason"),
            "changed_keys": t.get("changed_keys", []),
            "trial_root": t.get("trial_root"),
        }
        for t in ranked
    ]

    baseline_reference = None
    if ledger_first_valid is not None:
        baseline_reference = {
            "type": "ledger_first_valid_row",
            "used_config_path": ledger_first_valid.get("used_config_path"),
            "trial_root": ledger_first_valid.get("trial_root"),
            "score": safe_float(ledger_first_valid.get("score")),
            "decision": ledger_first_valid.get("decision"),
        }

    best_trial = None
    recommended_seed_trial = None
    if best_entry is not None:
        best_trial = {
            "trial_id": best_entry.get("trial_id"),
            "score": safe_float(best_entry.get("score")),
            "decision": best_entry.get("decision"),
            "decision_reason": best_entry.get("decision_reason"),
            "used_config_path": best_entry.get("used_config_path"),
            "trial_config_path": best_entry.get("trial_config_path"),
            "trial_record_path": best_entry.get("trial_record_path"),
            "score_json_path": best_entry.get("score_json_path"),
            "changed_keys": best_entry.get("changed_keys", []),
        }
        recommended_seed_trial = {
            "trial_id": best_entry.get("trial_id"),
            "reason": "Highest score among successful campaign trials.",
            "used_config_path": best_entry.get("used_config_path"),
        }

    param_freq = build_parameter_change_frequency(trial_entries)
    comparison_to_best = build_comparison_to_best(best_entry, trial_entries)

    summary = {
        "schema_version": "1.0",
        "campaign_name": campaign_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "campaign_root": str(campaign_root),
        "manifest_path": str(manifest_path),
        "results_path": str(results_path),
        "num_trials_declared": len(manifest_trials),
        "num_trials_found": len(trial_entries),
        "num_scored": num_scored,
        "num_success": num_success,
        "num_partial": num_partial,
        "num_failed": num_failed,
        "baseline_reference": baseline_reference,
        "best_trial": best_trial,
        "ranking": ranking,
        "kept_trials": [t["trial_id"] for t in ranked if t.get("decision") in {"keep", "baseline_keep"}],
        "discarded_trials": [t["trial_id"] for t in ranked if t.get("decision") == "discard"],
        "failed_trials": [t["trial_id"] for t in trial_entries if t.get("trial_status") in {"failed", "crash"}],
        "parameter_change_frequency": param_freq,
        "comparison_to_best": comparison_to_best,
        "recommended_seed_trial": recommended_seed_trial,
        "campaign_results_rows": campaign_rows,
        "trial_entries": trial_entries,
        "notes": [
            "Ranking is based on score descending (higher is better).",
            "Campaign rows are filtered from the global results.tsv using trial_root path detection.",
            "Missing outputs are tolerated and recorded in trial-level notes.",
        ],
    }

    summary_json_path = campaign_root / "campaign_summary.json"
    summary_md_path = campaign_root / "campaign_summary.md"
    save_json(summary_json_path, summary)
    summary_md_path.write_text(render_summary_md(summary), encoding="utf-8")

    print("=" * 90)
    print("Campaign summary generated")
    print(f"campaign_root        : {campaign_root}")
    print(f"campaign_name        : {campaign_name}")
    print(f"summary_json         : {summary_json_path}")
    print(f"summary_md           : {summary_md_path}")
    print(f"num_trials_found     : {len(trial_entries)}")
    print(f"num_scored           : {num_scored}")
    print(f"best_trial           : {best_trial['trial_id'] if best_trial else '<none>'}")
    print("=" * 90)


if __name__ == "__main__":
    main()