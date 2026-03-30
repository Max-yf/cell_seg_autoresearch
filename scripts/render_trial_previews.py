#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_trial_previews.py

Generate lightweight preview PNGs for campaign trials.

Important design choice
-----------------------
This script must be safe to call even when:
- no previewable source files exist,
- a trial has not completed,
- some TIFF files are missing/corrupted.

It should never crash the whole campaign finalization process just because
preview rendering is incomplete.

V1 behavior
-----------
- For each trial, try to locate:
  - crop_input.tif
  - step3_output/mask.tif
- If both exist and look valid, render a simple middle-z slice preview:
  raw / mask / overlay
- If not, record a missing preview reason in previews/index.json
- Always produce:
  - runs/<campaign_name>/previews/index.json
  - runs/<campaign_name>/previews/README_PREVIEWS.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    if p99 <= p1:
        if arr.max() > arr.min():
            out = (arr - arr.min()) / (arr.max() - arr.min())
            return out.astype(np.float32)
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - p1) / (p99 - p1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def choose_preview_z(mask: np.ndarray) -> int:
    if mask.ndim != 3:
        return 0
    counts = [(int(np.count_nonzero(mask[z] > 0)), z) for z in range(mask.shape[0])]
    counts.sort(reverse=True)
    if counts and counts[0][0] > 0:
        return counts[0][1]
    return mask.shape[0] // 2


def render_preview_png(raw_path: Path, mask_path: Path, out_path: Path) -> dict[str, Any]:
    raw = tifffile.imread(raw_path)
    mask = tifffile.imread(mask_path)

    if raw.ndim != 3:
        raise ValueError(f"Expected 3D raw TIFF, got shape={raw.shape} from {raw_path}")
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask TIFF, got shape={mask.shape} from {mask_path}")
    if raw.shape != mask.shape:
        raise ValueError(f"Raw/mask shape mismatch: raw={raw.shape}, mask={mask.shape}")

    z = choose_preview_z(mask)
    raw2d = normalize_image(raw[z])
    mask2d = (mask[z] > 0).astype(np.float32)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(raw2d, cmap="gray")
    ax1.set_title(f"raw z={z}")
    ax1.axis("off")

    ax2.imshow(mask2d, cmap="gray")
    ax2.set_title("mask > 0")
    ax2.axis("off")

    ax3.imshow(raw2d, cmap="gray")
    overlay = np.zeros((*mask2d.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask2d * 0.35
    ax3.imshow(overlay)
    ax3.set_title("overlay")
    ax3.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "z_index": int(z),
        "raw_shape": list(raw.shape),
        "mask_shape": list(mask.shape),
    }


def render_readme(index: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Preview generation report")
    lines.append("")
    lines.append(f"- Generated at: `{index['generated_at']}`")
    lines.append(f"- Campaign name: `{index['campaign_name']}`")
    lines.append(f"- Trials total: **{index['num_trials']}**")
    lines.append(f"- Preview success: **{index['num_preview_success']}**")
    lines.append(f"- Preview missing/failed: **{index['num_preview_missing_or_failed']}**")
    lines.append("")
    lines.append("## Per-trial status")
    lines.append("")
    for item in index.get("trials", []):
        lines.append(f"### {item.get('trial_id')}")
        lines.append("")
        lines.append(f"- preview_status: `{item.get('preview_status')}`")
        if item.get("preview_png"):
            lines.append(f"- preview_png: `{item.get('preview_png')}`")
        if item.get("reason"):
            lines.append(f"- reason: {item.get('reason')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render campaign trial preview PNGs.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    campaign_name = args.campaign_name
    campaign_root = repo_root / "runs" / campaign_name
    trials_root = campaign_root / "trials"
    previews_root = campaign_root / "previews"

    if not campaign_root.exists():
        raise FileNotFoundError(f"Campaign root not found: {campaign_root}")

    previews_root.mkdir(parents=True, exist_ok=True)
    trial_dirs = sorted([p for p in trials_root.iterdir() if p.is_dir()]) if trials_root.exists() else []

    trial_items: list[dict[str, Any]] = []
    preview_success = 0
    preview_missing_or_failed = 0

    for trial_dir in trial_dirs:
        trial_id = trial_dir.name
        item: dict[str, Any] = {
            "trial_id": trial_id,
            "trial_root": str(trial_dir),
            "preview_status": "unknown",
            "preview_png": None,
            "reason": None,
            "details": {},
        }

        try:
            crop_dirs = sorted((trial_dir / "crops").glob("*"))
            if not crop_dirs:
                item["preview_status"] = "missing_source_assets"
                item["reason"] = "No crop directories found."
                preview_missing_or_failed += 1
                trial_items.append(item)
                continue

            # V1: choose the first crop only
            crop_dir = crop_dirs[0]
            raw_path = crop_dir / "crop_input.tif"
            mask_path = crop_dir / "step3_output" / "mask.tif"

            if not raw_path.exists():
                item["preview_status"] = "missing_source_assets"
                item["reason"] = f"Missing raw TIFF: {raw_path}"
                preview_missing_or_failed += 1
                trial_items.append(item)
                continue

            if not mask_path.exists():
                item["preview_status"] = "missing_source_assets"
                item["reason"] = f"Missing mask TIFF: {mask_path}"
                preview_missing_or_failed += 1
                trial_items.append(item)
                continue

            out_png = previews_root / f"{trial_id}__{crop_dir.name}.png"
            details = render_preview_png(raw_path, mask_path, out_png)

            item["preview_status"] = "ok"
            item["preview_png"] = str(out_png)
            item["details"] = details
            preview_success += 1

        except Exception as exc:  # noqa: BLE001
            item["preview_status"] = "render_failed"
            item["reason"] = f"{type(exc).__name__}: {exc}"
            preview_missing_or_failed += 1

        trial_items.append(item)

    index = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "campaign_name": campaign_name,
        "campaign_root": str(campaign_root),
        "previews_root": str(previews_root),
        "num_trials": len(trial_dirs),
        "num_preview_success": preview_success,
        "num_preview_missing_or_failed": preview_missing_or_failed,
        "trials": trial_items,
        "notes": [
            "V1 renders at most one preview PNG per trial.",
            "V1 uses the first crop directory only.",
            "Preview rendering failures do not imply trial failure.",
        ],
    }

    save_json(previews_root / "index.json", index)
    (previews_root / "README_PREVIEWS.md").write_text(render_readme(index), encoding="utf-8")

    print("=" * 90)
    print("Trial previews rendered")
    print(f"campaign_root          : {campaign_root}")
    print(f"previews_root          : {previews_root}")
    print(f"num_trials             : {len(trial_dirs)}")
    print(f"preview_success        : {preview_success}")
    print(f"preview_missing_failed : {preview_missing_or_failed}")
    print("=" * 90)


if __name__ == "__main__":
    main()