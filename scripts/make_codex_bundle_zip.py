#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_codex_bundle_zip.py

Archive runs/<campaign_name>/codex_bundle/ into runs/<campaign_name>/codex_bundle.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zip one campaign codex_bundle directory.")
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
    bundle_root = campaign_root / "codex_bundle"

    if not campaign_root.exists():
        raise FileNotFoundError(f"Campaign root not found: {campaign_root}")
    if not bundle_root.exists():
        raise FileNotFoundError(f"Missing codex_bundle directory: {bundle_root}")

    zip_prefix = str(campaign_root / "codex_bundle")
    zip_path = Path(shutil.make_archive(zip_prefix, "zip", root_dir=bundle_root))

    zip_info = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "campaign_name": campaign_name,
        "bundle_root": str(bundle_root),
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size if zip_path.exists() else None,
    }
    save_json(campaign_root / "codex_bundle_zip_info.json", zip_info)

    print("=" * 90)
    print("Codex bundle ZIP created")
    print(f"campaign_root  : {campaign_root}")
    print(f"bundle_root    : {bundle_root}")
    print(f"zip_path       : {zip_path}")
    print(f"zip_size_bytes : {zip_info['zip_size_bytes']}")
    print("=" * 90)


if __name__ == "__main__":
    main()