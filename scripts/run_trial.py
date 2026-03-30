#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_trial.py

Purpose
-------
Run one autoresearch trial end-to-end on crop-level evaluation data.

Workflow
--------
1. Read used_config.json and crop_manifest.json
2. Extract fixed crops from the source 3D TIFF
3. Run Step1+Step2 via run_step12_pipeline.py
4. Run Step3 via run_infer_3d.py
5. Collect per-crop outputs and raw metrics
6. Write trial_record.json

Notes
-----
- This first-batch script focuses on robust orchestration and traceability.
- Scoring and TSV appending are intentionally separated into later scripts.
- It assumes your existing pipeline scripts are already working and only needs
  the correct script paths.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
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


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# -----------------------------------------------------------------------------
# Command helpers
# -----------------------------------------------------------------------------

def stringify(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    elapsed_sec: float
    stdout_path: str
    stderr_path: str


def run_command(cmd: list[str], cwd: Path | None, stdout_path: Path, stderr_path: Path) -> CommandResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open("w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, stdout=fout, stderr=ferr)
    elapsed = time.time() - start
    return CommandResult(
        cmd=cmd,
        returncode=int(proc.returncode),
        elapsed_sec=float(elapsed),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


# -----------------------------------------------------------------------------
# Safe file helpers
# -----------------------------------------------------------------------------

def safe_copy_file(src: Path, dst: Path) -> bool:
    """
    Copy src -> dst only when they are not the same file.

    Returns
    -------
    bool
        True if a copy was actually performed, False if skipped.
    """
    src_resolved = src.expanduser().resolve()
    dst_resolved = dst.expanduser().resolve()

    if src_resolved == dst_resolved:
        print(f"[skip-copy] source and destination are the same file: {src_resolved}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_resolved, dst_resolved)
    print(f"[copy] {src_resolved} -> {dst_resolved}")
    return True


# -----------------------------------------------------------------------------
# TIFF handling and crop extraction
# -----------------------------------------------------------------------------

def load_source_stack(path: Path, channel_index: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        arr = np.asarray(series.asarray())
        axes = getattr(series, "axes", "") or ""

    original = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "axes": axes,
    }

    if arr.ndim == 2:
        stack = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        stack = arr
    elif arr.ndim == 4:
        if "C" not in axes:
            raise ValueError(f"4D input is ambiguous without channel axis: shape={arr.shape}, axes={axes!r}")
        if channel_index is None:
            raise ValueError("Input has channels; config.input.channel_index is required.")
        c_axis = axes.index("C")
        if not (0 <= channel_index < arr.shape[c_axis]):
            raise ValueError(f"channel_index={channel_index} out of range for shape={arr.shape}, axes={axes!r}")
        stack = np.take(arr, indices=channel_index, axis=c_axis)
        remaining_axes = "".join(ax for ax in axes if ax != "C")
        if remaining_axes != "ZYX":
            raise ValueError(f"Expected ZYX after channel extraction, got {remaining_axes!r}")
    else:
        raise ValueError(f"Unsupported input shape={arr.shape}, axes={axes!r}")

    if stack.ndim != 3:
        raise ValueError(f"Expected extracted 3D ZYX stack, got shape={stack.shape}")
    return np.asarray(stack), original


def extract_crop(stack: np.ndarray, crop: dict[str, Any]) -> np.ndarray:
    z0, z1 = crop["z_range"]
    y0, y1 = crop["y_range"]
    x0, x1 = crop["x_range"]
    cropped = stack[z0:z1, y0:y1, x0:x1]
    if cropped.ndim != 3:
        raise ValueError(f"Crop extraction did not produce 3D stack: shape={cropped.shape}")
    return np.asarray(cropped)


# -----------------------------------------------------------------------------
# Config translation
# -----------------------------------------------------------------------------

STEP1_KEYS_TO_CONFIG_JSON = [
    "pixel_size_nm",
    "wavelength_nm",
    "effective_na",
    "sparse_iter",
    "fidelity",
    "z_continuity",
    "sparsity",
    "deconv_iter",
    "background_mode",
    "deblurring_method",
    "oversampling_method",
    "psf_integration_samples",
    "mode",
    "window_size",
    "halo",
    "backend",
    "gpu_device_index",
    "show_progress",
]


def build_step1_config_json(step1_cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in STEP1_KEYS_TO_CONFIG_JSON:
        if key in step1_cfg:
            out[key] = step1_cfg[key]
    return out


def step12_command(
    python_exe: str,
    step12_script: Path,
    crop_input_path: Path,
    step12_output_dir: Path,
    step1_cfg_json_path: Path,
    input_cfg: dict[str, Any],
    step1_cfg: dict[str, Any],
    step2_cfg: dict[str, Any],
) -> list[str]:
    cmd = [
        python_exe,
        str(step12_script),
        "--input",
        str(crop_input_path),
        "--output_dir",
        str(step12_output_dir),
        "--config_json",
        str(step1_cfg_json_path),
        "--ln_radius",
        stringify(step2_cfg["radius"]),
        "--ln_bias",
        stringify(step2_cfg["bias"]),
        "--ln_output_dtype",
        stringify(step2_cfg.get("output_dtype", "uint16")),
    ]
    channel_index = input_cfg.get("channel_index")
    if channel_index is not None:
        cmd.extend(["--channel_index", stringify(channel_index)])
    if bool(input_cfg.get("save_extracted_input", False)):
        cmd.append("--save_extracted_input")

    # Optional explicit overrides supported by run_step12_pipeline.py
    for key, flag in [
        ("mode", "--mode"),
        ("window_size", "--window_size"),
        ("halo", "--halo"),
        ("sparsity", "--sparsity"),
        ("backend", "--backend"),
        ("gpu_device_index", "--gpu_device_index"),
    ]:
        if key in step1_cfg and step1_cfg[key] is not None:
            cmd.extend([flag, stringify(step1_cfg[key])])
    if step1_cfg.get("show_progress", True) is False:
        cmd.append("--hide_progress")
    return cmd


def step3_command(
    python_exe: str,
    step3_script: Path,
    step3_input_path: Path,
    step3_output_dir: Path,
    model_cfg: dict[str, Any],
    step3_cfg: dict[str, Any],
) -> list[str]:
    cmd = [
        python_exe,
        str(step3_script),
        "--input",
        str(step3_input_path),
        "--output",
        str(step3_output_dir),
        "--model",
        stringify(model_cfg["model_path"]),
        "--config",
        stringify(model_cfg["config_path"]),
        "--cellprob_threshold",
        stringify(step3_cfg["cellprob_threshold"]),
        "--min_size",
        stringify(step3_cfg["min_size"]),
        "--anisotropy",
        stringify(step3_cfg["anisotropy"]),
        "--diameter",
        "None" if step3_cfg.get("diameter") is None else stringify(step3_cfg["diameter"]),
        "--rescale",
        stringify(step3_cfg["rescale"]),
        "--z_axis",
        stringify(step3_cfg["z_axis"]),
        "--batch_size_3d",
        stringify(step3_cfg["batch_size_3d"]),
        "--bsize",
        stringify(step3_cfg["bsize"]),
        "--stitch_threshold",
        stringify(step3_cfg["stitch_threshold"]),
        "--flow_threshold",
        stringify(step3_cfg["flow_threshold"]),
        "--tile_overlap",
        stringify(step3_cfg.get("tile_overlap", 0.1)),
    ]
    if bool(step3_cfg.get("use_gpu", False)):
        cmd.append("--use_gpu")
    if bool(step3_cfg.get("augment", False)):
        cmd.append("--augment")
    if bool(step3_cfg.get("save_flows", False)):
        cmd.append("--save_flows")
    # do_3D is store_true with default True in the current delivery script,
    # so there is no explicit false path. We only pass it when True for clarity.
    if bool(step3_cfg.get("do_3D", True)):
        cmd.append("--do_3D")
    return cmd


# -----------------------------------------------------------------------------
# Trial selection helpers
# -----------------------------------------------------------------------------

def choose_crop_ids(used_cfg: dict[str, Any], manifest: dict[str, Any], explicit_crop_ids: list[str] | None) -> list[str]:
    if explicit_crop_ids:
        return explicit_crop_ids
    req = used_cfg.get("evaluation_request", {})
    if isinstance(req, dict) and req.get("crop_ids"):
        return list(req["crop_ids"])
    return [c["crop_id"] for c in manifest.get("crops", [])]


# -----------------------------------------------------------------------------
# Trial execution
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one crop-level trial end-to-end.")
    parser.add_argument("--used-config", required=True, help="Path to used_config.json")
    parser.add_argument("--crop-manifest", required=True, help="Path to crop_manifest.json")
    parser.add_argument("--trial-root", required=True, help="Output root for this trial")
    parser.add_argument("--step12-script", default=None, help="Path to run_step12_pipeline.py")
    parser.add_argument("--step3-script", default=None, help="Path to run_infer_3d.py")
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocess calls")
    parser.add_argument(
        "--crop-ids",
        nargs="*",
        default=None,
        help="Optional explicit crop ids; overrides evaluation_request.crop_ids.",
    )
    parser.add_argument(
        "--copy-used-config",
        action="store_true",
        help="Copy used_config.json into the trial directory as used_config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    used_config_path = Path(args.used_config).expanduser().resolve()
    crop_manifest_path = Path(args.crop_manifest).expanduser().resolve()
    trial_root = Path(args.trial_root).expanduser().resolve()
    trial_root.mkdir(parents=True, exist_ok=True)

    used_cfg = load_json(used_config_path)
    manifest = load_json(crop_manifest_path)

    copied_used_config = False
    used_config_copy_path = trial_root / "used_config.json"
    if args.copy_used_config:
        copied_used_config = safe_copy_file(used_config_path, used_config_copy_path)

    input_cfg = used_cfg["input"]
    model_cfg = used_cfg["model"]
    step1_cfg = used_cfg["step1_sparse_sim"]
    step2_cfg = used_cfg["step2_local_normalization"]
    step3_cfg = used_cfg["step3_cellpose_3d"]

    step12_script = Path(args.step12_script or used_cfg.get("pipeline_paths", {}).get("run_step12_script", "")).expanduser()
    step3_script = Path(args.step3_script or used_cfg.get("pipeline_paths", {}).get("run_infer_3d_script", "")).expanduser()
    if not str(step12_script):
        raise ValueError("step12 script path is missing. Provide --step12-script or config.pipeline_paths.run_step12_script")
    if not str(step3_script):
        raise ValueError("step3 script path is missing. Provide --step3-script or config.pipeline_paths.run_infer_3d_script")
    step12_script = step12_script.resolve()
    step3_script = step3_script.resolve()

    source_path = Path(input_cfg["input_path"]).expanduser().resolve()
    source_stack, source_meta = load_source_stack(source_path, channel_index=input_cfg.get("channel_index"))

    requested_crop_ids = choose_crop_ids(used_cfg, manifest, explicit_crop_ids=args.crop_ids)
    manifest_crops = {c["crop_id"]: c for c in manifest.get("crops", [])}
    missing = [cid for cid in requested_crop_ids if cid not in manifest_crops]
    if missing:
        raise ValueError(f"Requested crop ids not found in crop manifest: {missing}")

    trial_start = now_iso()
    overall_start = time.time()
    per_crop_records: list[dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    for crop_id in requested_crop_ids:
        crop = manifest_crops[crop_id]
        crop_root = trial_root / "crops" / crop_id
        crop_root.mkdir(parents=True, exist_ok=True)
        crop_input_path = crop_root / "crop_input.tif"
        logs_dir = crop_root / "logs"
        step12_output_dir = crop_root / "step12_output"
        step3_output_dir = crop_root / "step3_output"
        step1_cfg_json_path = crop_root / "step1_sparse_config.json"

        crop_stack = extract_crop(source_stack, crop)
        tifffile.imwrite(crop_input_path, crop_stack)
        save_json(step1_cfg_json_path, build_step1_config_json(step1_cfg))

        crop_record: dict[str, Any] = {
            "crop_id": crop_id,
            "role": crop.get("role"),
            "crop_input_path": str(crop_input_path),
            "crop_shape_zyx": list(crop_stack.shape),
            "status": "pending",
            "commands": {},
            "metrics": {},
            "artifacts": {
                "step1_cfg_json": str(step1_cfg_json_path),
                "step12_output_dir": str(step12_output_dir),
                "step3_output_dir": str(step3_output_dir),
            },
        }

        try:
            # Step 1 + Step 2
            cmd12 = step12_command(
                python_exe=args.python,
                step12_script=step12_script,
                crop_input_path=crop_input_path,
                step12_output_dir=step12_output_dir,
                step1_cfg_json_path=step1_cfg_json_path,
                input_cfg=input_cfg,
                step1_cfg=step1_cfg,
                step2_cfg=step2_cfg,
            )
            res12 = run_command(
                cmd=cmd12,
                cwd=step12_script.parent,
                stdout_path=logs_dir / "step12.stdout.log",
                stderr_path=logs_dir / "step12.stderr.log",
            )
            crop_record["commands"]["step12"] = {
                "cmd": res12.cmd,
                "returncode": res12.returncode,
                "elapsed_sec": res12.elapsed_sec,
                "stdout_path": res12.stdout_path,
                "stderr_path": res12.stderr_path,
            }
            if res12.returncode != 0:
                raise RuntimeError(f"Step12 failed for crop {crop_id} with returncode={res12.returncode}")

            step12_meta_path = step12_output_dir / "step12_meta.json"
            if not step12_meta_path.exists():
                raise FileNotFoundError(f"Missing step12_meta.json for crop {crop_id}: {step12_meta_path}")
            step12_meta = load_json(step12_meta_path)

            step3_input_path = step12_output_dir / "step2_local_normalization.tif"
            if not step3_input_path.exists():
                raise FileNotFoundError(f"Missing Step3 input for crop {crop_id}: {step3_input_path}")

            # Step 3
            cmd3 = step3_command(
                python_exe=args.python,
                step3_script=step3_script,
                step3_input_path=step3_input_path,
                step3_output_dir=step3_output_dir,
                model_cfg=model_cfg,
                step3_cfg=step3_cfg,
            )
            res3 = run_command(
                cmd=cmd3,
                cwd=step3_script.parent,
                stdout_path=logs_dir / "step3.stdout.log",
                stderr_path=logs_dir / "step3.stderr.log",
            )
            crop_record["commands"]["step3"] = {
                "cmd": res3.cmd,
                "returncode": res3.returncode,
                "elapsed_sec": res3.elapsed_sec,
                "stdout_path": res3.stdout_path,
                "stderr_path": res3.stderr_path,
            }
            if res3.returncode != 0:
                raise RuntimeError(f"Step3 failed for crop {crop_id} with returncode={res3.returncode}")

            step3_meta_path = step3_output_dir / "meta.json"
            step3_params_path = step3_output_dir / "params.json"
            if not step3_meta_path.exists():
                raise FileNotFoundError(f"Missing Step3 meta.json for crop {crop_id}: {step3_meta_path}")
            step3_meta = load_json(step3_meta_path)
            step3_params = load_json(step3_params_path) if step3_params_path.exists() else {}

            crop_record["metrics"] = {
                "num_instances": step3_meta.get("num_instances"),
                "step12_elapsed_sec": res12.elapsed_sec,
                "step3_elapsed_sec": res3.elapsed_sec,
                "pipeline_elapsed_sec": res12.elapsed_sec + res3.elapsed_sec,
            }
            crop_record["step12_meta"] = step12_meta
            crop_record["step3_meta"] = step3_meta
            crop_record["step3_params"] = step3_params
            crop_record["status"] = "success" if bool(step3_meta.get("success", False)) else "failed"

            if crop_record["status"] == "success":
                success_count += 1
            else:
                fail_count += 1
        except Exception as exc:  # noqa: BLE001
            crop_record["status"] = "failed"
            crop_record["error_type"] = type(exc).__name__
            crop_record["error_message"] = str(exc)
            fail_count += 1

        per_crop_records.append(crop_record)
        save_json(crop_root / "crop_record.json", crop_record)

    overall_elapsed = time.time() - overall_start
    successful_counts = [
        rec.get("metrics", {}).get("num_instances")
        for rec in per_crop_records
        if rec.get("status") == "success" and rec.get("metrics", {}).get("num_instances") is not None
    ]
    trial_end = now_iso()

    aggregate = {
        "num_crops": len(requested_crop_ids),
        "success_count": int(success_count),
        "fail_count": int(fail_count),
        "cell_count_mean": float(np.mean(successful_counts)) if successful_counts else None,
        "cell_count_std": float(np.std(successful_counts)) if successful_counts else None,
        "cell_count_max": int(np.max(successful_counts)) if successful_counts else None,
        "cell_count_min": int(np.min(successful_counts)) if successful_counts else None,
        "runtime_total_sec": float(overall_elapsed),
    }

    trial_record = {
        "schema_version": "1.0",
        "trial_id": used_cfg.get("trial_id"),
        "run_tag": used_cfg.get("run_tag"),
        "timestamp_start": trial_start,
        "timestamp_end": trial_end,
        "used_config_path": str(used_config_path),
        "used_config_copy_path": str(used_config_copy_path),
        "used_config_copied": bool(copied_used_config),
        "crop_manifest_path": str(crop_manifest_path),
        "trial_root": str(trial_root),
        "python_executable": args.python,
        "pipeline_paths": {
            "run_step12_script": str(step12_script),
            "run_infer_3d_script": str(step3_script),
        },
        "input_source": {
            "path": str(source_path),
            "shape": source_meta["shape"],
            "dtype": source_meta["dtype"],
            "axes": source_meta["axes"],
        },
        "selected_crop_ids": requested_crop_ids,
        "aggregate_metrics": aggregate,
        "per_crop": per_crop_records,
        "status": "success" if fail_count == 0 else ("partial" if success_count > 0 else "failed"),
    }

    save_json(trial_root / "trial_record.json", trial_record)
    print("=" * 90)
    print("Trial finished")
    print(f"trial_id          : {used_cfg.get('trial_id')}")
    print(f"run_tag           : {used_cfg.get('run_tag')}")
    print(f"trial_root        : {trial_root}")
    print(f"used_config_path  : {used_config_path}")
    print(f"used_config_copy  : {used_config_copy_path}")
    print(f"used_config_copied: {copied_used_config}")
    print(f"status            : {trial_record['status']}")
    print(f"success/fail      : {success_count}/{fail_count}")
    print(f"cell_count_mean   : {aggregate['cell_count_mean']}")
    print(f"runtime_total     : {aggregate['runtime_total_sec']:.2f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()