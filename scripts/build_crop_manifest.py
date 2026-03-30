#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_crop_manifest.py

Purpose
-------
Automatically build a fixed crop evaluation set for autoresearch on a single
3D TIFF stack.

Design goals:
1. Deterministic and reproducible.
2. Explainable crop selection.
3. Works for both 3D ZYX input and 4D input with channels (e.g. ZCYX/ CZYX).
4. Outputs a crop_manifest.json that later trials can reuse as a fixed exam set.

The script computes a simple per-slice information score from:
- bright area ratio
- local variance proxy (slice variance)
- edge energy proxy (mean gradient magnitude)

Then it smooths the score along Z and selects up to three windows:
- high_density
- medium_density
- difficult

For each selected z-window it chooses one or more XY crops using a projection-
based energy map.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class SourceInfo:
    input_path: str
    image_shape: list[int]
    dtype: str
    detected_axes: str
    selected_channel_index: int | None
    extracted_shape_zyx: list[int]
    z_axis: int


@dataclass
class CropRecord:
    crop_id: str
    role: str
    z_range: list[int]
    y_range: list[int]
    x_range: list[int]
    center_zyx: list[int]
    window_score_mean: float
    projection_score: float
    notes: str


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64, copy=True)
    window = int(max(1, window))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(x.astype(np.float64), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def robust_minmax(x: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = float(np.percentile(x, q_low))
    hi = float(np.percentile(x, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.zeros_like(x, dtype=np.float64)
        if np.nanmax(x) > np.nanmin(x):
            out = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        return np.clip(out, 0.0, 1.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def integral_image(arr: np.ndarray) -> np.ndarray:
    return arr.cumsum(axis=0).cumsum(axis=1)


def rect_sum(ii: np.ndarray, y0: int, x0: int, h: int, w: int) -> float:
    y1 = y0 + h - 1
    x1 = x0 + w - 1
    total = ii[y1, x1]
    if y0 > 0:
        total -= ii[y0 - 1, x1]
    if x0 > 0:
        total -= ii[y1, x0 - 1]
    if y0 > 0 and x0 > 0:
        total += ii[y0 - 1, x0 - 1]
    return float(total)


def choose_non_overlapping_indices(candidates: list[int], min_gap: int) -> list[int]:
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(idx - old) >= min_gap for old in chosen):
            chosen.append(idx)
    return chosen


# -----------------------------------------------------------------------------
# TIFF loading / channel handling
# -----------------------------------------------------------------------------

def load_stack(path: Path, channel_index: int | None) -> tuple[np.ndarray, SourceInfo]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        arr = np.asarray(series.asarray())
        axes = getattr(series, "axes", "") or ""

    original_shape = list(arr.shape)
    dtype = str(arr.dtype)

    if arr.ndim == 2:
        stack = arr[np.newaxis, ...]
        selected_channel = None
        z_axis = 0
    elif arr.ndim == 3:
        # Assume already ZYX.
        stack = arr
        selected_channel = None
        z_axis = 0
    elif arr.ndim == 4:
        if "C" not in axes:
            raise ValueError(
                f"4D TIFF is ambiguous without a channel axis. shape={arr.shape}, axes={axes!r}"
            )
        c_axis = axes.index("C")
        if channel_index is None:
            raise ValueError("Input contains channels; please provide --channel-index.")
        if not (0 <= channel_index < arr.shape[c_axis]):
            raise ValueError(
                f"channel_index={channel_index} out of range for shape={arr.shape}, axes={axes!r}"
            )
        stack = np.take(arr, indices=channel_index, axis=c_axis)
        remaining_axes = "".join(ax for ax in axes if ax != "C")
        if remaining_axes != "ZYX":
            raise ValueError(
                f"After channel extraction expected remaining axes ZYX, got {remaining_axes!r}."
            )
        selected_channel = channel_index
        z_axis = 0
    else:
        raise ValueError(f"Unsupported TIFF shape={arr.shape}, axes={axes!r}")

    if stack.ndim != 3:
        raise ValueError(f"Expected extracted 3D stack, got shape={stack.shape}")

    info = SourceInfo(
        input_path=str(path.resolve()),
        image_shape=original_shape,
        dtype=dtype,
        detected_axes=axes,
        selected_channel_index=selected_channel,
        extracted_shape_zyx=list(stack.shape),
        z_axis=z_axis,
    )
    return np.asarray(stack), info


# -----------------------------------------------------------------------------
# Slice scoring
# -----------------------------------------------------------------------------

def normalize_slice_for_scoring(slc: np.ndarray) -> np.ndarray:
    slc = slc.astype(np.float32, copy=False)
    lo = np.percentile(slc, 1.0)
    hi = np.percentile(slc, 99.0)
    if hi <= lo:
        return np.zeros_like(slc, dtype=np.float32)
    return np.clip((slc - lo) / (hi - lo), 0.0, 1.0)


def compute_slice_metrics(stack: np.ndarray) -> dict[str, np.ndarray]:
    z = stack.shape[0]
    bright_area = np.zeros(z, dtype=np.float64)
    variance = np.zeros(z, dtype=np.float64)
    edge_energy = np.zeros(z, dtype=np.float64)
    foreground_est = np.zeros(z, dtype=np.float64)

    for zi in range(z):
        slc = normalize_slice_for_scoring(stack[zi])
        variance[zi] = float(np.var(slc))
        bright_thr = float(np.quantile(slc, 0.90))
        bright_area[zi] = float(np.mean(slc >= bright_thr))
        gy, gx = np.gradient(slc)
        edge_energy[zi] = float(np.mean(np.sqrt(gx * gx + gy * gy)))
        fg_thr = max(0.05, float(np.quantile(slc, 0.70)))
        foreground_est[zi] = float(np.mean(slc >= fg_thr))

    return {
        "bright_area": bright_area,
        "variance": variance,
        "edge_energy": edge_energy,
        "foreground_est": foreground_est,
    }


def compute_combined_slice_score(metrics: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bright_n = robust_minmax(metrics["bright_area"])
    var_n = robust_minmax(metrics["variance"])
    edge_n = robust_minmax(metrics["edge_energy"])
    fg_n = robust_minmax(metrics["foreground_est"])

    # Slightly favor “there is actual structure” over raw brightness.
    score = 0.25 * bright_n + 0.30 * var_n + 0.30 * edge_n + 0.15 * fg_n
    normalized = {
        "bright_area_norm": bright_n,
        "variance_norm": var_n,
        "edge_energy_norm": edge_n,
        "foreground_est_norm": fg_n,
    }
    return score.astype(np.float64), normalized


# -----------------------------------------------------------------------------
# Z-window and XY crop selection
# -----------------------------------------------------------------------------

def find_best_window_center(smoothed_score: np.ndarray, used_centers: Iterable[int], min_gap: int, target: str) -> int:
    z = len(smoothed_score)
    used_centers = list(used_centers)
    order_desc = list(np.argsort(smoothed_score)[::-1])
    order_asc = list(np.argsort(smoothed_score))

    if target == "high_density":
        candidates = order_desc
    elif target == "medium_density":
        median = float(np.median(smoothed_score))
        candidates = list(np.argsort(np.abs(smoothed_score - median)))
    elif target == "difficult":
        # Prefer below-median but not near-zero structure.
        median = float(np.median(smoothed_score))
        lower_half = [i for i in np.argsort(np.abs(smoothed_score - 0.7 * median)) if smoothed_score[i] <= median]
        candidates = lower_half + order_asc
    else:
        raise ValueError(f"Unknown target={target!r}")

    for idx in candidates:
        if all(abs(int(idx) - int(prev)) >= min_gap for prev in used_centers):
            return int(idx)

    # Fallback: first available.
    return int(candidates[0]) if candidates else int(z // 2)


def z_window_from_center(center: int, z_size: int, depth: int) -> tuple[int, int]:
    depth = min(depth, z_size)
    start = max(0, center - depth // 2)
    end = min(z_size, start + depth)
    start = max(0, end - depth)
    return int(start), int(end)


def projection_energy(window_stack: np.ndarray) -> np.ndarray:
    # Mean projection of normalized slices.
    norm = np.stack([normalize_slice_for_scoring(s) for s in window_stack], axis=0)
    proj = np.mean(norm, axis=0)
    gy, gx = np.gradient(proj)
    edge = np.sqrt(gx * gx + gy * gy)
    energy = 0.55 * robust_minmax(proj) + 0.45 * robust_minmax(edge)
    return energy.astype(np.float64)


def choose_xy_crop_from_energy(energy: np.ndarray, crop_h: int, crop_w: int) -> tuple[int, int, float]:
    h, w = energy.shape
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)
    ii = integral_image(energy)
    best_score = -math.inf
    best_yx = (0, 0)

    stride_y = max(8, crop_h // 6)
    stride_x = max(8, crop_w // 6)

    for y0 in range(0, h - crop_h + 1, stride_y):
        for x0 in range(0, w - crop_w + 1, stride_x):
            score = rect_sum(ii, y0, x0, crop_h, crop_w) / float(crop_h * crop_w)
            if score > best_score:
                best_score = score
                best_yx = (y0, x0)

    # Always also consider the final border-aligned positions.
    final_positions = {(h - crop_h, w - crop_w), (0, w - crop_w), (h - crop_h, 0)}
    for y0, x0 in final_positions:
        y0 = max(0, y0)
        x0 = max(0, x0)
        score = rect_sum(ii, y0, x0, crop_h, crop_w) / float(crop_h * crop_w)
        if score > best_score:
            best_score = score
            best_yx = (y0, x0)

    return int(best_yx[0]), int(best_yx[1]), float(best_score)


# -----------------------------------------------------------------------------
# Main build logic
# -----------------------------------------------------------------------------

def build_manifest(
    stack: np.ndarray,
    source_info: SourceInfo,
    output_path: Path,
    z_depth: int,
    crop_h: int,
    crop_w: int,
    num_xy_per_window: int,
    smoothing_window: int,
    min_center_gap: int,
) -> dict[str, Any]:
    metrics = compute_slice_metrics(stack)
    slice_score, normalized = compute_combined_slice_score(metrics)
    smoothed = moving_average(slice_score, smoothing_window)

    roles = ["high_density", "medium_density", "difficult"]
    selected_centers: list[int] = []
    crops: list[CropRecord] = []

    z_size, h, w = stack.shape
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    for role in roles:
        center = find_best_window_center(smoothed, selected_centers, min_center_gap=min_center_gap, target=role)
        selected_centers.append(center)
        z0, z1 = z_window_from_center(center, z_size, z_depth)
        window_stack = stack[z0:z1]
        energy = projection_energy(window_stack)

        # First crop = best projection-energy region.
        y0, x0, score0 = choose_xy_crop_from_energy(energy, crop_h, crop_w)
        candidates = [(y0, x0, score0, 0)]

        # Optional extra crop(s): choose around the center / alternate region.
        if num_xy_per_window > 1:
            cy = max(0, min(h - crop_h, (h - crop_h) // 2))
            cx = max(0, min(w - crop_w, (w - crop_w) // 2))
            center_score = float(np.mean(energy[cy : cy + crop_h, cx : cx + crop_w]))
            if abs(cy - y0) + abs(cx - x0) > min(crop_h, crop_w) // 4:
                candidates.append((cy, cx, center_score, 1))

        for y0, x0, proj_score, local_idx in candidates[: max(1, num_xy_per_window)]:
            crop_id = f"crop_z{role.split('_')[0]}_xy{local_idx}"
            y1 = y0 + crop_h
            x1 = x0 + crop_w
            rec = CropRecord(
                crop_id=crop_id,
                role=role,
                z_range=[int(z0), int(z1)],
                y_range=[int(y0), int(y1)],
                x_range=[int(x0), int(x1)],
                center_zyx=[int((z0 + z1) // 2), int((y0 + y1) // 2), int((x0 + x1) // 2)],
                window_score_mean=float(np.mean(smoothed[z0:z1])) if z1 > z0 else float(smoothed[center]),
                projection_score=float(proj_score),
                notes=f"Auto-selected {role} crop from smoothed z-profile and projection energy.",
            )
            crops.append(rec)

    manifest = {
        "schema_version": "1.0",
        "manifest_id": output_path.stem,
        "source_image": asdict(source_info),
        "generation": {
            "method_version": "v1",
            "description": "Auto-selected representative crops from per-slice score and projection energy.",
            "slice_score_formula": {
                "bright_area_weight": 0.25,
                "variance_weight": 0.30,
                "edge_energy_weight": 0.30,
                "foreground_est_weight": 0.15,
            },
            "z_smoothing": {
                "method": "moving_average",
                "window": int(smoothing_window),
            },
            "selection_policy": roles,
        },
        "global_crop_defaults": {
            "z_depth": int(z_depth),
            "xy_size": [int(crop_h), int(crop_w)],
            "num_xy_per_window": int(max(1, num_xy_per_window)),
        },
        "slice_metrics": {
            "bright_area": metrics["bright_area"].tolist(),
            "variance": metrics["variance"].tolist(),
            "edge_energy": metrics["edge_energy"].tolist(),
            "foreground_est": metrics["foreground_est"].tolist(),
            "combined_score": slice_score.tolist(),
            "combined_score_smoothed": smoothed.tolist(),
        },
        "slice_metrics_normalized": {k: v.tolist() for k, v in normalized.items()},
        "crops": [asdict(c) for c in crops],
    }
    return manifest


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crop_manifest.json from a 3D TIFF stack.")
    parser.add_argument("--input", required=True, help="Path to source TIFF.")
    parser.add_argument("--output", required=True, help="Output crop_manifest.json path.")
    parser.add_argument(
        "--channel-index",
        type=int,
        default=None,
        help="Required if the input TIFF contains channels and axes include C.",
    )
    parser.add_argument("--z-depth", type=int, default=24, help="Depth of each z-window.")
    parser.add_argument("--crop-height", type=int, default=384, help="Crop height.")
    parser.add_argument("--crop-width", type=int, default=384, help="Crop width.")
    parser.add_argument(
        "--num-xy-per-window",
        type=int,
        default=1,
        help="How many XY crops to keep per selected z-window.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=7,
        help="Moving-average window on the per-slice score.",
    )
    parser.add_argument(
        "--min-center-gap",
        type=int,
        default=18,
        help="Minimum z-gap between selected window centers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    stack, source_info = load_stack(input_path, channel_index=args.channel_index)

    manifest = build_manifest(
        stack=stack,
        source_info=source_info,
        output_path=output_path,
        z_depth=args.z_depth,
        crop_h=args.crop_height,
        crop_w=args.crop_width,
        num_xy_per_window=args.num_xy_per_window,
        smoothing_window=args.smoothing_window,
        min_center_gap=args.min_center_gap,
    )
    save_json(output_path, manifest)

    print("=" * 90)
    print("Crop manifest built successfully")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Crops  : {len(manifest['crops'])}")
    for crop in manifest["crops"]:
        print(
            f" - {crop['crop_id']}: role={crop['role']} "
            f"z={tuple(crop['z_range'])} y={tuple(crop['y_range'])} x={tuple(crop['x_range'])}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
