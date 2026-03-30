# 3.28 Autoresearch-lite for 3D Cell Segmentation

## Project Goal

This repository implements a local-orchestrated / cluster-executed trial workflow for a fixed 3-step 3D cell segmentation pipeline:

1. sparse deconvolution
2. local normalization
3. fixed finetuned Cellpose-SAM 3D inference

The goal is to search for better preprocessing and inference parameter combinations while keeping the scientific pipeline controlled and reproducible.

## Key Constraints

- The finetuned Cellpose-SAM model is fixed.
- `diameter` must remain `None` unless explicitly changed by a human.
- `do_3D` must remain `True`.
- `z_axis` must remain `0`.
- Large TIFF outputs should not be committed to git.

## Repository Structure

- `program.md`  
  Research goal, constraints, and scoring principles.

- `AGENTS.md`  
  Rules for Codex / AI-assisted editing.

- `baseline_config.json`  
  Baseline trial configuration.

- `trial_config.template.json`  
  Template for generating new trial configs.

- `crop_manifest.json`  
  Crop metadata used by trials.

- `results.tsv`  
  Lightweight summary table of completed trials.

- `scripts/`  
  Utility scripts for proposing trials, materializing configs, running trials, scoring trials, and appending results.

- `runs/`  
  Per-trial runtime metadata and outputs.

## Intended Workflow

### 1. Prepare baseline
Use `baseline_config.json` as the starting point.

### 2. Propose new trial
Generate a small mutation around baseline or current best trial.

### 3. Materialize trial config
Convert template + parameters into a runnable config file.

### 4. Run on Slurm cluster
The cluster only executes Python/Slurm jobs. It does not make scientific decisions.

### 5. Score trial
Produce `score.json` with metrics such as:
- cell count
- hollow metric
- runtime
- status

### 6. Record result
Append a lightweight summary into `results.tsv`.

## Philosophy

- local machine: orchestrates, edits, decides
- cluster: executes
- git: records lightweight history
- large artifacts stay out of git

## Current Status

This repository is under active refactoring toward a minimal reproducible autoresearch loop.