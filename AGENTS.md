# AGENTS.md

## Purpose

This repository manages autoresearch for a fixed 3-step 3D cell segmentation
pipeline:

1. sparse deconvolution
2. local normalization
3. fixed finetuned Cellpose-SAM 3D inference

The main goal is to improve segmentation parameters without redesigning the
scientific pipeline.

## Operating Model

- Codex runs locally.
- Heavy GPU execution happens on a remote Slurm cluster.
- Local orchestration should prepare configs, metadata, and Slurm job scripts.
- Cluster jobs should execute prepared trials and write results back into the
  corresponding trial directories.

## Hard Constraints

1. Do not change the fixed finetuned Cellpose-SAM model.
2. Do not set `diameter` to a numeric value. It must remain `None` unless the
   human explicitly changes that rule.
3. Keep `do_3D=True` and `z_axis=0`.
4. Prefer editing config generation, orchestration, logging, and evaluation
   helpers over rewriting the core scientific pipeline.
5. All completed trials must be recorded in `results.tsv`.
6. Large TIFF outputs and heavy run artifacts must not be committed.
7. Favor simple, reproducible, debuggable changes over fully autonomous logic.
8. If something is inferred rather than directly specified in files, say so
   clearly.

## Repository Conventions

- `baseline_config.json` is the baseline scientific configuration.
- `crop_manifest.json` defines the fixed crop evaluation set for a campaign.
- `trial_config.template.json` documents the override-style trial schema.
- `scripts/propose_trial.py` proposes trial edits.
- `scripts/materialize_config.py` merges baseline and trial config into a
  runnable `used_config.json`.
- `scripts/run_trial.py` is the cluster-side executor for one prepared trial.
- `scripts/score_trial.py` scores one finished trial.
- `scripts/append_results.py` appends one scored trial into `results.tsv`.
- `scripts/run_autoresearch_loop.py` is the local orchestration entry point for
  preparing a small batch of cluster-executed trials.

## Preferred Workflow

1. Prepare or refresh `crop_manifest.json`.
2. Use `run_autoresearch_loop.py` locally to create:
   - trial configs
   - materialized runnable configs
   - per-trial metadata
   - Slurm submission scripts
3. Copy or sync the prepared repository state to the cluster working area.
4. Submit the generated Slurm scripts on the cluster.
5. Let each cluster job run:
   - `run_trial.py`
   - `score_trial.py`
   - `append_results.py`
6. Inspect `results.tsv` and the generated trial directories before proposing
   the next batch.

## Notes For Future Edits

- Keep the first version of any orchestration feature minimal.
- Avoid introducing new heavy dependencies.
- Preserve the existing config schema where possible.
- Prefer writing extra metadata files instead of overloading scientific config
  sections with cluster-only details.


Rules for this repository:

1. Only read and modify files inside this project workspace.
2. Do not access parent directories or sibling projects.
3. Do not create, delete, move, or overwrite files without first explaining the plan.
4. Do not use network unless explicitly approved.
5. Before editing, list which files you plan to change and why.
6. If a requested action would touch anything outside this workspace, stop and ask first.
