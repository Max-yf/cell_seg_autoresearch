# program.md

## 3D Cell Segmentation Autoresearch Program

This repository is an experiment in autonomous parameter research for a fixed 3-step 3D cell segmentation pipeline.

The agent is **not** allowed to redesign the whole project.
The job is to **search for better parameter combinations** under clearly defined constraints, record the results, and keep only the improvements.

The pipeline is:

1. **Step 1**: Sparse-SIM style sparse deconvolution
2. **Step 2**: Slice-wise local normalization
3. **Step 3**: 3D Cellpose-SAM inference using a **fixed finetuned 2D model**

---

## 1. Core Goal

The goal is to find parameter settings that produce:

1. **more valid cell instances**
2. **fewer donut / hollow artifacts**
3. **stable and memory-safe 3D inference**

This is a **multi-objective optimization problem**, but the practical priority is:

* first: increase valid cell count
* second: reduce donut / hollow artifacts
* third: keep runtime / memory reasonable
* fourth: prefer simpler settings over unnecessarily complicated ones

A result with a slightly larger cell count is **not** considered better if it obviously creates many fake fragments, hollow cells, or unstable behavior.

---

## 2. What Is Fixed

### 2.1 Fixed model

The 2D Cellpose-SAM finetuned model is already considered optimal enough for now.

The model is **fixed**.
The agent must **not** search over:

* model weights
* model architecture
* training hyperparameters
* retraining / finetuning procedure

The current task is **not** model training research.
It is **pipeline parameter research** for 3D inference.

### 2.2 Fixed input type

The pipeline only supports:

* a **single 3D `.tif` / `.tiff`**
* default axis order: **`(Z, H, W)`** for Step 3 input
* `z_axis = 0`

### 2.3 Fixed Step 3 safety rule

To avoid triggering Cellpose-SAM internal scaling behavior and memory blow-up:

* **`diameter` must stay `None`**
* this is a hard rule, not a suggestion

The agent must never intentionally switch `diameter` from `None` to a numeric value unless the human explicitly changes this rule.

---

## 3. Philosophy of This Project

This is **not** a brute-force full-volume sweep.

The search must be:

1. **pipeline-aware**
2. **crop-based first**
3. **full-volume later**
4. **progressive**
5. **well logged**
6. **reproducible**

The key reality of this project is:

* Step 1 changes the image distribution
* Step 2 further changes local contrast / intensity structure
* therefore Step 3 optimal parameters may shift depending on Step 1 and Step 2

So the pipeline must be treated as **linked**.

However, linked does **not** mean opening every parameter at full range immediately.
The agent should perform **linked but staged search**.

---

## 4. In-Scope Files

The exact filenames may vary slightly depending on repo layout, but the agent should treat the following as the main in-scope logic:

* `README_HUMAN.md`
* `README_LLM.md`
* pipeline runner for Step1+2+3
* Step 1 sparse deconvolution script
* Step 2 local normalization script
* Step 3 3D inference script
* config templates
* experiment output folders
* `results.tsv`

The agent should read the pipeline code carefully before proposing new search settings.

---

## 5. What The Agent CAN Do

The agent may:

1. create and update experiment config files
2. create crop-based evaluation subsets
3. run pipeline trials on crops
4. run selected candidate settings on larger crops
5. run top candidates on full 3D volume
6. compute and compare metrics
7. log all experiments in `results.tsv`
8. keep or discard experiments based on the rules below
9. narrow the search space over time
10. add small helper scripts for:

* crop selection
* metric calculation
* result aggregation
* trial scheduling
* report generation

The preferred mode is:

* keep pipeline core stable
* modify configs / search logic / evaluation helpers
* do not casually rewrite core processing code

---

## 6. What The Agent CANNOT Do

The agent must **not**:

1. change the fixed 2D finetuned model
2. retrain the model
3. change the scientific goal from segmentation parameter search to something else
4. remove logging or reproducibility
5. silently change input assumptions
6. set `diameter` to a numeric value
7. disable 3D mode for the main search target
8. claim improvement using only one lucky crop without broader validation
9. promote a candidate to “best” if it only increases count by clearly introducing artifacts
10. install arbitrary new heavy dependencies unless the human explicitly allows it

---

## 7. Primary Optimization Objective

The real-world goal is:

> maximize useful cell detection while minimizing donut artifacts and avoiding memory blow-up.

The working score should combine:

* **cell count**
* **donut / hollow penalty**
* **failure penalty**
* **resource penalty**

A conceptual form is:

[
\text{score}
============

## w_c \cdot \text{count_score}

## w_h \cdot \text{hollow_penalty}

## w_f \cdot \text{failure_penalty}

## w_m \cdot \text{memory_penalty}

w_t \cdot \text{time_penalty}
]

At early stage, the score may be simplified.
But the agent must always remember:

* more count is good
* fake count is bad
* hollow / donut masks are bad
* crashes are bad
* OOM is bad
* unstable behavior is bad

---

## 8. Baseline First

The very first run must establish a baseline.

The baseline should use the currently validated default / recommended pipeline settings.

This baseline must be run on:

1. crop evaluation set
2. larger validation crop(s)
3. full-volume final check only when needed

The baseline result must be logged in `results.tsv` before any search begins.

No candidate can be called an improvement unless it beats the baseline under the same evaluation protocol.

---

## 9. Search Strategy

## 9.1 Do not start with full-volume search

The search should begin on **small representative crops**.

The order should be:

1. crop-level coarse search
2. crop-level local refinement
3. larger crop verification
4. full-volume verification for top candidates only

## 9.2 Linked but staged search

The pipeline is linked across Step 1 / 2 / 3, but the search should still be controlled.

Recommended approach:

### Phase A: safe linked coarse search

Open only a limited subset of important parameters across all three steps.

### Phase B: local refinement around promising regions

Take the top candidates from Phase A and refine nearby values.

### Phase C: robustness check

Test top candidates on multiple crop windows, not just one.

### Phase D: full-volume promotion

Only the most promising robust candidates are allowed to run on the full 3D stack.

---

## 10. Automatic Crop Selection Rules

The dataset is a single 3D TIFF with about 200 z-slices.
The agent should not choose crops randomly unless explicitly requested.

Instead, it should build a **crop evaluation set** automatically.

### 10.1 Purpose

The crop set should represent:

* dense cell regions
* medium-density regions
* difficult / low-contrast / ambiguous regions

### 10.2 Slice scoring

The agent should compute a simple per-slice information score using image-derived proxies such as:

* bright-area ratio
* local variance
* edge energy
* foreground occupancy estimate

The exact formula can evolve, but it must be documented.

### 10.3 Window selection

The agent should smooth the per-slice score along Z and select several Z-windows, for example:

* one high-density window
* one medium-density window
* one difficult window

### 10.4 Crop extraction

For each selected Z-window, extract one or more XY crops.

Recommended first-pass crop scale:

* Z depth: around 16–32 slices
* XY size: around 256–512 per side

The crop set should contain multiple representative samples, not just a single crop.

### 10.5 Fixed evaluation set

Once the initial crop evaluation set is created, it should remain fixed for the current run tag unless there is a strong reason to rebuild it.
This avoids “moving the exam paper”.

---

## 11. Parameters: What To Search

## 11.1 Step 1: sparse deconvolution

These are candidates for search:

* `sparse_iter`
* `fidelity`
* `z_continuity`
* `sparsity`
* `deconv_iter`

These should start with narrow, reasonable ranges.
Do not explode the search space too early.

Parameters tied mainly to physical meaning or implementation stability should usually remain fixed unless there is a clear reason:

* `pixel_size_nm`
* `wavelength_nm`
* `effective_na`
* `mode`
* `window_size`
* `halo`
* `backend`

## 11.2 Step 2: local normalization

These are good search candidates:

* `radius`
* `bias`

Keep Step 2 small and interpretable.

## 11.3 Step 3: 3D Cellpose-SAM

These are primary search candidates:

* `cellprob_threshold`
* `min_size`
* `anisotropy`
* `rescale`
* `flow_threshold`
* `tile_overlap`
* `batch_size_3d`
* `bsize`
* `augment`

### Hard fixed rules for Step 3

* `diameter = None`
* `do_3D = True`
* `z_axis = 0`
* `stitch_threshold = 0.0` unless the human explicitly changes strategy

---

## 12. Search Space Discipline

The agent must not open all parameters widely at once.

Use these rules:

1. start with a small number of important parameters
2. use narrow ranges first
3. prefer random / quasi-random / guided search over huge grid search
4. after finding promising areas, refine locally
5. drop clearly bad regions quickly
6. avoid wasting compute on obviously dangerous combinations

The search process should be intentional, not chaotic.

---

## 13. Safety Constraints

Any experiment must be treated as **invalid** if it violates one of these:

1. run crashed
2. output files missing
3. OOM occurred
4. runtime far exceeds reasonable budget for its evaluation level
5. output masks are obviously broken
6. `diameter != None`
7. input assumptions are violated
8. the candidate only “improves” by producing obvious garbage segmentation

The agent must prefer stable, reproducible, debuggable improvements.

---

## 14. Evaluation Levels

## Level 1: crop-level screening

Purpose:

* fast elimination
* coarse ranking
* identify sensitive parameters

## Level 2: multi-crop validation

Purpose:

* reject lucky overfit settings
* assess consistency across representative regions

## Level 3: larger crop verification

Purpose:

* test whether behavior survives larger spatial context

## Level 4: full-volume verification

Purpose:

* determine whether the candidate is truly useful in the real task

A setting should only be called “best current candidate” after passing at least multi-crop validation, and ideally larger-crop verification.

---

## 15. Metrics To Record

Every trial should record at least:

* run tag
* trial id
* config path
* git commit hash if applicable
* evaluation level
* crop identifiers
* status
* total score
* cell count
* hollow / donut metrics
* runtime
* peak memory
* brief description

If available, also record:

* count variance across crops
* failure count across crops
* notes about visible artifacts

---

## 16. Donut / Hollow Awareness

This project explicitly cares about donut / hollow artifacts.

A candidate should be penalized if it produces:

* hollow centers inside otherwise cell-like masks
* obvious ring-only detections
* severe fragmentation that inflates count artificially

If a precise hollow metric already exists in the repo, use it.
If not, create a simple, explainable proxy and improve it later.

The metric does not need to be perfect on day one, but it must be:

* documented
* repeatable
* comparable across trials

---

## 17. Keep / Discard Rules

### Keep

Mark a trial as `keep` if:

* it beats the current best score by a meaningful margin
* or it matches score while being simpler / safer / faster
* or it produces slightly fewer cells but clearly reduces hollow artifacts enough to improve overall usefulness

### Discard

Mark a trial as `discard` if:

* score is worse
* count gain is tiny but artifacts increase
* memory or runtime gets much worse without clear benefit
* behavior is inconsistent across crops

### Crash

Mark a trial as `crash` if:

* script failed
* OOM
* invalid output
* broken config
* impossible parameter combination
* safety rule violation

---

## 18. Simplicity Rule

All else equal, simpler wins.

A tiny gain is not worth:

* messy logic
* hard-to-maintain hacks
* opaque special cases
* fragile code paths

The best candidate is not just the one with the highest score.
It is the one with the best balance of:

* score
* simplicity
* robustness
* reproducibility

---

## 19. results.tsv Format

Use tab-separated format.

Suggested header:

```tsv
run_tag	trial_id	level	status	score	cell_count	hollow_metric	runtime_sec	peak_mem_gb	config	description
```

Where:

* `run_tag`: experiment campaign name
* `trial_id`: unique trial identifier
* `level`: crop / multicrop / largecrop / fullvolume
* `status`: keep / discard / crash
* `score`: overall score
* `cell_count`: main count metric
* `hollow_metric`: lower is better
* `runtime_sec`: elapsed time
* `peak_mem_gb`: peak memory
* `config`: config filename or path
* `description`: short description of what changed

Do not use comma-separated CSV for this log.

---

## 20. Trial Execution Pattern

For each trial:

1. decide the parameter proposal
2. save the trial config
3. run the pipeline on the designated evaluation set
4. collect metrics
5. compute score
6. compare with current best
7. log the result in `results.tsv`
8. keep or discard according to the rules
9. update the local search direction

The agent should always know:

* what changed
* why it changed
* what happened
* whether it was worth keeping

---

## 21. Promotion Rules

A candidate may be promoted from crop-level to larger validation only if:

* it is stable across the crop set
* it has no major safety issues
* it improves the combined objective, not just raw count

A candidate may be promoted to full-volume only if:

* it remains competitive after multi-crop evaluation
* it does not show obvious artifact inflation
* it stays within practical memory / runtime limits

---

## 22. Human Interaction Policy

The human wants the search to be thoughtful, traceable, and scientifically sane.

Do not repeatedly ask for confirmation once the rules are clear.
Do not stop after one or two trials just because the first ideas failed.

However, if a foundational ambiguity appears, surface it clearly in logs or notes.

The agent should behave like a disciplined research assistant:

* autonomous
* careful
* skeptical
* reproducible

---

## 23. Default Starting Plan

Unless the human overrides it, start like this:

### Stage 0

* read all relevant pipeline files
* identify baseline parameters
* verify fixed model path
* verify `diameter=None`
* build crop evaluation set
* initialize `results.tsv`

### Stage 1

* run baseline on crop evaluation set
* log baseline

### Stage 2

* run a limited linked coarse search across:

  * a few Step 1 parameters
  * Step 2 radius / bias
  * key Step 3 parameters

### Stage 3

* refine around the top candidates

### Stage 4

* validate on larger crops / multiple windows

### Stage 5

* run full-volume verification on the best few candidates

---

## 24. Final Output Expectations

At the end of a search campaign, the agent should be able to provide:

1. the current best config
2. the baseline config
3. a summary of what changed
4. the best observed cell count
5. the hollow / donut behavior summary
6. runtime / memory comparison
7. the recommended next search direction

The output should make it easy for a human collaborator to understand:

* what was tried
* what worked
* what failed
* what should be done next

---

## 25. Guiding Principle

This project is not about winning a toy benchmark.
It is about finding a practically useful 3D segmentation recipe for a fixed finetuned 2D Cellpose-SAM model, under real memory constraints, with attention to both **cell count** and **donut reduction**.

The agent should think like this:

* more cells, but not fake cells
* fewer donuts, but not by deleting half the dataset
* better settings, not more chaos
* stable improvements, not lucky accidents

