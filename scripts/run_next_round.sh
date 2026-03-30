#!/bin/bash
set -eo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/run_next_round.sh /path/to/next_round_upload [optional_campaign_name]"
  exit 1
fi

UPLOAD_DIR="$1"
CAMPAIGN_NAME="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASELINE_PATH="$REPO_ROOT/baseline_config.json"
CROP_MANIFEST_PATH="$REPO_ROOT/crop_manifest.json"
RESULTS_PATH="$REPO_ROOT/results.tsv"

# 这里沿用你当前工程里已经明确的 cluster 配置
CLUSTER_REPO_ROOT="$REPO_ROOT"
CLUSTER_PYTHON="/gpfs/share/home/2306391536/.local/share/mamba/envs/cpsm/bin/python"
CLUSTER_STEP12_SCRIPT="/gpfs/share/home/2306391536/projects/cell_seg/delivery_nuclei3d_pipeline/scripts/run_step12_pipeline.py"
CLUSTER_STEP3_SCRIPT="/gpfs/share/home/2306391536/projects/cell_seg/delivery_nuclei3d_pipeline/scripts/run_infer_3d.py"

echo "============================================================"
echo "Importing local next_round_upload package"
echo "UPLOAD_DIR       : $UPLOAD_DIR"
echo "REPO_ROOT        : $REPO_ROOT"
echo "BASELINE_PATH    : $BASELINE_PATH"
echo "CROP_MANIFEST    : $CROP_MANIFEST_PATH"
echo "RESULTS_PATH     : $RESULTS_PATH"
echo "============================================================"

CMD=(
  python scripts/import_local_candidates.py
  --upload-dir "$UPLOAD_DIR"
  --baseline "$BASELINE_PATH"
  --crop-manifest "$CROP_MANIFEST_PATH"
  --results "$RESULTS_PATH"
  --output-root "$REPO_ROOT/runs"
  --cluster-repo-root "$CLUSTER_REPO_ROOT"
  --cluster-python "$CLUSTER_PYTHON"
  --cluster-step12-script "$CLUSTER_STEP12_SCRIPT"
  --cluster-step3-script "$CLUSTER_STEP3_SCRIPT"
  --cluster-results-path "$RESULTS_PATH"
  --cluster-setup-line "source ~/.bashrc"
  --cluster-setup-line "micromamba activate cpsm"
)

if [ -n "$CAMPAIGN_NAME" ]; then
  CMD+=(--campaign-name "$CAMPAIGN_NAME")
fi

"${CMD[@]}"

# 找到刚导入的 campaign
if [ -n "$CAMPAIGN_NAME" ]; then
  NEW_CAMPAIGN="$CAMPAIGN_NAME"
else
  NEW_CAMPAIGN="$(python - <<'PY'
from pathlib import Path
runs = Path("runs")
cands = []
for p in runs.iterdir():
    if p.is_dir() and p.name.startswith("campaign_"):
        suffix = p.name[len("campaign_"):]
        if suffix.isdigit():
            cands.append((int(suffix), p.name))
print(sorted(cands)[-1][1] if cands else "")
PY
)"
fi

if [ -z "$NEW_CAMPAIGN" ]; then
  echo "Could not determine newly created campaign name."
  exit 1
fi

echo "============================================================"
echo "Submitting imported campaign"
echo "NEW_CAMPAIGN     : $NEW_CAMPAIGN"
echo "============================================================"

bash "runs/$NEW_CAMPAIGN/submit_all.sh"

echo "============================================================"
echo "Next round submitted successfully"
echo "campaign         : $NEW_CAMPAIGN"
echo "submit script    : runs/$NEW_CAMPAIGN/submit_all.sh"
echo "============================================================"