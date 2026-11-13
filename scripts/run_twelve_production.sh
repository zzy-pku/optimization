#!/usr/bin/env bash
# Sequential single-GPU training and final-checkpoint measurement; safe to resume.
set -Eeuo pipefail
ROOT="${RUN_ROOT:?Set RUN_ROOT to the dedicated run directory}"
PY="${PY:-/qiyuan_research_vepfs_001/opt/server_validation_20261025/.venv/bin/python}"
DATA="${DATA:-/qiyuan_research_vepfs_001/opt/s1_m1m2_20261025/data/packed}"
PROBE="${PROBE:-/qiyuan_research_vepfs_001/opt/seven_optimizers_20261105_1517/fixed_probes/FP1.pt}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(mkdir -p "$ROOT" && cd "$ROOT" && pwd)"
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/pipeline.lock"
flock -n 9 || { echo 'Pipeline already running for this directory' >&2; exit 1; }
cd "$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
phase=initializing
status() {
  local state="$1"
  printf '{"state":"%s","phase":"%s","pid":%s,"updated_utc":"%s"}\n' \
    "$state" "$phase" "$$" "$(date -u +%FT%TZ)" > "$ROOT/status.json.tmp"
  mv -f "$ROOT/status.json.tmp" "$ROOT/status.json"
}
trap 'rc=$?; status failed; echo "Pipeline failed in $phase with exit code $rc" >&2; exit "$rc"' ERR
status running
commit=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Refusing to start with a modified code checkout' >&2; false
fi
"$PY" - "$ROOT" "$DATA" "$PROBE" "$commit" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,data,probe,commit=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
manifest={'commit':commit,'train_sha256':sha(data/'train.npy'),
          'validation_sha256':sha(data/'validation.npy'),'probe_sha256':sha(probe),
          'steps':12208,'batch_tokens':8192,'train_microbatch':4,'measurement_microbatch':4,
          'seed':17,'single_layer':6,'pair':[6,7],'k':20,'probes':32}
p=root/'run_manifest.json'
if p.exists():
    if json.loads(p.read_text()) != manifest:
        raise SystemExit('Run manifest changed; use a new run directory')
else: p.write_text(json.dumps(manifest,indent=2)+'\n')
PY
phase=training
status running
resume=()
if [[ -f "$ROOT/train/latest.pt" ]]; then
  resume=(--resume "$ROOT/train/latest.pt")
fi
"$PY" -m ougd.train --config configs/twelve_layer.json \
  --data "$DATA/train.npy" --validation-data "$DATA/validation.npy" --validation-every 3000 \
  --output "$ROOT/train" --steps 12208 --batch-tokens 8192 --microbatch 4 \
  --layer 5 --measure-blocks seven --precision bf16 --optimizer adamw \
  --lr 0.0004 --min-lr 0.00004 --weight-decay 0.1 --beta1 0.9 --beta2 0.95 \
  --warmup 300 --seed 17 --measure-steps 0,3000,6000,9000,12207 --save-every 250 \
  "${resume[@]}" >> "$ROOT/logs/train.log" 2>&1

phase=verify_final_checkpoint
status running
"$PY" - "$ROOT/train/latest.pt" <<'PY'
import sys,torch
s=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
if s['step'] != 12208 or s['config']['layers'] != 12:
    raise SystemExit('Final checkpoint is not the expected 12-layer post-update step 12208')
print('verified final post-update checkpoint at step',s['step'])
PY
common=(--checkpoint "$ROOT/train/latest.pt" --probe-batch "$PROBE" --output "$ROOT/measurement"
        --single-layer 6 --pair 6 7 --k 20 --probes 32 --microbatch 4)

# Build the F-norm heatmap first so it is available even if a later spectrum needs a retry.
phase=frobenius
status running
"$PY" -m ougd.twelve_measure "${common[@]}" --phase frobenius >> "$ROOT/logs/frobenius.log" 2>&1
"$PY" -m ougd.twelve_plot "$ROOT/measurement" >> "$ROOT/logs/plot.log" 2>&1
phase=spectra
status running
"$PY" -m ougd.twelve_measure "${common[@]}" --phase spectra >> "$ROOT/logs/spectra.log" 2>&1
"$PY" -m ougd.twelve_plot "$ROOT/measurement" >> "$ROOT/logs/plot.log" 2>&1
phase=coverage
status running
"$PY" -m ougd.twelve_measure "${common[@]}" --phase coverage >> "$ROOT/logs/coverage.log" 2>&1
phase=completed
status completed
