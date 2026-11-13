#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:?set ROOT to experiment directory}"
PY="${PY:?set PY to Python executable}"
cd "$ROOT/repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/smoke"
"$PY" - "$ROOT/smoke" <<'PY'
import sys
from pathlib import Path
import numpy as np
import torch
p = Path(sys.argv[1]); rng = np.random.default_rng(1729)
np.save(p/'train.npy', rng.integers(0, 32, size=(16, 9), dtype=np.int64))
probe = rng.integers(0, 32, size=(2, 9), dtype=np.int64)
torch.save({'input_ids': torch.tensor(probe[:, :-1].copy()),
            'labels': torch.tensor(probe[:, 1:].copy())}, p/'probe.pt')
PY
"$PY" -m ougd.train --config configs/twelve_layer_smoke.json --data "$ROOT/smoke/train.npy" \
  --output "$ROOT/smoke/train" --steps 2 --batch-tokens 16 --microbatch 1 --layer 5 \
  --precision fp32 --optimizer adamw --warmup 1 --measure-steps 0,1 --save-every 1
"$PY" -m ougd.twelve_measure --checkpoint "$ROOT/smoke/train/latest.pt" --probe-batch "$ROOT/smoke/probe.pt" \
  --output "$ROOT/smoke/measurement" --single-layer 6 --pair 6 7 --k 1 --probes 2 \
  --solver-tol 1e-5 --residual-tol 1e-3 --maxiter 400
"$PY" -m ougd.twelve_plot "$ROOT/smoke/measurement"
