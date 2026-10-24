#!/usr/bin/env bash
#SBATCH --job-name=ougd
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
set -euo pipefail
# Run from repository root on a shared filesystem, with the CUDA venv activated.
mode=${1:?Usage: sbatch scripts/slurm.sh train|measure ARGS...}
shift
case "$mode" in train|measure) ;; *) exit 2 ;; esac
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c '
 exec python -m torch.distributed.run \
   --nnodes="$SLURM_NNODES" --nproc-per-node="$GPUS_PER_NODE" \
   --node-rank="$SLURM_PROCID" --rdzv-id="$SLURM_JOB_ID" \
   --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
   -m "$@"
' _ "ougd.$mode" "$@"
