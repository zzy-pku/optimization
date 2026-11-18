#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_llama120m_shampoo.sh pilot|train|analyze|all}
case "$mode" in pilot|train|analyze|all) ;; *) exit 2 ;; esac
data_dir=${DATA_DIR:-/data/zzy/twelve_layer_experiment/data}
root=${RUN_DIR:-runs/llama120m_shampoo}
python=${PYTHON:-python}
config=configs/llama2_120m.json
world=${WORLD_SIZE_TRAIN:-2}
validation_every=${VALIDATION_EVERY:-3000}
mkdir -p "$root"

if [[ "$mode" != analyze ]]; then
  if [[ -n "${TOKENIZER:-}" && -z "${PRETOKENIZED_PROVENANCE:-}" ]]; then
    input_source=(--tokenizer "$TOKENIZER")
  elif [[ -z "${TOKENIZER:-}" && -n "${PRETOKENIZED_PROVENANCE:-}" ]]; then
    input_source=(--pretokenized-provenance "$PRETOKENIZED_PROVENANCE")
  else
    printf '%s\n' 'Set exactly one of TOKENIZER or PRETOKENIZED_PROVENANCE.' >&2
    exit 2
  fi
  if [[ ! -f "$root/init.pt" ]]; then
    "$python" -m ougd.shampoo_plan init --config "$config" --output "$root/init.pt"
  fi
fi

train() {
  local scheme=$1 peak=$2 minimum=$3 steps=$4 directory=$5
  printf 'Training %s with peak LR %s and target %s updates in %s\n' "$scheme" "$peak" "$steps" "$directory"
  local resume=()
  if [[ -f "$directory/latest.pt" ]]; then
    resume=(--resume "$directory/latest.pt")
  elif [[ -f "$directory/t0.jsonl" ]]; then
    mv "$directory/t0.jsonl" "$directory/interrupted.jsonl"
  fi
  "$python" -m torch.distributed.run --standalone --nproc-per-node "$world" -m ougd.train \
    --config "$config" --optimizer shampoo --shampoo-partition "$scheme" \
    --initial-state "$root/init.pt" --data "$data_dir/train.npy" \
    --validation-data "$data_dir/validation.npy" --validation-every "$validation_every" \
    "${input_source[@]}" --fp1 "$data_dir/FP1.pt" \
    --output "$directory" --steps "$steps" --measure-steps 0,300,3000,6000,9000,12208 \
    --batch-tokens 8192 --microbatch 4 --lr "$peak" --min-lr "$minimum" "${resume[@]}"
}

if [[ "$mode" == pilot || "$mode" == all ]]; then
  for peak in 0.0001 0.0002 0.0004; do
    for scheme in M U; do
      directory="$root/pilot-$scheme-$peak"
      if ! train "$scheme" "$peak" "0.0000${peak: -1}" 400 "$directory"; then
        printf 'Pilot %s %s failed\n' "$scheme" "$peak" >&2
      fi
    done
  done
  "$python" -m ougd.shampoo_plan select-lr --root "$root"
fi

if [[ "$mode" == train || "$mode" == all ]]; then
  peak=$("$python" -m ougd.shampoo_plan select-lr --root "$root")
  for scheme in M U; do
    train "$scheme" "$peak" "0.0000${peak: -1}" 12208 "$root/train-$scheme"
  done
fi

if [[ "$mode" == analyze || "$mode" == all ]]; then
  analyze() {
    local directory=$1 step=$2 variant=$3 window=$4 ridge=$5
    shift 5
    printf 'Analyzing %s at step %s, variant %s\n' "$directory" "$step" "$variant"
    "$python" -m ougd.shampoo_measure --checkpoint "$directory/checkpoint_$step.pt" \
      --data "$data_dir/train.npy" --fp1 "$data_dir/FP1.pt" \
      --output "$directory/analysis/$step/$variant" --hessian-cache "$directory/hessian_cache" \
      --window "$window" --ridge "$ridge" "$@"
  }
  status=0
  for scheme in M U; do
    directory="$root/train-$scheme"
    for step in 000000 000300 003000 006000 009000 012208; do
      analyze "$directory" "$step" main 8 1e-6 || status=1
      if [[ "$step" == 003000 || "$step" == 012208 ]]; then
        analyze "$directory" "$step" w16 16 1e-6 --stages operator top effective || status=1
        analyze "$directory" "$step" ridge_1e-7 8 1e-7 --stages operator top effective || status=1
        analyze "$directory" "$step" ridge_1e-5 8 1e-5 --stages operator top effective || status=1
      fi
    done
  done
  "$python" -m ougd.shampoo_plan summarize --root "$root"
  exit "$status"
fi
