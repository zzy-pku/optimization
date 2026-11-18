# Llama 120M EMA-Shampoo experiment

The experiment trains M (84 matrices) and U (36 per-layer QK/VO/FFN groups) from one saved initial state. Each analysis checkpoint uses the same calibration sequence for M, U, and the G reference over the 99,090,432 Transformer matrix coordinates. Embedding, output head, and RMSNorm parameters use AdamW. The two-sided matrix powers are applied in the packed layout with D transposed and are mapped back to the original parameter coordinates.

## Run

Use the repository's PyTorch, SciPy, and optional plotting dependencies with `pip install -e '.[plots,test]'`. The full-domain eigensolves also require a Python PRIMME build with 64-bit workspace indexing. Training defaults to two ranks. Analysis is single-rank and uses disk-backed Hessian eigenvectors.

```bash
export TOKENIZER=/path/to/the/original/tokenizer.model
export DATA_DIR=/data/zzy/twelve_layer_experiment/data
export RUN_DIR=runs/llama120m_shampoo
bash scripts/run_llama120m_shampoo.sh pilot
bash scripts/run_llama120m_shampoo.sh train
bash scripts/run_llama120m_shampoo.sh analyze
```

The pilot runs each shared peak LR candidate for 400 updates. Its operational stability thresholds are a finite loss and update ratio, an update/parameter norm ratio at most 0.1, a root action error at most 1e-5, and no 20-step post-warmup median loss above twice the end-of-warmup median. The selection file records all six results. Both formal runs restart from `init.pt` at the largest LR that passes for both layouts.

The training manifest records calibration indices and hashes, the data, tokenizer and FP1 hashes, and the partition. The pilot and full runs log training and validation cross-entropy, gradient and update norms, optimizer time, throughput, and peak CUDA allocation. The periodic `latest.pt` and numbered checkpoints are both valid resume inputs.

For legacy pretokenized arrays, set `PRETOKENIZED_PROVENANCE` instead of `TOKENIZER`. The JSON manifest must include `data_sha256` and `source`. The trainer validates the hash and records the manifest alongside a null tokenizer hash. The launcher uses the previous 120M experiment's validation interval of 3000 steps. Set `VALIDATION_EVERY` to change the interval.

## Analyze one checkpoint

```bash
python -m ougd.shampoo_measure \
  --checkpoint "$RUN_DIR/train-M/checkpoint_003000.pt" \
  --data "$DATA_DIR/train.npy" --fp1 "$DATA_DIR/FP1.pt" \
  --output "$RUN_DIR/train-M/analysis/003000/main" \
  --hessian-cache "$RUN_DIR/train-M/hessian_cache"
```

Use `--stages operator` to measure isotropic and native-momentum inputs first. The other stages are `top`, `coverage`, `hessian_heatmaps`, `factor_heatmaps`, `effective`, and `effective_spectrum`. Reuse the output folder and identical settings to add stages. Use a separate output folder and the same Hessian cache for each W/ridge variant. The full launcher measures W=16 and W=8 with ridge 1e-7 and 1e-5 at steps 3000 and 12208 and writes `comparison.json`. For a staged run, inspect the stage statuses and the raw samples in `metrics.json`.

The Hessian uses the verified FP1 token-mean CE batch, float32 HVPs without TF32, and algebraic positive eigenpairs. The 720-dimensional comparison gives M 9 modes per attention matrix and 8 per FFN matrix, and U 20 per group. If one scheme has fewer accepted modes, the matched-rank comparison drops the lowest retained local eigenvalues from the larger space. Hessian heatmaps include the norm, squared fraction, matrix-element RMS, and squared-norm standard error per block pair.

The G left factor uses a float64 Gram eigendecomposition of the weighted history. The implementation omits Gram modes at or below 4e-8 times the ridge, reports the maximum omitted eigenvalue and gain bound, and applies the ridge-only gain on the orthogonal complement. S uses the inverse eighth-root factors. The two operator comparisons using random inputs report paired-bootstrap intervals, and S reports estimates of the first two spectral moments plus both spectral edges. Step 0 checkpoints report null momentum metrics.

Measure eigensolver, HVP and root-refresh memory and throughput on the target system before scaling. A failed stage is recorded as incomplete, with completed stages retained. Each result carries the analysis scope, so outputs from reduced development models can be identified.
