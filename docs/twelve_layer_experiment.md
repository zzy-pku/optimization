# 12 层 Transformer 跨层 Hessian 实验

## 方案与口径

模型使用 `configs/twelve_layer.json`：12 层、宽度 288、FFN 768、6 个 query heads / 2 个 KV heads、词表 16000、上下文 512，pre-RMSNorm、RoPE、SwiGLU、tied embedding，线性层无 bias，权重初始化 N(0, 0.02²)。每层 7 个矩阵 Q/K/V/O/U/G/D 共 884,736 个参数，12 层总计 10,616,832 个被测参数。各层 RMSNorm、embedding 和最后 norm 不在 Hessian 坐标中，训练时仍正常更新。相对原 4 层宽度 384 模型，深度和被测矩阵总参数量增加，但宽度缩小；不能作为只改变深度的严格对照。

建议复用 FineWeb 训练数据及独立 validation packed rows，AdamW 全参数，seed 17，global batch 8192 tokens，BF16 训练，峰值学习率 4e-4、末值 4e-5、warmup 300、weight decay 0.1、betas=(0.9,0.95)，共 12208 步（约 1 亿 token）。在 0、3000、6000、9000、12207 步保存更新前快照，另保存 12208 步更新后的 `latest.pt`；最后的测量以 `latest.pt` 为准。`pre_step_012207.pt` 比最后 checkpoint 少一次更新。正式训练前应先单独评估显存、吞吐与时长；本方案不保证指定时间内结束。

测量将模型设为 FP32、禁 TF32，以固定 held-out FP1 的 token 平均交叉熵定义 Hessian，不含 weight decay、优化器、裁剪或预条件；排除所有非七矩阵参数。训练批次与 held-out 的 packed 行来自既有划分，具体文档隔离以原始数据元数据为准。全局指 84 个矩阵组成的 Hessian 主子矩阵，而非所有模型参数的 Hessian。默认单层第 6 层，双层第 6 和第 7 层，编号从 1 开始，均可改。

每个 self 及联合目标求 LA top 21，记录 top20、残差、正值认证与第 20/21 谱隙。目标分别为 84 矩阵整体、第 6 层、第 6/7 两层；候选为各矩阵 self top20 的直和，另以各层联合 self top20 直和覆盖整体。每块记录逐模式和均值的参与度、覆盖贡献、条件块内捕获率。第 20/21 截断处近重根时相关数字是初步结果；缺少 20 个已认证正特征对时不生成覆盖结论。整体、各层和各矩阵谱均单独保存。H_ij 的绝对 F 范数来自每列 32 个独立 Rademacher 探针的平方范数估计，两个方向在平方范数上平均后开方；结果、平方、平方估计标准误、方向估计和原始样本全部保存。平方估计无偏，开方后的范数估计有偏。全部谱符号均包含在范数中，所有图像的颜色使用对数尺度但数值未归一化。图上格子不写数值，CSV 包含全部 7056 项。对小值或相对标准误大的块，应使用独立 seed 和更多探针复核。

## 训练与测量

以下路径需根据服务器环境设置；执行训练前请确认数据文件和磁盘空间：

```bash
export PY=/qiyuan_research_vepfs_001/opt/server_validation_20261025/.venv/bin/python
export DATA=/qiyuan_research_vepfs_001/opt/s1_m1m2_20261025/data/packed
export ROOT=/qiyuan_research_vepfs_001/opt/twelve_layer_hessian_20261113
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT/repo"
"$PY" -m ougd.train --config configs/twelve_layer.json --data "$DATA/train.npy" \
  --output "$ROOT/train" --steps 12208 --batch-tokens 8192 --microbatch 1 \
  --layer 5 --measure-blocks seven --precision bf16 --optimizer adamw \
  --seed 17 --measure-steps 0,3000,6000,9000,12207 --save-every 250

export PROBE=/qiyuan_research_vepfs_001/opt/seven_optimizers_20261105_1517/fixed_probes/FP1.pt
"$PY" -m ougd.twelve_measure --checkpoint "$ROOT/train/latest.pt" --probe-batch "$PROBE" \
  --output "$ROOT/measurement" --single-layer 6 --pair 6 7 --k 20 --probes 32 --microbatch 1
"$PY" -m ougd.twelve_plot "$ROOT/measurement"
```

`--phase spectra` / `frobenius` / `coverage` 可分阶段执行；相同参数重新运行可跳过已完成的谱或逐列范数。不要在同一目录并发写；参数、输入哈希、探针数或求解容差变化时必须使用新目录。未收敛的谱仍保存诊断，不应解读为已认证结果；需在新目录重新求解并检查每个 boundary 和 residual。`metadata.json` 标注数据哈希、checkpoint 步数和坐标；`operator_checks.json` 给出对称性、线性及逐样本 token 聚合检查。所有谱对应同一个 probe、checkpoint 和损失。

84 列×32 次 HVP 仅为范数部分；另需 98 个 Lanczos 求解（全局、12 层、双层、84 矩阵），每次会多次 HVP。全局 eigensolver 在 CPU 内存保留多份长度 10616832 的向量，可能需要数十 GB 内存及大量存储和 GPU 时间。该实现逐测量任务串行，支持中断恢复，不保证能在一张 A100 上于短期内完成。正式启动前应计时完整尺寸的少量 HVP、评估内存与预估总时长；若内存或运行时间不允许，需要增加资源或修改实现，而不是把 smoke test 耗时外推。

## Smoke test

`ROOT=... PY=... bash scripts/twelve_smoke.sh` 使用 12 层微型网络、合成数据、2 个训练更新、k=1 和 2 个探针，验证管线和文件，不能当作正式精度或吞吐证明。脚本要求空的 smoke 输出目录。更多数值验证在 `tests/test_twelve_measure.py`。
