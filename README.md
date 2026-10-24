# Transformer O/U/G/D Hessian 实验（M1 / M2）

本仓库依据 [实验协议 v2.0](docs/protocol_v2.md) 实现训练与 **M1、M2**；M3/M4 观测不在当前实现范围。层号在命令行中从 **0** 开始，因此协议 Layer 6 使用 `--layer 5`。

## 已实现

- Llama-style decoder：Pre-RMSNorm、RoPE、GQA、SwiGLU、tied embedding；S0/S1/S2 配置及独立 smoke 配置。
- `torchrun` 单机/多机 DDP 训练、BF16/FP32、AdamW、warmup/cosine、global gradient clipping、验证 CE、T0 日志及恢复。
- **更新前**测量检查点：模型/optimizer/RNG、global batch 的所有 tokens/labels、数据索引和游标、原始梯度、实际更新、文件 hash。checkpoint 内含实际 batch，不依赖原始语料仍在远端。
- 固定 ST / 完整 TB / 显式 FP 的 matrix-free HVP。TB 按 global 有效 prediction-token 总数归一化，各 rank 重放自己的 microbatches，再 SUM all-reduce；不平均 token 谱。ST 保留完整上下文。
- M1.1：十块 Frobenius norm²/SE、norm、delta-method SE、element RMS、coupling、cross energy、Hutchinson trace、梯度/更新方向的 signed curvature；行列外积投影及 Kronecker rank-r 误差、两套随机基对照。
- M1.2–4：四个 self 及十五种参数集合的 LA/SA（SciPy ARPACK implicitly restarted Lanczos）；六组 cross 的 symmetric-dilation SVD 和双侧 residual；谱间隔、正交误差、向量、self/cross principal angles、联合参与比例/尺寸 enrichment、十项 signed contribution、self-only、六组 cross 成对置零、interlacing margins。
- rank 0 统一产生随机 probes、驱动求解器并向其他 rank 广播 HVP 请求；不会由多个独立求解器发出不一致的 collective。
- JSON/NPZ 原始输出、coverage、Parquet 表与 SVG heatmap 导出。FP/T1 proxy、投影、未测/预算跳过、未收敛分别标记。

## 安装

在共享文件系统中的仓库目录执行，所有节点使用同一代码、数据和 Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
# GPU 集群先按照 https://pytorch.org/get-started/locally/ 安装匹配驱动的 CUDA PyTorch。
# 当前开发环境只有 CPU，因此验证时安装的是 CPU wheel，不能直接作为 GPU 环境使用。
pip install -e '.[data,reports]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

依赖只允许正常的 TLS/签名校验。无需 API key。下载 FineWeb/tokenizer 时需要相应 Hugging Face 网络访问；也可完全使用本地 JSONL/tokenizer。

## 准备数据

tokenizer 未在协议中固定，必须明确选择；S2 要求所有 token ID 小于 32000。可以使用自己的本地 32K tokenizer。准备程序不会训练或更换 tokenizer。

```bash
python -m ougd.prepare --fineweb --documents 1000000 \
  --tokenizer /shared/tokenizer-32k --context 2048 --vocab 32000 \
  --output /shared/data/fineweb
# 或 --jsonl /shared/text.jsonl（每行含 text 和可选 id）替换 --fineweb。
```

按文档内容 hash 先划分 train/validation，重复内容进入同一 split，再分别串接 EOS 和 packing。输出 memory-mapped `[N, context+1]` 整数 `.npy`，模型输入为前 context 个 token，labels 为后 context 个。相邻行重叠一个边界 token，不重复计算 prediction。尾部不足一行的 tokens 丢弃并在计数中排除。documents.jsonl 保存文档 ID、hash、split、token 数和 stream offset，可据 batch index/context 还原文档/position；metadata.json 保存 tokenizer 和数据 hash。默认 holdout 为文档的 1%，**不承诺自动等于 10M validation tokens**。正式实验应冻结 HF revisions 为 commit，并按实际 token 数准备足够数据。

## 训练

```bash
# 单机 8 卡；131072 / 2048 = 64 sequences，每 rank 累积 8 个 microbatch。
torchrun --standalone --nproc-per-node=8 -m ougd.train \
  --config configs/s2.json --data /shared/data/fineweb/train.npy \
  --validation-data /shared/data/fineweb/validation.npy \
  --output /shared/runs/s2 --steps 15259 --batch-tokens 131072 \
  --microbatch 1 --layer 5 --precision bf16
```

`batch-tokens` 必须整除 `context * world_size * microbatch`，不静默缩小 batch。没有 dropout，batch 游标由 `step * global_sequences` 唯一确定；数据按固定预处理顺序循环，耗尽后会重复，因此正式 2B-token 实验需要准备足够的唯一 tokens。15259 steps 为协议的约数，最终 token 计数以 t0.jsonl 为准。

`latest.pt` 是**更新后**恢复点（默认每 250 steps 及最终保存，可设 `--save-every`）；`pre_step_XXXXXX.pt` 是**更新前**测量点，其 updates 是这一步执行后回填的真实变化。只从可信来源加载 `.pt`。恢复命令保持原训练 horizon、LR、batch、配置和数据不变，加 `--resume /shared/runs/s2/latest.pt`。更改 horizon 会改变 cosine schedule，属于新实验。

`--measure-steps` 控制保存点，默认包含协议候选点和最终更新前一步；如需 T1 全程采样，传入包含每 250–500 steps 的列表。观测采用离线重放，**不会在训练循环里隐式插入昂贵谱测量**。

## M1 测量

```bash
# 双口径分别运行，output 必须为空，防止不同算子混写。
torchrun --standalone --nproc-per-node=8 -m ougd.measure \
  --checkpoint /shared/runs/s2/pre_step_000300.pt --scope TB \
  --output /shared/results/step300/TB --microbatch 1 \
  --k 10 --probes 16 --tol 1e-4 --maxiter 120

torchrun --standalone --nproc-per-node=8 -m ougd.measure \
  --checkpoint /shared/runs/s2/pre_step_000300.pt --scope ST --token 0:100 \
  --output /shared/results/step300/ST_seq0_pos100 --k 10 --probes 16

python -m ougd.report /shared/results/step300/TB
```

默认分析全部十五种集合，含所有 self-only、六组全联合 cross 消融、行列基各 4 维的两次投影。S2 建议先缩小配置/预算 profile：`--sets O,U,G,D,OU,OG,OD,UG,UD,GD,UGD,OUGD --projection-rank 0 --no-ablation`，省略项在 coverage 中记录。`--maxiter` 是 ARPACK restart 迭代预算，**不是 HVP 次数**；实际次数在 solver_matvecs 和 operator_meta.matvecs 中记录。向量内积/ARPACK 在 rank 0 CPU 上进行；HVP 在 GPU 上运行，整套参数向量需在设备间传送。

T1 使用固定 FP batch（保存含 input_ids/labels 的 torch 字典）和 `--scope FP --probe-batch fixed.pt --norms-only --probes 4`。可对预先固定的多个 ST tokens 分别调用 `--token`；当前不自动挑选 loss/confidence 分层 cohort，也不把不同 tokens 的谱相加。不同 seeds/precision 的复核使用不同 output。`--precision fp64` 用于小模型真值核验；正式 curvature 默认 FP32。

`operator_meta.json` 保存 scope、完整 batch/checkpoint hash、参数形状与 row-major O/U/G/D flatten 顺序、环境和成本。TB 记录 FP32/64 CE 重放梯度与训练原始梯度的差异，预设容差为 FP32 `1e-5` / BF16 `0.05`，超阈值会标记 `gradient_replay_passed=false`，必须调查后才可用于正式结论。BF16 训练梯度与 FP32 CE 导数不是逐 bit 相同。

`vectors/*.npz` 保存 eigenvectors / cross 左右向量及 residual；不生成完整 Hessian。范数少于 16 probes 标作 pilot，sqrt estimator 有偏，coupling 分母近零记 null。谱即使输出了数值也需检查 `converged` 和 residual；近简并需比较整个 subspace，不能仅凭单方向解释。投影稳定性只针对所选随机基，结构结论不能外推完整参数空间。重复 seed 稳定性由独立运行对照，当前不自动执行复核或冻结研究结论。

## 多机 / Slurm

`scripts/slurm.sh` 是每节点 8 卡、2 节点示例；按集群 partition/account/GPU 数修改 `#SBATCH`，并让 `GPUS_PER_NODE` 与资源申请一致：

```bash
sbatch scripts/slurm.sh train --config configs/s2.json \
  --data /shared/data/fineweb/train.npy --output /shared/runs/s2 \
  --layer 5 --steps 15259 --batch-tokens 131072
sbatch scripts/slurm.sh measure --checkpoint /shared/runs/s2/pre_step_000300.pt \
  --scope TB --output /shared/results/step300/TB
```

无 Slurm 时每台节点启动同一条 torchrun 命令，使用相同 `--nnodes` / `--rdzv-id` / `--rdzv-endpoint=主节点地址:端口`，各节点传对应 `--node-rank`。网络需支持 PyTorch rendezvous 和 NCCL 节点通信。训练复制模型与 optimizer，测量复制模型；本版本使用数据并行，**不支持 FSDP/张量并行**。如果单卡无法容纳一份模型及二阶图，需要先降低 microbatch/context/模型规模并 profile，不能指望增加卡数自动拆分单样本图。ST 只有一个 sequence，因此分布式 ST 的 HVP 实际只在一张卡计算，其他卡参与规约；TB 才能随 sequence 数并行。

## CPU 可复现验证

```bash
python -m unittest discover -s tests -v
torchrun --standalone --nproc-per-node=2 -m tests.distributed_check
```

数值测试使用 tiny FP64 Transformer 的小坐标 dense 真值，验证十五种子集 LA/SA、六组 cross SVD、signed 回和、Hessian 对称性、有效 token 加权及 ST→TB 聚合。此为 **M1 的 S0 子集**，不宣称完成协议中 M2–M4 的 S0 验收。

开发机为 CPU-only。已实际跑通双进程 DDP 训练、ST/TB 各十五种 joint、self-only/六组消融和投影。这里只证明实现路径和数值回归；**未运行 GPU/NCCL、多节点、FineWeb 下载或 S1/S2 正式训练**。正式运行前应使用目标 CUDA 环境、小规模配置和真实 context/batch 对 HVP 显存/耗时、BF16 梯度重放及通信进行 pilot，不将 smoke 性能外推到 100M 模型。


## M2：activation-anchor A/R 分解

M1 基线已保存在 `main` 的 `3449b51`（commit message: `version 1`）；M2 在独立 `m2` 分支开发。

```bash
torchrun --standalone --nproc-per-node=8 -m ougd.measure --module M2 \
  --checkpoint /shared/runs/s2/pre_step_000300.pt --scope TB \
  --output /shared/results/step300/M2_TB --k 10 --probes 16
# ST 使用相同 checkpoint，改为 --scope ST --token 0:100，并更换 output。
python -m ougd.report /shared/results/step300/M2_TB
```

锚点是所选 `blocks[layer]` 完成 FFN residual 相加后的全部 hidden states。完整前缀和后续网络均参与计算。`A = J^T B J` 使用 forward AD JVP、激活 BVP 和参数 VJP；`R = H-A`。A 是 activation-curvature propagation term，**可能不定，不是 PSD logits GGN**。每次 M2 运行先用独立 `Hessian(<stopgrad(q),z>)` 核验 residual，再开始观测；独立核验不通过时停止并留下失败记录。

输出内容：

- `H/`、`A/`、`R/`：各自十块 norm/SE、四 self、六 cross SVD、十五 joint 的 LA/SA、参与比例、signed block terms 及同一 seed 下的行/列投影。所有组件共享随机 probes 和投影基。
- `component_alignment.json`：A/R 内积、cosine、A/H 与 R/H norm ratio、配对 Frobenius 回和误差；各组件 cross energy 在子目录。
- `H_direction_contributions.json`：**固定 H 的原始 eigenvector** 后的 q^T A q、q^T R q、20 个有符号块项与回和误差。不将 A/R 排名相同的 eigenvalues 相加。
- `H_cross_contributions.json`：固定 H 的 singular triple，保存 u^T A v、u^T R v 及回和误差。
- `component_subspaces.json`：H/A/R 相同维数空间的 principal angles/overlap，以及 R→A、A→R 的绝对/相对 leakage 和谱间隔；cross 左右空间分别比较。
- `decomposition_checks.json` 和 `coverage.json`：独立 residual、对称性、数值容差及完成/未收敛/未测状态。独立核验默认 FP64 1e-9、FP32 1e-4，未根据测量结果放宽。

对当前 SwiGLU block-output anchor，z 对 U、D 分别仿射，所以 `R_UU=R_DD=0`。这两块使用解析零谱，保存的 canonical vectors 仅是任意零空间基；相关 sharp-space overlap/leakage 的不可识别比较记为 null，不据此解释方向。

`--sets`、`--k`、`--probes`、`--projection-rank` 和 `--norms-only` 同样适用。预算裁剪在 coverage 中保留。M2 不重复 M1 的逐 cross 消融；M4 的成分插值/因子消融仍未实现。R matvec 通常需要 H 和 A 两条路径，M2 的计算量显著高于 M1；当前仍重放真实 TB，不缓存完整二阶图，也不以小 probe batch 冒充 TB。

额外分布式回归：

```bash
torchrun --standalone --nproc-per-node=2 -m tests.distributed_components
```

这项测试逐一核对 TB/ST 下的 H、A、R、R_direct 与串行计算，包括 ST 空数据 worker。CUDA/NCCL 和多节点仍需要目标 GPU 集群实测。
