# 单层 Transformer 七块正端覆盖与完整 F 范数

### 优化器对照

训练入口 `--optimizer` 支持 `adamw`（默认）、`adam`、`sgd`、`muon`、`soap`。默认 AdamW 超参数与旧入口一致：`--lr 4e-4 --beta1 0.9 --beta2 0.95 --eps 1e-8 --weight-decay 0.1`。Adam 的 weight decay 为耦合 L2，AdamW 和两个混合优化器为解耦衰减；SGD 可设置 `--momentum`、`--nesterov`。不同优化器的学习率需分别试探并记录。

Muon 和 SOAP 采用混合分组：所有 Transformer block 内二维矩阵用所选优化器，embedding（与输出权重绑定）及所有 RMSNorm 参数使用 AdamW；它们仍参与正常训练，只是不纳入七块 Hessian 坐标。`--lr` 是主优化器峰值学习率，`--aux-lr` 是辅助 AdamW 峰值学习率，两者共享 warmup/cosine 的相对进度，末端辅助 LR 等于 `aux-lr * min-lr / lr`。可设置 `--muon-momentum`（默认 0.95）、`--soap-precondition-frequency`（默认 10）和 `--soap-max-precond-dim`（默认 1024）；SOAP 默认 beta 与本实验 AdamW 相同，可通过 `--beta1`、`--beta2` 调整。SOAP 的预条件矩阵会增加显存和计算开销，官方实现首次 step 用于初始化预条件器，主矩阵当步不更新。`run.json` 保存参数分组，checkpoint 保存两部分状态，恢复时验证优化器类型与关键超参数；日志记录主 LR 和辅助 LR。

示例：在下述训练命令中追加 `--optimizer muon --lr 0.02 --min-lr 0.002 --aux-lr 0.0004`，或追加 `--optimizer soap --lr 0.003 --min-lr 0.0003 --aux-lr 0.0004`。这些数字仅为可运行的起点，不代表已调优或已验证收敛。比较时建议固定初始化种子、数据顺序、训练 token 和测量 probe，分别记录验证损失及真实运行配置。CE Hessian 不包含优化器、weight decay 和裁剪项。

Muon 和 SOAP 代码分别固定于 [Muon 官方实现 f98f1ca](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806) 与 [SOAP 官方实现 a1e5535](https://github.com/nikhilvyas/SOAP/tree/a1e553530fde97d0e6b307d7c82ac6d38b072340)，保留各自 MIT 许可；在 DDP 下使用同步梯度后每个 rank 各自执行同一更新，不是官方分片优化器。

本实验使用 block 内的完整 Q/K/V/O/U/G/D 权重矩阵。embedding 和 RMSNorm 在训练中照常更新，重放 Hessian 时固定；完整网络前向和loss路径仍保留。只求正端领先特征方向，不求负端特征对。F范数是完整 H/A/R 的块范数，包含所有谱符号；不构造正谱截断F。

设计范围、对照和可复现条件见 [实验方案](seven_block_plan.md)。推荐在独立Python环境安装依赖：

```bash
pip install -e '.[plots]'
python -m unittest discover -v
```

`seven_measure` 是单进程入口，自动使用可见CUDA设备或CPU；不要用多进程torchrun启动。现有四块接口和训练默认行为保持兼容。

## 制作固定 held-out probe

```bash
python -m ougd.make_fixed_probes \
  --data /data/packed/validation.npy --output /runs/fixed_probes \
  --context 512 --vocab 16000 --tokens 8192 --cohorts 3 --seed 101
```

输出FP1/2/3和元数据，保存packed行索引、数据与batch哈希。行索引不重叠；没有文档ID时不能声明文档级互斥。输出目录必须不存在。

## 训练并保存七块更新前快照

```bash
python -m ougd.train --config configs/seven_block_single_layer.json \
  --data /data/packed/train.npy --validation-data /data/packed/validation.npy \
  --output /runs/seven_seed17 --steps 12208 --batch-tokens 8192 \
  --microbatch 4 --layer 0 --precision bf16 --measure-blocks seven --seed 17 \
  --measure-steps 0,3000,12207 --save-every 250
```

`--measure-blocks seven` 保存七块raw gradients和更新；默认`four`保持旧版兼容。训练包含embedding、norm等所有模型参数，Hessian测量只选七块。方案中的3个seed分别为17、23、29；FP1测3个checkpoint，末期再测真实TB和FP2/FP3，共18个条件。先使用一个条件的pilot测算资源，命令不会自动启动正式实验。

## 单条件测量

```bash
python -m ougd.seven_measure \
  --checkpoint /runs/seven_seed17/pre_step_012207.pt \
  --scope FP --probe-batch /runs/fixed_probes/FP1.pt \
  --output /runs/seven_seed17_measure/step12207_FP1 \
  --microbatch 4 --k 20 --probes 64 --random-repeats 100
```

真实训练batch使用`--scope TB`且不传`--probe-batch`。默认要求8192个有效目标token，单层模型、预更新checkpoint和层号0。主算子为关闭TF32的FP32重放；FP64 Lanczos、正交化和统计；`--precision fp64`适合tiny验证。与BF16训练梯度的差异单独记录。

可选：

- `--equal-dimension-control`：四块额外求top35，与七块各top20的140维候选比较；正向数量不足时标记不可用。
- `--diagonal-equal-dimension-control`：按需扩展各self正端谱，计算与S7实际维数相同的D_7-top-d；逐块检查未返回边界及残差，无法认证时标明失败。它不会将每块top20的拼接冒称全局D_7-top140。
- `--cross-randomizations 3`：生成3个人工cross方向随机化对照。它使用块内独立带符号置换，保持self和各cross的F范数及奇异值；这是合成矩阵，不是训练模型的真实Hessian，也不是Haar旋转。每个随机化矩阵作用目前约需15次底层HVP，可能非常昂贵。
- `--save-hp`保存H对候选基的作用；`--skip-frobenius`仅用于调试并显式标记跳过。F区间过宽时使用新seed、新输出目录和128/256 probes做独立复核。输出目录存在会拒绝覆盖。

## 输出与状态

- `metadata.json`、`operator_checks.json`、`status.json`：参数顺序、权重与数据哈希、精度、算子检查与状态。预检查包括H=A+直接R、HVP线性/对称和token加权聚合。
- `spectrum_{Q,K,V,O,U,G,D,H4,H7}.npz`、`spectra.json`：LA值、残差、符号和截断边界；只保存数值确认正向的特征向量。未收敛结果不作为已认证领先子空间。
- `coverage.json`：S4→H4、S4→H7、S7→H7的containment、逐模式捕获、全主角范围、分块贡献、H7四块坐标参与上限、分块Haar基线。
- `diagonal_baseline.json`：D_4/D_7正端top-k；`equal_dimension_control.json`、`diagonal_equal_dimension_control.json`和`cross_randomization.json`按需输出。扩谱结果保存在`diagonal_refined_*.npz`。
- `projected_H.npz`、`ritz.json`：受限H小矩阵；分别对S4→H4、S4→H7、S7→H7计算Ritz值和目标范围内的全空间残差。S4小矩阵在两种目标下相同，但H7残差包含Q/K/V分量。保存HP需启用`--save-hp`。
- `frobenius.json`：28个独立块的H/A/R完整F范数、paired probe样本与bootstrap区间、A/R内积和F余弦、H4/H7汇总。区间过宽的H块被单独列出。

`preliminary_boundary_or_insufficient_positive` 表示正向数量不足或截断边界不稳定；需要补算/按谱簇复核，不能当成已确认结论。非正或符号未决的LA值仅作为边界诊断保留在元数据中。未收敛的谱在输出中标明并使入口返回失败，不会被默认为通过。

block-output锚点对U、D分别仿射，故 \(R_{UU}=R_{DD}=0\)。程序按解析零值报告这两个self块，同时保存未强制置零的H−A差分以供检查；其F余弦为NA。Q/K/V不套用此零块假设。

## 可视化

报告只读取已保存的结果，可在本地具有matplotlib的环境运行，不需重新计算HVP：

```bash
python -m ougd.seven_report /runs/seven_seed17_measure/step12207_FP1 \
  --output /runs/seven_seed17_report/step12207_FP1
```

可传入多个测量目录。输出每个条件单独的覆盖曲线、逐模式捕获与完整主角、H/A/R F范数热图、Ritz谱、`coverage_summary.csv`及`report.md`。不同seed、checkpoint或FP/TB不被自动当作独立样本合并；报告保留初步/不可用状态与数据哈希。
