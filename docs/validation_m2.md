# M2 实现验证

2026-10-24，CPU-only，FP64 curvature。M1 基线提交：`3449b51`，message `version 1`，位于 GitHub `main`。

- 12 项 unittest 全部通过：新增 dense J^T B J、独立 dense residual、H=A+R、ST/TB 聚合、末层 anchor、矩阵无关 A/R 极端谱、共享 probes 下的抵消/Frobenius 恒等式、leakage、解析零谱和退化求解处理；保留 M1 回归。
- `torchrun --standalone --nproc-per-node=2 -m tests.distributed_components`：TB、ST 的 H/A/R/R_direct 分别与串行结果一致，绝对容差 1e-11。包含 ST 空数据 worker。
- ST 串行与 TB 双进程均完成完整 M2 smoke：H/A/R 各十五组 LA/SA、六组 cross SVD、独立谱方向对齐、30 个 H 极端方向的 A/R 归因、6 个 H singular triple 归因、双向 leakage、投影。
- smoke 使用 k=1、4 probes、行列基 rank=1；coverage 每个 scope 为 250 complete、3 pilot、3 skipped_budget，零 failed_convergence。pilot 是低 probe 范数；skipped_budget 是 M2 不重复执行的三套 M1 cross ablation，未宣称完成 M4。
- 独立 H=A+R_direct 最大相对误差：ST 4.8175587e-17，TB 8.6037416e-17。
- 固定 H eigenvector 的 A/R signed 回和最大绝对误差：ST 4.4408921e-16，TB 9.7144515e-17。
- 固定 H singular triple 的回和最大绝对误差：ST 1.1102230e-16，TB 1.3877788e-17。
- H/A/R 的投影基逐元素一致；R_UU/R_DD 解析零谱明确标记 fully_degenerate，不解释任意零空间基为 sharp direction。
- 每个 scope 成功导出 23 个 Parquet 文件以及 H/A/R 范数热图。

运行命令示例（保留在 git 忽略的 runs/m2 中）：

```bash
torchrun --standalone --nproc-per-node=2 -m ougd.measure --module M2 \
  --checkpoint runs/smoke/train/pre_step_000000.pt --output runs/m2/TB_v2 \
  --scope TB --k 1 --probes 4 --projection-rank 1 --basis-repeats 1 \
  --tol 1e-5 --maxiter 120 --precision fp64
```

当前没有 GPU，尚未执行 CUDA/NCCL、多节点、正式 S1/S2 或跨 seed 科学复现；这些 smoke 结果不代表正式实验完成。A 可能不定，此实现没有将它当作标准 PSD GGN。

## M1+M2 联合运行回归

默认入口改为 `--module M1+M2`，保留显式 M1/M2 单项模式。

- unittest 共 14 项通过；新增调度测试验证 H/A/R 各运行一次、只对 H 启用 M1 cross 消融，并正确保留显式预算裁剪。
- 同一 tiny FP64 模型检查点上，ST 单进程及 TB 两进程（各 rank 有效 token 数为 3/2）均完成联合测量；TB 不传 module 参数，验证默认值，ST 显式传 `--module M1+M2`。
- 每个 scope 完成 H/A/R 各十五组 LA/SA、各六组 cross SVD、H 的十二个 cross 消融谱（六组×LA/SA）、M2 归因和子空间比较；coverage 为 262 complete、3 pilot、2 skipped_budget，无失败。
- k=1、4 probes、投影 row/column rank=1。pilot 为低 probe 范数；两项 skipped_budget 是不适用于完整 M1 观测的 A/R cross 消融，H cross 消融已执行。
- 逐项确认 H/A/R 的 checkpoint/batch hash、layer、scope、precision 一致；投影基逐元素相同；M2 的固定 H 方向引用指向唯一的 H/vectors 文件。
- 两个 scope 的联合结果各成功导出 23 个 Parquet 文件。输出位于 git 忽略的 runs/joint/{ST,TB}。

仍是 CPU/Gloo smoke，未增加 CUDA/NCCL、多节点或正式科学实验的验收声明。
