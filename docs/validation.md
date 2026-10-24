# 本机实现验证记录

日期：2026-10-24。CPU-only；Python 3.12.14、PyTorch 2.14.1+cpu、Gloo 双进程。以下均为实现 smoke/regression，非论文实验结果。

- `python -m unittest discover -s tests -v`：4 项通过，含独立 dense 小坐标 Hessian、全部十五组合 LA/SA、六组 cross、signed 回和、ST/TB 与不同有效 token 数聚合、文档拆分及 packing。
- `torchrun --standalone --nproc-per-node=2 -m tests.distributed_check`：TB 的分布式梯度/HVP 与单进程真值一致（绝对容差 1e-11）。
- smoke 模型双进程 DDP/单进程各训练两步：最大权重差 3.7252903e-09。
- 从更新前 step 1 的 model/optimizer 恢复并重跑该步：与连续训练最终权重逐元素一致。
- ST 单进程和 TB 双进程均实际运行全部十五组 LA/SA、六组 cross SVD、self-only、六组 OUGD cross ablation、随机投影。该 smoke 使用 k=1，不代表正式 k=10 验收。
- TB 最终测量使用 16 probes、FP64 curvature、projection row/column rank 2，所有 72 个已记录数值项目通过；未进行跨 seed 稳定性正式验收。
- TB 重放梯度相对于 FP32 训练原始梯度误差：2.1265786e-07；signed contribution 最大回和绝对误差：9.7144515e-17。
- 另跑双进程 ST norm proxy，验证无本地样本的 worker 正确参与 collective。
- 成功导出 6 张 Parquet 表及 SVG 范数热图；Slurm 脚本通过 bash 语法检查。

未验证：CUDA/NCCL、多节点资源调度、真实 FineWeb 下载/远端 tokenizer、BF16 GPU 精度、S1/S2 的速度/显存/正式科学结果。测试输出位于被 git 忽略的 `runs/smoke/`；命令和运行方法见 README。
