---
title: "Transformer O/U/G/D Hessian：结构、分解与联合谱实验协议"
subtitle: "v2.0 · 按研究问题组织观测指标与计算方法，按模型规模实施"
date: "2026-10-24"
lang: zh-CN
---

> **文档性质：实验设计，尚无实测结果。** 本版以 v1.0 为蓝本重构研究模块和实施逻辑；保留十个 Hessian 块、两组件精确分解、激活 $B$/Jacobian $J$、15 种参数组合、两种主 loss 口径、训练全程追踪及 S0–S3 的规模安排。下文中“必测”指其所属规模阶段和指定深度测量点的覆盖要求，**不表示每个训练 step 都进行完整二阶分析**。

# 0. 研究目标与协议地图

## 0.1 研究问题

固定 decoder-only Transformer 的某一层，研究 attention output projection $O$ 和 SwiGLU 中的 up $U$、gate $G$、down $D$ 四个**完整参数矩阵**。不把单个权重矩阵拆成神经元、行或列作为研究基本单位；行/列基仅用于数值投影与结构诊断。

本研究分为四个**并行研究模块**，每个模块负责一组明确的科学问题和观测指标：

| 模块 | 核心问题 | 主要观测对象 | 方法主线 |
|:--|:--|:--|:--|
| **M1｜原始 Hessian 的块结构与联合谱** | 各矩阵内部有多大的曲率、矩阵间耦合多强、耦合如何改变联合 sharp directions？ | 10 个 $H_{ij}$、15 个 $H_S$ | 范数估计；self eigensolver；cross SVD；联合 eigensolver；self-only/cross ablation |
| **M2｜两项分解及其谱贡献** | $A^{(z)}$ 与 $R^{(z)}$ 各自具有怎样的大小、谱和方向？它们如何贡献 $H$ 的原始特征值与特征向量？ | 10 个 $A_{ij},R_{ij}$；$A_S,R_S$；相同 $H$ 方向上的贡献 | 成分范数/内积；独立谱；Rayleigh/双线性归因；子空间角度与 leakage |
| **M3｜激活曲率 $B$ 与 Jacobian $J$ 的机制** | 参数 Hessian 的传播项由什么激活曲率和参数—激活映射产生？ | 共同 activation anchor 的 $B^{(z)},J_O,J_U,J_G,J_D$；logits GGN 对照 | BVP、JVP/VJP、B 与 J 的独立谱、token 结构、$J_i^\top BJ_j$ 收缩 |
| **M4｜组件间联合作用与方向形成** | $A$ 和 $R$ 组合后为何产生特定联合谱？self 与 cross、$A$-cross 与 $R$-cross 分别造成什么变化？ | $A_S+tR_S$、cross-factorial 算子、共同候选子空间 | 连续插值；受控消融；Ritz 投影；谱轨迹与不变子空间比较 |

**贯穿四个模块的三个实验坐标轴**，不再另设重复的“样本模块”或“训练动力学模块”：

1. **Loss 口径**：单 token CE 与真实 optimizer-step train-batch 全有效 token 均值 CE；固定 probe 只用于附加对照。
2. **训练时间**：T0 每 step 常规日志、T1 跨全训练的低成本代理曲线、T2 指定 checkpoint 深度分析。
3. **规模阶段**：S0 数值真值测试、S1 全类别研究覆盖、S2 约 100M 主模型重点检验、S3 鲁棒性复核。

## 0.2 统一实验链路

在**同一模型、checkpoint、层、loss 口径和数据样本**上，先定义固定算子 $H$、$A$、$R$、$B$、$J$，再分别执行 M1–M4 的观测。不同模块可共享 matvec 与已收敛的特征向量，但**不按模块顺序划分实验阶段**。统计指标与谱指标始终从相同算子导出；不能先对单 token 特征值取平均再宣称得到训练均值的特征值。

---

# 1. 统一数学对象与记号

## 1.1 十块 Hessian 与十五种联合参数集合

记 $\mathcal I=\{O,U,G,D\}$、$w_i=\operatorname{vec}(W_i)$、$p_i=\dim w_i$。对给定标量 loss $\mathcal L$ 定义完整网络的参数 Hessian 子块

$$
H_{ij}=\nabla_{w_i}\nabla_{w_j}^{\mathsf T}\mathcal L\in\mathbb R^{p_i\times p_j},\qquad H_{ji}=H_{ij}^{\mathsf T}.
$$

十个独立块为四个 self：$OO,UU,GG,DD$；六个 cross：$OU,OG,OD,UG,UD,GD$。对任何非空 $S\subseteq\mathcal I$，$H_S=[H_{ij}]_{i,j\in S}$，共 15 种：4 singles、6 pairs、4 triples（$OUG,OUD,OGD,UGD$）、1 个 $OUGD$。

所有 Hessian 都是**完整模型 loss 对选定参数坐标的二阶导**：其余参数固定，但上下游网络均参与原始 forward/backward；不能把某层从网络中单独截断。参数 flatten 顺序固定并存档。

## 1.2 共同 activation anchor 的精确分解（主实验）

取被研究层 $\ell$ 的 **FFN residual 相加后、下一 Transformer block 之前**的完整 hidden states

$$
z=z_\ell(\theta)\in\mathbb R^{T\times d},\quad
q^{(z)}=\nabla_z\mathcal L,\quad B^{(z)}=\nabla_z^2\mathcal L,\quad
J_i^{(z)}=\frac{\partial\operatorname{vec}(z)}{\partial w_i}.
$$

由于这一共同锚点截获 $O/U/G/D$ 到输出 loss 的全部路径，链式法则给出**不附加结构假设**的恒等式

$$
\boxed{H_{ij}=A_{ij}^{(z)}+R_{ij}^{(z)}}
$$

$$
A_{ij}^{(z)}=(J_i^{(z)})^{\mathsf T}B^{(z)}J_j^{(z)},\qquad
R_{ij}^{(z)}=\sum_a q_a^{(z)}\nabla_{w_i}\nabla_{w_j}^{\mathsf T}z_a.
$$

$A^{(z)}$ 称为**激活曲率传播项**（下文简称 $A$），$R^{(z)}$ 称为**activation residual 项**（下文简称 $R$）。$B^{(z)}$ 含该层以下游网络产生的非线性，可能不定，因此 $A^{(z)}$ 也可能不定。这里**不能**把 $A^{(z)}$ 直接称为标准 PSD GGN；$A$、$R$ 的名称在全文只对应本锚点。

## 1.3 logits anchor（辅助、独立标注）

令 $f$ 为参与 loss 的全部预测 logits，$B^{(f)}=\nabla_f^2\mathcal L$、$J_i^{(f)}=\partial f/\partial w_i$。对 softmax 交叉熵，每个预测位置的 logits 曲率

$$B_t^{(f)}=\operatorname{diag}(p_t)-p_tp_t^{\mathsf T}\succeq0.$$

有另一套精确分解

$$
H_{ij}=G_{ij}^{(f)}+R_{ij}^{(f)},\qquad
G_{ij}^{(f)}=(J_i^{(f)})^{\mathsf T}B^{(f)}J_j^{(f)}.
$$

$G^{(f)}\succeq0$ 是**标准 logits-level GGN**，与 activation anchor 的 $A^{(z)}$ 不同；其 residual 亦与 $R^{(z)}$ 不同。全部结果标明 `anchor=block_output` 或 `anchor=logits`，不得交叉比较未对齐的成分值。

## 1.4 统一谱术语

- **self 块** $X_{ii}=X_{ii}^{\mathsf T}$：测 largest algebraic（LA）和 smallest algebraic（SA）两端的特征值与特征向量。最大绝对值谱另记为 LM，绝不替代 LA/SA。
- **cross 块** $X_{ij}\in\mathbb R^{p_i\times p_j}$：测 singular values、左/右 singular vectors，不能将 cross 的非对称特征值当作参数曲率。可使用 symmetric dilation $\left[\begin{smallmatrix}0&X_{ij}\\X_{ji}&0\end{smallmatrix}\right]$ 求 $\pm\sigma$。
- **joint 算子** $X_S=[X_{ij}]_{i,j\in S}$：是对称的参数主子矩阵，可进行 LA/SA eigendecomposition。$X\in\{H,A,R\}$ 时保持同一参数坐标与排序。
- **特征方向贡献**：先固定 $H$ 自身的 eigenvector/singular vectors 再计算 $A,R$ 的 signed contribution；$A$ 与 $R$ 独立求得的特征值通常**不能**直接相加。

---

# 2. Loss 口径、批量聚合及统一比较规则

## 2.1 两种必需口径与一种对照口径

设 $\ell_{s,t}$ 表示序列 $s$ 在位置 $t$ 的 next-token CE（保留完整有效上下文）。

$$
\mathcal L_{\mathrm{tok}(s,t)}=\ell_{s,t},\qquad
\mathcal L_{\mathrm{train},b}=\frac1{M_b}\sum_{(s,t)\in\mathcal B_b}\ell_{s,t},\qquad
\mathcal L_{\mathrm{probe}}=\frac1{M_P}\sum_{(s,t)\in\mathcal P}\ell_{s,t}.
$$

| 口径 | 构造方式 | 用途和约束 |
|:--|:--|:--|
| **ST：single-token** | 单个 target token 的标量 CE，保留原始 sequence/causal context | 观察 token 层曲率异质性；按 position、loss、confidence 分层选多个 token |
| **TB：true train-batch mean** | 真实一次 optimizer step 中**所有有效 prediction tokens** 的未裁剪 CE 加权均值，包括所有 gradient accumulation microbatches | 与实际训练目标严格一致；在该 step **更新前权重**上重放全部 batch |
| **FP：fixed-probe mean** | 预先固定 train/validation 文档、token、mask 形成的 token 均值 | 跨 checkpoint 对比；**不得**代称为 TB |

权重衰减项（decoupled AdamW）、gradient clipping 及 optimizer 状态**不**进入这里定义的 CE Hessian。TB 若由多个 microbatch 聚合，应按**实际有效 token 数**加权，而非简单平均 microbatch 结果；必须核对复算出的梯度与该 step 未裁剪训练梯度一致。

对 $X\in\{H,A,R,G^{(f)},R^{(f)}\}$，token-average 的线性性给出

$$X^{[\mathcal D]}=\sum_{(s,t)\in\mathcal D}\omega_{s,t}X^{(s,t)},\qquad \sum\omega_{s,t}=1.$$

**顺序必须是“固定样本集合 → 聚合算子 → 估计范数/谱/方向”。** 平均 $\|H_t\|$、平均 token 特征值或平均 token eigenvectors，都不是 $\bar H$ 的相应结果。

## 2.2 $B$ 与 $J$ 的 batch 定义（不能误做样本矩阵平均）

不同 sequence 的 $z_s$ 位于不同实例坐标空间。对于无跨样本耦合的模型，把它们组成直和 $z_\mathcal B=(z_1,\ldots,z_n)$，并令 $\mathcal L_\mathcal B=\sum_s\omega_s\mathcal L_s$（$\mathcal L_s$ 是该 sequence 内**有效 token 的平均 CE**，$\omega_s=M_s/\sum_r M_r$）。则

$$
B_{\mathcal B}^{(z)}=\operatorname{BlockDiag}(\omega_1 B_1^{(z)},\ldots,\omega_nB_n^{(z)}),\quad
J_{i,\mathcal B}^{(z)}=\begin{bmatrix}J_{i,1}^{(z)}\\\vdots\\J_{i,n}^{(z)}\end{bmatrix},
$$

$$
(J_{i,\mathcal B}^{(z)})^\top B_{\mathcal B}^{(z)}J_{j,\mathcal B}^{(z)}=\sum_s\omega_s A_{ij}^{(z,s)}.
$$

因此，**参数 Hessian 是加权均值、激活 Hessian 是带权直和、激活 Jacobian 是垂直堆叠**。TB 口径的参数空间 $A/H/R$ 完全可比较；TB 的 $B$ 谱是**直和算子**的谱，不是 $B_s$ 特征值的平均。不同 sequence 的 $B$ 仅在定义了共同 token 坐标或共同低维投影后才能做均值/子空间角度比较；**TB 直和 $B$ 的原始极端特征值还受到 sequence 权重 $\omega_s$ 的缩放**，不可与 ST 极端值不加校正地作强弱对比。同一个 sequence 内不同 target token 使用相同 $J_i$，但具有不同 $B$ 和 $R$。

## 2.3 可比性及记录原则

- ST/TB 配对比较须使用**同一 checkpoint**；TB 必须可以通过保存的 `data_cursor`、文档/position IDs、有效 mask、random seed 和所有 microbatch 重放。
- FP 在各 checkpoint 固定，且与 TB 在图表中分别标注；跨 checkpoint 的 ST 亦应保留一套固定 token cohort，另设随机分层复核。
- ST/TB 的各类二阶观测按四模块共用的 `scope` 字段保存；每项同时记录 `exact/full_batch`、`fixed_probe_proxy` 或 `projected/subsampled` 等测量属性。
- 对同一矩阵的所有成分尽量重用**相同随机 probe 方向**，以估计可比的范数、内积和误差。

---

# 3. 研究模块 M1｜原始 Hessian 的块结构与联合谱

**目的：** 先不讨论 $H=A+R$ 的来源，只回答 $H$ 的十块是什么形状，以及 O/U/G/D 的耦合如何影响曲率和联合特征方向。以下指标对 ST 与 TB 分别计算。

## M1.1 十个 $H_{ij}$：强度、结构及 cross coupling

| 观测指标 | 定义或具体输出 | 计算方法 |
|:--|:--|:--|
| 绝对强度 | $\|H_{ij}\|_F^2$、$\|H_{ij}\|_F$ 及估计标准误 | 对每个源参数块用共享 Rademacher probes；估计 $\mathbb E\|H_{ij}v\|^2$ |
| 参数规模归一化 | $\|H_{ij}\|_F/\sqrt{p_ip_j}$ | 从 Frobenius 估计计算；跨尺寸对比须同步报告原始值 |
| 成对耦合强度 | $c_{ij}=\|H_{ij}\|_F/\sqrt{\|H_{ii}\|_F\|H_{jj}\|_F}$ | 同 checkpoint/scope 估计三个块；分母近零为 NA，$c_{ij}$ 可大于 1 |
| 总 cross 能量 | $\eta_{\rm cross}=2\sum_{i<j}\|H_{ij}\|_F^2/(\sum_i\|H_{ii}\|_F^2+2\sum_{i<j}\|H_{ij}\|_F^2)$ | 十块平方范数组合，范围 $[0,1]$ |
| 有符号方向性 | $\operatorname{tr}(H_{ii})$、$v_i^\top H_{ii}v_i$、$v_i^\top H_{ij}v_j$ | Hutchinson trace、定向 HVP；$v$ 可取梯度/实际更新方向并归一化 |
| 块内可分离结构 | 投影热图、best rank-$r$ Kronecker 相对误差、投影基稳定性 | 以行/列正交基的外积构造低维 $P_i$；计算 $P_i^\top H_{ij}P_j$，再做 Kronecker rearrangement/SVD |

投影首轮取每矩阵行/列基各 4 维，即 16 个参数方向；正式检验可扩至 $8\times8$。若 $\widetilde H_{ij}=P_i^\top H_{ij}P_j$，记录

$$
S_{\mathrm{kron}}(\widetilde H_{ij})=\frac{\sigma_1(\mathcal R(\widetilde H_{ij}))^2}{\|\widetilde H_{ij}\|_F^2}
$$

及 rank-$r$ 误差。$\mathcal R$ 表示与所选行/列基一致的 Kronecker rearrangement。**投影结果只支持被观测子空间的结构结论，不能直接外推完整 Hessian 的可分离性。**

## M1.2 四个 self Hessian：正、负 sharp eigenspaces

| 观测指标 | 计算方法 | 保存 |
|:--|:--|:--|
| 正/负极端特征值 $\lambda_{\mathrm{LA},1:k},\lambda_{\mathrm{SA},1:k}$ | 对 self HVP 用 restarted/block Lanczos，两端分别求解 | eigenvalues、Ritz vectors、gap、residual、收敛标记 |
| 正负方向的谱能量与尺度 | Rayleigh quotient、谱间隔；可选 LM 与随机方向基线 | 保留 LA/SA 区分，不把绝对值大等同于正曲率 |
| 与训练动力学的关系 | $|q_k^\top g_i|/\|g_i\|$、$|q_k^\top\Delta w_i|/\|\Delta w_i\|$ | 同一步冻结梯度与更新；零范数时 NA |

默认 pilot $k=4$–6；正式 $k=10$，资源许可或谱近简并时扩至 20。若收敛不稳定，报告 spectral cluster，而不强行解释单个 eigenvector。

## M1.3 六个 cross Hessian：双侧奇异空间

对 $i<j$，使用 Golub–Kahan bidiagonalization 或对称 dilation 求

$$H_{ij}v_k=\sigma_ku_k,\qquad H_{ji}u_k=\sigma_kv_k.$$

观测 **top-$k$ singular values/双侧 singular vectors**、两侧 residual、$\sum_{r\le k}\sigma_r^2/\|H_{ij}\|_F^2$、左右 singular spaces 与相应 self $H_{ii},H_{jj}$ 的 LA/SA 子空间 principal angles 与 overlap。cross SVD 描述耦合强度与方向，**不直接说明 joint Hessian 存在 $-\sigma_k$ 的负特征值**。

## M1.4 十五个 $H_S$：联合谱及原始耦合效应

对所有 $\varnothing\neq S\subseteq\mathcal I$ 研究 $H_S$ 的 LA/SA eigenvalues、top eigenspaces、spectral gap 和数值 residual。核心比较为四个 self、六个 pairs、$UGD$ 与 $OUGD$；S1 的完整覆盖同时包含 $OUG/OUD/OGD$ 三个 triples。

联合 eigenvector $H_Sq=\lambda q$（$q=[q_i]_{i\in S}$、$\|q\|=1$）记录：

$$
P_i=\|q_i\|_2^2,\qquad E_i=\frac{P_i}{p_i/\sum_{j\in S}p_j},\qquad \sum_{i\in S}P_i=1,
$$

以及十块原始曲率的**有符号分解**：

$$
\boxed{\lambda=\sum_{i\in S}q_i^\top H_{ii}q_i+2\sum_{i<j,\ i,j\in S}q_i^\top H_{ij}q_j.}
$$

比较 $D_S=\operatorname{BlockDiag}(H_{ii})_{i\in S}$ 与 $H_S$：记录联合 vs self-only 的 LA/SA、top-space overlap 和参数参与比例；对六个 pair 可逐一将 $H_{ij},H_{ji}$ **成对置零**以测单一 cross 的作用。判别 $H_S$ 的谱是否出现超出 self Hessians 的共同 sharp direction，并检查

$$\lambda_{\max}(H_S)\ge\max_{i\in S}\lambda_{\max}(H_{ii}),\qquad \lambda_{\min}(H_S)\le\min_{i\in S}\lambda_{\min}(H_{ii}).$$

此处只讨论完整 $H$ 中的参数块之间的耦合；将 cross 分别归因于 $A$ 或 $R$ 属于 M4。比较 self-only 与 joint eigenspaces 时，统一总子空间维数或明确采用 containment 指标，不能直接比较 $4k$ 维和 $k$ 维空间的相同维数 overlap。

**M1 主产出：** 10 块 H norm/coupling heatmap；4 self 正负谱；6 cross SVD 与 self 对齐；15 种联合谱；joint eigenvector 的 O/U/G/D 参与比例及十项 signed contributions。

---

# 4. 研究模块 M2｜$A$、$R$ 的独立结构与对 $H$ 谱的贡献

**目的：** 固定 §1.2 的 activation anchor，对同一个 Hessian $H$ 的两项做**完整且对称的观测**：每一项分别有范数、self/cross/联合谱、特征向量；再在**原始 H 的方向**上测贡献。所有指标在 ST/TB 两种口径下分别执行。

## M2.1 十块成分大小、对齐和抵消

| 观测指标 | 定义与方法 | 解释限制 |
|:--|:--|:--|
| 成分规模 | 每个 self/cross 块的 $\|A_{ij}\|_F,\|R_{ij}\|_F$、元素 RMS、标准误 | 与相同 $H_{ij}$ 的数值并列 |
| 相对强度 | $\|A_{ij}\|_F/\|H_{ij}\|_F$、$\|R_{ij}\|_F/\|H_{ij}\|_F$ | 两比率不必加和为 1，可能大于 1 |
| 成分内积 | $\langle A_{ij},R_{ij}\rangle_F$、Frobenius cosine | 用共享 probe 的输出内积估计 |
| 抵消/协同恒等式 | $\|H_{ij}\|_F^2=\|A_{ij}\|_F^2+\|R_{ij}\|_F^2+2\langle A_{ij},R_{ij}\rangle_F$ | 检验每块分解数值正确性及误差 |
| 组件自身 cross 比例 | 对 $A_S,R_S$ 分别计算 §M1.1 的 cross energy、coupling ratio | 防止仅报告 $H$ 的 cross 比例而忽略成分 |

## M2.2 $A$、$R$ 各自的 self、cross 与联合谱

**对 $A$ 和 $R$ 各自完整重复 M1 的谱观测**，方法相同、输入算子不同：

- 4 个 self 块：LA/SA eigenvalues、eigenvectors、spectral gaps、Ritz residual。
- 6 个 cross 块：top singular values、左右 singular vectors、双侧 residual、captured Frobenius energy。
- 15 个 joint 参数组合：LA/SA 谱、eigenvectors、O/U/G/D 参数参与比例；S1 覆盖全部组合，S2 以主要组合为优先。
- 同一块/组合对照 $H,A,R$ 的 positive/negative sharp subspaces：principal angles、固定维数的 $\Omega(Q_1,Q_2)=\|Q_1^\top Q_2\|_F^2/k$；cross 的左右奇异空间分别比较。
- 结构诊断仍沿用 M1 的同一组行/列投影基 $P_i$，对 $A_{ij},R_{ij}$ 分别估计 Kronecker separability 与投影基稳定性；**同一块 $H/A/R$ 必须使用相同基**才能比较。

$A$ 和 $R$ 均可能不定。求得 $\lambda_k(A)$、$\lambda_k(R)$ 或 $\sigma_k(A)$、$\sigma_k(R)$ 后，**不得逐序号相加**作为 $H$ 的谱归因。

## M2.3 两项对**原始 $H$ 特征值**的精确贡献

对任意 self/joint $H_S q_k=\lambda_k^H q_k$，保留**这一 $H$ 的同一个特征向量**，记录

$$
\boxed{\lambda_k^H=q_k^\top A_Sq_k+q_k^\top R_Sq_k.}
$$

进一步按矩阵块展开

$$
\boxed{\lambda_k^H=\sum_{i\in S}(q_i^\top A_{ii}q_i+q_i^\top R_{ii}q_i)+2\sum_{i<j}(q_i^\top A_{ij}q_j+q_i^\top R_{ij}q_j).}
$$

$OUGD$ 对应**十个 $A$ 贡献项 + 十个 $R$ 贡献项，共 20 个 signed terms**；记录每项原始数值、代数和、回和误差。负值是合法贡献，不能直接当非负百分比。对 cross $H_{ij}$ 的 singular triple $(\sigma_k,u_k,v_k)$，记录

$$\boxed{\sigma_k=u_k^\top A_{ij}v_k+u_k^\top R_{ij}v_k.}$$

此处向量必须来自 **$H_{ij}$**，并采用 $u_k^\top H_{ij}v_k=\sigma_k\ge0$ 的方向约定。$\lambda$ 或 $\sigma$ 近零时只报告 signed 贡献与误差，不给不稳定的“百分比”。

## M2.4 两项对原始**特征方向**的作用

除了比较各自 top space 与 $H$ 的 overlap，还测 $R$ 将 $A$ 的已选不变子空间“推向外部”的作用：

$$
P_A=Q_AQ_A^\top,\qquad
L_{R\to A}=\|(I-P_A)R_SQ_A\|_F,\qquad
L_{\rm rel}=\frac{L_{R\to A}}{\|R_SQ_A\|_F+\varepsilon}.
$$

对称地计算 $L_{A\to R}$。该量是**子空间 leakage 而不是旋转角本身**，应结合对应 eigengap 解读。若存在近简并，比较 stable invariant cluster 而非单个 eigenvector；cross SVD 对左、右空间分别测 leakage/overlap。

**M2 主产出：** 10 块 A/R 的 norm、对齐及抵消；两组件各自的 self/cross/15-combination 联合谱；A/R/H 主子空间角度；固定 H 方向的两项与 20 项有符号特征值贡献。

---

# 5. 研究模块 M3｜$B$ 与 $J$ 的传播机制

**目的：** 不只统计 $A_{ij}$ 的最终数值，还观察构成 $A_{ij}=J_i^\top BJ_j$ 的激活曲率、参数敏感性及其对 cross coupling 的影响。B/J 的测量须严格符合 §2.2 的多样本坐标定义。

## M3.1 共同激活 Hessian $B^{(z)}$：大小、谱、token 耦合

固定单个 sequence 长度 $T$，令 $z\in\mathbb R^{T\times d}$，则 $B^{(z)}\in\mathbb R^{Td\times Td}$。目标观察：

| 观测指标 | 计算方法 | 输出/注意事项 |
|:--|:--|:--|
| 总曲率与符号 | $\|B\|_F^2$、$\operatorname{tr}B$、LA/SA Ritz values/vectors、谱间隔 | BVP + stochastic norm/trace + Lanczos；不预设 PSD |
| token-token 块能量 | $\|B_{r,r'}\|_F^2$ 的 diagonal/off-diagonal 能量比例 | token-block probes/抽样；注明抽样位置与覆盖率 |
| 时间距离结构 | 按 $|r-r'|$ 分组的平均/分布、局部低维 heatmap | 只在当前 sequence 的 activation 坐标比较 |
| 低维几何 | 随机投影谱、Kronecker/tensor separability、不同 token loss 的 B 主空间角度 | 解释为 projected 结构；不得冒充完整 B 谱 |
| ST 与 TB | ST 的单 sequence $B$；TB 的带权 block-diagonal $B_{\mathcal B}$ | 以直和规则分析；**禁止**无对齐的 $\bar B_N=\frac1N\sum B_s$ |

注意：单 token CE 仍使用完整 sequence 的中间激活，因此 $B_{r,r'}$ 可能在不同上下文位置之间非零；单 token 不等于 $d\times d$ 的“单位置 B”。

## M3.2 四个 $J_i^{(z)}$：参数—激活敏感性与共享输出空间

对 $i\in\{O,U,G,D\}$，$J_i\in\mathbb R^{Td\times p_i}$，观测：

| 观测指标 | 计算方法 |
|:--|:--|
| $\|J_i\|_F$ 与 $\|J_i\|_F/\sqrt{p_i}$ | JVP stochastic probes，记录误差 |
| top singular values、左右 singular vectors、有效秩 proxy | JVP/VJP + Golub–Kahan 或 Gram matvec；保留双侧 residual |
| $J_i$ 的输出 token 响应与位置分布 | $J_iv$ 的 token-resolved 范数；固定 probe 与参数单位方向 |
| 跨矩阵 Jacobian 耦合 $\|J_i^\top J_j\|_F$ | 组合 JVP/VJP + 随机 probe |
| 输出 Gram 共享空间 | 比较 $J_iJ_i^\top$ 与 $J_jJ_j^\top$ 的 Frobenius cosine；以及左奇异子空间角度 |

共享输出空间的一个指标为

$$
\operatorname{Overlap}_{ij}^{J}=\frac{\langle J_iJ_i^\top,J_jJ_j^\top\rangle_F}{\|J_iJ_i^\top\|_F\|J_jJ_j^\top\|_F}.
$$

$J_i$ 对同一 sequence **不依赖选择哪个 target token 的 loss**，但其活跃的后续曲率方向会因 $B$ 的改变而不同。TB 的 $J$ 必须是堆叠后的算子；不能把 $J_i$ 的不同 sequence 实例直接逐元素平均。**TB 原始 $\|J\|_F$ 会随堆叠样本数改变**，与 ST 比较时应报告每 sequence 的分布或按有效 token 数归一化的辅助指标，而不能将原始数值差异直接解释为参数敏感性变化。若比较跨样本共享空间，应规定形状/位置对齐或报告各样本指标的分布。

## M3.3 $B$ 与 $J$ 如何形成 $A_{ij}$

对每个 self/cross 块检查 $A_{ij}=J_i^\top BJ_j$：

1. **几何覆盖**：求 $B$ 的 LA/SA 方向 $u_r$，观察 $\|J_i^\top u_r\|$，并在固定投影维数内测 $\|P_{B,\pm}J_i\|_F$；区分激活正/负曲率在参数空间是否可达。
2. **传播前后耦合**：对相同 $i,j$ 比较 $J_i^\top J_j$ 和 $J_i^\top BJ_j$ 的谱与耦合尺度；另外分别以各自 Frobenius norm 标准化比较方向，**不将两者原始范数直接解释为放大倍数**。
3. **正负曲率分量（可选）**：在已测低维 $B$ 谱上构造 $B_+$/$B_-$，近似解释 $A\approx J^\top B_+J-J^\top B_-J$；明确投影截断误差。因 $B$ 不定，不能用全空间实数 $B^{1/2}J$ Gram 解释整个 $A$。
4. **闭环核验**：抽取随机参数方向 $v_j$，检查 $J_i^\top B(J_jv_j)$ 是否等于独立得到的 $A_{ij}v_j$。

## M3.4 logits GGN 对照：区分两类曲率来源

对固定 token 的 logits 概率 $p$，$a_r=\sum_v p_v^r$，记录

$$\operatorname{tr}B_t^{(f)}=1-a_2,\qquad \|B_t^{(f)}\|_F^2=a_2-2a_3+a_2^2.$$

记录 softmax entropy/confidence、$B_t^{(f)}$ top eigenvalue 与 effective rank；对 $J_i^{(f)}$ 重复核心范数/Gram-overlap。因为 $B^{(f)}\succeq0$，对 $G_S^{(f)}=K^\top K$（$K=(B^{(f)})^{1/2}J_S^{(f)}$）可比较其非零特征值与 $K$ 的 squared singular values。对照 $H=G^{(f)}+R^{(f)}$ 与 $H=A^{(z)}+R^{(z)}$ 的相同 $H$ 方向贡献，解释将 anchor 从 logits 前移到 block output 后曲率归因如何变化；**这是两套不同分解，不是四个可以直接相加的成分**。

**M3 主产出：** activation B 的正/负谱和 token-token 结构；四个 J 的敏感性与共享响应空间；$B$-weighted 与 unweighted Jacobian 耦合；两类 anchor 下的 H 谱贡献差异。

---

# 6. 研究模块 M4｜组件之间的联合谱作用（控制实验）

**目的：** M2 已说明 $A/R$ 各自谱及它们在 H 原方向的代数贡献。本模块进一步探究：为什么组合后会形成某个 eigenvector/eigenspace？**只在固定 checkpoint、固定数据和固定算子上修改数学组合**，不把这些谱路径误解为真实模型训练过程。

## M4.1 两组件的谱插值 $A_S+tR_S$

$$H_S(t)=A_S+tR_S,\qquad t\in\{0,0.25,0.5,0.75,1\}.$$

观测 LA/SA eigenvalue trajectories、相邻 $t$ 的 top eigenspace principal angles、谱间隔、block participation、方向在 A/R 子空间中的投影。对 simple eigenvalue 有

$$\frac{d\lambda_k(t)}{dt}=q_k(t)^\top R_Sq_k(t).$$

需要追踪同一谱分支；遇到近简并/交叉时转为 cluster-subspace 分析，**不得**逐 $t$ 按 eigenvalue 序号机械计算向量转角。保留必要的有限差分和 Ritz residual。

## M4.2 原始参数 cross coupling 路径

对于 M1 已确定的 $D_S^H=\operatorname{BlockDiag}(H_{ii})$、$C_S^H=H_S-D_S^H$，定义

$$H_S^{\rm cross}(\alpha)=D_S^H+\alpha C_S^H,\quad \alpha\in\{0,0.25,0.5,0.75,1\}.$$

这里将 M1 的“是否存在 cross effect”扩展为谱路径：观测极端 eigenvalues、subspace trajectories、block participation、谱分支交换及 nonlinear response。对指定 pair 的单 cross 消融继续采用**同时删除转置共轭的两侧块**。

## M4.3 $A$-cross 与 $R$-cross 的四条件因子消融

$$
C_S^A=A_S-\operatorname{BlockDiag}(A_{ii})_{i\in S},\qquad
C_S^R=R_S-\operatorname{BlockDiag}(R_{ii})_{i\in S},
$$

$$
\boxed{H_S(\alpha,\beta)=D_S^H+\alpha C_S^A+\beta C_S^R,\quad
(\alpha,\beta)\in\{(0,0),(1,0),(0,1),(1,1)\}.}
$$

四条件分别表示 self-only、仅加入 $A$-cross、仅加入 $R$-cross、恢复完整 $H$。比较极端谱、top-space principal angles、参数参与及变化交互；可补充更细致的 $(\alpha,\beta)$ 网格。变化并非简单的非负可加“贡献率”，当谱分支交换时需以不变子空间为比较对象。

## M4.4 $A/R$ 联合候选 sharp subspace

收集两项各自的 LA/SA 极端方向

$$Q_{AR}=\operatorname{orth}([Q_{A,+},Q_{A,-},Q_{R,+},Q_{R,-}]),\quad
\widetilde H=Q_{AR}^\top H_SQ_{AR}.$$

对 $\widetilde H$ 求小矩阵 Ritz eigenpairs，再投回参数空间与独立求得的 $H_S$ LA/SA 对比：记录 eigenvalue error、Ritz residual、$H$ top space containment $\|Q_{AR}^\top Q_{H,k}\|_F^2/k$、投影总维数及相同维数随机 baseline；另与仅 $Q_A$、仅 $Q_R$ 比较。辅助记录 $Q_A^\top R_SQ_A$ 与 $Q_R^\top A_SQ_R$ 的非对角项、$[A_S,R_S]=A_SR_S-R_SA_S$ 的低维/算子 proxy。此法检验两项提供的特征方向是否足以覆盖 H，而**不是**把两项当成两个物理参数块。

**M4 主产出：** $A+tR$ 与 $D+\alpha C$ 两条谱轨迹；$A$-cross/$R$-cross 四条件方向变化；联合候选子空间对原始 H sharp space 的覆盖与误差。

---

# 7. 共用横向实验：单 token、样本聚合与训练演化

## 7.1 ST → sequence → 多 sequence → TB

此项对 M1–M4 的相关指标提供**额外解释维度**，不再与四个模块平级。对 $X\in\{H,A,R\}$ 的同一参数块或联合算子，观测单 token 的范数²分布、不同 token 算子 Frobenius cosine，以及按有效 token 权重合成的 $X_N$ 的范数与谱。取 $N\in\{1,2,4,8,16,32\}$ 个 sequence subsets，多次独立抽样并报告误差；最终与**真实 TB** 比较。

定义 $\bar X_N=\sum_s\omega_sX_s$、$\sum\omega_s=1$，并计算

$$
C_X(N)=\frac{\|\bar X_N\|_F^2}{\sum_s\omega_s\|X_s\|_F^2},\qquad 0\le C_X(N)\le1
$$

（分母为零时记 NA）。$C_X$ 为跨样本算子一致性指标，单独不能区分不同方向分散与符号抵消。**先加权累积 HVP，再求平方范数或做 eigensolver**。在相同 $N$ 重复下比较 LA/SA 谱、子空间稳定性、Kronecker 投影结构、20 项贡献与 O/U/G/D 参与比例。

对于 $B/J$，按 §2.2 **分别使用 TB 直和/堆叠**，或比较已对齐的 per-sequence 统计分布；**不定义**跨 sequence 的“原始 $B$ eigenvector 算术平均”或无对齐的 $\bar B_N$。对 logits $B^{(f)}$ 可报告按预测 token 的解析统计分布及平均，但需与直和谱区分。

## 7.2 训练时间维度与目标曲线

每个训练阶段均同时追踪 M1–M4 的代表性解释变量；但按成本区分日志与深入观测：

- **动力学基础**：CE/validation loss、LR、token count、吞吐、各矩阵参数/梯度/实际更新范数，以及 gradient clipping 触发率。
- **曲率代理时间序列**：十块 H/A/R norm² 与 cross energy 的低 probe 估计、A/R alignment、主要 self/joint 的粗 LA/SA 界、$B/J$ 的低成本 norm/trace；明确代理数据来源与置信误差。
- **关键节点正式谱**：同一 checkpoint 成对比较 ST/TB 的完整 M1–M4，含 signed contribution、eigenspace rotation、sample-size dependence，并在 FP 上提供跨 checkpoint 的稳定对照。

T0/T1/T2 的具体频率、阶段适用性和最低完成标准在 §10 统一规定，不在研究模块中重复指定。

---

# 8. 共用数值方法、算子接口与成本控制

## 8.1 Matrix-free 计算（唯一实现后端）

对固定 $\mathcal L$、checkpoint、参数块 $i,j$ 和输入方向 $v_j$：

$$H_{ij}v_j=\nabla_{w_i}\langle\nabla_{w_j}\mathcal L,v_j\rangle.$$

传播项按以下三步计算：

$$u=J_j^{(z)}v_j\ \xrightarrow{\ B^{(z)}\ }\ b=B^{(z)}u\ \xrightarrow{\ J_i^{(z)\top}\ }\ A_{ij}^{(z)}v_j=(J_i^{(z)})^\top b.$$

$R_{ij}v_j=H_{ij}v_j-A_{ij}v_j$；**独立核验**方式：在同一 checkpoint 计算 $q_0=\operatorname{stopgrad}(\nabla_z\mathcal L)$，构造 $\psi(\theta)=\langle q_0,z_\ell(\theta)\rangle$，则 $\nabla_{w_i}\nabla_{w_j}^\top\psi\,v_j=R_{ij}v_j$。

使用以下统一接口：

```text
H_matvec(S, scope, v)
A_act_matvec(S, scope, v)
R_act_matvec(S, scope, v)
B_act_matvec(scope, u)                 # activation-coordinate operator
J_act_jvp(i, scope, v)
J_act_vjp(i, scope, u)
G_logits_matvec(S, scope, v)           # auxiliary GGN
R_logits_matvec(S, scope, v)          # auxiliary residual
```

`scope` 必须绑定不可变的 loss/operator 规格（batch IDs、masks、loss weights、anchor、参数版本）。对 $S$ 的 joint matvec 复用一套 OUGD 源/目标坐标接口，通过零填充与切片选择子集。**同一次 eigensolver 迭代所有 matvec 必须作用于完全相同的固定算子，不能每次更换随机 batch。**

## 8.2 统一估计器与求解器

Rademacher 向量 $v$ 满足 $\mathbb E[vv^\top]=I$，因此对自块或矩形 cross 块均有

$$\|X\|_F^2=\mathbb E_v\|Xv\|_2^2,\qquad
\langle X,Y\rangle_F=\mathbb E_v[(Xv)^\top(Yv)].$$

- **范数与内积**：T1 每个源空间用 2–4 probes 做 *noisy proxy*，T2 从 16 probes 起并按标准误扩至 32/64；计算 $\sqrt{\widehat{\|X\|_F^2}}$ 时注明平方根引入的偏差，报告原始平方范数与误差更稳健。
- **对称 self/joint / $B$ 谱**：restarted/block Lanczos，分别求 LA/SA；可以做 stochastic Lanczos quadrature 估谱密度，但**它不提供已验证的特征向量**。
- **非对称 cross / $J$ 奇异谱**：Golub–Kahan 或 dilation；同时使用正向与转置 matvec 验证两侧 residual。
- **小投影算子**：dense `eigh`/`svd`，必须标记为 `projected`。
- **迭代预算**：pilot $k=4$–6，约 30–60 matvec；正式初始 $k=10$，约 60–120 matvec。以上仅用于 profiling，是否可报告取决于 residual 与收敛，而不是迭代次数。

对求得的 $Xq\approx\lambda q$、$Xv\approx\sigma u$、$X^\top u\approx\sigma v$，保存绝对/相对 Ritz residual、正交误差、迭代数、重复 seed 稳定性；near-degenerate cluster 以 projector/subspace 为单位比较。H/A/R 的范数和贡献尽量用同一批随机方向，避免测量误差造成假性的加减不一致。

## 8.3 规模、内存与时间上界的现实约束

主模型中 $p_O=589,824$、$p_U=p_G=p_D=1,572,864$，故 $p_{OUGD}=5,308,416$。完整 FP32 联合 Hessian 需要约 **113 TB**；单个 FFN self Hessian 约 **9.9 TB**。长度 2048、hidden 768 时单序列 $B^{(z)}$ 已是 $(1,572,864)\times(1,572,864)$，也不可显式存储。实际应只计算 matvec 和小规模投影。

正式实验前在 **microbatch=1、context=2048、FP32 curvature、math attention backend** 下 profile HVP、BVP、JVP/VJP 的显存/耗时；统计完整 TB 的 microbatch 数 $m$，一个 TB matvec 通常需要重放 $m$ 个 microbatches。总时间近似按

$$\text{GPU-time}\ \propto\ \sum_{\text{solver runs}}(\text{matvec 次数})\times(\text{每次重放真实数据的成本})$$

核算，而不是用 16K-token FP 的成本代替 131,072-token TB。80GB GPU 仅是起始 profiling 环境，**不是内存/耗时一定可行的保证**。资源不足时依次压缩正式 checkpoint 数、ST token replications、谱方向数 $k$、插值网格密度；保留 S1 关键科学覆盖及 TB 的真实定义，未完成项必须在 coverage sheet 中标明。

---

# 9. 模型、数据与训练配置

## 9.1 主模型（S2）预注册配置

| 项目 | 计划配置 |
|:--|:--|
| 网络 | decoder-only Llama-style；Pre-RMSNorm、RoPE、GQA、SwiGLU；tied output embedding |
| 深度 / hidden / FFN | 12 / 768 / 2048，实际参数量按代码统计（目标约 100M） |
| Query / KV heads | 12 / 4；head dim 64 |
| Context / vocab | 2048 / 32,000 |
| O / U / G / D 形状 | $768\times768$ / $2048\times768$ / $2048\times768$ / $768\times2048$ |
| 预训练语料 | FineWeb-Edu `sample-10BT`；文档级 train/validation 划分；tokenizer 后重新计数 |
| 训练 / 验证 tokens | 2B 有效预测 tokens / 约 10M validation tokens |
| AdamW | peak LR $4\times10^{-4}$，$\beta=(0.9,0.95)$，weight decay 0.1 |
| LR / clipping | 300-step warmup + cosine decay 至 $4\times10^{-5}$；global clip 1.0 |
| Global batch | 131,072 有效 prediction tokens；预计约 15,259 optimizer steps |
| 数值精度 | BF16 训练；FP32 curvature（attention 优先 math backend） |
| 主层与复核层 | Layer 6 主分析；Layer 1 和 Layer 12 复核 |
| 候选完整 checkpoint | 0、10、50、150、300、1000、3000、7500、12000、final |

训练前先做短跑 profiling 与数值核验，再冻结配置；实际 batch 构成、学习率 schedule、token 计数应以运行配置及代码为准。保存模型和 optimizer state、**optimizer-step 更新前的**数据游标、所有 gradient-accumulation microbatches、RNG state、tokenizer hash、document IDs、mask、learning rate 和可复算 loss weights。

## 9.2 模型变化的控制

S0/S1 可缩减 width、FFN、context、vocab，但保持 causal attention、Pre-RMSNorm、残差连接、SwiGLU、O/U/G/D 的完整拓扑。不同规模、层、loss 及参数维度下的 Hessian 绝对范数不可直接视作“相同结构强弱”的公平比较；先提供尺寸归一化和 matched settings，再讨论尺度演化。S3 可使用公开 SmolLM2-135M checkpoint 或其他模型作为外部验证，明确架构、tokenizer 和训练历史差异，不与本实验自训 checkpoint 连成单一时间序列。

---

# 10. 实验组织：规模阶段与训练采样频率

## 10.1 S0–S3 是**唯一实施阶段**

| 阶段 | 模型与样本 | 四模块执行范围 | 升级门槛或交付 |
|:--|:--|:--|:--|
| **S0 数学真值验证** | 2 层 tiny Transformer，hidden 32/64、FFN 64/192、context 16–32、小 vocab；小 batch | ST/TB；选定**小参数坐标窗口**下 dense H/A/R/B/J，十块、15 种子集合、谱归因及 M4 受控算子，对照 matrix-free | $H=A+R=G^{(f)}+R^{(f)}$、HVP/JVP/VJP、谱及归因全部通过数值回归 |
| **S1 中小模型完整研究** | 建议 4 层、hidden 384、FFN 1024、heads 6/KV 2、context 512、vocab 16K；约 10–20M，约 0.1–0.3B tokens | 在至少**一个预先指定的深度测量 checkpoint**、同一中间层，对 ST/TB **完成 M1–M4 全类别**：十块 H/A/R、各自 self/cross 谱、15 个 H/A/R 联合谱、B/J、归因、插值及因子消融 | 完整双口径数据包与成本表；形成可供 S2 预注册的主要发现和优先假设 |
| **S2 约 100M 主模型** | §9.1 配置、FineWeb-Edu 2B tokens | T0/T1 覆盖整个训练；T2 Layer 6 优先两口径的 10 块 H/A/R norm、self/cross 谱、OUGD/UGD 与重要 pairs 的 joint 谱及 M2–M4 机制；15 组合全覆盖为资源允许的**扩展目标** | 检验 S1 观察是否延伸到现实规模；透明报告覆盖率、数值误差与 GPU 成本 |
| **S3 跨层/复现/模型外推** | Layer 1/12、额外 seed，或外部模型（可选） | 针对 S1/S2 确认的重要现象复核 M1–M4；敏感性分析 | 结果的稳定性、适用范围和局限 |

**覆盖政策：** S1 的“全覆盖”是预注册的科学测量要求，不代表每个训练 checkpoint 都要做全部 15 组合；S2 是**重点复核**而非默认同等完整。最小完成单元为 `(stage, model, layer, checkpoint, scope, parameter_set, operator, metric)`，每个单元存 `complete/pilot/pending/skipped_budget/failed_convergence`。未测组合不可默认为通过。

## 10.2 T0/T1/T2 是**同一训练中的观测频率**

| 等级 | 计划频率与数据 | 观测内容 | 与四模块对应关系 |
|:--|:--|:--|:--|
| **T0 每 step 日志** | 全部约 15K optimizer steps；真实 TB | train CE、训练 tokens、LR、throughput、global/各矩阵梯度范数、参数范数、实际更新范数、clipping；validation loss 按周期评估 | M1/M2 对曲率—梯度—更新关系的基础解释变量，无额外 HVP |
| **T1 全程低成本曲率曲线** | warmup 0/10/50/150/300，之后每约 250–500 steps；固定小 FP/分层样本，**必须标记 proxy** | 10 块 H/A/R noisy norm²、cross energy、A/R cosine；选定 self/joint 粗谱，B 的 trace/norm 与 J 的 norm 的少量 probe；logits CE 解析统计 | 在整个训练范围捕捉趋势，不能冒充 TB 深度谱 |
| **T2 深度观测点** | 主候选 0/300/3000/7500/final（按实际 step 对齐），Layer 6；S2 成本决定重点数量 | 对每个正式选定点分别建立 ST 与完整 TB 固定算子，辅以 FP；M1–M4 按所在阶段覆盖要求做收敛的谱、贡献和机制分析 | 审核数值误差、coverage 与 token 代表性 |

**阶段与频率相互独立**：S0 可以在少量 step 做 T2；S1 可在至少一个 checkpoint 全覆盖；S2 可以只在若干关键 checkpoint 做选定组合的深度分析。若 T1 曲线发现异常转变，可在已记录数据和资源允许条件下增加 T2 点，并把它标为 `adaptive`，避免将事后增加的点误报为预注册测量。

## 10.3 双口径必测矩阵（正式深度测量点）

| 测量系列 | ST | 真实 TB | FP |
|:--|:--:|:--:|:--:|
| M1：10 块 H 的 norm/结构、4 self、6 cross、选定 15 joint | **要求** | **要求** | 跨 checkpoint 对照 |
| M2：10 块 A/R 的 norm 与各自谱、H signed contribution、子空间 leakage | **要求** | **要求** | 重要组合推荐 |
| M3：$B/J$ 范数/谱/传播机制、logits GGN 对照 | **要求** | **要求**（B/J 按 batch 直和定义） | 按对齐规则可选 |
| M4：$A+tR$、cross 插值/因子消融、共同子空间 | **要求** | **要求** | 重要组合推荐 |
| §7：ST→sequence→$N$ sequences→真实 TB | ST token 分布 | TB 终点 | $N=1,2,4,8,16,32$ 对照 |

“要求”是**在所属规模阶段的选定覆盖范围内**对两种口径同时执行，并不取消 S2 的预算裁剪。S1 完成所有 15 组合；S2 以 ten blocks、核心 pairs/UGD/OUGD 及关键机制为正式承诺，扩展三参数组合列为 stretch。任何裁剪都记录在实际 coverage 表中，而非默默删除模块。

---

# 11. 数值验收、数据组织及论文产出

## 11.1 数值正确性与质量控制

S0 dense 对照与 S1/S2 的 matrix-free 检查必须覆盖：

1. **精确分解**：同一 anchor/loss 下 $H_{ij}v=A_{ij}v+R_{ij}v$，以及 $H_{ij}v=G_{ij}^{(f)}v+R_{ij}^{(f)}v$；独立 direct residual RVP 对照；$A$ 可不定而 $G^{(f)}$ PSD。
2. **scope 聚合**：参数空间 ST operator 加权聚合与独立 TB HVP 一致；所有 microbatch 的有效 token 权重、loss normalization 及 gradients 可复算。
3. **双侧性和对称性**：$H_{ji}=H_{ij}^\top$、$A_{ji}=A_{ij}^\top$、$R_{ji}=R_{ij}^\top$，JVP/VJP adjointness；cross SVD 双侧 residual。
4. **self/joint 谱**：dense vs Lanczos 的 LA/SA eigenvalues/eigenspaces；15 种小坐标 joint 谱；interlacing 极值约束、正交误差、近简并 cluster。
5. **范数及 signed 归因**：Frobenius 恒等式，$H$ eigenvalue 2 项/20 项贡献回和，cross singular value 归因回和，projection truncation error。
6. **B/J 完整性**：共同 anchor 捕获 O/U/G/D 所有路径；B 的 token-token off-diagonal dense 检查；多 sequence 的 B 直和/J 堆叠收缩；logits GGN PSD 检查。
7. **随机估计可靠性**：probe 数量、standard error、相同 seed 重复、solver budget、误差阈值、precision/backend 与耗时/显存均留痕。未通过阈值者只报告探索性 proxy，不进入确定的 eigenspace 结论。

默认记录归一化对称 Ritz residual $\|Xq-\lambda q\|/(\|Xq\|+|\lambda|+\varepsilon)$；cross 记录左右残差 $\|Xv-\sigma u\|$ 和 $\|X^\top u-\sigma v\|$，同时保存归一化形式。求解器是否收敛由测试阈值和重复性决定；具体阈值在 S0 profiling 后**预先冻结**，不能根据结果事后放宽。

## 11.2 统一数据键及存储布局

```text
experiment/
  configs/{model,training,measurement}.yaml
  tracking/t0_step_metrics.parquet
  tracking/t1_curvature_proxies.parquet
  checkpoints/step_XXXXXX/{model,optimizer,rng,data_cursor,batch_manifest}
  measurements/step_XXXXXX/layer_YY/scope_ST|TB|FP/
    coverage.parquet
    norms_and_coupling.parquet         # M1/M2/B/J strength + uncertainty
    eigenvalues.parquet               # symmetric self/joint H/A/R/B/GGN
    singular_values.parquet           # cross H/A/R and Jacobians
    contributions.parquet             # 10/20 signed terms and cross SVD
    subspace_metrics.parquet          # angle, overlap, leakage, containment
    sample_aggregation.parquet        # ST -> sequence -> N -> TB
    paths/component_interpolation.parquet
    paths/cross_interpolation.parquet
    paths/cross_factorial.parquet
    vectors/{operator}/{parameter_set}/...
    operator_meta.json
  reports/
```

最小记录维度包括 `run_id, stage, model_hash, step, pre_update_state, layer, scope, batch_id/token_key, valid_tokens, parameter_set, block, operator, anchor, metric, eig_end(LA/SA/LM/SVD), k, probe_seed, precision, backend, measurement_status, solver_residual, stderr, gpu_seconds, peak_memory`。每项 eigenvector 必须带 flatten 规则、参数 shape、来源算子和是否 full/projected 的元数据；不同 scope 与 checkpoint 不能重名覆盖。

## 11.3 核心图表与结论边界

最终结果按**科学问题**组织，而不是“先算 norm，后算 spectrum”的计算顺序：

| 图表组 | 支持的研究问题 |
|:--|:--|
| M1：10 块 H norm/cross heatmap、4 self 正负谱、6 cross SVD、15 joint vs self-only、O/U/G/D block participation | 耦合是否显著改变 sharp eigenspaces？ |
| M2：10 块 A/R 强度和内积、两项独立谱与 H overlap、20 项 signed contributions、leakage | 哪个成分在数值和方向上主导？二者是否抵消或重塑 sharp directions？ |
| M3：B 正负谱与 token coupling、J 输出 Gram overlap、B-weighted/unweighted coupling、logits GGN 对照 | 激活曲率与参数映射如何产生所见谱结构？ |
| M4：$A+tR$、$D+\alpha C$ 谱轨迹、cross-factorial、联合候选 Ritz space vs H | 特征方向如何在成分和 cross 耦合的共同作用下形成？ |
| 横向：ST/TB/FP 对比、$N$-dependent 一致性、T0/T1/T2 曲线、跨层和跨模型复核 | 结果是否反映真实训练均值，能否沿训练与规模保持？ |

**最终完成定义：** S0 的关键恒等式和数值回归通过；S1 至少一个正式深度点对 ST/TB 完成 M1–M4 全类别及全部 15 种参数组合；S2 在 Layer 6 对两主 scope 完成已预注册的重点观测，并以实测 coverage 明示缺项；所有确认性谱结论均提供收敛残差与不确定性。**不能把 S1 的全覆盖表述为 100M 主模型的全覆盖。**

---

# 12. 实施前检查清单与参考实现

- 冻结共同 activation anchor 的准确 module path；通过自动微分验证所有四组参数对 anchor 的路径完整性。
- 冻结 CE reduction、token masks、gradient accumulation 加权规则、数据顺序与更新前 checkpoint 的重放接口。
- 完成 S0 dense 谱/分解/多样本直和回归，确认 BF16 训练图在 FP32 curvature 路径的 AD 支持。
- 完成 TB HVP、BVP、JVP/VJP 的 microbatch profiling，按真实迭代次数核算 GPU-time。
- 冻结 T2 首批 checkpoint、token 分层方案、solver tolerance 和 coverage 承诺；将额外分析标记 exploratory。
- 记录 PyTorch/CUDA、attention backend、源码 commit、tokenizer/data hashes、参数扁平顺序、随机种子、显存峰值和测量失败原因。

实现参考：

- PyTorch JVP：https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html
- PyTorch VJP：https://docs.pytorch.org/docs/stable/generated/torch.func.vjp.html
- PyTorch Hessian：https://docs.pytorch.org/docs/stable/generated/torch.func.hessian.html
- SciPy symmetric partial spectrum：https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
- SciPy partial SVD：https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html

> **一页执行原则：** M1 观察原始 Hessian 的十块/联合谱；M2 观察每块的两项成分及对原谱的贡献；M3 观察激活 $B$ 和各 Jacobian 的形成机制；M4 观察成分与 cross coupling 的联合谱响应。ST 和真实 TB 是全部模块的两种主口径，FP 是对照；T0/T1/T2 决定追踪密度，S0/S1/S2/S3 决定实施规模。所有理论结论必须同时标注 anchor、scope、模型规模与数值误差。
