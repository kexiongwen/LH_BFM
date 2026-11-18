# CAVI 的谱初始化方案

两种谱初始化都用数据的谱分解确定载荷均值，用剩余二阶矩确定噪声种子和正定的载荷协方差，再通过条件更新生成其余变分参数。`init_method="spectral"` 直接使用谱载荷方向，`init_method="spectral_varimax"` 则在生成得分前对谱载荷做一次正交 varimax 旋转。每次拟合只使用指定的一个起点。

当前默认使用 `spectral` 初始化和 `float32` 推断。

## 1. 数据与记号

设原始数据为 $X_{\mathrm{raw}}\in\mathbb R^{n\times p}$，截断因子数为 $K$。要求 $n\ge 2$、$p\ge 1$、$K\ge 1$，数据有限，且每个变量在所选推断精度下仍有非零样本变异。

`center=True` 时，先在主机端用 float64 计算并减去各列均值，再将结果转换为推断精度；`center=False` 时不减去均值。以下 $X$ 均指完成这一预处理后的数据。谱初始化函数不再中心化，也不额外标准化变量。

模型与高斯变分因子写为

$$
x_i=B\eta_i+\varepsilon_i,
\qquad
\eta_i\sim N_K(0,I_K),
\qquad
\varepsilon_i\sim N_p(0,\operatorname{diag}(\sigma_1^2,\ldots,\sigma_p^2)),
$$

$$
q(B_{j\cdot}^{\mathsf T})=N_K(\mu_j,V_j),
\qquad
q(\eta_i)=N_K(w_i,\Phi).
$$

记 $M\in\mathbb R^{p\times K}$ 的第 $j$ 行为 $\mu_j^{\mathsf T}$，$W\in\mathbb R^{n\times K}$ 的第 $i$ 行为 $w_i^{\mathsf T}$。代码中的 `mu`、`V`、`W`、`Phi` 分别对应这些均值和协方差；所有数学列下标 $k$ 从 $1$ 开始。

## 2. 谱载荷均值

对 $X$ 的 float64 副本作薄 SVD：

$$
X=UDQ^{\mathsf T},
\qquad
d_1\ge d_2\ge\cdots\ge 0.
$$

采用与推断精度一致的数值秩阈值

$$
\tau=\max(n,p)\,\epsilon_{\mathrm{dtype}}\,d_1,
\qquad
r_X=\#\{\ell:d_\ell>\tau\},
\qquad
r=\min(K,r_X),
$$

其中 $\epsilon_{\mathrm{dtype}}$ 是 float32 或 float64 的机器精度。若 $r=0$，初始化报错。SVD 在 float64 下计算，并不意味着默认路径采用 float64 的数值秩阈值。

定义未加尾列扰动的谱均值矩阵

$$
M^{\mathrm{svd}}_{\cdot,1:r}
=Q_rD_r/\sqrt n,
\qquad
M^{\mathrm{svd}}_{\cdot,r+1:K}=0.
$$

这一缩放与因子先验的单位二阶矩匹配：若暂取 $W_*=\sqrt n\,U_r$，则

$$
\frac1n W_*^{\mathsf T}W_*=I_r,
\qquad
W_*(M^{\mathrm{svd}}_{\cdot,1:r})^{\mathsf T}
=U_rD_rQ_r^{\mathsf T}.
$$

$W_*$ 用于解释 $\sqrt n$ 的缩放。实际的初始得分通过第 6 节的条件更新生成。

## 3. 噪声种子与正定的载荷协方差

先计算每个变量的经验二阶矩及谱载荷解释的部分：

$$
s_j=\frac1n\sum_{i=1}^n X_{ij}^2,
\qquad
h_j=\sum_{k=1}^r (M^{\mathrm{svd}}_{jk})^2.
$$

取

$$
\widetilde\sigma_j^2
=\max\{s_j-h_j,\;0.05s_j\},
\qquad
V_j^{(0)}=\frac{\widetilde\sigma_j^2}{n}I_K.
$$

噪声下限防止保留较多谱方向时将初始噪声压到零；$V_j^{(0)}$ 则保留正的载荷不确定性。若全部非零奇异方向均被保留，精确算术下剩余二阶矩为零，因此噪声种子等于 $0.05s_j$。只保留数值秩对应的方向时，还可能留下低于阈值的谱残差。

这些噪声种子在添加尾列扰动之前确定，之后不重新估计。它们只用于生成初始状态，不是最终的噪声后验均值。`center=False` 时，$s_j$ 是关于零的二阶矩，不是减去样本均值后的方差。

### 3.1 可选的 varimax 旋转

选用 `spectral_varimax` 时，取 $A=Q_rD_r/\sqrt n$，将这 $r$ 列转换到所选推断精度，再从单位矩阵出发迭代求正交矩阵 $R$，以提高 varimax 准则。记 $H=AR$，该准则为

$$
\sum_{k=1}^{r}\left[
  \sum_{j=1}^{p}H_{jk}^{4}
  -\frac{1}{p}\left(\sum_{j=1}^{p}H_{jk}^{2}\right)^2
\right].
$$

这一准则使用原始载荷尺度，各行保留其原有权重。每一步用 SVD 计算所需的正交极因子。若 $r=1$，则直接使用 $R=I_1$。旋转后用 $AR$ 作为前 $r$ 列的均值。正交性保证每行的谱解释量 $h_j$ 在精确算术下相同，因此两个起点使用相同的噪声种子和 $V_j^{(0)}$。

## 4. 仅给额外列加入小扰动

令 $L=K-r$。若 $L=0$，将选定方向的谱载荷直接送入第 5 节的排序。此时初始化完全由数据和数值设置确定。

若 $L>0$，在选定的 $r$ 个谱方向之后生成额外列。两种方法都用下式生成这些列，因此相同种子给出相同的尾列扰动。

$$
M_{jk}
=\left[\widetilde\sigma_j^2
\min\left\{\frac1n,\frac{0.01}{L}\right\}\right]^{1/2}Z_{jk},
\qquad
Z_{jk}\overset{\mathrm{iid}}{\sim}N(0,1),
\qquad k=r+1,\ldots,K.
$$

使用局部生成器 `np.random.default_rng(init_seed)`，默认 `init_seed=2026`，不改变全局 NumPy 随机数状态。对固定数据，额外载荷每行的期望总能量为

$$
\mathbb E_Z\!\left[\sum_{k=r+1}^K M_{jk}^2\right]
=\widetilde\sigma_j^2\min\left\{\frac{L}{n},0.01\right\}
\le 0.01\widetilde\sigma_j^2.
$$

这是期望界，不是每次随机实现都满足的硬上界。它防止尾列数量增大时，总扰动能量随之无限累积。

扰动用于打破零均值列的固定状态：在当前对角初始协方差下，如果某一整列载荷均值为零，则相应得分均值也为零，后续更新可能一直保持该列为零。只赋予该列正的变分方差不足以打破这一状态。因此，仅当缺少可用谱方向时才添加非零尾列；不对全部载荷加入随机抖动。

## 5. 按收缩统计量排列列次序

将均值及协方差对角元转换为所选推断精度后，计算

$$
t_k=\sum_{j=1}^p
\left\{M_{jk}^2+(V_j^{(0)})_{kk}\right\}^{1/4}.
$$

实际排序统计量在这些已存储数值的 float64 副本上计算。采用稳定的降序排列，把 $t_k$ 较大的列放在前面，并将排序后的均值记为 $M^{(0)}$。由于 $V_j^{(0)}$ 是单位矩阵的倍数，同步置换其坐标后仍保持原值。此时尚未生成得分，后续直接依据排序后的载荷生成 $W^{(0)}$ 和 $\Phi^{(0)}$。

该排序使用辅助因子更新所依赖的四次根二阶矩之和，而非载荷列的平方范数。对于 $c_1,c_2>0$，前列具有较小的 $2p+a+k^{c_1}$ 和较大的 $k^{-c_2}$，这一排列与模型按列递增的收缩结构对应。

更具体地，在固定高斯因子并同步置换载荷与得分坐标、刷新辅助分布的有限置换问题中，相关 ELBO 项为

$$
-\sum_{k=1}^K (2p+a+k^{c_1})
\log\left(t_{\pi(k)}+k^{-c_2}\right).
$$

在精确算术下，对给定方向的载荷，降序排列最大化这一置换目标。varimax 可能改变各列的 $t_k$，因此选择旋转起点时，在旋转和尾列生成之后对全部 $K$ 列排序一次。两种方法均根据最终排好的载荷生成初始得分并刷新辅助因子和噪声因子。

## 6. 用噪声种子生成初始得分

设

$$
C^{\mathrm{seed}}
=\operatorname{diag}(\widetilde\sigma_1^{-2},\ldots,\widetilde\sigma_p^{-2}).
$$

用得分的条件更新公式生成

$$
\Phi^{(0)}
=\left[
I_K+(M^{(0)})^{\mathsf T}C^{\mathrm{seed}}M^{(0)}
+\sum_{j=1}^p\frac{V_j^{(0)}}{\widetilde\sigma_j^2}
\right]^{-1},
$$

$$
W^{(0)}=X C^{\mathrm{seed}}M^{(0)}\Phi^{(0)},
\qquad
S^{(0)}=(W^{(0)})^{\mathsf T}W^{(0)}+n\Phi^{(0)}.
$$

特别地，在精确算术下

$$
\sum_{j=1}^p\frac{V_j^{(0)}}{\widetilde\sigma_j^2}
=\frac pn I_K.
$$

因此不能在得分协方差中忽略载荷不确定性：当 $p\gg n$ 时，其累计贡献并不小。实现通过 Cholesky 分解计算所需协方差和对数行列式。

这里可将噪声种子精度理解为临时逆 Gamma 因子 $\operatorname{IG}(\alpha_n,\alpha_n\widetilde\sigma_j^2)$ 的期望精度，其中 $\alpha_n=a_\sigma+n/2$；代码直接使用 $C^{\mathrm{seed}}$，不需要保存这个临时因子。

## 7. 刷新辅助因子与噪声因子

先依据当前载荷矩刷新辅助因子。令

$$
E_{jk}^{(0)}=(\mu_{jk}^{(0)})^2+(V_j^{(0)})_{kk},
\qquad
m_k^{(0)}=\sum_{j=1}^p(E_{jk}^{(0)})^{1/4}+k^{-c_2},
$$

$$
\rho_{jk}^{(0)}
=\mathbb E_q\!\left[\frac{\lambda_k^4}{\tau_{jk}^2}\right]
=\frac{2p+a+k^{c_1}}
{2(E_{jk}^{(0)})^{3/4}m_k^{(0)}}.
$$

这与正文 Proposition 4.1 的条件最优辅助分布对应；实现保存后续更新需要的 $\rho$ 和 $m$，不单独抽样辅助变量。计算分数次幂时，代码对 $E_{jk}^{(0)}$ 使用所选浮点类型的最小正规正数作为下限。

再根据同一组高斯矩更新噪声因子：

$$
R_j^{(0)}
=\left\|X_{\cdot j}-W^{(0)}\mu_j^{(0)}\right\|_2^2
+n(\mu_j^{(0)})^{\mathsf T}\Phi^{(0)}\mu_j^{(0)}
+\operatorname{tr}\!\left(V_j^{(0)}S^{(0)}\right),
$$

$$
\alpha_n=a_\sigma+\frac n2,
\qquad
\beta_j^{(0)}=b_\sigma+\frac12 R_j^{(0)},
\qquad
c_j^{(0)}=\frac{\alpha_n}{\beta_j^{(0)}}.
$$

此时 $q_j(\sigma_j^2)=\operatorname{IG}(\alpha_n,\beta_j^{(0)})$，逆 Gamma 的密度约定为 $q_j(s)\propto s^{-\alpha_n-1}\exp(-\beta_j^{(0)}/s)$，$s>0$。因此 $c_j^{(0)}=\mathbb E_q[1/\sigma_j^2]$，而噪声后验均值为 $\beta_j^{(0)}/(\alpha_n-1)$。

实现直接计算残差平方范数，不将它展开成三个较大数的加减，以减少单精度下的消去误差。随后以零为下限保护残差二阶矩的舍入误差。

刷新后的 $c_j^{(0)}$ 一般不同于 $\widetilde\sigma_j^{-2}$。初始化在这里不再次更新 $W^{(0)},\Phi^{(0)}$，也不更新载荷均值和协方差；正式 CAVI 的第一次得分更新才使用刷新后的 $c^{(0)}$。

## 8. 初始 ELBO 与进入 CAVI 的状态

完成以上刷新后，初始 ELBO 的可变部分为

$$
\begin{aligned}
\mathcal L_{\mathrm{var}}^{(0)}
={}&-\alpha_n\sum_{j=1}^p\log\beta_j^{(0)}
+\frac12\sum_{j=1}^p\log\det V_j^{(0)}
+\frac n2\log\det\Phi^{(0)}\\
&-\frac12\operatorname{tr}S^{(0)}
-\sum_{k=1}^K(2p+a+k^{c_1})\log m_k^{(0)}.
\end{aligned}
$$

报告的绝对 ELBO 为 $\mathcal L^{(0)}=C_0+\mathcal L_{\mathrm{var}}^{(0)}$，其中

$$
\begin{aligned}
C_0={}&p\left[\log\Gamma(\alpha_n)-\log\Gamma(a_\sigma)
+a_\sigma\log b_\sigma\right]
-\frac{np}{2}\log(2\pi)\\
&+\frac{pK}{2}\{1+\log(2\pi)\}+\frac{nK}{2}\\
&+\sum_{k=1}^K\left[
\log\Gamma(2p+a+k^{c_1})-\log\Gamma(a+k^{c_1})
-c_2(a+k^{c_1})\log k-p\log4
\right].
\end{aligned}
$$

该简化表达式要求噪声与辅助因子已经针对当前高斯矩完成更新，即 $\beta_j^{(0)}$ 和 $m_k^{(0)}$ 与当前 $M^{(0)},V^{(0)},W^{(0)},\Phi^{(0)}$ 一致。它不要求所有变分块同时达到坐标最优，也不要求得分已经针对刷新后的噪声重新更新。不能直接以临时噪声种子代替 $\beta_j^{(0)}$ 计算这一 ELBO。

完整初始化顺序为：

```text
预处理数据 X
    → 薄 SVD 和数值秩
    → 谱载荷均值
    → 固定噪声种子、正定的载荷协方差
    → 若选择 spectral_varimax，对前 r 个谱载荷方向做正交旋转
    → 仅在 K > r 时生成额外载荷列
    → 按四次根二阶矩之和降序排列
    → 用噪声种子精度生成 W、Phi、S
    → 刷新辅助因子的 rho、m
    → 刷新噪声因子的 alpha_n、beta、c
    → 计算初始 ELBO 并检查状态有效性
    → 进入原有 CAVI 循环
```

正式 CAVI 的更新顺序保持为“得分 → 载荷 → 辅助因子 → 噪声”。初始化返回 `mu, V, W, Phi, S, alpha_n, beta, c, rho, m, elbo_var`，以及状态检查标志和 `initialization_info`。

## 9. 默认设置、数值约定与调用

| 项目 | 当前设置 |
|---|---|
| 默认初始化 | `init_method="spectral"` |
| 可选旋转起点 | `init_method="spectral_varimax"` |
| 默认推断精度 | `precision="float32"` |
| 尾列随机种子 | `init_seed=2026` |
| 噪声种子下限比例 | `0.05`，当前为内部常量 |
| 尾列期望能量上限比例 | `0.01`，当前为内部常量 |
| 载荷协方差尺度 | $V_j^{(0)}=\widetilde\sigma_j^2 I_K/n$，无额外可调倍数 |
| 默认中心化 | `center=True` |
| 收缩先验 | `a=10.0, c1=2.3, c2=1.0` |
| 噪声先验 | `a_sigma=1.0, b_sigma=1.0` |
| CAVI 停止设置 | `max_iter=1000, tol=0.1, drop_budget=0.5` |

SVD 和种子统计在主机端以 float64 计算，输入是已经转换为推断精度的数据；因此提升为 float64 不会恢复先前转换时丢失的精度。状态数组、先验列权重、条件更新及 ELBO 可变部分采用所选推断精度。常数 $C_0$ 在主机端计算并加到报告历史中，停止判据使用可变部分的绝对变化。

显式选择 `precision="float64"` 时，预处理后的数据和后续推断均使用 float64，不经过 float32 中间转换；这会开启进程级 JAX x64 标志并保持开启。默认 float32 调用仍显式使用 float32 数组。

常数列会被拒绝并报告列索引，包括在推断精度转换后变成常数的列。噪声种子、载荷方差及种子精度必须是有限的正规正浮点数；整列载荷均值下溢到零、状态非有限或分解检查失败时，初始化报错，而不静默删除变量或改变数据尺度。

```python
from bfm import CaviBFM

# 当前默认方案：一次谱初始化，单精度 CAVI。
vi = CaviBFM(K=50).fit(X)

# 将初始化及精度设置写明，便于复现实验。
vi = CaviBFM(
    K=50,
    init_method="spectral",
    init_seed=2026,
    precision="float32",
    center=True,
).fit(X)

# 在谱载荷上应用 varimax，然后运行一次 CAVI。
vi_rot = CaviBFM(K=50, init_method="spectral_varimax").fit(X)
```

两种谱初始化都从数据计算初始秩和协方差尺度。若已有初始变分参数，可通过 `initial_state` 传入。

拟合后可查看 `vi.initialization_method_`、`vi.initialization_info_` 和 `vi.initialization_sec_`，分别核对实际初始化方法、数值秩及尾列等信息、初始化时间。`spectral_varimax` 的 `initialization_info_["varimax_n_iter"]` 记录旋转迭代次数。若希望按最终 ELBO 选择起点，可用相同数据、先验和数值设置分别拟合两次，再比较各自的 `elbo_[-1]`。

## 10. 实现位置

| 内容 | 文件与函数 |
|---|---|
| CAVI 与 Gibbs 共用的载荷和噪声种子 | [_spectral.py](bfm/_spectral.py) 中的 `spectral_seeds` |
| 从谱种子生成 CAVI 初始变分参数 | [spectral_initialization.py](bfm/_vi/spectral_initialization.py) 中的 `spectral_initialization` |
| 用谱载荷与逐变量噪声种子启动 Gibbs 采样 | [GS_LH.py](bfm/_gs/GS_LH.py) 中的 `GS_LH`，指定 `init_method="spectral_varimax"` |
| 正交 varimax | [varimax.py](bfm/_rsp/kernels/varimax.py) 中的 `varimax_batch` |
| 中心化与精度转换 | [base.py](bfm/base.py)：`BaseBFM._prepare_X` |
| 得分更新 | [eta.py](bfm/_vi/eta.py)：`update_eta` |
| 辅助因子矩 | [shrinkage.py](bfm/_vi/shrinkage.py)：`update_shrinkage` |
| 噪声条件更新 | [sigma2.py](bfm/_vi/sigma2.py)：`update_sigma2` |
| ELBO 及其常数 | [elbo.py](bfm/_vi/elbo.py)：`elbo`、`elbo_const` |
| 方法选择与 CAVI 驱动 | [VI.py](bfm/_vi/VI.py)：`CAVI` |
| 公共接口 | [cavi.py](bfm/cavi.py)：`CaviBFM` |
