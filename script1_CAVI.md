# Demo1 CAVI 50 次重复实验计划

## 1. 目标

新增脚本 `experiment/run_demo1_cavi.py`，使用 `random_seeds.txt` 中的 50 个随机种子重复运行 `examples/demo1.ipynb` 的模拟实验。每次仅使用 CAVI 拟合模型，最终汇报以下四项指标在 50 次实验中的平均值和样本标准差：

1. FDR（false discovery rate）
2. FNDR（false non-discovery rate）
3. 激活的因子数
4. 协方差估计的 Frobenius norm 误差

不修改 `examples/demo1.ipynb` 或 `bfm` 源码。

## 2. 数据生成

- 按文件顺序读取 `random_seeds.txt` 中的 50 个种子，验证种子数量、唯一性及取值范围。
- 固定 demo1 的模拟设定：$n=100$、$p=1956$、真实因子数 $r=5$。
- 按 demo1 构造块状真实载荷矩阵 $B_0$：第 $k$ 列在从第 $364(k-1)$ 行开始的 500 个连续位置上取 1，其余位置取 0。
- 固定真实协方差矩阵 $\Sigma_0=B_0B_0^\top+I_p$。
- 每次使用 `np.random.RandomState(seed)` 生成一份新的数据矩阵 $X$，与 notebook 使用的 NumPy 随机数机制保持一致。
- CAVI 的初始化和更新均为确定性过程，因此种子仅用于生成数据。

## 3. CAVI 设置

每次运行采用以下配置：

```python
vi = CaviBFM(
    K=50,
    a=10.0,
    c1=2.0,
    c2=1.0,
    a_sigma=1.0,
    b_sigma=1.0,
    center=False,
    max_iter=1000,
    tol=1e-1,
).fit(X)
```

50 次实验顺序运行，不运行 Gibbs，也不进行 RSP 对齐。保存每次的收敛状态、停止原因、迭代次数和运行时间，以便诊断异常。

## 4. 指标定义

### 4.1 FDR 和 FNDR

使用 `vi.Sigma_mean_` 估计协方差稀疏结构，阈值设为 `1e-2`：

- 非对角元素满足 $|\widehat\Sigma_{ij}|<10^{-2}$ 时估计为零；等于阈值时仍视为非零。
- 真实零元素由 `Sigma_true == 0.0` 确定。
- 仅统计排除对角线的上三角元素，避免对称元素重复计数。
- 将非零元素视为发现，得到 TP、FP、FN 和 TN。

计算：

$$
\mathrm{FDR}=\frac{FP}{TP+FP},
\qquad
\mathrm{FNDR}=\frac{FN}{FN+TN}.
$$

优先调用现有的 `vi.covariance_sparsity_metrics()`，避免指标口径与包内实现不一致。逐次结果同时保存 TP、FP、FN 和 TN，以便后续核算其他指标。

### 4.2 激活的因子数

采用 CAVI 的 95% 边际高斯可信区间规则：

```python
n_active = vi.n_factors(method="credible", level=0.95)
```

至少一个载荷元素的可信区间排除零时，该因子列被计为激活。主结果不使用 RSP 对齐后的载荷。

### 4.3 Frobenius norm 误差

使用变分后验均值协方差的绝对 Frobenius 误差：

$$
\left\|\texttt{vi.Sigma\_mean\_}-\Sigma_0\right\|_F.
$$

该指标使用绝对误差，不除以 $\|\Sigma_0\|_F$。

## 5. 汇总方式

- 对四项指标的 50 个逐次结果分别计算算术平均值。
- 标准差采用样本标准差，即 `np.std(..., ddof=1)`；不使用均值标准误。
- 激活因子数直接对 50 个整数计数求均值和样本标准差。
- 只有 50 次实验全部成功时才生成完整实验汇总；若有失败或缺失，明确报告，不将不足 50 次的结果标记为完整实验。

## 6. 输出文件

- `experiment/results/demo1_cavi_runs.csv`：每个种子一行，包含种子、FDR、FNDR、激活因子数、Frobenius norm 误差、TP、FP、FN、TN，以及收敛和运行诊断。
- `experiment/results/demo1_cavi_summary.csv`：四项指标的 `mean` 和 `std`。
- 终端打印逐次进度和最终汇总表。每次完成后保存进度，以支持中断后继续。

## 7. 验证

1. 先用第一个种子按完整参数运行一次，检查数据形状、模型输出、指标计算和文件格式。
2. 正式运行后核对结果包含恰好 50 个唯一种子，所有用于汇总的指标均为有限值。
3. 从逐次 CSV 独立重算四项指标的均值和样本标准差，并与汇总文件核对。
4. 检查随机种子、CAVI 参数和指标阈值均被完整记录，确保实验可复现。
