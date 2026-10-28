# PBMC3k 固定工作流程

## 一、当前固定方案

当前确认并固定的工作流程为：

```text
原始 counts
→ 统一质控与归一化
→ 2000 个高变基因上的 X_scaled
→ CAVI / Gibbs（K=50，center=False）
→ RSP 对齐
→ 载荷列 L2 范数加权
→ 前 20 列 score 构建欧氏邻域图
→ Leiden（resolution=0.45）
→ CAVI 与 Gibbs 均为 9 簇
→ Wilcoxon top 5 markers、UMAP 和诊断报告
```

PCA 仅作为参照，不视为细胞类型真值。ARI/NMI（其中 NMI 实际使用 AMI）只表示聚类一致性，不称为准确率。9 簇是当前固定表示下的调参结果，不直接解释为真实细胞类型数或生物学最优。

## 二、数据预处理

实现文件：`PBMC/data_processing.py`。固定原始输入为 `PBMC/data/pbmc3k_raw.h5ad`，原始维度为 **2700 × 32738**。保存输入路径、SHA-256、软件版本和分阶段配置。

### 1. 过滤与质控

严格按以下顺序执行：

1. `sc.pp.filter_cells(adata, min_genes=200)`。
2. `sc.pp.filter_genes(adata, min_cells=3)`。
3. 以 `MT-` 前缀标记线粒体基因，调用 `sc.pp.calculate_qc_metrics(qc_vars=["mt"], percent_top=None, log1p=False)`。
4. 保留满足 `200 < n_genes_by_counts < 2500` 且 `pct_counts_mt < 5` 的细胞，质控后为 **2638 × 13714**。
5. 在归一化前执行 `adata.layers["counts"] = adata.X.copy()`。

### 2. 归一化与高变基因

1. 执行 `sc.pp.normalize_total(adata, target_sum=1e4)`。
2. 执行 `sc.pp.log1p(adata)`，并保存 `lognorm` 层供 marker 分析使用。
3. 使用原始 `counts` 层选择 **2000** 个高变基因，固定参数为：
   - `flavor="seurat_v3"`；
   - `n_top_genes=2000`；
   - `min_mean=0.0125`；
   - `max_mean=3`；
   - `min_disp=0.5`。
4. 固定细胞条形码、高变基因标识及顺序。

### 3. 模型输入 `X_scaled`

只在选定的 2000 个高变基因上构造模型矩阵：

```python
adata1 = adata[:, adata.var.highly_variable].copy()
adata1.layers["scaled"] = adata1.X.toarray()
sc.pp.regress_out(adata1, ["total_counts", "pct_counts_mt"], layer="scaled")
sc.pp.scale(adata1, max_value=10, layer="scaled")
X = adata1.layers["scaled"]
```

固定结果：

- 输入矩阵为 **2638 × 2000**，全部有限，取值范围为 **[-10, 10]**；
- `X_scaled` 是 PCA、CAVI 和 Gibbs 的统一模型输入；
- `lognorm` 仅用于 Wilcoxon marker 分析；
- CAVI 和 Gibbs 均使用 `center=False`，不再附加 BFM 居中；
- 稠密数组仅针对 2000 个高变基因生成。

固定预处理输出：

- `PBMC/data/notebook_scaled_preprocessed.h5ad`
- `PBMC/data/notebook_scaled_preprocessed_model_inputs.npz`
- `PBMC/figures/notebook_scaled_qc/`

## 三、模型拟合与 RSP 对齐

### 1. 共同设置

| 项目 | 固定值 |
|---|---|
| 输入 | `X_scaled` |
| 截断上限 | `K=50` |
| BFM 额外居中 | `center=False` |
| 先验 | `a=10, c1=2, c2=1, a_sigma=1, b_sigma=1` |
| 对齐 | RSP：varimax + signed-permutation/reference 更新 |
| 得分权重 | 当前坐标系下载荷列的 L2 范数 |

RSP 对齐前后均保存载荷列范数图：CAVI 比较 `mu_` 与 `mu_aligned_`，Gibbs 比较原始载荷后验均值与对齐后载荷后验均值。

### 2. CAVI

固定配置与结果：

- 确定性 PPCA 初始化；
- `tol=0.1, drop_budget=0.5, max_iter=1000`；
- 第 **465** 轮满足停止准则；
- RSP `status_code=1`；
- 完整模型活动因子数为 **34**。

模型文件：

- `PBMC/data/notebook_scaled_cavi_norm_d10_k50.h5ad`
- `PBMC/data/notebook_scaled_cavi_norm_d10_k50_model.npz`
- `PBMC/figures/notebook_scaled_cavi_norm_d10_k50/loading_column_norms_cavi_scaled_norm_d10.png`

### 3. 单链 Gibbs

固定采样配置：

| 项目 | 固定值 |
|---|---|
| `n_iter` | 20000 |
| `burn_in` | 10000 |
| `thin` | 10 |
| `seed` | 2026 |
| 保留样本 | 1000，轮次 `10010–20000` |
| Gamma 路径 | exact |

固定结果：

- RSP `status_code=1`；
- 完整模型活动因子数为 **26**；
- `review_required=True`，载荷和噪声方差的最小 bulk ESS 分别约为 **16.06** 和 **80.40**，仍需保留采样诊断警示。

条件均值按参数样本逐批计算：

```math
M_s=(I_K+B_s^\top\Omega_s^{-1}B_s)^{-1}B_s^\top\Omega_s^{-1}X_c^\top.
```

每个样本使用自己的 RSP 变换 \(R_s\)，累计 \(R_s^\top M_s\) 后取平均，不保存完整 `(S,K,n)` 分数数组。

模型文件：

- `PBMC/data/notebook_scaled_gibbs_norm_d10_n20000_b10000_t10_k50.h5ad`
- `PBMC/data/notebook_scaled_gibbs_norm_d10_n20000_b10000_t10_k50_model.npz`
- `PBMC/figures/notebook_scaled_gibbs_norm_d10_n20000_b10000_t10_k50/loading_column_norms_gibbs_scaled_norm_d10.png`

上述 `d10` 文件用于保存模型拟合、参数样本和诊断；最终聚类表示统一使用前 20 列，见下一节。

## 四、前 20 列加权 score 与 9 簇聚类

### 1. 固定 score 构造

先加权，再取前 20 列，顺序不得交换：

- CAVI：`W_aligned_` 乘以 `mu_aligned_` 各列的 L2 范数；
- Gibbs：后验条件均值 score 乘以当前坐标系下 `B_hat_` 各列的 L2 范数；
- 两种方法均取加权结果的列 `0:20`，得到 **2638 × 20** 的表示；
- 前 20 列在两种模型中均为活动因子列。

前 20 列结果存档：`PBMC/data/notebook_scaled_scores_d20_cavi_gibbs_k50.h5ad`。

### 2. 固定邻域图与 Leiden 参数

| 项目 | 固定值 |
|---|---|
| 距离 | Euclidean |
| `n_neighbors` | 10 |
| `resolution` | 0.45 |
| `random_state` | 0 |
| `flavor` | `igraph` |
| `n_iterations` | 2 |
| `directed` | False |

CAVI 和 Gibbs 使用共同分辨率 **0.45**，均得到 **9 个簇**。分辨率搜索记录保存在最终结果 `uns['leiden9_d20']` 中。仅调整 Leiden 标签时，复用相同邻域图和 UMAP 坐标。

图和报告中的簇编号从 **1** 开始；底层结果和 marker 文件中的原始簇编号仍从 0 开始，二者通过“展示编号 = 原始编号 + 1”对应。

### 3. Marker 与图形

- 在全部质控基因的 `lognorm` 上执行 Wilcoxon 排序，`use_raw=False`；
- 每簇输出 top 5 markers，包括 score、log fold change、原始 p 值和校正后 p 值；
- UMAP 仅用于展示，不单独作为算法优劣的证据。

固定 marker 文件：

- `PBMC/data/notebook_scaled_cavi_norm_d20_r0p45_k9_top5_markers.csv/.md`
- `PBMC/data/notebook_scaled_gibbs_norm_d20_r0p45_k9_top5_markers.csv/.md`

## 五、固定结果

| 项目 | PCA_scaled（参照） | CAVI | Gibbs |
|---|---:|---:|---:|
| 簇数 | 8 | 9 | 9 |
| `K` | — | 50 | 50 |
| 活动因子数（完整模型） | — | 34 | 26 |
| 活动因子数（前 20 列） | — | 20 | 20 |
| RSP `status_code` | — | 1 | 1 |
| 与 PCA_scaled 的 ARI | 1.0000 | 0.6468 | 0.6259 |
| 与 PCA_scaled 的 AMI | 1.0000 | 0.8020 | 0.7821 |

CAVI 与 Gibbs 的 9 簇结果之间：ARI **0.8844**、AMI **0.9062**。

最终 9 簇结果：

- `PBMC/data/notebook_scaled_scores_d20_leiden9_cavi_gibbs_k50.h5ad`
- `PBMC/figures/notebook_scaled_clustering_cavi_vs_gibbs_d20_leiden9_k50.png`
- `PBMC/figures/notebook_scaled_clustering_pca_cavi_gibbs_k50.png`
- 汇总报告：`PBMC.ipynb`

## 六、固定复现入口

在 Conda base 环境中运行：

```bash
python PBMC/data_processing.py PBMC/data/pbmc3k_raw.h5ad --output PBMC/data/notebook_scaled_preprocessed.h5ad --qc-figure-dir PBMC/figures/notebook_scaled_qc
python PBMC/PCA.py PBMC/data/pbmc3k_raw.h5ad --output PBMC/data/notebook_scaled_pca_baseline.h5ad --figure-dir PBMC/figures/notebook_scaled_pca
python PBMC/CAVI.py PBMC/data/notebook_scaled_pca_baseline.h5ad --output PBMC/data/notebook_scaled_cavi_norm_d20_k50.h5ad --figure-dir PBMC/figures/notebook_scaled_cavi_norm_d20_k50
python PBMC/GS.py PBMC/data/notebook_scaled_cavi_norm_d20_k50.h5ad --output PBMC/data/notebook_scaled_gibbs_norm_d20_n20000_b10000_t10_k50.h5ad --figure-dir PBMC/figures/notebook_scaled_gibbs_norm_d20_n20000_b10000_t10_k50
```

上述脚本固定完成以下步骤，不再提供其他输入或下游分支：

1. 按载荷列 L2 范数加权 score，并取前 20 列；
2. 使用 `n_neighbors=10` 构建欧氏邻域图；
3. 使用 `resolution=0.45` 和固定 Leiden 参数生成 9 簇；
4. 计算 top 5 markers、RSP 列范数图和 UMAP 对比图；
5. 执行并保存 `PBMC.ipynb` 汇总报告。

当前工作流程、参数和输出以此文件为准。
