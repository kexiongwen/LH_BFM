# BFM — Bayesian Factorization via L₁/₂ Shrinkage

**A unified, scikit-learn-style interface to the JAX Gibbs sampler and CAVI for
the Bayesian factor model with an L₁/₂ shrinkage prior.**

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/powered%20by-JAX-orange)](https://github.com/jax-ml/jax)
[![License](https://img.shields.io/badge/license-GPL--2.0--only-blue)](LICENSE)

`bfm` (distribution name `bfm-lh`) packages the two inference algorithms of the
paper *Bayesian factorization via L₁/₂ shrinkage* behind one API. It is a
single-precision JAX implementation with optional GPU acceleration, RSP factor
alignment, ESS diagnostics, and exact posterior covariance moments.

| Estimator | Inference | Speed | Best for |
|---|---|---|---|
| `GibbsBFM` | exact Gibbs sampling (MCMC) | slower, exact | posterior uncertainty, ESS, factor scores |
| `CaviBFM` | coordinate-ascent variational inference | fast, deterministic | large problems, quick exploration, ELBO diagnostics |

Both estimators share loading/noise/covariance accessors, optional RSP factor
alignment, and support `center=True` preprocessing.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Common API](#common-api)
- [Model covariance moments](#model-covariance-moments)
- [Factor alignment (RSP)](#factor-alignment-rsp)
- [Factor scores](#factor-scores)
- [Effective sample size](#effective-sample-size)
- [Parameters and conventions](#parameters-and-conventions)
- [Examples](#examples)
- [Notes](#notes)
- [License](#license)

---

## Install

```bash
# from the repository root
pip install -e .
```

Requirements: Python ≥ 3.12, JAX/JAXlib, NumPy. For GPU, install the CUDA
build of JAX first (see the JAX installation guide), then install this package.

Optional extras:

```bash
pip install -e ".[plot]"   # matplotlib: CaviBFM.plot_elbo
pip install -e ".[ess]"    # arviz: ESS diagnostics
pip install -e ".[rsp]"    # optax: RSP assignment backend for K > 7
```

The `"scipy"` RSP assignment backend needs `scipy`, which is not in the `rsp`
extra; install it separately if you use that backend.

## Quick start

```python
import numpy as np
from bfm import GibbsBFM, CaviBFM

X = ...  # (n, p) data; column-centered by default

# Exact MCMC
gs = GibbsBFM(K=10, n_iter=2000, burn_in=1000, thin=1, seed=2025).fit(X)
gs.B_samples_        # (S, p, K) kept draws
gs.B_hat_            # posterior-mean loadings
gs.Sigma_            # plug-in covariance: B_hat_ B_hat_^T + diag(sigma2_hat_)
gs.n_factors()       # number of active factors
gs.ess_summary()     # per-entry ESS (requires arviz)

# Fast variational approximation (deterministic)
vi = CaviBFM(K=10, max_iter=1000, tol=1e-1).fit(X)
vi.B_hat_
vi.Sigma_
vi.elbo_             # absolute ELBO history, length n_iter_ + 1
vi.converged_, vi.stop_reason_, vi.n_iter_
vi.plot_elbo(show=True)  # requires matplotlib
```

Both estimators are fitted with scikit-learn conventions (`fit` returns
`self`) and accept `center=True` by default. The center means are computed
and subtracted in float64, then the centered matrix is cast to float32 for
the single-precision JAX inference.

## Common API

After `fit`, both estimators expose:

| Attribute / method | Meaning |
|---|---|
| `B_hat_` | loading point estimate: posterior mean for `GibbsBFM`, variational mean for `CaviBFM` |
| `sigma2_hat_` | noise-variance point estimate |
| `Sigma_` | `B_hat_ @ B_hat_.T + diag(sigma2_hat_)` |
| `n_factors(level=0.95, ...)` | number of active factor columns from marginal credible intervals; `CaviBFM` also supports `method="linf"` (see below) |
| `active_entries(...)` | entrywise active mask behind `n_factors()` |
| `x_offset_` | column means removed when `center=True`, stored as float64 |
| `raw_output_` | the underlying driver's full return value |
| `align_factors(...)` | RSP post-processing (see below) |

`GibbsBFM` additionally provides `B_samples_`, `sigma2_samples_`, `ess()`,
`ess_summary()`, `sample_factors()`, and Monte Carlo covariance moments.
`CaviBFM` additionally provides the ELBO history, convergence diagnostics, and
exact mean-field covariance moments.

### CAVI factor selection by column magnitude

`CaviBFM` also supports an infinity-norm criterion on the variational
loading mean.  Column `k` is active exactly when
`max(abs(mu_[:, k])) > threshold`; equality is inactive.

```python
vi.n_factors()                              # existing credible-interval rule
vi.n_factors(method="linf")                 # threshold=1e-1
vi.n_factors(method="linf", threshold=0.05)  # configurable cutoff
vi.active_entries(method="linf")            # matching entrywise activity mask
```

The magnitude cutoff must be finite and nonnegative and is compared in
the loading means' precision.  It depends on the data scale and does not
use posterior uncertainty.  `level` and `interval` apply only to the
default `method="credible"`.  After alignment, `use_aligned=True` selects
`mu_aligned_`; rotation can change the count.  The `linf` method is
available only on `CaviBFM`: passing `method=` to `GibbsBFM.n_factors()`
or `GibbsBFM.active_entries()` raises `TypeError`.

### Importing driver submodules

`bfm._gs` and `bfm._vi` intentionally re-export same-named driver functions.
As a result, **`bfm._gs.GS_LH`**, **`bfm._vi.elbo`**, and
**`bfm._vi.initialization`** are functions, not modules. Use `importlib` or a
from-submodule import to reach the module itself:

```python
import importlib

gs_lh = importlib.import_module("bfm._gs.GS_LH")   # module
from bfm._gs.GS_LH import _accept_chi2             # module-level symbol
from bfm._vi.elbo import elbo_const

from bfm._gs import GS_LH                          # supported: function
from bfm._vi import elbo                           # supported: function
```

`bfm._rsp` has no such name collisions.

## Model covariance moments

`Sigma = B B^T + Omega` is the model covariance. Both estimators expose
posterior summaries of it.

### Variational (mean-field) moments — `CaviBFM`

Exact moments of `Sigma` under the mean-field variational posterior, in
float64. Mean-field usually underestimates posterior variance.

```python
vi = CaviBFM(K=10).fit(X)
vi.Sigma_mean_    # E_q[Sigma] = Sigma_ + diag(tr V_1, ..., tr V_p)
vi.Sigma_var_     # entrywise Var_q(Sigma_ij), computed lazily
vi.Sigma_sd_
vi.cov_Sigma([((i, j), (k, l)), ...])   # selected entry-pair covariances
```

`Sigma_var_`/`cov_Sigma` require `alpha_n = a_sigma + n/2 > 2`. The `p^2`-entry
variance matrix is capped at `2e7` entries by default (`p <= 4472`); use
`cov_Sigma` on selected pairs for larger `p`.

### Monte Carlo moments — `GibbsBFM`

Consistent estimates from the retained draws (`ddof=0`), computed lazily:

```python
gs = GibbsBFM(K=10).fit(X)
gs.Sigma_mean_      # posterior mean; alias of Sigma_mc_
gs.Sigma_var_       # entrywise posterior variance
gs.Sigma_sd_
gs.Sigma_diag_var_  # cheap diagonal variances, always uses all draws
gs.cov_Sigma([...])
gs.Sigma_draws_used_
```

The full `p x p` scan costs `O(S p^2 K)`. Above the compute budget,
`Sigma_var_` uses a deterministic equidistant subsample and warns;
`Sigma_draws_used_` records the count. Call `posterior_sigma_var(max_draws=None)`
to force all draws. Draws are autocorrelated, so use `ess()` to assess the Monte
Carlo error.

## Factor alignment (RSP)

The likelihood and reconstruction are invariant to matching orthogonal
transformations of loadings and scores, but the shrinkage prior is not.
Rotation-Sign-Permutation (RSP) post-processing picks an orientation for
display and comparison; it does not make the posterior invariant.

```python
gs = GibbsBFM(K=10, n_iter=2000, burn_in=1000, seed=2025).fit(X)
gs.align_factors()          # varimax + sign/permutation alignment

gs.B_hat_                   # aligned posterior mean
gs.B_aligned_               # (S, p, K) aligned draws; B_samples_ stays raw
gs.rsp_result_              # losses, signs, permutations, transforms, diagnostics
```

Key semantics:

- `B_hat_` becomes the aligned mean; raw draws and `ess()` are unchanged.
- `Sigma_` changes under alignment; `Sigma_mc_` (per-draw-invariant Monte Carlo
  covariance mean) does not.
- `n_factors(use_aligned=True)` uses aligned draws; the credible rule is
  invariant to signs/permutations but not to the varimax rotation.
- `status_code 4` raises `FloatingPointError` with **no estimator state change**
  (the failed result is attached as `exc.rsp_result`); `status_code 3` warns
  but accepts the finite result.
- With the default `assignment_backend="auto"`, `K > 7` needs `optax`
  (`pip install -e ".[rsp]"`).
- With the default `rotate=True`, an explicit `reference=` must already be in
  the **varimax-rotated** coordinate system. Pass `rotate=False` for a
  reference in the original loading coordinates.
- `precision="float64"` enables JAX's process-wide `jax_enable_x64` flag and
  does not restore it; use it in a dedicated process if that is undesirable.

`CaviBFM.align_factors()` applies RSP to the variational loading mean as a
singleton input and exposes transformed moments separately
(`mu_aligned_`, `W_aligned_`, `V_aligned_`, `Phi_aligned_`); the original VI
parameters and ELBO stay unchanged.

## Factor scores

`GibbsBFM` samples factor scores on demand (not stored on the estimator):

```python
raw = gs.sample_factors(X, seed=2025)
eta = raw["eta_sample"]                         # (S, K, n)

gs.align_factors()
aligned = gs.sample_factors(X, return_eta=False,
                            return_eta_aligned=True, seed=2025)
eta_aligned = aligned["eta_sample_aligned"]     # (S, K, n)
```

For each retained draw, scores are sampled conditional on that draw's
loadings/noise and the supplied observations. The training `x_offset_` is
subtracted. Aligned scores use each draw's RSP transform, so the reconstructed
common component matches up to numerical precision. `batch_size` bounds
intermediate memory; each output occupies `(S, K, n)` host entries.

## Effective sample size

`GibbsBFM` provides optional ArviZ-based ESS diagnostics:

```python
gs.ess()                                   # per-entry bulk ESS
gs.ess(method="tail", prob=0.9)            # tail ESS
gs.ess_summary()                           # compact min/median/mean/max
```

Multi-chain arrays `(C, S, p, K)` / `(C, S, p)` are accepted. For `p == 1`,
sigma2 `(C, S)` is interpreted using B's layout; pass `(C, S, p)` explicitly
to be unambiguous. Constant entries are forwarded from ArviZ as
`ESS = total draws`; treat them as degenerate, not as evidence of good mixing.

## Parameters and conventions

High-level estimators map one-to-one onto the underlying drivers:

- `GibbsBFM(...)` → `bfm._gs.GS_LH(X, K, ...)`
- `CaviBFM(...)` → `bfm._vi.CAVI(X, K, ...)`

Two deliberate wrapper-level differences:

1. **Centering is done once** in the shared base class. The driver receives the
   already-centered matrix (`GS_LH` never centers; `CAVI` is called with
   `center=False`).
2. **Point estimates are unified**: `sigma2_hat_` is the posterior mean for
   `GibbsBFM` and the inverse-gamma posterior mean for `CaviBFM`.

Notable defaults:

- `CaviBFM` uses absolute tolerance `tol=1e-1` on the variable part of the
  ELBO, with a separate `drop_budget=0.5`; a larger ELBO decrease raises
  `FloatingPointError`.
- Hyperparameters violating the paper's conditions (`a >= 4` and
  `4(c1 + c2) > 1`) trigger one `UserWarning` but still run.
- `thin` follows the `GS_LH` convention: it keeps the **last** sweep of
  each block. The first retained sweep after burn-in is `burn_in + thin`, and
  the number of kept draws is `(n_iter - burn_in) // thin`.
- Inference runs in float32; JAX x64 is only enabled by explicit opt-ins
  (e.g. RSP `precision="float64"`).
- On a GPU backend, GS may use a chi-square Gamma approximation when the shape
  rounding is negligible and the estimated normal-draw cost is below
  `chi2_max_normals` (default `8e6`). `GibbsBFM.gamma_mode_` and
  `GibbsBFM.chi2_normals_` report the choice; lower `chi2_max_normals` forces
  exact Gamma. `chi2_chunk_elems` bounds the sampling chunk memory.
- `graph=True` uses XLA command buffers on GPU and warns/falls back elsewhere.

## Examples

The notebooks in [`examples/`](examples/) run both estimators on the same
synthetic data and cover factor-dimension selection, noise/covariance recovery,
CAVI ELBO trajectories, ESS diagnostics, and RSP alignment:

- [`demo1.ipynb`](examples/demo1.ipynb) — block loadings, homoscedastic noise (`n=100, p=1956, r=5`)
- [`demo2.ipynb`](examples/demo2.ipynb) — sparse random loadings, heteroscedastic noise (`n=100, p=1000, r=5`)

Both notebooks start with a CPU/GPU selection cell (`MODE = "cpu"` by default).
To use an NVIDIA GPU, set `MODE = "gpu"` in a CUDA-enabled Linux/WSL2 Jupyter
kernel, then restart the kernel and run all cells. Saved outputs reflect the
last execution; rerun the notebooks to reproduce results.

## Notes

<details>
<summary><strong>Precision and memory notes</strong></summary>

- Inference runs in float32. The preprocessing centering uses a float64 host
  intermediate, then casts the centered matrix to float32.
- `CaviBFM` posterior covariance moments are computed and returned in float64.
- `precision="float64"` in RSP is opt-in and changes the process-wide JAX x64
  flag without restoring it.
- RSP varimax matrix products are pinned to JAX's `highest` matmul precision,
  so float32 RSP results are not limited by backend defaults such as TF32.
- Large `p` covariance matrices are guarded by entry limits; use the selected
  pair APIs when `p` is large.

</details>

## License

`bfm-lh` is distributed under the **GNU General Public License v2.0 only**
(`GPL-2.0-only`). The full text is in the repository-root [`LICENSE`](LICENSE)
file, and the package metadata declares `license = "GPL-2.0-only"` with
`license-files = ["LICENSE"]`.
