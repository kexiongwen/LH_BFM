"""A single spectral start with noise-scaled variational covariances.

The driver supplies the prepared data and handles centering.
The SVD and seed statistics are computed on the host in float64;
the returned variational state uses the selected inference precision.
Float32 remains the default; float64 explicitly enables JAX's x64 flag.

The initial loading means are right singular vectors scaled by d / sqrt(n).
Residual second moments, with a 5% per-variable floor, seed the noise and
the loading covariances V_j = noise_j / n * I. Optional varimax rotates
the retained spectral columns at this point. Only columns beyond the
numerical data rank receive fixed-seed perturbations.  Their expected
total squared magnitude per row is at most 1% of that row's noise seed.
Columns are then ordered by their fourth-root second-moment sums, before
the factor-score, noise, and auxiliary distributions are generated.

These constants specify a reproducible candidate initialization; they do
not assert improved convergence or statistical accuracy.
"""

from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np

from .._spectral import _positive_floating, spectral_seeds
from .B import log_det_V
from ._precision import resolve_precision
from .eta import update_eta
from .shrinkage import update_shrinkage
from .sigma2 import mean_inverse, update_sigma2

__all__ = ["spectral_initialization"]

def spectral_initialization(
    X: jax.Array,
    K: int,
    a_sigma: float,
    b_sigma: float,
    shape_k: jax.Array,
    kneg_c2: jax.Array,
    *,
    seed: int = 2026,
    precision: str = "float32",
    varimax: bool = False,
) -> dict:
    """Build the refreshed initial state for one CAVI run.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Prepared data.  This function never centers it again.  Every
        variable must have nonzero sample variation; constant columns
        are rejected instead of being silently dropped.
    K : positive int
        Loading truncation.  The spectral rank is
        min(K, numerical_rank(X)), with numerical-rank threshold
        max(n, p) * eps_selected_precision * largest_singular_value.
    a_sigma, b_sigma : positive float
        Inverse-gamma noise-prior parameters.
    shape_k, kneg_c2 : array-like, shape (K,)
        Column weights 2p + a + k**c1 and k**(-c2), respectively.
    seed : nonnegative int, default 2026
        Local NumPy generator seed used only when K exceeds the
        numerical data rank.  The global NumPy RNG is not changed.
    precision : {"float32", "float64"}, default "float32"
        Floating dtype of the prepared input and returned variational state.
        Float64 enables the process-wide JAX x64 flag and leaves it enabled.
        Host SVD/seed arithmetic remains float64 for either setting.
    varimax : bool, default False
        Rotate the retained SVD loadings using raw orthogonal varimax in
        the inference dtype after computing the noise seeds. Generate any
        extra columns afterwards, then order all columns together.

    Returns
    -------
    dict
        The same recurrence arrays as the original initializer, plus
        ``initialization_info`` with compact seed diagnostics.  Noise
        rates and auxiliary factors have been refreshed for the returned
        Gaussian moments, so the simplified ELBO is valid at this state.
    """
    dtype = resolve_precision(precision)
    for name, value in (("a_sigma", a_sigma), ("b_sigma", b_sigma)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    a_sigma_seed = _positive_floating(a_sigma, "a_sigma", dtype)
    b_sigma_seed = _positive_floating(b_sigma, "b_sigma", dtype)

    mu_seed, _, variance_seed, precision_seed, diagnostics = spectral_seeds(
        X, K, seed=seed, dtype=dtype, varimax=varimax,
    )
    weights = []
    for name, value in (("shape_k", shape_k), ("kneg_c2", kneg_c2)):
        arr = np.asarray(value)
        if arr.shape != (K,):
            raise ValueError(f"{name} must have shape ({K},), got {arr.shape}")
        weights.append(jnp.asarray(_positive_floating(arr, name, dtype), dtype=dtype))
    shape_k, kneg_c2 = weights

    mu = jnp.asarray(mu_seed, dtype=dtype)
    V = (jnp.asarray(variance_seed)[:, None, None]
         * jnp.eye(K, dtype=dtype)[None, :, :])
    # The CAVI driver already supplies X on device in the inference dtype.
    # For other inputs, retain the host float64-to-inference conversion used
    # for the seeds, including numeric lists and object arrays.
    if isinstance(X, jax.Array) and X.dtype == dtype:
        X_device = X
    else:
        X_device = jnp.asarray(np.asarray(X, dtype=np.float64).astype(dtype, copy=False))
    W, Phi, S, elbo_eta, ok_eta = update_eta(
        X_device, mu, V, jnp.asarray(precision_seed)
    )
    rho, m, elbo_shrink = update_shrinkage(mu, V, shape_k, kneg_c2)
    alpha_n, beta, elbo_sigma2 = update_sigma2(
        X_device, W, Phi, mu, V,
        jnp.asarray(a_sigma_seed), jnp.asarray(b_sigma_seed), S=S,
    )
    c = mean_inverse(alpha_n, beta)
    ldV, ok_V = log_det_V(V)
    elbo_var = elbo_eta + 0.5 * ldV.sum() + elbo_shrink + elbo_sigma2

    state = {
        "mu": mu, "V": V, "W": W, "Phi": Phi, "S": S,
        "alpha_n": alpha_n, "beta": beta, "c": c,
        "rho": rho, "m": m, "elbo_var": elbo_var,
        "ok": ok_eta & ok_V,
    }
    bad = [name for name, value in state.items()
           if name != "ok" and not np.isfinite(np.asarray(value)).all()]
    if bad:
        raise FloatingPointError(
            "spectral initialization produced non-finite "
            f"{', '.join(bad)}; rescale the data or hyperparameters"
        )
    state["initialization_info"] = diagnostics
    return state
