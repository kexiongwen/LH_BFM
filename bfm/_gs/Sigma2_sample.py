"""
Sigma2_sample.py  (JAX, single precision)

Idiosyncratic-variance step of the Gibbs sampler (Section 3 of the
paper "Bayesian factorization via L_{1/2} shrinkage"):
sample the idiosyncratic error variances sigma_j^2, j = 1, ..., p,
independently from their inverse-gamma full conditional

    sigma_j^2 | X_{.j}, B_j, eta
        ~ IG( a_sigma + n/2,
              b_sigma + (1/2) * || X_{.j} - eta^T B_j ||^2 ).

Conventions (shared with the whole pipeline):
    B      : (p, K) loading matrix (as returned by B_sample.py).
    eta    : (K, n) latent factor matrix, column i is eta_i (as returned
             by eta_sample.py), so that eta^T B_j is row j of B @ eta.
    return : (p,) vector of variances sigma_j^2, the diagonal of Omega.

Sampling scheme: reciprocal-Gamma identity.  With residuals
RSS_j = || X_{.j} - eta^T B_j ||^2 and g_j ~ Ga(a_sigma + n/2, 1)
(shape-rate), independent across j,

    sigma_j^2 = (b_sigma + RSS_j / 2) / g_j
              ~ IG(a_sigma + n/2, b_sigma + RSS_j/2),

since c / g ~ IG(alpha, c) when g ~ Ga(alpha, 1).  The residual sum of
squares is formed directly (no ||X||^2 - 2 B^T eta X + ... identity),
avoiding catastrophic cancellation when the fit is good; b_sigma > 0
keeps every draw strictly positive.

JAX-specific notes
------------------
* Single precision (float32) throughout; takes an explicit
  jax.random.PRNGKey as its first argument.
* jax.random.gamma is a pure XLA primitive, so -- unlike the PyTorch
  version -- the chi-square Gamma approximation of Chi2.py is never
  mandatory.  The draw is exact unless the driver passes a chi2_plan
  (its _accept_chi2 decision, GPU backend only); see Chi2.py.
* JAX implementation of the sigma2-sample step of the reference
  sampler.
"""

import jax
import jax.numpy as jnp

from .Chi2 import gamma_sample, sigma2_concentration

__all__ = ["sigma2_sample"]

_F32 = jnp.float32


def sigma2_sample(
    key: jax.Array,
    X: jax.Array,
    B: jax.Array,
    eta: jax.Array,
    a_sigma: float,
    b_sigma: float,
    chi2_plan: tuple | None = None,
) -> jax.Array:
    r"""One exact draw of the diagonal variances sigma_j^2, j = 1..p.

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : array-like, shape (n, p)
        Data matrix.
    B : array-like, shape (p, K)
        Loading matrix at the current iteration (output of B_sample()).
    eta : array-like, shape (K, n)
        Latent factors, column i is eta_i (output of eta_sample()).
    a_sigma, b_sigma : float
        Fixed inverse-gamma hyperparameters, a_sigma > 0, b_sigma > 0.
    chi2_plan : tuple or None
        Row-chunked plan from
        make_chi2_plan_scalar(a_sigma + n/2, p) for the chi-square
        Gamma approximation (see Chi2.py).  None draws exact Gamma.
        Supplied by the driver when its _accept_chi2 decision accepts
        the rounding perturbation and the cost gate passes (GPU only).

    Returns
    -------
    jnp.ndarray, shape (p,), float32
        The drawn variances sigma_j^2 (diagonal of Omega), independent
        across j and strictly positive.
    """
    X = jnp.asarray(X, dtype=_F32)
    B = jnp.asarray(B, dtype=_F32)
    eta = jnp.asarray(eta, dtype=_F32)
    n = X.shape[0]

    # RSS_j = || X_{.j} - eta^T B_j ||^2, row j of the (p, n) residual
    rss = jnp.sum((X.T - B @ eta) ** 2, axis=1)

    rate = b_sigma + 0.5 * rss

    # sigma_j^2 = rate_j / g_j,  g_j ~ Ga(a_sigma + n/2, 1) iid across j
    # (gamma_sample: exact Gamma unless the driver passes a chi2_plan)
    g = gamma_sample(key, sigma2_concentration(a_sigma, n),
                     jnp.ones_like(rate), chi2_plan=chi2_plan)
    return rate / g
