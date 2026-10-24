"""Closed-form ELBO of the mean-field approximation (Appendix B2).

L = Const
    - alpha_n * sum_j log(beta_j)
    + 1/2 * sum_j log det V_j
    + n/2 * log det Phi
    - 1/2 * (||W||_F^2 + n Tr(Phi))
    - sum_k (2p + a + k^{c1}) log(m_k),

with alpha_n = a_sigma + n/2 and

Const = p [log Gamma(alpha_n) - log Gamma(a_sigma) + a_sigma log b_sigma]
        - (np/2) log(2 pi) + (pK/2)(1 + log(2 pi)) + (nK/2)
        + sum_k [log Gamma(2p + a + k^{c1}) - log Gamma(a + k^{c1})
                 - c2 (a + k^{c1}) log k - p log 4].

Single precision (float32) throughout.  The stopping decision never
touches this file's absolute ELBO: it uses the variable part only
(the four block byproducts, without Const), which keeps the large
invariant constant out of the float32 difference (migration plan
section 8.2; review finding F1).  The constant is combined on the
host, once per chunk, for the reported ELBO history.

This module provides the standalone diagnostic evaluator :func:`elbo`
(re-factorizes Phi and V at a saved state) and the host-side constant
:func:`elbo_const`.  The production path assembles the same value from
the update byproducts; the independent float64 reference (five
separate B2 terms) is not included in this repository.
"""

import math

import jax
import jax.numpy as jnp

from ._linalg import cholesky_checked, logdet_from_chol
from .B import log_det_V
from .shrinkage import column_index, elbo_column_constant, elbo_column_variable

__all__ = ["elbo_const", "elbo"]


def elbo_const(
    n: int,
    p: int,
    K: int,
    a: float,
    c1: float,
    c2: float,
    a_sigma: float,
    b_sigma: float,
) -> float:
    """Constant term of the closed-form ELBO (Appendix B2).

    Host-side Python float computation, called once per run.  It
    depends only on the dimensions and the hyperparameters, all fixed
    during a CAVI run.  (It cancels in the ELBO differences used for
    the stopping criterion, but is kept so that absolute ELBO values
    can be reported.)

    Parameters
    ----------
    n, p, K : int
        Observation count, dimension, truncation level.
    a, c1, c2, a_sigma, b_sigma : float
        Prior hyperparameters.

    Returns
    -------
    float
    """
    alpha_n = a_sigma + n / 2.0
    return (
        p * (math.lgamma(alpha_n) - math.lgamma(a_sigma) + a_sigma * math.log(b_sigma))
        - 0.5 * n * p * math.log(2.0 * math.pi)
        + 0.5 * p * K * (1.0 + math.log(2.0 * math.pi))
        + 0.5 * n * K
        + elbo_column_constant(p, a, c1, c2, K)
    )


def elbo(
    W: jax.Array,
    Phi: jax.Array,
    beta: jax.Array,
    V: jax.Array,
    m: jax.Array,
    a: float,
    c1: float,
    a_sigma: float,
    const: float,
) -> tuple[jax.Array, jax.Array]:
    """Standalone closed-form ELBO at a saved state (diagnostic path).

    Recomputes log det Phi and log det V_j by fresh Cholesky
    factorizations of the saved covariances (no byproduct reuse), so it
    is an independent check of the production assembly.  The closed
    form assumes the state is self-consistent: beta_j = b_sigma +
    R_j / 2 and m_k from the current E_q[B_jk^2] (these identities were
    substituted into the derivation in Appendix B2); evaluating it at
    non-self-consistent states is meaningless.

    Parameters
    ----------
    W : jax.Array, shape (n, K)
        Variational means of the factor scores.
    Phi : jax.Array, shape (K, K)
        Common covariance of the factor scores.
    beta : jax.Array, shape (p,)
        Inverse-gamma rates of the noise variances.
    V : jax.Array, shape (p, K, K)
        Variational covariances of the loading rows.
    m : jax.Array, shape (K,)
        Column aggregates of the shrinkage block.
    a, c1, a_sigma : float
        Prior hyperparameters (c2 enters only through the precomputed
        ``const``; m is taken as given).
    const : float
        Precomputed ELBO constant from :func:`elbo_const`.

    Returns
    -------
    L : jax.Array, scalar
        The absolute ELBO (Const included).
    ok : jax.Array, scalar bool
        Validity flags of the two diagnostic factorizations.
    """
    n = W.shape[0]
    p, K = V.shape[0], V.shape[1]
    alpha_n = a_sigma + n / 2.0

    # only shape_k = 2p + a + k^{c1} is needed; m is taken as given
    shape_k = 2.0 * p + a + column_index(K) ** c1

    L_phi, ok_phi = cholesky_checked(Phi)
    log_det_Phi = logdet_from_chol(L_phi)
    ldV, ok_V = log_det_V(V)

    L = (
        const
        - alpha_n * jnp.log(beta).sum()
        + 0.5 * ldV.sum()
        + 0.5 * n * log_det_Phi
        - 0.5 * ((W * W).sum() + n * jnp.trace(Phi))
        - elbo_column_variable(m, shape_k)
    )
    return L, ok_phi & ok_V
