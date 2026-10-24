"""Factor loading matrix B (JAX, single precision).

Variational factor for the loading rows:

    q_j(B_j) = N_K(mu_j, V_j),  j = 1, ..., p,

with
    V_j   = ( c_j * S + diag(rho_j1..rho_jK) )^{-1},
    mu_j  = c_j * V_j (W^T X)_.j,

where
    c_j = E_q[1 / sigma_j^2] = alpha_n / beta_j   (see sigma2.py),
    S   = E_q[eta eta^T] = W^T W + n Phi          (see eta.second_moment),
    rho_jk = E_q[lambda_k^4 / tau_jk^2]           (see shrinkage.py).

The covariance is computed in WHITENED form: with w_jk = rho_jk^{-1/2}
and W_j = diag(w_j),

    V_j = W_j ( I_K + c_j (w_j w_j^T) . S )^{-1} W_j,

so the factorized matrix is I + PSD by construction, with condition
number near 1 even on heavily shrunk coordinates -- the raw precision,
whose diagonal entries rho_jk can reach ~1e31 there, is never formed
(the scheme is the whitened perturb-and-solve of the companion Gibbs
sampler in :mod:`bfm._gs`).

Tensor conventions (as in eta.py):
    X : (n, p), W : (n, K), Phi : (K, K), c : (p,),
    rho : (p, K), mu : (p, K), V : (p, K, K).

Reference: main text eq. (update_B) and Algorithm 1, B update.

ELBO byproduct: the B block contributes

    (1/2) sum_j log det V_j

to the closed-form ELBO (Appendix B2, Term 5).  Each log det V_j comes
free from the byproduct Cholesky factor of the whitened matrix
(log det V_j = 2 sum_k log w_jk - 2 sum_k log L_jkk, consistent with V
by construction), so the ELBO no longer re-factorizes V.
"""

import jax
import jax.numpy as jnp

from ._linalg import cholesky_checked, cholesky_inverse_checked
from .eta import second_moment

__all__ = ["compute_V", "compute_mu", "update_B", "log_det_V"]


def compute_V(
    c: jax.Array, S: jax.Array, rho: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Covariances V_j = (c_j * S + diag(rho_j))^{-1} for all rows.

    Computed in whitened form: with w_jk = rho_jk^{-1/2} and
    W_j = diag(w_j),

        V_j = W_j ( I_K + c_j (w_j w_j^T) . S )^{-1} W_j.

    The factorized matrix is I + PSD by construction, so its condition
    number stays near 1 even on heavily shrunk coordinates (huge rho),
    where the raw precision matrix would be severely ill-conditioned;
    w_jk -> 0 decouples coordinate k exactly (M_j[k, k] -> 1), so no
    overflow or inf/NaN can arise.

    Parameters
    ----------
    c : jax.Array, shape (p,)
        E_q[1 / sigma_j^2].
    S : jax.Array, shape (K, K)
        Second moment E_q[eta eta^T] = W^T W + n Phi.
    rho : jax.Array, shape (p, K)
        Shrinkage moments E_q[lambda_k^4 / tau_jk^2].

    Returns
    -------
    V : jax.Array, shape (p, K, K)
        Updated covariances.
    log_det_V : jax.Array, shape (p,)
        log det V_j = 2 sum_k log w_jk - 2 sum_k log L_jkk, from the
        same Cholesky factors, hence consistent with V by construction.
    ok : jax.Array, scalar bool
        Device-side validity flag over the whole batch (see _linalg).
    """
    # Whitening scales w_jk = rho_jk^{-1/2} (the prior sd of B_jk).
    w = jnp.reciprocal(jnp.sqrt(rho))
    # M_j = I_K + c_j (w_j w_j^T) . S, by pure broadcasting.
    M = c[:, None, None] * S * w[:, :, None] * w[:, None, :]
    M = M + jnp.eye(S.shape[0], dtype=S.dtype)
    # M_j = I + PSD by construction: plain batched Cholesky + flag.
    L, ok = cholesky_checked(M)
    # V_j = W_j M_j^{-1} W_j = (w_j w_j^T) . M_j^{-1}, via the shared
    # cholesky_inverse_checked helper (blocked matmul inverse on CPU for
    # K >= 32, cho_solve against the identity otherwise).
    V, ok = cholesky_inverse_checked(L, ok)
    V = V * w[:, :, None] * w[:, None, :]
    ldV = 2.0 * (
        jnp.log(w).sum(axis=1)
        - jnp.log(jnp.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
    )
    return V, ldV, ok


def compute_mu(c: jax.Array, V: jax.Array, WtX: jax.Array) -> jax.Array:
    """Means mu_j = c_j * V_j (W^T X)_.j for all rows.

    Parameters
    ----------
    c : jax.Array, shape (p,)
        E_q[1 / sigma_j^2].
    V : jax.Array, shape (p, K, K)
        Covariances from :func:`compute_V`.
    WtX : jax.Array, shape (K, p)
        Precomputed W^T X (column j is W^T X_.j).  The driver computes
        it once per sweep.

    Returns
    -------
    jax.Array, shape (p, K)
    """
    mu = jnp.einsum("jkl,jl->jk", V, WtX.T)  # V_j (W^T X)_.j
    return c[:, None] * mu


def update_B(
    X: jax.Array,
    W: jax.Array,
    Phi: jax.Array,
    c: jax.Array,
    rho: jax.Array,
    WtX: jax.Array | None = None,
    S: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One CAVI sweep of the B block.

    Parameters
    ----------
    X : jax.Array, shape (n, p)
        Data matrix.
    W : jax.Array, shape (n, K)
        Variational means of the factor scores.
    Phi : jax.Array, shape (K, K)
        Common covariance of the factor scores.
    c : jax.Array, shape (p,)
        E_q[1 / sigma_j^2] = alpha_n / beta_j.
    rho : jax.Array, shape (p, K)
        Shrinkage moments E_q[lambda_k^4 / tau_jk^2].
    WtX : jax.Array, shape (K, p), optional
        Precomputed W^T X shared with the sigma2 update; computed
        internally when omitted.
    S : jax.Array, shape (K, K), optional
        Precomputed second moment W^T W + n Phi shared with the eta and
        sigma2 updates; computed internally when omitted.

    Returns
    -------
    mu : jax.Array, shape (p, K)
        Updated variational means of the loading rows.
    V : jax.Array, shape (p, K, K)
        Updated variational covariances of the loading rows.
    elbo_B : jax.Array, scalar
        This block's ELBO contribution (1/2) sum_j log det V_j, with
        the log determinants taken from the byproduct Cholesky factors.
    ok : jax.Array, scalar bool
        Device-side validity flag of the block's factorizations.
    """
    if S is None:
        S = second_moment(W, Phi)
    if WtX is None:
        WtX = W.T @ X
    V, ldV, ok = compute_V(c, S, rho)
    mu = compute_mu(c, V, WtX)
    elbo_B = 0.5 * ldV.sum()
    return mu, V, elbo_B, ok


def log_det_V(V: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Log determinants log det V_j for all rows (ELBO, Term 5).

    Diagnostic path: re-factorizes V with a plain batched Cholesky and
    returns the validity flag alongside (no jitter fallback anywhere in
    this package; a failed diagnostic factorization is reported).

    Parameters
    ----------
    V : jax.Array, shape (p, K, K)
        Variational covariances, symmetric positive definite.

    Returns
    -------
    log_det_V : jax.Array, shape (p,)
    ok : jax.Array, scalar bool
    """
    L, ok = cholesky_checked(V)
    return 2.0 * jnp.log(jnp.diagonal(L, axis1=1, axis2=2)).sum(axis=1), ok
