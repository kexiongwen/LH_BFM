"""Latent factor scores eta (JAX, single precision).

Variational factor for the latent factors:

    q_i(eta_i) = N_K(w_i, Phi),  i = 1, ..., n,

with common covariance

    Phi = (I_K + mu^T C mu + A)^{-1},
    w_i = Phi mu^T C x_i,

where
    mu   : (p, K) variational means of the loading rows B_j,
    C    : diag(c), c_j = E_q[1 / sigma_j^2] = alpha_n / beta_j,
    A    : (K, K) sum_j c_j V_j.

Tensor conventions used throughout the package:
    X   : (n, p)      data matrix,
    mu  : (p, K)      variational means of the loading rows,
    V   : (p, K, K)   variational covariances of the loading rows,
    c   : (p,)        E_q[1 / sigma_j^2],
    W   : (n, K)      row i is w_i^T,
    Phi : (K, K)      common covariance.

Reference: main text eq. (update_eta) and Algorithm 1, eta update.

ELBO byproduct: the eta block contributes

    (n/2) log det Phi - (1/2) tr(S),   S = W^T W + n Phi,

to the closed-form ELBO (Appendix B2).  log det Phi comes free from the
Cholesky factor of the precision matrix (log det Phi = -2 sum log diag
L, consistent with Phi by construction), and S is also shared with the
B and sigma2 updates, so it is computed only once per sweep.
"""

import jax
import jax.numpy as jnp

from ._linalg import cholesky_checked, cholesky_inverse_checked

__all__ = ["compute_A", "compute_Phi", "compute_W", "second_moment", "update_eta"]


def compute_A(c: jax.Array, V: jax.Array) -> jax.Array:
    """Compute A = sum_j c_j V_j.

    Parameters
    ----------
    c : jax.Array, shape (p,)
        E_q[1 / sigma_j^2] for j = 1, ..., p.
    V : jax.Array, shape (p, K, K)
        Variational covariances of the loading rows.

    Returns
    -------
    jax.Array, shape (K, K)
    """
    return jnp.einsum("j,jkl->kl", c, V)


def compute_Phi(
    mu: jax.Array,
    V: jax.Array,
    c: jax.Array,
    mu_c: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Precision and covariance of the eta block, with a validity flag.

    Computes Phi = (I_K + mu^T diag(c) mu + A)^{-1} together with its
    log determinant.  The matrix to invert is I + PSD by construction,
    so a plain Cholesky is used; ``ok`` is the device-side
    factorization/inversion validity flag (see _linalg).

    Parameters
    ----------
    mu : jax.Array, shape (p, K)
        Variational means of the loading rows.
    V : jax.Array, shape (p, K, K)
        Variational covariances of the loading rows.
    c : jax.Array, shape (p,)
        E_q[1 / sigma_j^2].
    mu_c : jax.Array, shape (p, K), optional
        Precomputed mu * diag(c), shared with :func:`compute_W`;
        computed internally when omitted.

    Returns
    -------
    Phi : jax.Array, shape (K, K)
        Common covariance.
    log_det_Phi : jax.Array, scalar
        log det Phi = -2 sum log diag(L), from the same Cholesky factor
        L of the precision matrix, hence consistent with Phi by
        construction.
    ok : jax.Array, scalar bool
        Device-side validity flag of the factorization and inversion.
    """
    K = mu.shape[1]
    if mu_c is None:
        mu_c = mu * c[:, None]
    M = jnp.eye(K, dtype=mu.dtype) + mu_c.T @ mu + compute_A(c, V)
    # M = I + PSD by construction: plain Cholesky + flag, no fallback.
    L, ok = cholesky_checked(M)
    Phi, ok = cholesky_inverse_checked(L, ok)
    log_det_Phi = -2.0 * jnp.log(jnp.diagonal(L)).sum()
    return Phi, log_det_Phi, ok


def compute_W(
    X: jax.Array,
    mu: jax.Array,
    c: jax.Array,
    Phi: jax.Array,
    mu_c: jax.Array | None = None,
) -> jax.Array:
    """Variational means W = X diag(c) mu Phi.

    Row i of the result is w_i^T = x_i^T diag(c) mu Phi.

    Parameters
    ----------
    X : jax.Array, shape (n, p)
        Data matrix.
    mu : jax.Array, shape (p, K)
        Variational means of the loading rows.
    c : jax.Array, shape (p,)
        E_q[1 / sigma_j^2].
    Phi : jax.Array, shape (K, K)
        Common covariance from :func:`compute_Phi`.
    mu_c : jax.Array, shape (p, K), optional
        Precomputed mu * diag(c), shared with :func:`compute_Phi`;
        computed internally when omitted.

    Returns
    -------
    jax.Array, shape (n, K)
    """
    # X diag(c) mu == X @ (c[:, None] * mu): scaling the (p, K) matrix
    # mu avoids materializing an (n, p) scaled copy of X at every sweep.
    if mu_c is None:
        mu_c = mu * c[:, None]
    return X @ mu_c @ Phi


def second_moment(W: jax.Array, Phi: jax.Array) -> jax.Array:
    """E_q[eta eta^T] = W^T W + n Phi.

    Parameters
    ----------
    W : jax.Array, shape (n, K)
        Variational means of the factor scores.
    Phi : jax.Array, shape (K, K)
        Common covariance.

    Returns
    -------
    jax.Array, shape (K, K)
    """
    n = W.shape[0]
    return W.T @ W + n * Phi


def update_eta(
    X: jax.Array, mu: jax.Array, V: jax.Array, c: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One CAVI sweep of the eta block.

    Parameters
    ----------
    X : jax.Array, shape (n, p)
        Data matrix.
    mu : jax.Array, shape (p, K)
        Variational means of the loading rows.
    V : jax.Array, shape (p, K, K)
        Variational covariances of the loading rows.
    c : jax.Array, shape (p,)
        E_q[1 / sigma_j^2] = alpha_n / beta_j.

    Returns
    -------
    W : jax.Array, shape (n, K)
        Updated variational means of the factor scores.
    Phi : jax.Array, shape (K, K)
        Updated common covariance.
    S : jax.Array, shape (K, K)
        Second moment E_q[eta eta^T] = W^T W + n Phi, computed once here
        and shared with the B and sigma2 updates of the same sweep.
    elbo_eta : jax.Array, scalar
        This block's ELBO contribution (n/2) log det Phi - (1/2) tr(S),
        with log det Phi taken from the byproduct Cholesky factor.
    ok : jax.Array, scalar bool
        Device-side validity flag of the block's factorization.
    """
    mu_c = mu * c[:, None]  # scaled loadings, shared by both updates
    Phi, log_det_Phi, ok = compute_Phi(mu, V, c, mu_c=mu_c)
    W = compute_W(X, mu, c, Phi, mu_c=mu_c)
    S = second_moment(W, Phi)
    elbo_eta = 0.5 * W.shape[0] * log_det_Phi - 0.5 * jnp.trace(S)
    return W, Phi, S, elbo_eta, ok
