"""Initialization for the JAX CAVI driver (single precision).

Two entries:

* :func:`PPCA` -- deterministic thin-SVD probabilistic-PCA estimate of
  (B, sigma2), the JAX analog of the randomized-SVD PPCA.  A plain
  jnp.linalg.svd is deterministic (no RNG anywhere in this package)
  and exact; it runs ONCE per CAVI call, so the randomized low-rank
  variant only pays off for p far beyond the intended use (same
  reasoning as the :mod:`bfm._gs` one-shot eigh initialization).
* :func:`initialization` -- build the self-consistent initial
  variational state (mu, V, W, Phi, S, beta, c, rho, m and the ELBO
  block byproducts), mirroring the reference initialization.

Noise-floor convention (migration plan section 6.3): the trailing-eigenvalue
denominator is min(n - 1, p) - r -- column-centered X has rank at most
min(n - 1, p), so this is exactly the number of trailing eigenvalues that
can be nonzero.  The :mod:`bfm._gs` ``Initialization`` uses the same
denominator (the conventions were unified; it previously used
min(n, p) - 1 - r,
which differed when p < n).
"""

import jax
import jax.numpy as jnp

from .B import log_det_V
from .eta import update_eta
from .shrinkage import update_shrinkage
from .sigma2 import mean_inverse, update_sigma2

__all__ = ["PPCA", "initialization"]

_F32 = jnp.float32


def PPCA(X: jax.Array, r: int) -> tuple[jax.Array, float]:
    r"""Probabilistic PCA (PPCA) initialization for (B, sigma2).

    Closed-form maximum likelihood estimate of probabilistic PCA
    (Tipping & Bishop, 1999) -- the homoscedastic special case
    Omega = sigma2 * I of the factor model -- with the identity
    rotation and descending variance order (which suits the
    increasing-shrinkage prior), and with the trailing-eigenvalue
    denominator adapted to the high-dimensional regime.

    Via the thin SVD of M = X.T / sqrt(n - 1) (X is assumed
    column-centered, as the model has zero mean), with singular values
    s_i (descending; s_i^2 are the eigenvalues of the sample
    covariance M M^T):

        sigma2_hat = mean of the trailing eigenvalues {s_i^2}_{i > r},
        B0         = U_r diag( sqrt(s_1^2 - sigma2_hat, ...) ),

    so that B0 B0^T = U_r diag(s_i^2 - sigma2_hat) U_r^T is the rank-r
    PSD approximation of Sigma_hat - sigma2_hat I.

    Implementation notes:
    * the trailing sum uses the exact trace identity
      sum_{i > r} s_i^2 = ||M||_F^2 - sum_{i <= r} s_i^2,
      with the denominator min(n - 1, p) - r (see module docstring);
    * s_i^2 - sigma2_hat is floored at 0: top-r components below the
      noise floor initialize to a zero column instead of NaN;
    * deterministic: no seed, no power iterations, no RNG consumption.

    Parameters
    ----------
    X : jax.Array, shape (n, p)
        Column-centered data matrix.
    r : int
        Number of leading factors, 1 <= r <= min(n, p) - 2.

    Returns
    -------
    B0 : jax.Array, shape (p, r), float32
        Initial loading matrix.
    sigma2_hat : float
        Noise-floor variance estimate (scalar; broadcast to (p,) by
        the caller for the initial Omega diagonal).
    """
    X = jnp.asarray(X, dtype=_F32)
    n, p = X.shape
    if not 1 <= r <= min(n, p) - 2:
        raise ValueError(
            f"r must be in [1, min(n, p) - 2] = [1, {min(n, p) - 2}], got {r}"
        )

    M = X.T / jnp.sqrt(jnp.asarray(n - 1, dtype=_F32))  # M M^T = sample cov
    U, S, _ = jnp.linalg.svd(M, full_matrices=False)    # S descending
    Ur, Sr = U[:, :r], S[:r]

    # mean trailing eigenvalue (trace identity for the discarded
    # spectrum; exactly min(n - 1, p) - r trailing eigenvalues can be
    # nonzero for column-centered X)
    tail = (M * M).sum() - (Sr * Sr).sum()
    sigma2_hat = jnp.maximum(tail / (min(n - 1, p) - r), 0.0)

    B0 = Ur * jnp.sqrt(jnp.maximum(Sr * Sr - sigma2_hat, 0.0))[None, :]
    return B0, float(sigma2_hat)


def initialization(
    X: jax.Array,
    K: int,
    a_sigma: float,
    b_sigma: float,
    shape_k: jax.Array,
    kneg_c2: jax.Array,
    r: int | None = None,
    v0: float = 1.0,
) -> dict:
    """Initialize the variational state for :func:`VI.CAVI`.

    The loading means mu start at the thin-SVD PPCA estimate from
    :func:`PPCA` (the first r columns; remaining columns are zero and
    are shrunk away by the prior), and the loading covariances start
    at v0 * I_K.  The very first eta update uses precision
    1 / sigma2_hat; the noise rates beta then take one sigma2 update,
    so that the initial state is self-consistent and the closed-form
    ELBO (which substitutes beta_j = b_sigma + R_j / 2 and
    m_k = sum_j E[B_jk^2]^{1/4} + k^{-c2}) is exact when evaluated at
    it.

    X must already be column-centered (CAVI handles that before
    calling).

    Parameters
    ----------
    X : jax.Array, shape (n, p)
        Column-centered data matrix, float32.
    K : int
        Truncation level (number of loading columns).
    a_sigma, b_sigma : float
        Noise-variance inverse-gamma hyperparameters.
    shape_k, kneg_c2 : jax.Array, shape (K,)
        Column weights from shrinkage.column_weights (computed once by
        the driver and shared with the sweeps); the shrinkage
        hyperparameters a, c1, c2 enter only through these arrays.
    r : int or None, default None
        Number of leading factors used by the PPCA initialization;
        defaults to min(K, min(n, p) - 2).  Values larger than K are
        clipped to K.
    v0 : float, default 1.0
        Initial loading covariance scale, V_j = v0 * I_K.

    Returns
    -------
    dict with keys
        mu : (p, K) initial loading means,
        V : (p, K, K) initial loading covariances,
        W : (n, K) initial factor-score means,
        Phi : (K, K) initial common covariance of the factor scores,
        S : (K, K) initial second moment W^T W + n Phi,
        alpha_n : scalar, noise inverse-gamma shape a_sigma + n/2,
        beta : (p,) initial noise inverse-gamma rates,
        c : (p,) E_q[1 / sigma_j^2] = alpha_n / beta_j,
        rho : (p, K) initial shrinkage moments,
        m : (K,) initial column aggregates,
        elbo_var : scalar, variable part of the ELBO at the initial
        state (sum of the four block byproducts; the driver adds the
        constant on the host),
        ok : scalar bool, validity flags of the factorizations.
    """
    n, p = X.shape
    r_init = K if r is None else min(r, K)
    r_init = min(r_init, min(n, p) - 2)
    if r_init < 1:
        raise ValueError(
            f"need min(n, p) - 2 >= 1 for the PPCA initialization, "
            f"got n={n}, p={p}"
        )

    B0, sigma2_hat = PPCA(X, r_init)
    sigma2_hat = max(sigma2_hat, float(jnp.finfo(_F32).eps))

    mu = jnp.zeros((p, K), dtype=_F32).at[:, :r_init].set(B0)
    V = jnp.broadcast_to(v0 * jnp.eye(K, dtype=_F32), (p, K, K))

    alpha_n = jnp.asarray(a_sigma + n / 2.0, dtype=_F32)
    # precision for the very first eta update: E_q[1/sigma_j^2] = 1/sigma2_hat
    c = jnp.full((p,), 1.0 / sigma2_hat, dtype=_F32)
    W, Phi, S, elbo_eta, ok_eta = update_eta(X, mu, V, c)
    rho, m, elbo_shrink = update_shrinkage(mu, V, shape_k, kneg_c2)
    # beta consistent with the initial state, so the closed-form ELBO
    # (which substitutes beta_j = b_sigma + R_j / 2) is exact at evaluation
    alpha_n, beta, elbo_sigma2 = update_sigma2(X, W, Phi, mu, V, a_sigma, b_sigma)
    c = mean_inverse(alpha_n, beta)
    # the B block term at init: V has not been updated yet, so its log
    # determinants are taken from the standalone factorization (one-off)
    ldV, ok_V = log_det_V(V)
    elbo_var = elbo_eta + 0.5 * ldV.sum() + elbo_shrink + elbo_sigma2

    return {
        "mu": mu,
        "V": V,
        "W": W,
        "Phi": Phi,
        "S": S,
        "alpha_n": alpha_n,
        "beta": beta,
        "c": c,
        "rho": rho,
        "m": m,
        "elbo_var": elbo_var,
        "ok": ok_eta & ok_V,
    }
