"""
B_sample.py  (JAX, single precision)

Loading-matrix step of the Gibbs sampler (Section 3 of the paper
"Bayesian factorization via L_{1/2} shrinkage"):
sample each row B_j (j = 1, ..., p) of the factor loading matrix B (p x K)
independently from its Gaussian full conditional

    B_j | X_{.j}, eta, lambda, tau_j^2, sigma_j^2
        ~ N_K(Lam_j^{-1} b_j, Lam_j^{-1}),

    Lam_j = eta eta^T / sigma_j^2 + diag(lambda^4 / tau_j^2),
    b_j   = eta X_{.j} / sigma_j^2.

Conventions (shared with the whole pipeline):
    eta    : (K, n) latent factor matrix, column i is eta_i (as returned
             by eta_sample.py; matches the paper's eta eta^T, eta X_{.j}
             notation in this step).
    D      : (p, K) matrix with entries w_jk = sqrt(tau_jk^2) / lambda_k^2,
             i.e. EXACTLY the output of shrinkage() in Shrinkage.py.  The
             prior precision enters as diag(w_j)^{-2}
             = diag(lambda_k^4 / tau_jk^2).
    sigma2 : (p,) diagonal variances of Omega, as produced by
             Sigma2_sample.py.
    return : (p, K) matrix, row j is the drawn B_j (same orientation as
             the `param` argument of shrinkage()).

Sampling scheme: whitened perturb-and-solve, batched over all p rows
(no Python loop).  Write W_j = diag(w_j) and G = eta eta^T.  G is formed
ONCE (O(n K^2)) and shared by all rows: the whitened system

    M_j = (w_j w_j^T) . (G / sigma_j^2) + I_K = W_j Lam_j W_j

is built by pure broadcasting -- O(p K^2) in memory and flops, versus
O(p K n) and O(p K^2 n) for the direct per-row outer products C_j C_j^T
with C_j = W_j eta / sigma_j.  The perturbed right-hand side

    phi_j = W_j eta (X_{.j} + sigma_j z_j) / sigma_j^2 + eps_j,
    z_j ~ N(0, I_n), eps_j ~ N(0, I_K),

has noise covariance (w_j w_j^T) . (G/sigma_j^2) + I_K = M_j (the term
W_j eta z_j / sigma_j is C_j z_j), and data and noise contributions are
folded into a single eta @ (X + Z^T sigma) matmul.  The draw is mapped
back by B_j = w_j . gamma_j with gamma_j = M_j^{-1} phi_j.  Exactness:

    mean: W M^{-1} W b = W (W Lam W)^{-1} W b = Lam^{-1} b,
    cov : W M^{-1} M M^{-1} W = W (W Lam W)^{-1} W = Lam^{-1}.

Numerical merits:
    * the raw precision Lam_j has diagonal entries lambda_k^4/tau_jk^2
      = 1/w_jk^2 up to ~1e76 on heavily shrunk coordinates (ill
      conditioned); whitening turns those into near-zero data terms in
      M_j = I + PSD;
    * w_jk -> 0 decouples coordinate k (M_j[k,k] -> 1, gamma_k ~ N(0,1),
      B_jk = w_jk gamma_k -> 0): exact zeros in D cannot produce inf/NaN;
    * M_j = I + PSD by construction, so the batched Cholesky needs no
      fallback branch.

JAX-specific notes
------------------
* Single precision (float32) throughout; takes an explicit
  jax.random.PRNGKey as its first argument.
* jnp.linalg.cholesky and jax.scipy.linalg.cho_solve both handle the
  leading batch dimension p natively (the rhs is passed with a trailing
  singleton axis, phi[..., None], the non-deprecated batched form).
* JAX implementation of the B-sample step of the reference sampler.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

__all__ = ["B_sample"]

_F32 = jnp.float32


def B_sample(
    key: jax.Array,
    X: jax.Array,
    D: jax.Array,
    eta: jax.Array,
    sigma2: jax.Array,
) -> jax.Array:
    r"""One exact draw of the loading matrix B given (eta, D, sigma2).

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : array-like, shape (n, p)
        Data matrix.
    D : array-like, shape (p, K)
        Prior scales w_jk = sqrt(tau_jk^2)/lambda_k^2, i.e. the output
        of shrinkage().  Must be strictly positive (guaranteed
        by shrinkage()).
    eta : array-like, shape (K, n)
        Latent factors, column i is eta_i (output of eta_sample()).
    sigma2 : array-like, shape (p,)
        Diagonal variances sigma_j^2 of Omega (output of
        Sigma2_sample.py).  Floored at tiny**0.5 before use (sigma_j >=
        tiny**0.25), so a perfect fit cannot overflow the divisions;
        the perturbation is invisible (binds only for sigma_j^2 <
        tiny**0.5 ~ 1.1e-19 in float32).

    Returns
    -------
    jnp.ndarray, shape (p, K), float32
        The drawn loading matrix; rows are independent across j.
    """
    X = jnp.asarray(X, dtype=_F32)
    D = jnp.asarray(D, dtype=_F32)
    eta = jnp.asarray(eta, dtype=_F32)
    sigma2 = jnp.asarray(sigma2, dtype=_F32)

    K, n = eta.shape
    p = X.shape[1]

    s2 = jnp.maximum(sigma2, jnp.finfo(_F32).tiny ** 0.5)

    # ---- whitened system M_j = (w_j w_j^T) . (G / sigma_j^2) + I_K --------
    # G = eta eta^T formed once and reused by all p rows (replaces the
    # O(p K^2 n) batched outer product and O(p K n) intermediate C).
    G = eta @ eta.T                                            # (K, K)
    M = D[:, :, None] * D[:, None, :] * (G / s2[:, None, None])
    M = M + jnp.eye(K, dtype=_F32)                             # M_j += I_K

    # ---- perturbed right-hand side -----------------------------------------
    # phi_j = w_j . eta (X_{.j} + sigma_j z_j) / sigma_j^2 + eps_j; the
    # noise w_j (eta z_j)/sigma_j + eps_j has covariance M_j.
    key_z, key_e = jax.random.split(key)
    Z = jax.random.normal(key_z, (p, n), dtype=_F32)
    E = X + Z.T * jnp.sqrt(s2)[None, :]      # X_{.j} + sigma_j z_j, (n, p)
    phi = D * ((eta @ E) / s2).T + jax.random.normal(
        key_e, (p, K), dtype=_F32
    )

    # ---- solve and map back: B_j = w_j . M_j^{-1} phi_j --------------------
    # Plain Cholesky: M_j = I + PSD by construction, so no fallback
    # branch.  The driver audits finiteness of the chain per chunk.
    L = jnp.linalg.cholesky(M)                                # (p, K, K)
    gamma = jsla.cho_solve((L, True), phi[..., None]).squeeze(-1)
    return D * gamma
