"""
eta_sample.py  (JAX, single precision)

Latent-factor step of the Gibbs sampler (Section 3 of the paper
"Bayesian factorization via L_{1/2} shrinkage"):
sample the latent factors eta_i (i = 1, ..., n) independently from
their Gaussian full conditional

    eta_i | B, Omega, x_i ~ N_K(m_i, S),

    S   = (I_K + B^T Omega^{-1} B)^{-1},
    m_i = S B^T Omega^{-1} x_i.

Notation (paper Section 3):
    B     : p x K factor loading matrix
    Omega : p x p diagonal covariance, diag(sigma_1^2, ..., sigma_p^2)
    x_i   : p x 1 observation i (row i of the n x p data matrix X)

Sampling scheme: perturb-and-solve (Rue & Held, 2005, GMRF sampling;
Papandreou & Yuille, 2010).  With C = B^T Omega^{-1/2}, Lambda = C C^T + I_K
and z1 ~ N(0, I_p), z2 ~ N(0, I_K),

    Cov(C z1 + z2) = C C^T + I_K = Lambda,

so that

    Lambda^{-1} (B^T Omega^{-1} x_i + C z1 + z2)
        ~ N_K(Lambda^{-1} B^T Omega^{-1} x_i, Lambda^{-1})  =  N_K(m_i, S),

an EXACT draw: the precision system Lambda is I + PSD by construction,
so it is solved by plain Cholesky with no fallback branch.  One
factorization produces all n draws simultaneously, and the data and
noise contributions are folded into a single matmul via
C (x_i/s + z1_i) = B^T Omega^{-1} x_i + C z1_i.

Orientation: the latent factors are returned as a (K, n) matrix whose
i-th COLUMN is eta_i.  This is the transpose of the paper's n x K eta
matrix and is the convention used throughout the pipeline: it gives
eta @ eta.T and eta @ X_{.j} without transposes in B_sample.py, and
eta.T @ B_j in Sigma2_sample.py.

JAX-specific notes
------------------
* Single precision (float32) throughout: array inputs are coerced on
  entry (jax's x64 mode is neither required nor enabled by this
  package).
* Randomness is explicit: every function takes a jax.random.PRNGKey as
  its first argument.
* JAX implementation of the eta-sample step of the reference sampler;
  the CUDA-graph considerations of the PyTorch reference are moot here -- the
  function is jit/scan-compiled together with the rest of the sweep.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

__all__ = ["eta_sample"]

_F32 = jnp.float32


def eta_sample(
    key: jax.Array,
    X: jax.Array,
    B: jax.Array,
    sigma2: jax.Array,
) -> jax.Array:
    r"""One exact draw of all latent factors eta_i | B, Omega, x_i.

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : array-like, shape (n, p)
        Data matrix.
    B : array-like, shape (p, K)
        Factor loading matrix at the current iteration.
    sigma2 : array-like, shape (p,)
        DIAGONAL VARIANCES sigma_j^2 of Omega (as produced by
        Sigma2_sample.py).  Floored at tiny**0.5 (i.e. sigma_j >=
        tiny**0.25) before the square root, so a perfect fit cannot
        overflow the divisions or the C @ C.T product below; the model
        perturbation is invisible (it binds only for sigma_j^2 <
        tiny**0.5 ~ 1.1e-19 in float32).

    Returns
    -------
    jnp.ndarray, shape (K, n), float32
        Column i is one draw of eta_i | B, Omega, x_i; columns are
        independent.  NOTE the (K, n) orientation (transpose of the
        paper's n x K eta matrix) -- see module docstring.
    """
    X = jnp.asarray(X, dtype=_F32)
    B = jnp.asarray(B, dtype=_F32)
    sigma2 = jnp.asarray(sigma2, dtype=_F32)

    K = B.shape[1]
    n = X.shape[0]
    # sigma_j >= tiny**0.25: one more squaring happens in C @ C.T, and
    # (B / tiny**0.25)^2 ~ 9e18 per entry, summed over p stays ~1e22,
    # far below the 3.4e38 float32 ceiling.
    s = jnp.sqrt(jnp.maximum(sigma2, jnp.finfo(_F32).tiny ** 0.5))

    C = B.T / s                              # B^T Omega^{-1/2},  (K, p)
    Lam = C @ C.T
    Lam = Lam + jnp.eye(K, dtype=_F32)       # Lam = C C^T + I_K

    # Perturbed right-hand side folded into ONE matmul:
    #   rhs_i = C (x_i / s + z1_i) + z2_i
    #         = B^T Omega^{-1} x_i + C z1_i + z2_i,
    # whose noise covariance is C C^T + I = Lam, so the solved output has
    # covariance Lam^{-1} = S around the exact mean.
    key_z1, key_z2 = jax.random.split(key)
    z1 = jax.random.normal(key_z1, X.shape, dtype=_F32)
    E2 = z1 + X * (1.0 / s)[None, :]         # x_i / s + z1_i,   (n, p)
    rhs = C @ E2.T + jax.random.normal(key_z2, (K, n), dtype=_F32)

    # Plain Cholesky: Lam = I + PSD by construction, so no fallback
    # branch.  The driver audits finiteness of the chain per chunk.
    L = jnp.linalg.cholesky(Lam)
    return jsla.cho_solve((L, True), rhs)
