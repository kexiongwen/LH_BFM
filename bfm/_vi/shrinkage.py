"""L_{1/2} shrinkage block (JAX, single precision).

The optimal variational factor q_k(tau_.k^2, lambda_k) has no standard
form, but the moment needed by the B-update is available in closed form
(Proposition 4.1):

    rho_jk = E_q[lambda_k^4 / tau_jk^2]
           = (2p + a + k^{c1}) / ( 2 * (E_q[B_jk^2])^{3/4} * m_k ),

with the column aggregates

    m_k = sum_j (E_q[B_jk^2])^{1/4} + k^{-c2},

and E_q[B_jk^2] = mu_jk^2 + (V_j)_kk.

ELBO contributions of this block (Appendix B2, Term 4):
    variable part:  -sum_k (2p + a + k^{c1}) * log(m_k),
    constant part:   sum_k [ log Gamma(2p + a + k^{c1}) - log Gamma(a + k^{c1})
                             - c2 (a + k^{c1}) log k - p log 4 ].

Column indices k are 1-based throughout, as in the paper.  The column
weights (2p + a + k^{c1}) and (k^{-c2}) are computed ONCE per run by
the driver (:func:`column_weights`) and passed in as arrays; the block
itself is a pure function of the variational state.

Tensor conventions (as in eta.py):
    mu : (p, K), V : (p, K, K), E2 : (p, K), rho : (p, K), m : (K,).

Reference: main text eq. (moment_tau_lambda), Proposition 4.1,
Appendix B2 Term 4.

Numerical notes:
* E2 is floored at jnp.finfo(dtype).tiny inside second_moment_B: it
  feeds the fractional powers below, where a zero or subnormal entry
  would yield inf / severely degraded rho.  The floor is far below any
  statistically relevant scale (a dead column's m_k is dominated by
  k^{-c2}, its E2^{1/4} contributing only ~1e-10), so the guard has no
  statistical effect.  No upper cap on rho is needed: the shrinkage
  feedback rho -> V_jkk ~ 1/rho -> E2 ~ 1/rho has the finite fixed
  point rho* = ((2p + a + k^{c1}) / (2 m_k))^4 (<= ~1e31 in float32,
  far below the 3.4e38 ceiling), and huge rho columns are nearly
  decoupled in the whitened B system (see B.py).
* Fractional powers use sqrt identities, which are faster and more
  accurate than pow: E2^{1/4} = sqrt(sqrt(E2)) and E2^{3/4} = E2 /
  sqrt(sqrt(E2)).
"""

import math

import jax
import jax.numpy as jnp

__all__ = [
    "column_index",
    "column_weights",
    "second_moment_B",
    "compute_m",
    "compute_rho",
    "update_shrinkage",
    "elbo_column_variable",
    "elbo_column_constant",
]

_F32 = jnp.float32


def column_index(K: int) -> jax.Array:
    """1-based column indices k = 1, ..., K.

    Returns
    -------
    jax.Array, shape (K,), float32
    """
    return jnp.arange(1, K + 1, dtype=_F32)


def column_weights(
    p: int, K: int, a: float, c1: float, c2: float
) -> tuple[jax.Array, jax.Array]:
    """Column weights of the shrinkage prior, computed once per run.

    Returns
    -------
    shape_k : jax.Array, shape (K,)
        (2p + a + k^{c1})_{k=1}^K.
    kneg_c2 : jax.Array, shape (K,)
        (k^{-c2})_{k=1}^K.
    """
    k = column_index(K)
    return 2.0 * p + a + k**c1, k ** (-c2)


def second_moment_B(mu: jax.Array, V: jax.Array) -> jax.Array:
    """Second moments E_q[B_jk^2] = mu_jk^2 + (V_j)_kk.

    The result is floored at jnp.finfo(dtype).tiny: it feeds the
    fractional powers in :func:`compute_m` and :func:`compute_rho`,
    where a zero or subnormal entry would produce inf or severely
    imprecise rho.  The floor is far below any statistically relevant
    scale, so the guard has no statistical effect.

    Parameters
    ----------
    mu : jax.Array, shape (p, K)
        Variational means of the loading rows.
    V : jax.Array, shape (p, K, K)
        Variational covariances of the loading rows.

    Returns
    -------
    jax.Array, shape (p, K)
    """
    E2 = mu * mu + jnp.diagonal(V, axis1=1, axis2=2)
    return jnp.maximum(E2, jnp.finfo(E2.dtype).tiny)


def compute_m(E2: jax.Array, kneg_c2: jax.Array) -> jax.Array:
    """Column aggregates m_k = sum_j E2_jk^{1/4} + k^{-c2}.

    Parameters
    ----------
    E2 : jax.Array, shape (p, K)
        Second moments E_q[B_jk^2], must be strictly positive.
    kneg_c2 : jax.Array, shape (K,)
        Column weights (k^{-c2}) from :func:`column_weights`.

    Returns
    -------
    jax.Array, shape (K,)
    """
    # E2^{1/4} = sqrt(sqrt(E2)): faster and more accurate than pow(0.25)
    return jnp.sqrt(jnp.sqrt(E2)).sum(axis=0) + kneg_c2


def compute_rho(E2: jax.Array, m: jax.Array, shape_k: jax.Array) -> jax.Array:
    """Shrinkage moments rho_jk = E_q[lambda_k^4 / tau_jk^2].

    rho_jk = (2p + a + k^{c1}) / (2 * E2_jk^{3/4} * m_k).

    Parameters
    ----------
    E2 : jax.Array, shape (p, K)
        Second moments E_q[B_jk^2], must be strictly positive.
    m : jax.Array, shape (K,)
        Column aggregates from :func:`compute_m`.
    shape_k : jax.Array, shape (K,)
        Column weights (2p + a + k^{c1}) from :func:`column_weights`.

    Returns
    -------
    jax.Array, shape (p, K)
    """
    r4 = jnp.sqrt(jnp.sqrt(E2))  # E2^{1/4}
    # E2^{3/4} = E2 / r4: cheaper and more accurate than pow(0.75)
    return shape_k[None, :] / (2.0 * (E2 / r4) * m[None, :])


def update_shrinkage(
    mu: jax.Array, V: jax.Array, shape_k: jax.Array, kneg_c2: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One CAVI sweep of the shrinkage block.

    Parameters
    ----------
    mu : jax.Array, shape (p, K)
        Variational means of the loading rows.
    V : jax.Array, shape (p, K, K)
        Variational covariances of the loading rows.
    shape_k, kneg_c2 : jax.Array, shape (K,)
        Column weights from :func:`column_weights`.

    Returns
    -------
    rho : jax.Array, shape (p, K)
        Moments E_q[lambda_k^4 / tau_jk^2].
    m : jax.Array, shape (K,)
        Column aggregates, reused by the ELBO.
    elbo_shrink : jax.Array, scalar
        This block's ELBO contribution
        -sum_k (2p + a + k^{c1}) log(m_k) (Appendix B2, Term 4), which
        already merges the loading prior and the auxiliary-variable
        prior/entropy through the log Z_k identity (Proposition 4.1),
        so it must not be counted twice.
    """
    E2 = second_moment_B(mu, V)
    m = compute_m(E2, kneg_c2)
    rho = compute_rho(E2, m, shape_k)
    elbo_shrink = -elbo_column_variable(m, shape_k)
    return rho, m, elbo_shrink


def elbo_column_variable(m: jax.Array, shape_k: jax.Array) -> jax.Array:
    """Variable ELBO term sum_k (2p + a + k^{c1}) * log(m_k).

    It enters the ELBO with a negative sign (Appendix B2, final
    display).

    Parameters
    ----------
    m : jax.Array, shape (K,)
        Column aggregates from :func:`compute_m`.
    shape_k : jax.Array, shape (K,)
        Column weights (2p + a + k^{c1}).

    Returns
    -------
    jax.Array, scalar
    """
    return (shape_k * jnp.log(m)).sum()


def elbo_column_constant(p: int, a: float, c1: float, c2: float, K: int) -> float:
    """Constant ELBO term of the shrinkage block (Appendix B2, Term 4).

    sum_k [ log Gamma(2p + a + k^{c1}) - log Gamma(a + k^{c1})
            - c2 (a + k^{c1}) log k - p log 4 ].

    Host-side Python float computation, called once per run.

    Returns
    -------
    float
    """
    total = 0.0
    for k in range(1, K + 1):
        k_c1 = k**c1
        total += (
            math.lgamma(2.0 * p + a + k_c1)
            - math.lgamma(a + k_c1)
            - c2 * (a + k_c1) * math.log(k)
            - p * math.log(4.0)
        )
    return total
