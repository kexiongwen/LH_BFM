"""
Shrinkage.py  (JAX, single precision)

Shrinkage-parameter step of the Gibbs sampler (Section 3 of the paper
"Bayesian factorization via L_{1/2} shrinkage"):
sample the global shrinkage parameters lambda_k and the local shrinkage
parameters tau_jk^2 of the L_{1/2} shrinkage prior.

Global shrinkage parameters (independent across k = 1, ..., K):

    lambda_k | B_{.k}
        ~ Ga( 2p + a + k^{c_1},
              sum_{j=1}^p |B_jk|^{1/2} + k^{-c_2} ),

with shape-rate parameterization and hyperparameters a, c_1, c_2 > 0.

Local shrinkage parameters via the Gaussian-mixture representation
(Eq. (GM) of the paper), sampling pi(tau_jk^2, v_jk | lambda_k, B_jk)
by two inverse-Gaussian draws:

    1 / v_jk     | B_jk, lambda_k ~ InvGauss( 1 / (2 lambda_k |B_jk|^{1/2}), 1/2 ),
    1 / tau_jk^2 | v_jk, B_jk, lambda_k
                 ~ InvGauss( 1 / (lambda_k^2 v_jk |B_jk|), 1 / v_jk^2 ).

Both InvGauss draws are reduced to a shape-1 inverse-Gaussian generator
via the scaling law

    X ~ InvGauss(mu, lam0)  =>  c X ~ InvGauss(c mu, c lam0),

giving (as in the reference implementation shrinkage_sample.py of GS_LH,
Ke & Fan, 2024, JCGS, DOI: 10.1080/10618600.2024.2374579):

    1 / v_jk     = InvGauss(1/(lambda_k |B_jk|^{1/2}), 1) / 2,
    v_jk^2 / tau_jk^2 = InvGauss(v_jk / (lambda_k^2 |B_jk|), 1),
    hence tau_jk = v_jk / sqrt(.).

Design notes
------------
* The lambda -> v -> tau chain is a marginal-then-conditional draw given B,
  so (lambda, v, tau) need NOT be carried as sampler state; only the
  composite scale w_jk = sqrt(tau_jk^2) / lambda_k^2 (the prior sd of B_jk,
  since B_jk | tau, lambda ~ N(0, tau_jk^2 / lambda_k^4)) is returned.
  The full conditional of B then uses precision lambda_k^4 / tau_jk^2
  = 1 / w_jk^2 (see B_sample.py).

JAX-specific notes
------------------
* Single precision (float32) throughout; every public function takes an
  explicit jax.random.PRNGKey as its first argument.
* jax.random.gamma is a pure XLA primitive (no host synchronization),
  so -- unlike the PyTorch reference -- the chi-square Gamma
  approximation of Chi2.py is never mandatory.  The lambda step draws
  exact Gamma unless the driver passes a chi2_plan (its _accept_chi2
  decision, GPU backend only; see Chi2.py).
* JAX implementation of the shrinkage-parameter step of the reference
  sampler.
"""

import jax
import jax.numpy as jnp

from .Chi2 import gamma_sample, lambda_concentration

__all__ = ["inv_gauss", "shrinkage"]

_F32 = jnp.float32
_TINY = jnp.finfo(_F32).tiny


def inv_gauss(key: jax.Array, mu: jax.Array) -> jax.Array:
    r"""Elementwise draw from InvGauss(mean=mu, shape=1), mu > 0.

    Michael-Schucany-Haas algorithm specialized to shape lambda0 = 1:
    with nu ~ N(0, 1), y = nu^2,
        x = mu + mu^2 y / 2 - (mu / 2) sqrt(4 mu y + mu^2 y^2),
    return x with probability mu / (mu + x), else mu^2 / x.  Here `a` below
    is x / mu (note (ink+2)^2 - 4 = ink^2 + 4 ink with ink = mu * y), so
    x = mu * a and mu^2 / x = mu / a, with acceptance probability 1/(1+a).

    The textbook branch  a = 1 + (ink - sqrt(ink^2 + 4 ink)) / 2  suffers
    catastrophic cancellation for large ink -- frequent here, since
    heavily shrunk |B_jk| makes mu huge.  We use the identical
    rationalized form of the GS_LH reference implementation

        a = w / (1 + w),   w = 2 / (ink * (1 + sqrt(1 + 4 / ink))),

    which never forms ink^2 and stays accurate and positive in float32.

    Two safeguards (union of the GS_LH reference and the factor-model
    stress tests):

    * ink is floored at 8*tiny: float32 normal draws return an exact 0
      with nonzero probability, and ink = 0 makes w NaN via 0 * inf.
      The floor activates only for nu^2 < 8*tiny/mu, a region of
      negligible probability whose draw is unchanged in practice
      (a ~= 1 anyway).

    * mu is clamped into [f, 1/f] with f = tiny**0.25 (~3e-10 in
      float32).  Without the upper bound, mu = inf (e.g. from v / ink^2
      when ink^2 underflows to 0 -- reachable in the factor model when a
      column mixes hugely different |B_jk| scales) gives w = 0, a = 0
      and mu * a = inf * 0 = NaN.  With mu <= 1/f ~ 3e9, neither
      ink = mu * nu^2 nor mu / a ~ mu^2 nu^2 can overflow for any
      practically possible nu^2.  The clamp is statistically invisible:
      for mu beyond the range the MSH draw is x1 ~= 1/nu^2 + O(1/mu)
      regardless of mu, with the alternative branch selected with
      probability ~= 1/mu either way.

    Parameters
    ----------
    key : jax.random.PRNGKey
    mu : array-like
        Positive means, any shape.

    Returns
    -------
    jnp.ndarray, float32, same shape as `mu`.
    """
    mu = jnp.asarray(mu, dtype=_F32)
    f = _TINY ** 0.25
    mu = jnp.clip(mu, f, 1.0 / f)

    key_n, key_u = jax.random.split(key)
    ink = mu * jax.random.normal(key_n, mu.shape, dtype=_F32) ** 2
    # Guard the nu = 0 draw: 4/ink must stay finite (a float32 normal
    # draw can be an exact 0; ink = 0 makes w NaN via 0 * inf).
    ink = jnp.maximum(ink, 8 * _TINY)

    w = 2 / (ink * (1 + jnp.sqrt(1 + 4 / ink)))
    a = w / (1 + w)

    u = jax.random.uniform(key_u, mu.shape, dtype=_F32)
    return jnp.where((1 / (1 + a)) >= u, mu * a, mu / a)


def shrinkage(
    key: jax.Array,
    param: jax.Array,
    a: float,
    weight1: jax.Array,
    weight2: jax.Array,
    chi2_plan: tuple | None = None,
) -> jax.Array:
    r"""One joint draw of the L_{1/2} shrinkage parameters given B.

    Parameters
    ----------
    key : jax.random.PRNGKey
    param : array-like, shape (p, K)
        Factor loading matrix B at the current iteration.
    a : float
        Fixed hyperparameter a > 0 of the lambda hyperprior.
    weight1, weight2 : array-like, shape (K,)
        Column weights (k^{c_1})_{k=1}^K and (k^{c_2})_{k=1}^K.
    chi2_plan : tuple or None
        Bucketed/chunked plan from
        make_chi2_plan(2p + a + weight1) for the chi-square Gamma
        approximation of the lambda draw (see Chi2.py); None draws
        exact Gamma.  Supplied by the driver when its _accept_chi2
        decision accepts the rounding perturbation and the cost gate
        passes (GPU backend only).

    Returns
    -------
    jnp.ndarray, shape (p, K), float32
        w_jk = sqrt(tau_jk^2) / lambda_k^2 = sqrt(tau_jk^2 / lambda_k^4),
        the standard deviation of the Gaussian-mixture component
        B_jk | tau_jk^2, lambda_k ~ N(0, tau_jk^2 / lambda_k^4).
        Squaring the reciprocal gives the precision lambda_k^4 / tau_jk^2
        required by the full conditional of B (see B_sample.py).

    Notes
    -----
    * lambda step: lambda_k | B_{.k}
      ~ Ga(2p + a + k^{c_1}, sum_j |B_jk|^{1/2} + k^{-c_2}),
      drawn independently across k (conditional independence property of
      the prior, unlike MGP/CSP posteriors).  Exact Gamma by default;
      with chi2_plan the shape is rounded to the nearest half-integer
      and drawn as chi2(nu) / (2*rate) from a bucketed/chunked plan
      (see Chi2.py).
    * v step: 1/v_jk ~ InvGauss(1/(2 lambda_k |B_jk|^{1/2}), 1/2) is drawn
      as X/2 with X ~ InvGauss(1/(lambda_k |B_jk|^{1/2}), 1)
      (scaling law, c = 1/2).
    * tau step: 1/tau_jk^2 ~ InvGauss(1/(lambda_k^2 v_jk |B_jk|), 1/v_jk^2)
      is drawn via v_jk^2/tau_jk^2 ~ InvGauss(v_jk/(lambda_k^2 |B_jk|), 1)
      (scaling law, c = v_jk^2), hence tau_jk = v_jk / sqrt(.).
    * Entries of `param` are floored at the smallest positive normal
      float32 before the square root, so exact zeros cannot produce
      1/0 = inf; thanks to the rationalized `inv_gauss` this safeguard
      is enough to keep every downstream draw finite.
    """
    param = jnp.asarray(param, dtype=_F32)
    weight1 = jnp.asarray(weight1, dtype=_F32)
    weight2 = jnp.asarray(weight2, dtype=_F32)
    P = param.shape[0]

    key_lam, key_v, key_tau = jax.random.split(key, 3)

    # ---- Sample lambda (independent across k) ------------------------------
    # lambda_k | B_{.k} ~ Ga(2p + a + k^{c_1}, sum_j |B_jk|^{1/2} + k^{-c_2})
    # (gamma_sample: exact Gamma unless the driver passes a chi2_plan)
    ink = jnp.sqrt(jnp.maximum(jnp.abs(param), _TINY))

    lam = gamma_sample(key_lam, lambda_concentration(P, a, weight1),
                       ink.sum(0) + 1.0 / weight2, chi2_plan=chi2_plan)

    ink = ink * lam  # now ink = lambda_k * sqrt(|B_jk|)

    # ---- Sample v ----------------------------------------------------------
    # 1/v_jk = InvGauss(1/ink, 1) / 2
    v = 2 / inv_gauss(key_v, 1 / ink)

    # ---- Sample tau --------------------------------------------------------
    # v_jk^2 / tau_jk^2 = InvGauss(v_jk / ink^2, 1);  tau_jk = v_jk / sqrt(.)
    tau = v / jnp.sqrt(inv_gauss(key_tau, v / ink ** 2))

    # sqrt(tau_jk^2) / lambda_k^2  =  prior std of B_jk
    return tau / lam ** 2
