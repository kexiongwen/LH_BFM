"""
Chi2.py  (JAX, single precision)

Chi-square approximation of Gamma draws: Ga(alpha, rate) with the shape
rounded to the nearest half-integer (dof nu = round(2*alpha),
|delta| <= 0.25) drawn via the identity

      Ga(nu/2, rate) = chi2(nu) / (2*rate),   chi2(nu) = sum of nu N(0,1)^2,

composed purely of normal draws and arithmetic.  In the PyTorch
reference, torch._standard_gamma synchronizes the host, so the
chi-square path is mandatory under
CUDA-graph capture and ~15% faster in eager GPU execution.  In JAX the
exact jax.random.gamma is a pure XLA primitive, so the approximation is
only an accuracy-for-speed tradeoff; the driver's _accept_chi2 decides
whether the rounding perturbation is negligible and opts in
automatically on the GPU backend (everything else uses exact Gamma).

Plan-based GPU path (memory / O(K * nu_max) fix)
------------------------------------------------
The earlier mask implementation materialized a dense (K, nu_max) normal
tensor and a matching (K, nu_max) float32 mask for the lambda step, i.e.
O(K * nu_max) work, where nu_max = round(2(2p + a + K^{c1})).  Since
nu_max grows like K^{c1}, that is superlinear in K and can exhaust GPU
memory.

Instead the driver builds a *plan* once per run with the host-side
builders below; the jitted body then draws exactly the dof each element
needs:

  * make_chi2_plan(concentration, max_elems): per-element dof (lambda
    step).  Elements are sorted by dof and grouped into power-of-two
    buckets (cap = 2^{ceil(log2 nu)}); an entry covers nb elements with
    work nb*cap <= 2 * sum(nu), and entries are split so that
    nb*cap <= max_elems to bound transient device memory.
  * make_chi2_plan_scalar(concentration, n_elements, max_elems): one
    shared dof (sigma2 step).  Work is exactly n_elements*nu, split
    into row chunks bounded by max_elems.

A plan is a tuple of (idx, nu_rows, template) entries:

    idx      : int32 indices of the elements covered by the entry
    nu_rows  : int32 exact dof of each covered element
    template : (0, cap) int8; only the static shape is used.  A
               zero-element array carries cap in template.shape[1]
               without allocating cap bytes per bucket.  The dense
               mask is generated inside jit from nu_rows
               (arange(cap) < nu_rows[:, None]), so no (K, nu_max)
               mask is ever stored or transferred.

The jitted sampler draws z ~ N(0, I) of shape (nb, cap), masks, sums
z^2 over the last axis and scatter-sets the result by idx.  Every index
appears exactly once, so the result is deterministic (no atomics).
Compared with the dense version the lambda step uses <= 2 * sum(nu_k)
normal draws instead of K * nu_max, and peak transient memory is
bounded by max_elems up to a single element whose dof exceeds it
(such an element is necessarily sampled alone); the sigma2 step is
unchanged in draw count but chunked in memory.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "gamma_sample",
    "make_chi2_mask",
    "make_chi2_plan",
    "make_chi2_plan_scalar",
    "chi2_plan_cost",
    "lambda_concentration",
    "sigma2_concentration",
]

_F32 = jnp.float32
_DEFAULT_MAX_ELEMS = 1 << 22


def lambda_concentration(p: int, a, weight1):
    """Shape of the full conditional lambda_k | B_{.k}.

    Single source of truth for the shape used by BOTH the exact-Gamma
    kernel (Shrinkage.shrinkage, sigma2_sample) and the chi2 plan /
    _accept_chi2 decision in the driver:

        alpha_k = 2 p + a + k^{c_1}.

    ``p`` must be the number of rows of B and ``weight1`` the vector
    k^{c_1} (k = 1..K), float32 in the sampler.
    """
    return 2 * int(p) + a + weight1


def sigma2_concentration(a_sigma, n: int):
    """Shape of the Gamma draw behind the sigma2 full conditional.

    Single source of truth for

        alpha_sigma = a_sigma + n / 2,

    used by the exact-Gamma kernel (Sigma2_sample.sigma2_sample), the chi2
    plan and the _accept_chi2 rounding decision.
    """
    return a_sigma + 0.5 * n


def make_chi2_mask(concentration) -> jax.Array:
    """LEGACY dense 0/1 mask (kept for comparison / external callers).

    Row k has ones in the first nu_k = round(2 * concentration_k)
    columns and zeros elsewhere; nu_max = max_k nu_k.  The driver no
    longer uses this function: it uses make_chi2_plan() /
    make_chi2_plan_scalar(), which never materialize a dense mask.
    """
    conc = np.asarray(concentration, dtype=np.float32)
    nu = np.round(2.0 * conc)
    nu_max = int(nu.max())
    mask = np.arange(nu_max)[None, :] < nu.reshape(-1)[:, None]
    return jnp.asarray(mask.astype(np.float32))


def _round_dof(concentration) -> np.ndarray:
    conc = np.asarray(concentration, dtype=np.float32)
    if not np.isfinite(conc).all():
        raise ValueError("concentration must contain only finite values")
    nu = np.round(2.0 * conc).astype(np.int64).reshape(-1)
    if nu.size < 1 or (nu < 1).any():
        raise ValueError(f"chi2 dof must be >= 1, got {nu}")
    return nu


def _entry(idx: np.ndarray, nu: np.ndarray, cap: int):
    """One plan entry: (idx, nu_rows, template); cap is the static width.

    template is (0, cap): only its shape carries the width, so no
    storage is allocated for unused mask values.
    """
    return (
        jnp.asarray(np.asarray(idx, dtype=np.int32)),
        jnp.asarray(np.asarray(nu, dtype=np.int32)),
        jnp.zeros((0, int(cap)), dtype=jnp.int8),
    )


def make_chi2_plan(concentration, max_elems=_DEFAULT_MAX_ELEMS):
    """Bucket plan for per-element concentration (the lambda step).

    Parameters
    ----------
    concentration : array-like, shape (K,)
        Gamma shape parameters (dof = round(2 * concentration)).
    max_elems : int, default 1 << 22
        Maximum element count of one plan entry (nb * cap), bounding the
        transient device memory of the sampler.

    Returns
    -------
    tuple of (idx, nu_rows, template) entries, in bucket order.
    """
    if not isinstance(max_elems, (int, np.integer)) or int(max_elems) < 1:
        raise ValueError("max_elems must be a positive integer")
    max_elems = int(max_elems)
    nu = _round_dof(concentration)
    order = np.argsort(nu, kind="stable")
    plan = []
    i = 0
    while i < nu.size:
        cap = 1 << int(np.ceil(np.log2(max(1, int(nu[order[i]])))))
        j = i
        while j < nu.size and nu[order[j]] <= cap:
            j += 1
        step = max(1, max_elems // cap)
        for start in range(i, j, step):
            stop = min(j, start + step)
            idx = order[start:stop]
            plan.append(_entry(idx, nu[idx], cap))
        i = j
    return tuple(plan)


def make_chi2_plan_scalar(concentration, n_elements, max_elems=_DEFAULT_MAX_ELEMS):
    """Row-chunk plan for a scalar concentration (the sigma2 step).

    All n_elements outputs share the same dof, so the work is exactly
    n_elements * nu (no padding); chunks only bound the memory.
    """
    if not isinstance(n_elements, (int, np.integer)) or int(n_elements) < 1:
        raise ValueError("n_elements must be a positive integer")
    if not isinstance(max_elems, (int, np.integer)) or int(max_elems) < 1:
        raise ValueError("max_elems must be a positive integer")
    n_elements = int(n_elements)
    max_elems = int(max_elems)
    nu = int(np.round(2.0 * float(concentration)))
    if nu < 1:
        raise ValueError(f"chi2 dof must be >= 1, got {nu}")
    step = max(1, max_elems // nu)
    plan = []
    for start in range(0, n_elements, step):
        stop = min(n_elements, start + step)
        plan.append(_entry(np.arange(start, stop), np.full(stop - start, nu), nu))
    return tuple(plan)


def chi2_plan_cost(plan) -> int:
    """Estimated number of normal draws in a plan (0 for None / exact)."""
    if plan is None:
        return 0
    total = 0
    for idx, _nu_rows, template in plan:
        total += int(idx.shape[0]) * int(template.shape[1])
    return total


def _chi2_from_plan(key: jax.Array, plan, n_out: int) -> jax.Array:
    """chi2(nu_k) vector from a plan; no dense (K, nu_max) tensor."""
    out = jnp.zeros((int(n_out),), dtype=_F32)
    for b, (idx, nu_rows, template) in enumerate(plan):
        nb = int(idx.shape[0])
        cap = int(template.shape[1])
        z = jax.random.normal(jax.random.fold_in(key, b), (nb, cap), _F32)
        mask = jnp.arange(cap, dtype=jnp.int32)[None, :] < nu_rows[:, None]
        chi2 = jnp.sum(z * z * mask, axis=1)
        out = out.at[idx].set(chi2)      # each index occurs once: deterministic
    return out


def gamma_sample(
    key: jax.Array,
    concentration,
    rate: jax.Array,
    chi2_plan=None,
) -> jax.Array:
    r"""Draw from Ga(concentration, rate) (shape-rate), elementwise.

    Parameters
    ----------
    key : jax.random.PRNGKey
    concentration : float or array-like
        Shape parameter(s), scalar or broadcastable with `rate`.  The
        chi2 path takes the dof from `chi2_plan` (built from the same
        concentration) and only uses `rate` for scaling.
    rate : array-like
        Rate parameter(s).  The chi2 path requires a 1-D rate vector.
    chi2_plan : tuple or None
        Plan from :func:`make_chi2_plan` / :func:`make_chi2_plan_scalar`.
        None draws EXACT Gamma via jax.random.gamma; a plan selects the
        chi-square approximation with the shape rounded to the nearest
        half-integer (see the module docstring).

    Returns
    -------
    jnp.ndarray, float32.
    """
    rate = jnp.asarray(rate, dtype=_F32)
    if chi2_plan is None:
        shape = jnp.broadcast_shapes(jnp.shape(concentration), rate.shape)
        return jax.random.gamma(key, concentration, shape=shape,
                                dtype=_F32) / rate
    if rate.ndim != 1:
        raise ValueError("the chi2 plan path requires a 1-D rate vector")
    n_out = int(rate.shape[0])
    # Validate plan coverage on the host (plans are built on host and closed
    # over by the jitted samplers).  Without this check a too-short plan
    # silently leaves trailing elements at 0 and a too-long plan silently
    # drops out-of-bounds scatter updates.
    if not isinstance(chi2_plan, (tuple, list)):
        raise TypeError(
            "chi2_plan must be a tuple of (idx, nu_rows, template) entries"
        )
    covered = []
    traced_plan = False
    for entry in chi2_plan:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise TypeError("each chi2_plan entry must be (idx, nu_rows, template)")
        try:
            covered.append(np.asarray(entry[0], dtype=np.int64).ravel())
        except (jax.errors.TracerArrayConversionError, TypeError):
            # Inside jit (e.g. the GS_LH sweep) the plan entries are traced
            # arguments, so their values are not available in Python; the
            # driver builds and validates such plans on the host before the
            # compiled loop, and this public check applies to concrete plans.
            traced_plan = True
            break
    if not traced_plan:
        covered = (
            np.concatenate(covered) if covered else np.empty(0, dtype=np.int64)
        )
        if covered.size != n_out:
            raise ValueError(
                f"chi2_plan covers {covered.size} elements but rate has {n_out}"
            )
        if np.unique(covered).size != covered.size:
            raise ValueError("chi2_plan contains duplicate element indices")
        if covered.min() < 0 or covered.max() >= n_out:
            raise ValueError(
                f"chi2_plan element indices must lie in [0, {n_out})"
            )
    chi2 = _chi2_from_plan(key, chi2_plan, n_out)
    return chi2 / (2.0 * rate)
