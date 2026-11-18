"""Fixed-structure state containers for the compiled CAVI loop.

Every leaf keeps a fixed shape/dtype across one compiled run (migration
plan section 9.1): floating leaves share the selected precision, error
enumerations are int32, logical flags are bool. Both
containers are plain NamedTuples, hence JAX PyTrees by default.

Error codes (host maps them to exceptions at chunk boundaries):
    0  ERR_NONE
    1  ERR_NONFINITE        -- non-finite rho / m / beta / ELBO
    2  ERR_FACTORIZATION    -- Cholesky / inversion validity flag fired
    3  ERR_ELBO_DROP        -- negative ELBO change beyond the jitter budget

Module ids (which block first failed):
    0 none, 1 eta, 2 B, 3 shrinkage, 4 sigma2, 5 ELBO/stopping
"""

from typing import NamedTuple

import jax

__all__ = [
    "Carry",
    "Inputs",
    "ERR_NONE",
    "ERR_NONFINITE",
    "ERR_FACTORIZATION",
    "ERR_ELBO_DROP",
    "MODULE_NAMES",
]

ERR_NONE = 0
ERR_NONFINITE = 1
ERR_FACTORIZATION = 2
ERR_ELBO_DROP = 3

MODULE_NAMES = {0: "none", 1: "eta", 2: "B", 3: "shrinkage", 4: "sigma2", 5: "elbo"}


class Carry(NamedTuple):
    """The loop carry: minimal recurrence state plus outputs and control.

    mu, V, rho, beta : the recurrence core of one CAVI sweep.
    W, Phi, m        : outputs of the last completed sweep (kept in the
                       carry so the returned state needs no recomputation).
    prev_elbo        : variable part of the ELBO (no constant) of the
                       last completed sweep, selected floating dtype.
    converged        : bool, |delta| < tol at the last completed sweep.
    error            : int32 error code (0 = none).
    error_module     : int32 module id of the first failure.
    """

    mu: jax.Array        # (p, K)
    V: jax.Array         # (p, K, K)
    rho: jax.Array       # (p, K)
    beta: jax.Array      # (p,)
    W: jax.Array         # (n, K)
    Phi: jax.Array       # (K, K)
    m: jax.Array         # (K,)
    prev_elbo: jax.Array  # selected floating dtype
    converged: jax.Array  # bool scalar
    error: jax.Array     # int32 scalar
    error_module: jax.Array  # int32 scalar


class Inputs(NamedTuple):
    """Run-constant inputs (traced, so value changes do not recompile).

    X          : (n, p) data in the selected floating precision.
    shape_k    : (K,) (2p + a + k^{c1}) column weights.
    kneg_c2    : (K,) (k^{-c2}) column weights.
    alpha_n    : floating scalar, a_sigma + n/2.
    a_sigma    : floating scalar noise prior shape.
    b_sigma    : floating scalar noise prior rate.
    tol        : floating scalar absolute ELBO (variable part) tolerance.
    drop_budget: floating scalar; a negative ELBO change beyond this
                 budget is a numerical anomaly (jitter budget, plan 8.3).
    """

    X: jax.Array
    shape_k: jax.Array
    kneg_c2: jax.Array
    alpha_n: jax.Array
    a_sigma: jax.Array
    b_sigma: jax.Array
    tol: jax.Array
    drop_budget: jax.Array
