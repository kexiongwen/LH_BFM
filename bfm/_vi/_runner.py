"""Compiled execution layer: pure sweep, chunked while_loop runners.

Three execution levels (migration plan section 9.2):

1. :func:`sweep` -- the pure CAVI sweep (eta -> B -> shrinkage ->
   sigma2 -> ELBO + device-side diagnostics).  Never jitted on its
   own; it is inlined into the runners below (matching :mod:`bfm._gs`).
2. :func:`sweep_jit` -- single-sweep JIT kept for interactive debugging
   and fixed-count cross-backend comparisons.  Not exported (not in
   ``__all__``) and not used by the driver, which only runs the
   chunked runner.
3. :func:`make_run_chunk` -- a compiled lax.while_loop advancing the
   state until the chunk is full, the global max_iter is reached,
   convergence is detected, or a numerical failure is flagged.

Device-side diagnostics (plan section 10, review findings F2/F3):
each sweep checks the factorization validity flags (eta and B), the
finiteness of rho / m / beta / the ELBO variable part, and that the
ELBO did not drop by more than the jitter budget.  A failing sweep
does NOT overwrite the last valid state and is NOT counted in the
chunk's ELBO history; it only records (error, error_t, error_module),
which the host turns into an exception at the chunk boundary.

The chunk runner writes the variable-part ELBO of each successful
sweep into a fixed-length device buffer with a dynamic count, so tail
chunks reuse the same compiled executable (no recompilation).  The
constant is added back on the host for the reported history.
"""

import jax
import jax.numpy as jnp

from ._linalg import cholesky_checked, logdet_from_chol
from ._state import (
    ERR_ELBO_DROP,
    ERR_FACTORIZATION,
    ERR_NONE,
    ERR_NONFINITE,
    Carry,
    Inputs,
)
from .B import log_det_V, update_B
from .eta import update_eta
from .shrinkage import update_shrinkage
from .sigma2 import update_sigma2

__all__ = ["sweep", "make_run_chunk", "GRAPH_COMPILER_OPTIONS"]

# XLA:GPU command-buffer (CUDA graph) capture of the whole while_loop
# chunk: with CONDITIONAL+WHILE enabled, the entire chunk executable is
# recorded as ONE CUDA graph on first execution and replayed afterwards
# (the :mod:`bfm._gs` options, GPU only, passed per-compile
# so graph mode is a per-call choice).
GRAPH_COMPILER_OPTIONS = {
    "xla_gpu_enable_command_buffer":
        "FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE",
    "xla_gpu_graph_min_graph_size": 1,
}


def sweep(carry: Carry, inp: Inputs) -> Carry:
    """One full CAVI sweep (eta -> B -> shrinkage -> sigma2), pure.

    Same block order as the PyTorch reference driver and the paper's
    Algorithm 1: shrinkage and sigma2 come last so that m and beta are
    consistent
    with the current (W, Phi, mu, V) at the ELBO evaluation point,
    making the evaluated variable part the exact variable part of the
    current state (non-decreasing up to float32 jitter).

    On numerical failure the returned carry keeps the OLD variational
    state (the failing candidate does not overwrite it) and only the
    error fields are set; t and prev_elbo stay at the last valid
    sweep.  The while_loop's cond then stops the chunk.
    """
    c = inp.alpha_n / carry.beta
    W, Phi, S, elbo_eta, ok_eta = update_eta(inp.X, carry.mu, carry.V, c)
    WtX = W.T @ inp.X  # computed once per sweep, used by the B update
    mu, V, elbo_B, ok_B = update_B(inp.X, W, Phi, c, carry.rho, WtX=WtX, S=S)
    # shrinkage and sigma2 come last so that m and beta are consistent
    # with the current (W, Phi, mu, V) at the ELBO evaluation point
    rho, m, elbo_shrink = update_shrinkage(mu, V, inp.shape_k, inp.kneg_c2)
    _, beta, elbo_sigma2 = update_sigma2(
        inp.X, W, Phi, mu, V, inp.a_sigma, inp.b_sigma, S=S
    )
    elbo_var = elbo_eta + elbo_B + elbo_shrink + elbo_sigma2
    delta = elbo_var - carry.prev_elbo

    # ---- device-side diagnostics -------------------------------------
    chol_ok = ok_eta & ok_B
    fin_ok = (
        jnp.isfinite(rho).all()
        & jnp.isfinite(m).all()
        & jnp.isfinite(beta).all()
        & jnp.isfinite(elbo_var)
    )
    drop_ok = delta >= -inp.drop_budget
    ok = chol_ok & fin_ok & drop_ok

    error = jnp.where(
        chol_ok,
        jnp.where(fin_ok, jnp.where(drop_ok, ERR_NONE, ERR_ELBO_DROP), ERR_NONFINITE),
        ERR_FACTORIZATION,
    ).astype(jnp.int32)
    shrink_bad = ~(jnp.isfinite(rho).all() & jnp.isfinite(m).all())
    beta_bad = ~jnp.isfinite(beta).all()
    fin_module = jnp.where(shrink_bad, 3, jnp.where(beta_bad, 4, 5))
    module = (
        jnp.where(~ok_eta, 1, 0)
        + jnp.where(ok_eta & ~ok_B, 2, 0)
        + jnp.where(chol_ok & ~fin_ok, fin_module, 0)
        + jnp.where(chol_ok & fin_ok & ~drop_ok, 5, 0)
    ).astype(jnp.int32)

    new_t = carry.t + jnp.int32(1)
    # The independent drop check must also pass before accepting convergence.
    converged = jnp.abs(delta) < inp.tol

    # On failure keep the OLD variational state; set only error fields.
    def pick(new, old):
        return jnp.where(ok, new, old)

    return Carry(
        mu=pick(mu, carry.mu),
        V=pick(V, carry.V),
        rho=pick(rho, carry.rho),
        beta=pick(beta, carry.beta),
        W=pick(W, carry.W),
        Phi=pick(Phi, carry.Phi),
        m=pick(m, carry.m),
        prev_elbo=pick(elbo_var, carry.prev_elbo),
        t=pick(new_t, carry.t),
        converged=pick(converged, carry.converged),
        error=jnp.where(ok, carry.error, error),
        error_t=jnp.where(ok, carry.error_t, new_t),
        error_module=jnp.where(ok, carry.error_module, module),
    )


# Single-sweep JIT for debugging and fixed-count cross-backend checks.
sweep_jit = jax.jit(sweep)


def _run_chunk_impl(carry: Carry, buf: jax.Array, inp: Inputs):
    """Advance over one chunk of sweeps (compiled lax.while_loop).

    Stops at the first of: chunk full (``i == len(buf)``), a numerical
    error flag, convergence, or the global max_iter.  Only successful
    sweeps are written into ``buf`` (their variable-part ELBO) and
    counted; the returned count is the valid prefix length.
    """
    chunk_len = buf.shape[0]

    def cond(cs):
        c, b, i = cs
        return (
            (i < chunk_len)
            & (c.error == ERR_NONE)
            & (~c.converged)
            & (c.t < inp.max_iter)
        )

    def body(cs):
        c, b, i = cs
        c = sweep(c, inp)
        good = c.error == ERR_NONE
        b = b.at[i].set(jnp.where(good, c.prev_elbo, jnp.float32(0.0)))
        i = i + good.astype(jnp.int32)
        return c, b, i

    return jax.lax.while_loop(cond, body, (carry, buf, jnp.int32(0)))


def make_run_chunk(graph: bool = False):
    """Return the compiled chunk runner (plain JIT or XLA command buffers).

    ``graph=True`` requests XLA command buffers (CUDA graphs): the
    whole while_loop chunk is recorded as one CUDA graph and replayed,
    the JAX analog of the PyTorch reference's CUDA-graph sweep.  It is a
    per-compile choice for GPU backends; callers must only pass
    graph=True when jax.default_backend() == "gpu" (the driver warns
    and falls back otherwise).
    """
    if graph:
        return jax.jit(_run_chunk_impl, compiler_options=GRAPH_COMPILER_OPTIONS)
    return jax.jit(_run_chunk_impl)


def elbo_variable_of_state(carry_arrays: dict, inp: Inputs):
    """Variable-part ELBO of a carried state, recomputed independently.

    Standalone diagnostic helper (not exported in ``__all__`` and not
    used by the driver, whose warm-start path normalizes inline):
    re-factorizes the stored covariances instead of trusting cached
    byproducts, and returns the validity flags alongside.  Requires the
    state to be self-consistent (beta, m as stored).

    Parameters
    ----------
    carry_arrays : dict with W, Phi, beta, V, m.
    inp : Inputs (X unused).

    Returns
    -------
    elbo_var : jax.Array, scalar
    ok : jax.Array, scalar bool
    """
    n = carry_arrays["W"].shape[0]
    L_phi, ok_phi = cholesky_checked(carry_arrays["Phi"])
    log_det_Phi = logdet_from_chol(L_phi)
    ldV, ok_V = log_det_V(carry_arrays["V"])
    S = carry_arrays["W"].T @ carry_arrays["W"] + n * carry_arrays["Phi"]
    elbo_var = (
        0.5 * n * log_det_Phi
        - 0.5 * jnp.trace(S)
        + 0.5 * ldV.sum()
        - inp.alpha_n * jnp.log(carry_arrays["beta"]).sum()
        - (inp.shape_k * jnp.log(carry_arrays["m"])).sum()
    )
    return elbo_var, ok_phi & ok_V
