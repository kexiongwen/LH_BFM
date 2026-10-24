"""Coordinate ascent variational inference (CAVI) driver -- JAX, float32.

One sweep updates, in order, q(eta), q(B), the shrinkage moments rho,
and q(sigma2), then evaluates the variable part of the closed-form
ELBO (Appendix B2); the constant is added back on the host for the
reported history.  This block order matches the paper's Algorithm 1:
the closed form is exact only at states where beta_j = b_sigma +
R_j / 2 and m_k = sum_j E[B_jk^2]^{1/4} + k^{-c2} hold for the current
(W, Phi, mu, V), and ending the sweep with the shrinkage and sigma2
updates guarantees that.

Single precision (float32) throughout -- the same philosophy as :mod:`bfm._gs`:
X is coerced on entry and every module derives its dtype from it;
JAX's x64 mode is neither required nor enabled by this package.  The
stopping criterion uses the VARIABLE PART of the ELBO only
(delta_t = L_var_t - L_var_{t-1}, |delta_t| < tol), which keeps the
large invariant constant out of the float32 difference -- the main
cancellation defence under single precision (migration plan 8.2;
review finding F1).  The float32 ELBO resolution is on the order of
|ELBO_var| * 1e-7; tolerances near that scale can respond to rounding
jitter.  BFM uses a default tol of 1e-1 and an independent absolute
drop_budget of 0.5 (the earlier driver used tol 1e-2 and a
tolerance-dependent drop budget).

Execution (migration plan 9.2): the pure sweep is inlined into a
compiled lax.while_loop chunk runner; chunks sync with the host only
at their boundaries (ELBO history transfer, audits, progress).  AOT
compilation (lower().compile()) happens before the timed loop, so
compile time is never counted in runtime_sec.  graph=True on a GPU
backend compiles the same runner with XLA command buffers (CUDA
graphs); elsewhere it warns and falls back.  jit=False runs the sweep
eagerly for debugging and step-by-step cross-backend comparison.

Device-side numerical diagnostics (plan section 10, review findings
F2/F3): each sweep checks the factorization validity flags, the
finiteness of rho / m / beta / the ELBO variable part, and that the
ELBO did not drop by more than the jitter budget.  A failing sweep
keeps the last valid state, records (error, sweep, module), and the
host raises at the chunk boundary -- failures are never returned as
successful results.
"""

import math
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from ._linalg import cholesky_checked, logdet_from_chol
from ._runner import make_run_chunk, sweep
from ._state import (
    ERR_ELBO_DROP,
    ERR_FACTORIZATION,
    ERR_NONFINITE,
    MODULE_NAMES,
    Carry,
    Inputs,
)
from .B import log_det_V
from .elbo import elbo, elbo_const
from .eta import second_moment
from .initialization import PPCA, initialization
from .shrinkage import column_weights, update_shrinkage
from .sigma2 import mean_inverse, update_sigma2

__all__ = ["CAVI", "PPCA", "elbo", "elbo_const", "plot_elbo", "initialization"]

_F32 = jnp.float32

_ERROR_MESSAGES = {
    ERR_NONFINITE: "non-finite shrinkage moments / rates / ELBO",
    ERR_FACTORIZATION: "Cholesky factorization or inversion failed",
    ERR_ELBO_DROP: "ELBO dropped by more than the jitter budget (numerical anomaly)",
}


def _validate_args(
    X: jax.Array,
    K: int,
    a: float,
    c1: float,
    c2: float,
    a_sigma: float,
    b_sigma: float,
    tol: float,
    max_iter: int,
    v0: float,
    chunk_size: int,
    audit_every: int,
) -> np.ndarray:
    """Validate the CAVI arguments and return X as a float64 host array.

    The float64 conversion happens here so the caller can center in
    float64 before the single-precision coercion (casting first would
    quantize away the within-column variation when the column offsets are
    large).

    Returns
    -------
    numpy.ndarray
        ``X`` converted to float64.

    Raises
    ------
    ValueError
        On non-2-D or non-finite data, non-positive hyperparameters,
        or invalid tol / max_iter / K / chunk_size / audit_every.
    """
    try:
        X = np.asarray(X, dtype=np.float64)  # convert first: lists should fail
        # with ValueError, not AttributeError on X.ndim (a ragged list fails
        # here in numpy)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"X must be convertible to a float64 array, got "
            f"{type(X).__name__}: {exc}"
        ) from exc
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {tuple(X.shape)}")
    if not bool(np.isfinite(X).all()):
        raise ValueError("X must contain only finite values (found NaN or inf)")
    try:
        K_int = int(K)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"K must be an integer >= 1, got {K!r}") from exc
    if isinstance(K, (bool, np.bool_)) or K != K_int or K_int < 1:
        raise ValueError(f"K must be an integer >= 1, got {K!r}")
    for name, val in (
        ("a", a), ("c1", c1), ("c2", c2),
        ("a_sigma", a_sigma), ("b_sigma", b_sigma), ("v0", v0),
    ):
        # math.isfinite raises a bare TypeError on non-numeric values;
        # report the documented ValueError instead.
        try:
            valid = math.isfinite(val) and val > 0
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a positive finite number, got {val}")
    try:
        tol_ok = math.isfinite(tol) and tol > 0
    except TypeError:
        tol_ok = False
    if isinstance(tol, (bool, np.bool_)) or not tol_ok:
        raise ValueError(f"tol must be a positive finite number, got {tol}")
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError(f"max_iter must be an integer >= 1, got {max_iter}")
    if (
        isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError(f"chunk_size must be an integer >= 1, got {chunk_size}")
    if (
        isinstance(audit_every, (bool, np.bool_))
        or not isinstance(audit_every, (int, np.integer))
        or audit_every < 0
    ):
        raise ValueError(
            f"audit_every must be an integer >= 0 (0 disables), got {audit_every}"
        )
    if a < 4 or 4.0 * (c1 + c2) <= 1.0:
        warnings.warn(
            f"a={a}, c1={c1}, c2={c2}: the theoretical guarantees "
            f"(well-defined covariance, truncation bound, stochastic "
            f"increasing shrinkage, posterior contraction) require "
            f"a >= 4 and 4(c1 + c2) > 1; the algorithm still runs, but "
            f"these guarantees do not apply",
            UserWarning,
            stacklevel=3,
        )
    return X


def _make_carry(state: dict) -> Carry:
    """Pack an initialization/warm-start dict into the loop carry."""
    return Carry(
        mu=state["mu"],
        V=state["V"],
        rho=state["rho"],
        beta=state["beta"],
        W=state["W"],
        Phi=state["Phi"],
        m=state["m"],
        prev_elbo=jnp.asarray(state["elbo_var"], dtype=_F32),
        t=jnp.int32(0),
        converged=jnp.bool_(False),
        error=jnp.int32(0),
        error_t=jnp.int32(0),
        error_module=jnp.int32(0),
    )


def _normalize_warm_start(
    X: jax.Array,
    initial_state: dict,
    K: int,
    a_sigma: float,
    b_sigma: float,
    shape_k: jax.Array,
    kneg_c2: jax.Array,
) -> dict:
    """Validate and normalize an external initial state.

    Required fields: mu, V, W, Phi.  After shape / finiteness / SPD
    checks, S, E2, m, rho, beta, c and the initial ELBO are recomputed
    so the state is self-consistent in the end-of-sweep sense (the
    entry point does not promise to recover arbitrary non-consistent
    midway states bitwise; cached values are never authoritative).
    """
    n, p = X.shape
    missing = {"mu", "V", "W", "Phi"} - set(initial_state)
    if missing:
        raise ValueError(f"initial_state is missing required fields: {sorted(missing)}")
    mu = jnp.asarray(initial_state["mu"], dtype=_F32)
    V = jnp.asarray(initial_state["V"], dtype=_F32)
    W = jnp.asarray(initial_state["W"], dtype=_F32)
    Phi = jnp.asarray(initial_state["Phi"], dtype=_F32)
    if mu.shape != (p, K):
        raise ValueError(f"initial_state['mu'] must have shape {(p, K)}, got {mu.shape}")
    if V.shape != (p, K, K):
        raise ValueError(
            f"initial_state['V'] must have shape {(p, K, K)}, got {V.shape}"
        )
    if W.shape != (n, K):
        raise ValueError(f"initial_state['W'] must have shape {(n, K)}, got {W.shape}")
    if Phi.shape != (K, K):
        raise ValueError(
            f"initial_state['Phi'] must have shape {(K, K)}, got {Phi.shape}"
        )
    for name, arr in (("mu", mu), ("V", V), ("W", W), ("Phi", Phi)):
        if not bool(jnp.isfinite(arr).all()):
            raise ValueError(f"initial_state['{name}'] contains non-finite values")

    # SPD validity of the supplied covariances (checked, no jitter).
    ldV, ok_V = log_det_V(V)
    L_phi, ok_Phi = cholesky_checked(Phi)
    if not bool(ok_V & ok_Phi):
        raise ValueError(
            "initial_state covariances are not positive definite "
            "(V and Phi must be SPD)"
        )

    S = second_moment(W, Phi)
    rho, m, elbo_shrink = update_shrinkage(mu, V, shape_k, kneg_c2)
    alpha_n, beta, elbo_sigma2 = update_sigma2(X, W, Phi, mu, V, a_sigma, b_sigma)
    c = mean_inverse(alpha_n, beta)
    elbo_eta = 0.5 * n * logdet_from_chol(L_phi) - 0.5 * jnp.trace(S)
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
        "ok": jnp.bool_(True),
    }


def _audit_host(state_arrays: dict, upto: int) -> None:
    """Full finiteness audit of the state at a chunk boundary (host).

    Covers mu, V, W, Phi, beta, rho, m and c (review finding F3; the
    PyTorch reference driver's audit was later extended to the same coverage).
    A non-finite state signals a numerical anomaly outside the guarded
    range; fail loudly.
    """
    bad = [k for k, a in state_arrays.items() if not np.isfinite(np.asarray(a)).all()]
    if bad:
        raise FloatingPointError(
            f"non-finite variational state ({', '.join(bad)}) detected in "
            f"the chunk ending at sweep {upto}; the run is numerically "
            f"contaminated and must be discarded. This should not happen "
            f"(all factorized systems are I + PSD by construction and the "
            f"device-side flags did not fire) -- please report the data "
            f"and hyperparameters."
        )


def CAVI(
    X: jax.Array,
    K: int,
    a: float = 10.0,
    c1: float = 2.0,
    c2: float = 1.0,
    a_sigma: float = 1.0,
    b_sigma: float = 1.0,
    tol: float | None = 1e-1,
    max_iter: int = 1000,
    r: int | None = None,
    v0: float = 1.0,
    center: bool = True,
    initial_state: dict | None = None,
    jit: bool = True,
    chunk_size: int = 50,
    graph: bool = False,
    audit_every: int = 50,
    drop_budget: float | None = 0.5,
    verbose: bool = False,
    plot: bool = False,
) -> dict:
    """Coordinate ascent variational inference (Algorithm 1), JAX float32.

    Parameters
    ----------
    X : jax.Array, shape (n, p)
        Data matrix.  The model has zero mean, so columns are centered
        by default (``center=True``).  Centering is done in float64 on
        the host, then the centered matrix is coerced to float32 (casting
        first would quantize away the within-column variation when the
        column offsets are large).
    K : int
        Truncation level (upper bound on the number of factors).
    a, c1, c2 : float
        Hyperparameters of the L_{1/2} shrinkage prior.  Defaults a=10,
        c1=2.0, c2=1.0 satisfy a >= 4 and q = 4(c1 + c2) > 1.
    a_sigma, b_sigma : float
        Inverse-gamma hyperparameters of the noise variances.
    tol : float or None, default 1e-1
        Absolute ELBO stopping tolerance.  After each complete sweep,
        stop when abs(Lvar_t - Lvar_prev) < tol, where Lvar excludes
        the invariant ELBO constant.  Must be positive and finite;
        None also resolves to 1e-1.  Float32 ELBO resolution is on the
        order of |ELBO_var| * 1e-7; tolerances near that scale can
        respond to rounding jitter.
    max_iter : int, default 1000
        Maximum number of CAVI sweeps.
    r : int or None, default None
        Number of leading factors used by the PPCA initialization;
        defaults to min(K, min(n, p) - 2).  Values larger than K are
        clipped to K.  Ignored with ``initial_state``.
    v0 : float, default 1.0
        Initial loading covariance scale, V_j = v0 * I_K.  Ignored
        with ``initial_state``.
    center : bool, default True
        Subtract the column means of X before running.  The means are
        computed and subtracted in float64 on the host, then the centered
        matrix is cast to float32.  With ``center=False`` the data are
        ASSUMED already column-centered: the PPCA initialization (thin SVD
        with the n-1 denominator) relies on it, so running uncentered data
        with ``center=False`` biases the initial noise floor.  This affects
        only the initial values (and possibly the sweep count), not the
        CAVI updates themselves; the model has zero mean, so centering the
        data is strongly recommended.
    initial_state : dict or None, default None
        Warm start: must contain mu (p, K), V (p, K, K), W (n, K),
        Phi (K, K); validated (shape, finiteness, SPD) and normalized
        into a self-consistent state (S, rho, m, beta, c and the
        initial ELBO are recomputed).
    jit : bool, default True
        Compile the sweep into chunked lax.while_loop runners.  False
        runs the sweep eagerly, one host-visible step at a time (for
        debugging and fixed-count cross-backend comparison).
    chunk_size : int, default 50
        Sweeps per compiled chunk: bounds the host sync frequency
        (ELBO history transfer, audits, progress).  Does NOT change
        the stopping point -- convergence is checked on the device
        after every sweep.
    graph : bool, default False
        On a GPU backend, compile the chunk runner with XLA command
        buffers (CUDA graphs): the whole chunk is recorded as one CUDA
        graph and replayed.  Elsewhere a RuntimeWarning is issued and
        the run proceeds with the plain runner.
    audit_every : int, default 50
        Host-side full-state finiteness audit threshold.  Audits run ONLY at
        compiled chunk boundaries (see ``chunk_size``) and when the host
        regains control; the check is ``t - last_audit >= audit_every`` at
        those points.  The actual spacing is therefore the next chunk
        boundary at/after ``audit_every`` sweeps -- roughly
        ``max(chunk_size, audit_every)`` sweeps (and exactly one chunk when
        ``chunk_size <= audit_every``), not necessarily every
        ``audit_every`` sweeps.  Set ``chunk_size <= audit_every`` to get the
        requested cadence.  0 disables the periodic audit; the mandatory
        init/final audits and the per-sweep device-side checks are
        unaffected.
    drop_budget : float or None, default 0.5
        Maximum permitted absolute ELBO decrease per sweep, independent
        of tol.  A larger decrease raises FloatingPointError,
        even if it satisfies the stopping threshold.  Must be finite
        and nonnegative; 0 requires non-decreasing ELBO.  None also
        resolves to the fixed default 0.5.
    verbose : bool, default False
        Print the ELBO at chunk boundaries; warn if it decreases
        beyond the jitter budget (a numerical issue).
    plot : bool, default False
        After the loop ends, plot the ELBO trajectory with
        :func:`plot_elbo` (requires matplotlib).

    Raises
    ------
    ValueError
        On invalid data, hyperparameters, or run parameters, or on an
        invalid ``initial_state``.
    UserWarning
        If the hyperparameters violate the theory conditions a >= 4
        and 4(c1 + c2) > 1 (the algorithm still runs).
    FloatingPointError
        On a device-side numerical failure (non-finite state, failed
        factorization, ELBO drop beyond the jitter budget) or a failed
        host audit; the message carries the sweep and module.

    Returns
    -------
    dict with keys
        mu : (p, K) variational means of the loading rows,
        V : (p, K, K) variational covariances of the loading rows,
        W : (n, K) variational means of the factor scores,
        Phi : (K, K) common covariance of the factor scores,
        alpha_n : float, noise inverse-gamma shape a_sigma + n/2,
        beta : (p,) noise inverse-gamma rates,
        c : (p,) E_q[1 / sigma_j^2] = alpha_n / beta_j,
        rho : (p, K) shrinkage moments E_q[lambda_k^4 / tau_jk^2],
        m : (K,) column aggregates,
        elbo : list of float, absolute ELBO after initialization and
        each sweep (constant included; length n_iter + 1),
        n_iter : int, number of sweeps performed,
        converged : bool, whether the ELBO tolerance was reached,
        stop_reason : "tol" or "max_iter",
        execution_mode : "while_loop", "command_buffer" or "eager",
        backend, device : execution environment summary,
        compile_sec : AOT compilation wall time (0 in eager mode),
        runtime_sec : sweep-loop wall time (excludes compilation).
    """
    if tol is None:
        tol = 1e-1
    X64 = _validate_args(X, K, a, c1, c2, a_sigma, b_sigma, tol,
                         max_iter, v0, chunk_size, audit_every)
    if drop_budget is None:
        drop_budget = 0.5
    try:
        budget_ok = math.isfinite(drop_budget) and drop_budget >= 0
    except TypeError:
        budget_ok = False
    if isinstance(drop_budget, (bool, np.bool_)) or not budget_ok:
        raise ValueError(
            f"drop_budget must be a nonnegative finite number, got {drop_budget}"
        )
    if r is not None and (
        isinstance(r, (bool, np.bool_))
        or not isinstance(r, (int, np.integer))
        or int(r) < 1
    ):
        raise ValueError(f"r must be a positive integer or None, got {r!r}")

    # Center in float64 on the host and cast once to float32.  Casting the
    # uncentered matrix first would round entries at the scale of the column
    # offset, destroying the within-column variation (see BaseBFM._prepare_X).
    if center:
        X64 = X64 - X64.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X = jnp.asarray(X64, dtype=_F32)
    if not bool(jnp.all(jnp.isfinite(X))):
        raise ValueError(
            "the (centered) X matrix overflows the float32 coercion; "
            "rescale the data (the estimation model is single precision)"
        )
    n, p = X.shape
    K = int(K)

    shape_k, kneg_c2 = column_weights(p, K, a, c1, c2)
    const = elbo_const(n, p, K, a, c1, c2, a_sigma, b_sigma)

    # --- initialization ------------------------------------------------
    if initial_state is None:
        state = initialization(
            X, K, a_sigma, b_sigma, shape_k, kneg_c2, r=r, v0=v0
        )
    else:
        state = _normalize_warm_start(
            X, initial_state, K, a_sigma, b_sigma, shape_k, kneg_c2
        )
    if not bool(state["ok"]):
        raise FloatingPointError(
            "initialization produced a numerically invalid state "
            "(factorization flags fired); please report the data."
        )
    # mandatory init audit (covers states that never reach a periodic point)
    _audit_host(
        {"mu": state["mu"], "V": state["V"], "W": state["W"], "Phi": state["Phi"],
         "beta": state["beta"], "rho": state["rho"], "m": state["m"],
         "c": state["c"]},
        0,
    )

    carry = _make_carry(state)
    inp = Inputs(
        X=X,
        shape_k=shape_k,
        kneg_c2=kneg_c2,
        alpha_n=jnp.asarray(a_sigma + n / 2.0, dtype=_F32),
        a_sigma=jnp.asarray(a_sigma, dtype=_F32),
        b_sigma=jnp.asarray(b_sigma, dtype=_F32),
        tol=jnp.asarray(tol, dtype=_F32),
        drop_budget=jnp.asarray(drop_budget, dtype=_F32),
        max_iter=jnp.int32(max_iter),
    )

    history = [const + float(state["elbo_var"])]
    if verbose:
        print(f"[CAVI] init      ELBO = {history[0]:.6f}")

    # --- runner selection ----------------------------------------------
    use_graph = graph and jax.default_backend() == "gpu"
    if graph and not use_graph:
        warnings.warn(
            f"graph=True requires a GPU backend (got "
            f"'{jax.default_backend()}'); running with the plain runner",
            RuntimeWarning,
        )
    execution_mode = "command_buffer" if use_graph else ("while_loop" if jit else "eager")

    def _raise_on_error(c: Carry) -> None:
        err = int(c.error)
        if err != 0:
            msg = _ERROR_MESSAGES.get(err, f"unknown numerical error {err}")
            mod = MODULE_NAMES.get(int(c.error_module), "?")
            raise FloatingPointError(
                f"{msg} at sweep {int(c.error_t)} (module: {mod}); the run "
                f"is numerically contaminated and must be discarded."
            )

    def _state_arrays(c: Carry) -> dict:
        return {"mu": c.mu, "V": c.V, "W": c.W, "Phi": c.Phi, "beta": c.beta,
                "rho": c.rho, "m": c.m, "c": inp.alpha_n / c.beta}

    # --- CAVI sweeps ----------------------------------------------------
    compile_sec = 0.0
    last_audit = 0
    if not jit:
        # eager debug path: one host-visible sweep at a time
        t_start = time.perf_counter()
        while (
            int(carry.t) < max_iter
            and not bool(carry.converged)
            and int(carry.error) == 0
        ):
            carry = sweep(carry, inp)
            _raise_on_error(carry)
            history.append(const + float(carry.prev_elbo))
            if verbose:
                v, prev = history[-1], history[-2]
                print(f"[CAVI] sweep {len(history) - 1:4d}  ELBO = {v:.6f}  "
                      f"d = {v - prev:.3e}")
                if v < prev - drop_budget:
                    print("[CAVI] warning: ELBO decreased (numerical issue)")
            if audit_every and int(carry.t) - last_audit >= audit_every:
                _audit_host(_state_arrays(carry), int(carry.t))
                last_audit = int(carry.t)
        runtime_sec = time.perf_counter() - t_start
    else:
        run_chunk = make_run_chunk(graph=use_graph)
        buf = jnp.zeros((chunk_size,), dtype=_F32)
        # AOT compilation (not timed): one executable, reused by every
        # chunk including the shorter tail (dynamic count).
        t_compile = time.perf_counter()
        exe = run_chunk.lower(carry, buf, inp).compile()
        compile_sec = time.perf_counter() - t_compile

        t_start = time.perf_counter()
        while (
            int(carry.t) < max_iter
            and not bool(carry.converged)
            and int(carry.error) == 0
        ):
            carry, buf, used = exe(carry, buf, inp)
            used = int(used)
            _raise_on_error(carry)
            if used:
                vals = np.asarray(buf[:used]).astype(np.float64) + const
                base = len(history) - 1
                history.extend(vals.tolist())
                if verbose:
                    for k, v in enumerate(vals):
                        prev = history[base + k]
                        print(f"[CAVI] sweep {base + k + 1:4d}  ELBO = {v:.6f}  "
                              f"d = {v - prev:.3e}")
                        if v < prev - drop_budget:
                            print("[CAVI] warning: ELBO decreased (numerical issue)")
            if audit_every and int(carry.t) - last_audit >= audit_every:
                _audit_host(_state_arrays(carry), int(carry.t))
                last_audit = int(carry.t)
        runtime_sec = time.perf_counter() - t_start

    stop_reason = "tol" if bool(carry.converged) else "max_iter"

    # mandatory final audit (covers runs that end before any periodic point)
    _audit_host(_state_arrays(carry), int(carry.t))

    c = mean_inverse(inp.alpha_n, carry.beta)
    out = {
        "mu": carry.mu,
        "V": carry.V,
        "W": carry.W,
        "Phi": carry.Phi,
        "alpha_n": float(inp.alpha_n),
        "beta": carry.beta,
        "c": c,
        "rho": carry.rho,
        "m": carry.m,
        "elbo": history,
        "n_iter": int(carry.t),
        "converged": bool(carry.converged),
        "stop_reason": stop_reason,
        "execution_mode": execution_mode,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "compile_sec": compile_sec,
        "runtime_sec": runtime_sec,
    }

    if plot:
        plot_elbo(history, show=True)
    return out


def plot_elbo(history, ax=None, logx=False, show=False, save=None, title=None):
    """Plot the ELBO trajectory of a CAVI run against the iteration count.

    Iteration 0 is the ELBO after initialization; iteration t is the
    ELBO after sweep t.  matplotlib is imported lazily, so VI.py does
    not depend on it unless this function (or CAVI with ``plot=True``)
    is used.

    Parameters
    ----------
    history : dict or sequence of float
        The dict returned by :func:`CAVI` (its ``"elbo"`` entry is
        used) or the ELBO sequence itself.
    ax : matplotlib.axes.Axes, optional
        Drawn into this axes; a new figure is created when omitted.
    logx : bool, default False
        Logarithmic iteration axis (useful for long runs); sweep 0 is
        dropped in that case since log(0) is undefined.
    show : bool, default False
        Call ``plt.show()`` for interactive use.
    save : str or Path, optional
        Save the figure to this path (e.g. ``"elbo.png"``).
    title : str, optional
        Figure title; defaults to a summary with the final ELBO and
        the number of sweeps.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the history is empty.
    ImportError
        If matplotlib is not installed.
    """
    if isinstance(history, dict):
        history = history["elbo"]
    history = list(history)
    if len(history) == 0:
        raise ValueError("empty ELBO history")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_elbo requires matplotlib (e.g. pip install matplotlib)"
        ) from exc

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    xs = list(range(len(history)))
    if logx:
        xs, history = xs[1:], history[1:]
        ax.set_xscale("log")
    ax.plot(xs, history, lw=1.5)
    ax.set_xlabel("sweep" + (" (log scale)" if logx else ""))
    ax.set_ylabel("ELBO")
    ax.set_title(
        title
        or f"CAVI ELBO trajectory (final {history[-1]:.6g}, "
           f"{len(history) - (0 if logx else 1)} sweeps shown)"
    )
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
