"""
GS_LH.py  (JAX, single precision)

Gibbs sampler driver for the Bayesian factor model with the L_{1/2}
shrinkage prior (paper: "Bayesian factorization via L_{1/2} shrinkage",
Section 3).

Orchestrates one full sweep of the sampler by calling, in order
(each step is an exact full conditional; see Section 3):

    1. eta_sample    : eta_i | B, Omega, x_i                 (i = 1..n)
    2. shrinkage     : lambda_k | B_{.k}; tau_jk^2 via InvGauss draws
    3. sigma2_sample : sigma_j^2 | X_{.j}, B_j, eta          (j = 1..p)
    4. B_sample      : B_j | X_{.j}, eta, lambda, tau_j^2, sigma_j^2

Model:
    x_i = B eta_i + eps_i,
    eta_i ~ N_K(0, I_K),
    eps_i ~ N_p(0, Omega),  Omega = diag(sigma_1^2, ..., sigma_p^2).

Prior:
    pi(B_jk | lambda_k) = (lambda_k^2 / 4) exp(-lambda_k |B_jk|^{1/2}),
    lambda_k ~ Ga(a + k^{c_1}, k^{-c_2})   (shape-rate),
    sigma_j^2 ~ IG(a_sigma, b_sigma).

Chain structure: the Markov state carried between sweeps is only
(B, sigma2) -- the lambda/v/tau chain is a marginal-then-conditional
draw given B and is regenerated every sweep, so it is not sampler state.

JAX-specific notes
------------------
* Single precision (float32) throughout: X is coerced on entry and
  every module derives its dtype from it.  jax's x64 mode is neither
  required nor enabled by this package.
* Randomness is explicit: `seed` seeds one jax.random.PRNGKey that is
  split into one key per sweep up front; the same seed reproduces the
  same chain.
* The chain is executed as jax.lax.scan over the sweep keys, in chunks
  of at most _CHUNK sweeps: each chunk is ONE compiled XLA While loop
  (the full 4-step sweep inside), with no Python dispatch or host
  synchronization inside.  Between chunks the draws are copied to the
  host (bounded device memory, progress reports, finiteness audit).
  Two compiled runners are used: a burn-in runner that advances the
  state without collecting draws, and a collecting runner for the kept
  phase; compilation is cached per (chunk length, shapes) across GS_LH
  calls and is done ahead of time (lower().compile()), so no
  compilation is counted in runtime_sec.  With graph=True (GPU only),
  the runners are compiled with XLA command buffers, so each chunk is
  additionally recorded as ONE CUDA graph and replayed -- the JAX
  analog of the PyTorch reference's CUDA-graph sweep.
* jax.random.gamma is a pure XLA primitive -- exact and capturable by
  CUDA graphs on every backend -- so the chi-square Gamma approximation
  of Chi2.py is never MANDATORY here (in the PyTorch reference
  torch._standard_gamma syncs the host, forcing chi-square under graph
  capture).  On the GPU backend the driver still offers it as an
  accuracy-for-speed tradeoff: _accept_chi2 applies the same rule as
  the PyTorch reference (half-integer rounding of the Gamma shapes
  negligible, e.g. large n/p or integer shapes), and the approximation
  then applies to both the plain and the graph runners; otherwise, and
  on every non-GPU backend, exact Gamma is used.
* Posterior draws are stored on the host and returned as numpy arrays
  (the JAX analog of the PyTorch reference's CPU tensors).

Implementation: JAX-based, functional style.
"""

import os
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .B_sample import B_sample
from .Chi2 import (
    chi2_plan_cost,
    lambda_concentration,
    make_chi2_plan,
    make_chi2_plan_scalar,
    sigma2_concentration,
)
from .eta_sample import eta_sample
from .Shrinkage import shrinkage
from .Sigma2_sample import sigma2_sample

__all__ = ["GS_LH", "Initialization"]

_F32 = jnp.float32

# Sweeps per lax.scan chunk (shrunk adaptively for large p*K, see
# _chunk_size): bounds device-side draw storage per chunk and gives one
# progress report / finiteness audit per chunk.
_CHUNK = 500
# Cap on device-side collected draws per chunk, in bytes.
_CHUNK_BYTES = 256 * 2 ** 20


def Initialization(X: jax.Array, r: int) -> tuple[jax.Array, float]:
    r"""Principal-factor initialization for (B, sigma2).

    Eigen-decomposition of the sample covariance (X is assumed
    column-centered, as the model has zero mean): with
    M = X.T / sqrt(n - 1), M M^T = X^T X / (n - 1) = Sigma_hat, and
    eigenvalues s_i^2 of Sigma_hat (descending),

        sigma2_hat = mean of the trailing eigenvalues {s_i^2}_{i > r},
        B0         = U_r diag( sqrt(s_1^2 - sigma2_hat, ...) ),

    so that B0 B0^T = U_r diag(s_i^2 - sigma2_hat) U_r^T is the rank-r
    PSD approximation of Sigma_hat - sigma2_hat I (the factor model
    decomposes Sigma = B B^T + sigma2 I at initialization).

    Implementation notes (JAX implementation):
    * the top-r eigenpairs come from one jnp.linalg.eigh of the p x p
      covariance -- deterministic and exact; the PyTorch reference uses
      torch.pca_lowrank (randomized SVD), which only pays off for p far
      beyond the intended use, and this runs ONCE per chain;
    * the trailing sum uses the exact trace identity
      sum_{i > r} s_i^2 = tr(Sigma_hat) - sum_{i <= r} s_i^2,
      with the denominator min(n - 1, p) - r: column-centered X has rank
      at most min(n - 1, p), so this is exactly the number of trailing
      eigenvalues that can be nonzero (the convention shared with
      the :mod:`bfm._vi` PPCA initialization);
    * s_i^2 - sigma2_hat is floored at 0: top-r components below the
      noise floor initialize to a zero column instead of NaN.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Column-centered data matrix.
    r : int
        Number of leading factors, 1 <= r <= min(n, p) - 2.

    Returns
    -------
    B0 : jnp.ndarray, shape (p, r), float32
        Initial loading matrix.
    sigma2_hat : float
        Noise-floor variance estimate (scalar; broadcast to (p,) by the
        caller for the initial Omega diagonal).
    """
    X = jnp.asarray(X, dtype=_F32)
    n, p = X.shape
    if not 1 <= r <= min(n, p) - 2:
        raise ValueError(
            f"r must be in [1, min(n, p) - 2] = [1, {min(n, p) - 2}], got {r}"
        )

    M = X.T / jnp.sqrt(jnp.asarray(n - 1, dtype=_F32))  # (p, n)
    C = M @ M.T                                          # sample cov (p, p)
    evals, evecs = jnp.linalg.eigh(C)                    # ascending order

    # top-r eigenpairs, descending
    s2 = evals[::-1][:r]
    U = evecs[:, ::-1][:, :r]

    # mean trailing eigenvalue (trace identity for the discarded
    # spectrum; exactly min(n - 1, p) - r trailing eigenvalues can be
    # nonzero for column-centered X)
    tail = jnp.trace(C) - s2.sum()
    sigma2_hat = jnp.maximum(tail / (min(n - 1, p) - r), 0.0)

    B0 = U * jnp.sqrt(jnp.maximum(s2 - sigma2_hat, 0.0))[None, :]
    return B0, float(sigma2_hat)


def _initial_loadings(B0: jax.Array, p: int, K: int, r: int) -> jax.Array:
    """Assemble the (p, K) starting B, keeping inactive columns non-zero.

    ``Initialization`` only provides the first r columns.  The remaining
    K-r columns used to start at exactly 0; at B_jk = 0 the inverse-
    Gaussian parameterization of the conditional
    pi(tau_jk^2, v_jk | lambda_k, B_jk) degenerates (the mean parameters
    of both InvGauss draws blow up, although the conditional itself has
    the proper Gamma limits tau_jk^2 | v_jk ~ Ga(1/2, 1/(2 v_jk^2)) and
    v_jk ~ Ga(1/2, 1/4)), and ``shrinkage`` then relied on a
    ``finfo.tiny`` floor.  A tiny deterministic alternating jitter
    (order eps * max|B0|) keeps the first sweep at a proper, generic point
    of the parameter space; it is far below the active loadings and is
    forgotten after a few sweeps.  No RNG key is consumed, so fixed-seed
    chains keep their key schedule.
    """
    B = jnp.zeros((p, K), dtype=_F32).at[:, :r].set(B0)
    if K > r:
        scale = jnp.maximum(jnp.max(jnp.abs(B0)), _F32(1.0))
        jitter = jnp.finfo(_F32).eps * scale
        cols = jnp.arange(K, dtype=jnp.int32)[None, :]
        rows = jnp.arange(p, dtype=jnp.int32)[:, None]
        signs = jnp.where((rows + cols) % 2 == 0, _F32(1.0), _F32(-1.0))
        B = B + jnp.where(cols < r, _F32(0.0), jitter * signs)
    return B


def _accept_chi2(
    p: int,
    a: float,
    weight1: jax.Array,
    a_sigma: float,
    n: int,
    tol: float = 1e-2,
    verbose: bool = False,
) -> bool:
    r"""Compute the chi-square rounding perturbation and decide on it.

    Same decision rule as the PyTorch reference.  The chi-square Gamma
    path (Chi2.py) rounds the shapes to the nearest half-integer
    (|delta| <= 0.25 on each shape).  The two Gamma steps of a sweep
    then suffer relative shape perturbations

        lambda_k: alpha_k = 2p + a + k^{c_1},
                  rel_k = |round(2*alpha_k)/2 - alpha_k| / alpha_k,
        sigma2  : alpha_s = a_sigma + n/2,
                  rel_s = |round(2*alpha_s)/2 - alpha_s| / alpha_s.

    Returns True  -- max(rel_lam, rel_s2) <= tol: the approximation is
    accurate enough (negligible for large n/p; integer shapes give
    rel = 0) and is used automatically on the GPU backend;
    Returns False -- otherwise: refused; a one-line note is printed
    when verbose=True and exact Gamma is used everywhere.
    """
    alpha_lam = lambda_concentration(p, a, weight1)
    delta = jnp.abs(jnp.round(2 * alpha_lam) / 2 - alpha_lam)
    rel_lam = float(jnp.max(delta / alpha_lam))
    alpha_s2 = sigma2_concentration(a_sigma, n)
    rel_s2 = abs(round(2 * alpha_s2) / 2 - alpha_s2) / alpha_s2
    rel = max(rel_lam, rel_s2)
    if rel > tol:
        if verbose:
            print(f"GS_LH: chi-square approximation refused (shape "
                  f"perturbation {rel:.1e} > {tol:g}); using exact Gamma",
                  flush=True)
        return False
    return True


def _sweep(state, key, X, a, weight1, weight2, a_sigma, b_sigma,
           chi2_plan_lam, chi2_plan_s2):
    """One full Gibbs sweep: eta -> shrinkage -> sigma2 -> B.

    Scalar hyperparameters are traced, so changing them does not
    trigger recompilation.  This stays UN-jitted: it is compiled inline
    into the scan runners below.
    """
    B, sigma2 = state
    k_eta, k_sh, k_s2, k_B = jax.random.split(key, 4)
    # 1. latent factors
    eta = eta_sample(k_eta, X, B, sigma2)
    # 2. shrinkage parameters (lambda, tau), returned as prior stds D
    D = shrinkage(k_sh, B, a, weight1, weight2, chi2_plan=chi2_plan_lam)
    # 3. idiosyncratic variances
    sigma2 = sigma2_sample(k_s2, X, B, eta, a_sigma, b_sigma,
                           chi2_plan=chi2_plan_s2)
    # 4. loading matrix
    B = B_sample(k_B, X, D, eta, sigma2)
    return (B, sigma2), (B, sigma2)


def _run_burn_impl(state, keys, X, a, weight1, weight2, a_sigma, b_sigma,
                   chi2_plan_lam, chi2_plan_s2):
    """Advance the chain over a chunk of burn-in keys, collecting
    nothing.  Returns only the final state."""
    def f(state, k):
        state, _ = _sweep(state, k, X, a, weight1, weight2, a_sigma,
                          b_sigma, chi2_plan_lam, chi2_plan_s2)
        return state, None

    state, _ = jax.lax.scan(f, state, keys)
    return state


def _run_keep_impl(state, keys, X, a, weight1, weight2, a_sigma, b_sigma,
                   chi2_plan_lam, chi2_plan_s2):
    """Advance the chain over a chunk of keys, collecting (B, sigma2)
    of every sweep.  Returns (final state, (B_chunk, sigma2_chunk))."""
    def f(state, k):
        state, draws = _sweep(state, k, X, a, weight1, weight2,
                              a_sigma, b_sigma, chi2_plan_lam, chi2_plan_s2)
        return state, draws

    return jax.lax.scan(f, state, keys)


def _run_keep_thin_impl(state, keys, X, a, weight1, weight2, a_sigma,
                        b_sigma, chi2_plan_lam, chi2_plan_s2):
    """Advance the chain over a chunk of (n_blocks, thin) sweep-key
    blocks, collecting (B, sigma2) only at the LAST sweep of each block
    (the documented keep-last thinning convention: retained sweeps are
    thin-1, 2*thin-1, ... in 0-based kept-phase indices, so the first
    retained sweep is burn_in + thin).  Returns
    (final state, (B_blocks, sigma2_blocks)): the non-kept sweeps never
    leave the device, so a run with thin > 1 no longer pays the
    device-to-host transfer and the host copy of every sweep."""
    def one_block(state, kblock):
        def f(state, k):
            state, _ = _sweep(state, k, X, a, weight1, weight2, a_sigma,
                              b_sigma, chi2_plan_lam, chi2_plan_s2)
            return state, None

        state, _ = jax.lax.scan(f, state, kblock)
        return state, state

    return jax.lax.scan(one_block, state, keys)


# XLA:GPU command-buffer (CUDA graph) capture of the whole scan While
# loop: with CONDITIONAL+WHILE enabled, the entire chunk executable is
# recorded as ONE CUDA graph on first execution and replayed afterwards,
# removing all per-kernel launch overhead between sweeps.  Every op in
# the sweep body (GPU RNG, cuBLAS/cuSOLVER custom calls, fusions) is
# capturable, and the draws are bitwise-reproducible for a fixed seed.
# Passed per-compile so graph mode is a per-call choice (no XLA_FLAGS
# needed before importing jax).
_GRAPH_COMPILER_OPTIONS = {
    "xla_gpu_enable_command_buffer":
        "FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE",
    "xla_gpu_graph_min_graph_size": 1,
}

_run_burn = jax.jit(_run_burn_impl)
_run_keep = jax.jit(_run_keep_impl)
_run_keep_thin = jax.jit(_run_keep_thin_impl)
_run_burn_graph = jax.jit(_run_burn_impl,
                          compiler_options=_GRAPH_COMPILER_OPTIONS)
_run_keep_graph = jax.jit(_run_keep_impl,
                          compiler_options=_GRAPH_COMPILER_OPTIONS)
_run_keep_thin_graph = jax.jit(_run_keep_thin_impl,
                               compiler_options=_GRAPH_COMPILER_OPTIONS)


def _chunk_sizes(total: int, p: int, K: int) -> list[int]:
    """Split `total` sweeps into chunk lengths bounded by _CHUNK and by
    the per-chunk device memory cap _CHUNK_BYTES (the collecting runner
    stores p*K + p floats per sweep).  This bounds only the COLLECTED
    draws; one sweep's workspace -- O(p*K^2) for the batched Cholesky
    of B_sample plus O((p+K)*n) data/noise intermediates -- is
    independent of the chunk length (see the GS_LH memory note)."""
    per_sweep = (p * K + p) * 4
    chunk = max(1, min(_CHUNK, _CHUNK_BYTES // max(per_sweep, 1)))
    sizes, left = [], total
    while left > 0:
        sizes.append(min(chunk, left))
        left -= sizes[-1]
    return sizes


def _audit_host(arrs: tuple[np.ndarray, ...], upto: int) -> None:
    """Finiteness audit of one host-side chunk (the JAX analog of the
    torch version's periodic device audit).

    All sampled systems are I + PSD by construction, so a non-finite
    state signals a numerical anomaly OUTSIDE the guarded range; fail
    loudly rather than silently contaminating the chain.
    """
    if not all(np.isfinite(a).all() for a in arrs):
        raise FloatingPointError(
            f"non-finite state (B or sigma2) detected in the chunk ending "
            f"at sweep {upto}; the chain is numerically contaminated and "
            "must be discarded. This should not happen (all systems are "
            "I + PSD by construction) -- please report the data and "
            "hyperparameters."
        )


def GS_LH(
    X: jax.Array,
    K: int,
    a: float = 10.0,
    c1: float = 2.0,
    c2: float = 1.0,
    a_sigma: float = 1.0,
    b_sigma: float = 1.0,
    n_iter: int = 2000,
    burn_in: int = 1000,
    thin: int = 1,
    seed: int | None = None,
    verbose: bool = True,
    graph: bool = False,
    chi2_max_normals: int = 8_000_000,
    chi2_chunk_elems: int = 1 << 22,
) -> dict:
    r"""Gibbs sampler for the L_{1/2}-shrunk Bayesian factor model (JAX).

    Runs n_iter sweeps on the truncated model with K columns.  Each
    sweep updates, in order (all exact full conditionals, cf. Section 3):

        1. eta    ~ pi(eta | B, sigma2, X)            [eta_sample]
        2. D      ~ pi(lambda, tau | B)               [shrinkage; D_jk =
                   sqrt(tau_jk^2)/lambda_k^2, regenerated from B every
                   sweep -- the lambda/v/tau chain is not sampler state]
        3. sigma2 ~ pi(sigma2 | B, eta, X)            [sigma2_sample]
        4. B      ~ pi(B | eta, sigma2, D, X)         [B_sample]

    Parameters
    ----------
    X : array-like, shape (n, p)
        Column-centered data matrix, coerced to float32.  Must be 2-D and
        finite; NaN/inf entries (or values that overflow the float32
        coercion) raise ValueError before the chain starts.
    K : int
        Truncation level of the loading matrix (number of columns);
        must be an integer >= 1 (integral floats are coerced).
    a, c1, c2 : float
        Hyperparameters of the L_{1/2} prior's lambda hyperprior
        Ga(a + k^{c_1}, k^{-c_2}); all must be > 0.  Defaults a = 10,
        c1 = 2.0, c2 = 1.0 satisfy the theory conditions a >= 4 and
        q = 4(c1 + c2) > 1; violating them triggers a UserWarning
        (the sampler on the K-truncated model remains valid, but the
        paper's asymptotic theory need not apply).
    a_sigma, b_sigma : float
        Inverse-gamma hyperparameters of sigma_j^2 ~ IG(a_sigma, b_sigma);
        both must be > 0.
    n_iter, burn_in, thin : int
        Total sweeps, discarded initial sweeps, and keep-every-thin.
        Thinning keeps the LAST sweep of each thin block: with 0-based
        kept-phase indices the retained draws are thin-1, 2*thin-1, ...,
        so the first retained sweep is burn_in + thin (not burn_in),
        matching the documented GS_LH convention; the sample count is
        (n_iter - burn_in) // thin and thin=1 keeps every sweep.  With
        thin > 1 the selection happens on the device (the thinned kept
        runner emits only each block's last sweep), so the non-kept
        sweeps are never transferred to the host; the chain itself
        (keys, sweep order, final state) is unchanged.
    seed : int or None
        Seed for jax.random.PRNGKey.  None draws a random seed.  The
        same seed (on the same backend) reproduces the same chain,
        including eta_last / D_last.
    verbose : bool
        Print progress about every 10% of the sweeps (reports are
        emitted at chunk boundaries, so small runs print less often).
    graph : bool
        If True and the backend is GPU, compile the scan runners with
        XLA command buffers (CUDA graphs): the whole chunk -- all its
        sweeps, RNG included -- is recorded as one CUDA graph on first
        execution and replayed afterwards, removing the per-kernel
        launch overhead of the scan loop (~2x faster in practice; the
        JAX analog of the PyTorch reference's graph=True).  The math is
        unchanged: the same kernels are replayed, so the graph runner is
        numerically identical to the plain runner (we verify bitwise
        equality on the same GPU backend); this is not a cross-backend
        bitwise guarantee.  Requires a GPU backend; on CPU/TPU a
        RuntimeWarning is issued and the run proceeds with the plain
        runners.
    chi2_max_normals : int, default 8_000_000
        Upper bound on the estimated number of normal draws of the GPU
        chi-square Gamma approximation; above it the whole run falls
        back to exact jax.random.gamma.  Ignored on non-GPU backends.
    chi2_chunk_elems : int, default 1 << 22
        Maximum element count of one chi-square sampling chunk (about
        16 MB in float32), bounding the transient device memory of the
        GPU approximation.

    Returns
    -------
    dict with keys
        "B"       : (S, p, K) kept loading matrices, numpy float32 on
                    the host, S = (n_iter - burn_in) // thin;
        "sigma2"  : (S, p) kept variance vectors, likewise;
        "eta_last": (K, n) latent factors drawn from the FINAL state
                    (one extra partial sweep; same marginal law as the
                    torch version's last-sweep eta), for diagnostics or
                    warm starts;
        "D_last"  : (p, K) shrinkage scales drawn from the final state;
        "seed"    : the seed actually used;
        "runtime_sec" : wall-clock seconds of the scan loop (excludes
                    the ahead-of-time JIT compilation);
        "gamma_mode" : "exact" or "chi2" (the Gamma-draw path used);
        "chi2_normals": int, estimated normal draws of the chi2 path
                    (0 when exact).

    Initialization: principal-factor initialization (see Initialization)
    with r = min(K, min(n, p) - 2) leading factors; remaining columns of
    B start at a tiny deterministic jitter (not exact zero, where the
    conditional's inverse-Gaussian parameterization degenerates; see
    _initial_loadings), and
    every sigma_j^2 starts at the noise-floor estimate.  Memory note: the
    B chain costs S*p*K*4 bytes in float32 on the host -- use thinning for
    very large p.  Device memory note:
    _chunk_sizes bounds only the per-chunk COLLECTED draws; one sweep's
    device workspace is O(p*K^2) (batched Cholesky of the p whitening
    systems in B_sample) plus O((p+K)*n) (data and noise intermediates),
    independent of the chunk length.  Very large p*K^2 can exceed device
    memory even with small chunks; reduce K in that regime.

    Chi-square approximation: on the GPU backend the driver applies the
    _accept_chi2 rule (same as the PyTorch reference): if the
    half-integer rounding of the Gamma shapes is negligible (max
    relative perturbation <= 1e-2, e.g. large n/p or integer shapes)
    AND the estimated normal-draw cost stays below chi2_max_normals,
    the lambda/sigma2 draws use the chi-square identity (see Chi2.py)
    in BOTH the plain and the graph runners; otherwise, and on non-GPU
    backends, exact Gamma is used.  The chi2 draws use a bucketed /
    chunked plan (make_chi2_plan / make_chi2_plan_scalar): at most
    2 * sum_k nu_k normals for the lambda step instead of K * nu_max,
    and transient memory bounded by chi2_chunk_elems.  An accepted
    chi-square run is still bitwise-reproducible from `seed` (the plan
    is fixed per run), but its chain differs bitwise from the exact-
    Gamma chain; gamma_mode / chi2_normals report the choice.

    Numerical safety: all sampled systems are I + PSD by construction,
    so the modules run plain Cholesky with no per-solve flag checks;
    instead, the chain is audited for finiteness on the host at every
    chunk boundary -- the carried state after each burn-in chunk, every
    collected draw of the kept phase (with thin > 1, the retained draw
    of each thin block; intermediate non-kept sweeps stay on the device
    and are not host-audited), and the final state after any trailing
    leftover sweeps (the JAX analog of the PyTorch reference's `audit_every`
    device audit).  A non-finite state raises FloatingPointError instead
    of silently contaminating the chain.
    """
    # Deterministic/scalar control arguments are validated before any chain
    # work.  bool is an int subclass in Python; reject it explicitly so that
    # e.g. max_iter=True cannot silently mean one sweep.
    if seed is not None and (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= int(seed) < 2**32
    ):
        raise ValueError("seed must be an integer in [0, 2**32) or None")
    for name, value in (("n_iter", n_iter), ("burn_in", burn_in), ("thin", thin)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    n_iter, burn_in, thin = int(n_iter), int(burn_in), int(thin)
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    else:
        seed = int(seed)
    key = jax.random.PRNGKey(seed)

    # single precision throughout: every module derives dtype from X.
    # Validate the data before any chain work: without this check a NaN/inf
    # entry is only found by the first host audit (up to 500 sweeps later)
    # and is then reported as an internal "please report this bug" anomaly.
    try:
        with warnings.catch_warnings():
            # finite float64 values can overflow the float32 cast; the
            # explicit isfinite check below reports that case.
            warnings.simplefilter("ignore", RuntimeWarning)
            X = jnp.asarray(X, dtype=_F32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"X must be a numeric 2-dimensional array, got "
            f"{type(X).__name__}: {exc}"
        ) from exc
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {tuple(X.shape)}")
    if not bool(jnp.all(jnp.isfinite(X))):
        raise ValueError(
            "X must contain only finite values (found NaN or inf, or values "
            "that overflow the float32 coercion)"
        )

    # ---- hyperparameter validation ----------------------------------------
    # integral floats (K = 10.0) are coerced; a fractional K would be
    # silently truncated by the jnp.arange below (K = 2.5 -> 3 columns).
    # int(K) is wrapped because int(inf)/int(None) raise OverflowError/
    # TypeError instead of a user-facing ValueError.
    try:
        K_int = int(K)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"K must be an integer >= 1, got {K!r}") from exc
    if isinstance(K, (bool, np.bool_)) or K != K_int or K_int < 1:
        raise ValueError(f"K must be an integer >= 1, got {K!r}")
    K = K_int
    for name, val in (("a", a), ("c1", c1), ("c2", c2),
                      ("a_sigma", a_sigma), ("b_sigma", b_sigma)):
        # Non-numeric values (str, None, ...) raise a bare TypeError on
        # the comparison; report the documented ValueError instead.
        try:
            valid = bool(val > 0)
        except (TypeError, ValueError):
            valid = False
        if not valid:  # also rejects NaN
            raise ValueError(f"{name} must be > 0, got {name} = {val}")
    if a < 4 or 4.0 * (c1 + c2) <= 1.0:
        warnings.warn(
            f"hyperparameters (a={a}, c1={c1}, c2={c2}) violate the prior-"
            "regularity conditions of the paper's Lemmas 1-2 (a >= 4 and "
            "q = 4*(c1 + c2) > 1): the asymptotic theory need not apply, "
            "although the sampler on the K-truncated model remains valid",
            UserWarning,
        )

    if not 0 <= burn_in < n_iter:
        raise ValueError(f"need 0 <= burn_in < n_iter, got {burn_in}/{n_iter}")
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")
    if (n_iter - burn_in) // thin == 0:
        raise ValueError(
            f"no draws would be kept: (n_iter - burn_in) // thin = "
            f"({n_iter} - {burn_in}) // {thin} = 0; reduce thin or burn_in"
        )

    n, p = X.shape
    if min(n, p) < 3:
        raise ValueError(
            f"need min(n, p) >= 3 for initialization, got (n, p) = ({n}, {p})"
        )

    # column weights (k^{c_1}), (k^{c_2}), k = 1..K
    k = jnp.arange(1, K + 1, dtype=_F32)
    weight1, weight2 = k ** c1, k ** c2

    # ---- initialize ---------------------------------------------------------
    r = min(K, min(n, p) - 2)
    B0, s2_0 = Initialization(X, r)
    B = _initial_loadings(B0, p, K, r)
    sigma2 = jnp.full((p,), s2_0, dtype=_F32)

    n_keep = (n_iter - burn_in) // thin

    # ---- chi-square decision (GPU backend only) --------------------------
    # Accepted -> the lambda/sigma2 Gamma draws use the chi-square
    # identity with half-integer-rounded dof in BOTH the plain and the
    # graph runners (see Chi2.py / _accept_chi2); refused, over the cost
    # gate, or non-GPU -> exact Gamma everywhere.  The plan (bucketed
    # lambda dof + row-chunked sigma2 dof) is built ONCE on host here;
    # no dense (K, nu_max) mask is materialized.
    if not isinstance(chi2_max_normals, (int, np.integer)) or int(chi2_max_normals) < 1:
        raise ValueError("chi2_max_normals must be a positive integer")
    if not isinstance(chi2_chunk_elems, (int, np.integer)) or int(chi2_chunk_elems) < 1:
        raise ValueError("chi2_chunk_elems must be a positive integer")
    chi2_max_normals = int(chi2_max_normals)
    chi2_chunk_elems = int(chi2_chunk_elems)
    chi2_ok = jax.default_backend() == "gpu" and _accept_chi2(
        p, a, weight1, a_sigma, n, verbose=verbose)
    chi2_plan_lam = chi2_plan_s2 = None
    chi2_normals = 0
    if chi2_ok:
        chi2_plan_lam = make_chi2_plan(
            lambda_concentration(p, a, weight1), max_elems=chi2_chunk_elems)
        chi2_plan_s2 = make_chi2_plan_scalar(
            sigma2_concentration(a_sigma, n), p, max_elems=chi2_chunk_elems)
        chi2_normals = (chi2_plan_cost(chi2_plan_lam)
                        + chi2_plan_cost(chi2_plan_s2))
        if chi2_normals > chi2_max_normals:
            chi2_ok = False
            chi2_plan_lam = chi2_plan_s2 = None
            if verbose:
                print(f"GS_LH: chi2 approximation refused (estimated "
                      f"{chi2_normals} normals > {chi2_max_normals}); "
                      "using exact Gamma", flush=True)
            chi2_normals = 0
    args = (X, a, weight1, weight2, a_sigma, b_sigma,
            chi2_plan_lam, chi2_plan_s2)

    # one key per sweep, split up front: same seed -> same chain.  An
    # (n_iter + 1)-th child is reserved for the eta_last / D_last partial
    # sweep at the end of the run; children of one split are distinct by
    # construction, so it never collides with an in-chain sweep key (in
    # current jax, split(key, n)[i] == fold_in(key, i), so the first
    # n_iter children -- hence every fixed-seed chain -- are unchanged).
    keys_all = jax.random.split(key, n_iter + 1)
    keys, key_extra = keys_all[:n_iter], keys_all[n_iter]

    # ---- runner selection (graph mode = XLA command buffers on GPU) ---------
    use_graph = graph and jax.default_backend() == "gpu"
    if graph and not use_graph:
        warnings.warn(
            f"graph=True requires a GPU backend (got "
            f"'{jax.default_backend()}'); running with the plain runners",
            RuntimeWarning,
        )
    run_burn = _run_burn_graph if use_graph else _run_burn
    run_keep = _run_keep_graph if use_graph else _run_keep
    run_keep_thin = _run_keep_thin_graph if use_graph else _run_keep_thin

    # ---- ahead-of-time compilation (not timed) ------------------------------
    # One compilation per DISTINCT chunk length per runner (only shapes
    # are traced, so which keys slice feeds lower() is irrelevant: a valid
    # slice of the required length is enough, no running offset is tracked.
    burn_sizes = _chunk_sizes(burn_in, p, K)
    for c in set(burn_sizes):
        run_burn.lower((B, sigma2), keys[:c], *args).compile()
    if thin == 1:
        keep_sizes = _chunk_sizes(n_iter - burn_in, p, K)
        for c in set(keep_sizes):
            run_keep.lower((B, sigma2), keys[burn_in:burn_in + c],
                           *args).compile()
    else:
        # device-side thinning: chunk lengths count KEPT draws (blocks of
        # `thin` sweeps each); sweeps that do not complete a trailing
        # block advance the chain burn-style afterwards, so the final
        # state is still sweep n_iter.
        n_leftover = (n_iter - burn_in) - n_keep * thin
        keep_sizes = _chunk_sizes(n_keep, p, K)
        for c in set(keep_sizes):
            kslice = keys[burn_in:burn_in + c * thin].reshape(c, thin, 2)
            run_keep_thin.lower((B, sigma2), kslice, *args).compile()
        if n_leftover:
            run_burn.lower((B, sigma2), keys[n_iter - n_leftover:],
                           *args).compile()

    # ---- run ------------------------------------------------------------------
    state = (B, sigma2)
    B_keep = np.empty((n_keep, p, K), dtype=np.float32)   # stored on host
    s2_keep = np.empty((n_keep, p), dtype=np.float32)

    t_start = time.perf_counter()
    report = max(1, n_iter // 10)
    last_report = 0

    # burn-in: advance without collecting (audit the carried state per chunk)
    start = 0
    for c in burn_sizes:
        state = run_burn(state, keys[start:start + c], *args)
        start += c
        _audit_host((np.asarray(state[0]), np.asarray(state[1])), start)
        if verbose and (start - last_report >= report or start == n_iter):
            print(f"GS_LH: sweep {start}/{n_iter}", flush=True)
            last_report = start

    kept = 0
    if thin == 1:
        # kept phase, thin == 1: collect every sweep per chunk
        for c in keep_sizes:
            state, (Bc, s2c) = run_keep(state, keys[start:start + c], *args)
            Bc, s2c = np.asarray(Bc), np.asarray(s2c)
            _audit_host((Bc, s2c), start + c)
            B_keep[kept:kept + c] = Bc
            s2_keep[kept:kept + c] = s2c
            kept += c
            start += c
            if verbose and (start - last_report >= report or start == n_iter):
                print(f"GS_LH: sweep {start}/{n_iter}", flush=True)
                last_report = start
    else:
        # kept phase, thin > 1: the thinned runner emits only the LAST
        # sweep of each thin block (the documented keep-last convention),
        # so the non-kept sweeps are never transferred to the host.  Only
        # the kept draws are host-audited in this phase; the carried
        # state is still audited at every burn-in chunk and after the
        # trailing leftover sweeps below.
        for c in keep_sizes:
            kslice = keys[start:start + c * thin].reshape(c, thin, 2)
            state, (Bc, s2c) = run_keep_thin(state, kslice, *args)
            Bc, s2c = np.asarray(Bc), np.asarray(s2c)
            _audit_host((Bc, s2c), start + c * thin)
            B_keep[kept:kept + c] = Bc
            s2_keep[kept:kept + c] = s2c
            kept += c
            start += c * thin
            if verbose and (start - last_report >= report or start == n_iter):
                print(f"GS_LH: sweep {start}/{n_iter}", flush=True)
                last_report = start
        # trailing sweeps that do not complete a thin block: advance the
        # chain to sweep n_iter (collecting nothing) so eta_last / D_last
        # are drawn from the true final state.
        if n_leftover:
            state = run_burn(state, keys[start:start + n_leftover], *args)
            start += n_leftover
            _audit_host((np.asarray(state[0]), np.asarray(state[1])), start)
            if verbose:
                print(f"GS_LH: sweep {start}/{n_iter}", flush=True)
                last_report = start

    runtime = time.perf_counter() - t_start
    if kept != n_keep:
        raise RuntimeError(
            f"kept {kept} draws but expected {n_keep}: the thinning "
            "bookkeeping is inconsistent -- please report this bug"
        )

    # eta_last / D_last: one extra partial sweep from the final state
    # (steps 1-2 only; same marginal law as the last in-chain draws).
    # key_extra is the reserved (n_iter + 1)-th child of the up-front
    # split (see above), so these draws are independent of every sweep
    # key and the whole run stays reproducible from `seed` alone.
    k_eta, k_sh = jax.random.split(key_extra)
    eta_last = eta_sample(k_eta, X, state[0], state[1])
    # pass the run's chi2 plan so D_last uses the same lambda-draw rule
    # (exact Gamma vs chi-square) as the in-chain D draws
    D_last = shrinkage(k_sh, state[0], a, weight1, weight2,
                       chi2_plan=chi2_plan_lam)

    return {
        "B": B_keep,
        "sigma2": s2_keep,
        "eta_last": np.asarray(eta_last),
        "D_last": np.asarray(D_last),
        "seed": seed,
        "runtime_sec": runtime,
        "gamma_mode": "chi2" if chi2_ok else "exact",
        "chi2_normals": int(chi2_normals),
    }
