"""Spectral loading and noise seeds shared by finite-factor estimators."""

from numbers import Integral

import jax.numpy as jnp
import numpy as np


_NOISE_FLOOR_FRACTION = 0.05
_TAIL_ENERGY_FRACTION = 0.01


def _positive_floating(values, name, dtype):
    """Require finite, normal positive scales in inference precision."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.asarray(values, dtype=dtype)
    if not np.isfinite(result).all() or np.any(result < np.finfo(dtype).tiny):
        raise ValueError(
            f"spectral initialization requires {name} to be finite and "
            f"positive normal {dtype.name} values; rescale the data or "
            "hyperparameters"
        )
    return result


def spectral_seeds(X, K, *, seed=2026, dtype, varimax=False):
    """Build sorted loading means and per-variable seeds from prepared data.

    X has shape (n, p), with centering controlled by the caller. The caller
    supplies the resolved inference dtype and enables JAX x64 for float64
    varimax. The host SVD and seed statistics use float64 after coercing the
    data to the inference dtype. Only extra columns use the local RNG.

    Returns (mu, noise_seed, variance_seed, precision_seed, info). The first
    four values are NumPy arrays in the inference dtype: mu has shape (p, K)
    and the other arrays have shape (p,). The latter contain noise, noise / n
    and inverse noise, each rounded separately from the host seed. The info
    dictionary records the method, numerical rank and column ordering.
    """
    dtype = np.dtype(dtype)
    info = np.finfo(dtype)
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, Integral) or K < 1:
        raise ValueError(f"K must be a positive integer, got {K!r}")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, Integral) or seed < 0):
        raise ValueError(f"seed must be a nonnegative integer, got {seed!r}")
    if not isinstance(varimax, (bool, np.bool_)):
        raise ValueError(f"varimax must be a boolean, got {varimax!r}")
    K, seed = int(K), int(seed)
    try:
        X64 = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("X must be a finite two-dimensional numeric array") from exc
    if X64.ndim != 2 or X64.shape[0] < 2 or X64.shape[1] < 1:
        raise ValueError(
            "spectral initialization requires X with shape (n, p), "
            f"n >= 2 and p >= 1; got {X64.shape}"
        )
    if not np.isfinite(X64).all():
        raise ValueError("X must contain only finite values")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        X_prepared = X64.astype(dtype)
    if not np.isfinite(X_prepared).all():
        raise ValueError(f"X overflows {dtype.name}; rescale the data")
    # Match inference precision, without rounding float64 calls through float32.
    X64 = X_prepared.astype(np.float64)
    n, p = X64.shape
    constant_columns = np.flatnonzero(np.ptp(X64, axis=0) == 0.0)
    if constant_columns.size:
        preview = constant_columns[:10].tolist()
        suffix = " ..." if constant_columns.size > 10 else ""
        raise ValueError(
            "spectral initialization requires nonzero-variance variables; "
            f"constant column indices: {preview}{suffix}. Remove or handle "
            "these variables before fitting."
        )
    # Thin SVD avoids constructing a p x p sample covariance matrix.
    _, singular_values, right_vectors = np.linalg.svd(X64, full_matrices=False)
    rank_tolerance = max(n, p) * info.eps * singular_values[0]
    numerical_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    r = min(K, numerical_rank)
    if r < 1:
        raise ValueError(
            f"X has no resolved nonzero singular direction in {dtype.name}; "
            "rescale the data or reduce its dimensions"
        )
    mu64 = np.zeros((p, K), dtype=np.float64)
    mu64[:, :r] = right_vectors[:r].T * (singular_values[:r] / np.sqrt(n))

    # Fix these seeds before perturbing means: there is no circular
    # dependence between the tail perturbations and the residual noise.
    second_moment_x = np.einsum("ij,ij->j", X64, X64) / n
    explained = np.einsum("jk,jk->j", mu64, mu64)
    noise64 = np.maximum(
        second_moment_x - explained,
        _NOISE_FLOOR_FRACTION * second_moment_x,
    )
    noise_seed = _positive_floating(noise64, "noise seeds", dtype)
    variance_seed = _positive_floating(noise64 / n, "loading covariance diagonals", dtype)
    precision_seed = _positive_floating(1.0 / noise64, "noise seed precisions", dtype)

    varimax_n_iter = 0
    if varimax and r > 1:
        from ._rsp.kernels.varimax import varimax_batch

        spectral_loadings = jnp.asarray(mu64[:, :r], dtype=dtype)
        rotated, _, iterations = varimax_batch(
            spectral_loadings[None, :, :], polar="svd", return_n_iter=True,
        )
        mu64[:, :r] = np.asarray(rotated[0])
        varimax_n_iter = int(iterations[0])

    tail_columns = K - r
    if tail_columns:
        rng = np.random.default_rng(seed)
        tail_scale = np.sqrt(
            noise64 * min(1.0 / n, _TAIL_ENERGY_FRACTION / tail_columns)
        )
        mu64[:, r:] = tail_scale[:, None] * rng.standard_normal((p, tail_columns))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mu_seed = mu64.astype(dtype)
    if not np.isfinite(mu_seed).all():
        raise ValueError(f"spectral loading means overflow {dtype.name}; rescale the data")
    if np.any(np.all(mu_seed == 0.0, axis=0)):
        raise ValueError(
            "a spectral initial loading column underflowed to zero in "
            f"{dtype.name}; rescale the data"
        )

    # Use the actual stored means/covariance diagonals for stable sorting.
    E2 = mu_seed.astype(np.float64) ** 2 + variance_seed.astype(np.float64)[:, None]
    column_sums = np.sqrt(np.sqrt(E2)).sum(axis=0)
    permutation = np.argsort(-column_sums, kind="stable")
    diagnostics = {
        "method": "spectral_varimax" if varimax else "spectral",
        "precision": dtype.name,
        "seed": seed,
        "numerical_rank": numerical_rank,
        "initial_rank": r,
        "tail_columns": tail_columns,
        "rank_tolerance": float(rank_tolerance),
        "rank_precision": dtype.name,
        "svd_precision": "float64",
        "noise_floor_fraction": _NOISE_FLOOR_FRACTION,
        "tail_energy_fraction": _TAIL_ENERGY_FRACTION,
        "covariance_scale": "noise_seed / n",
        "noise_seed_min": float(noise_seed.min()),
        "noise_seed_max": float(noise_seed.max()),
        "column_permutation": permutation.tolist(),
    }
    if varimax:
        diagnostics["varimax_n_iter"] = varimax_n_iter
    return mu_seed[:, permutation], noise_seed, variance_seed, precision_seed, diagnostics
