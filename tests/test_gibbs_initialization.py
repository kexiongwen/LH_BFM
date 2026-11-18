"""Check the state and random keys consumed by the first actual Gibbs sweep."""

import importlib
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bfm import GibbsBFM
from bfm._gs import GS_LH, Initialization
from bfm._vi.shrinkage import column_weights
from bfm._vi.spectral_initialization import spectral_initialization


SHORT_CHAIN = dict(n_iter=2, burn_in=0, verbose=False)


@pytest.fixture
def first_chunk(monkeypatch):
    driver = importlib.import_module("bfm._gs.GS_LH")
    runner = Mock(wraps=driver._run_keep)
    monkeypatch.setattr(driver, "_run_keep", runner)
    return runner


def test_spectral_varimax_chain_starts_from_sorted_loadings_and_residual_noise(first_chunk):
    X = np.random.default_rng(7).normal(size=(12, 6)) * np.arange(1, 7) + 1000
    K, seed = 3, 41
    fit = GibbsBFM(K, init_method="spectral_varimax", seed=seed, **SHORT_CHAIN).fit(X)
    (B, sigma2), keys, prepared, *_ = first_chunk.call_args.args
    Xc = (X - X.mean(axis=0)).astype(np.float32)
    np.testing.assert_array_equal(prepared, Xc)

    shape, rate = column_weights(X.shape[1], K, fit.a, fit.c1, fit.c2)
    vi = spectral_initialization(Xc, K, 1.0, 1.0, shape, rate, varimax=True)
    np.testing.assert_array_equal(B, vi["mu"])
    # Gibbs starts with residual noise, before CAVI's inverse-gamma update.
    X64 = Xc.astype(np.float64)
    _, s, vt = np.linalg.svd(X64, full_matrices=False)
    second = np.mean(X64 ** 2, axis=0)
    signal = vt[:K].T * (s[:K] / np.sqrt(X.shape[0]))
    noise = np.maximum(second - np.sum(signal ** 2, axis=1), 0.05 * second)
    np.testing.assert_allclose(sigma2, noise, rtol=2e-6)
    assert B.dtype == sigma2.dtype == np.float32
    expected_keys = jax.random.split(jax.random.PRNGKey(seed), SHORT_CHAIN["n_iter"] + 1)
    np.testing.assert_array_equal(keys, expected_keys[:-1])
    assert fit.initialization_method_ == fit.initialization_info_["method"] == "spectral_varimax"
    assert fit.initialization_info_["column_permutation"] == vi["initialization_info"]["column_permutation"]
    assert fit.initialization_info_["varimax_n_iter"] > 0

    # With no surplus columns, changing only the initializer seed has no effect.
    other = GS_LH(Xc, K, seed=seed, init_method="spectral_varimax", init_seed=99, **SHORT_CHAIN)
    np.testing.assert_array_equal(other["B"], fit.B_samples_)
    np.testing.assert_array_equal(other["sigma2"], fit.sigma2_samples_)


def test_rank_one_tail_depends_only_on_init_seed(first_chunk):
    X = np.arange(-4, 5, dtype=np.float32)[:, None] * np.array([1, 4, -2], np.float32)

    def run(seed, init_seed):
        out = GS_LH(X, 3, seed=seed, init_method="spectral_varimax", init_seed=init_seed,
                    **SHORT_CHAIN)
        (B, sigma2), keys, *_ = first_chunk.call_args.args
        info = out["initialization_info"]
        assert info["initial_rank"] == 1 and info["tail_columns"] == 2
        assert info["varimax_n_iter"] == 0
        return np.asarray(B)[:, np.argsort(info["column_permutation"])], sigma2, keys, out["B"]

    B, sigma2, keys, draws = run(11, 17)
    B_tail, sigma2_tail, same_keys, _ = run(11, 19)
    np.testing.assert_array_equal(B[:, 0], B_tail[:, 0])
    assert not np.array_equal(B[:, 1:], B_tail[:, 1:])
    np.testing.assert_array_equal(sigma2, sigma2_tail)
    np.testing.assert_array_equal(keys, same_keys)

    B_again, sigma2_again, new_keys, new_draws = run(7, 17)
    np.testing.assert_array_equal(B, B_again)
    np.testing.assert_array_equal(sigma2, sigma2_again)
    assert not np.array_equal(keys, new_keys)
    assert not np.array_equal(draws, new_draws)


def test_default_chain_preserves_principal_factor_start_and_key_schedule(first_chunk):
    X = np.random.default_rng(65).normal(size=(10, 5)).astype(np.float32)
    K, seed = 4, 29
    default = GS_LH(X, K, seed=seed, **SHORT_CHAIN)
    (B, sigma2), keys, *_ = first_chunk.call_args.args
    r = min(X.shape) - 2
    signal, noise = Initialization(X, r)
    np.testing.assert_array_equal(B[:, :r], signal)
    jitter = jnp.finfo(jnp.float32).eps * jnp.maximum(jnp.abs(signal).max(), 1)
    signs = np.where((np.arange(X.shape[1]) + r) % 2 == 0, 1, -1)
    np.testing.assert_array_equal(B[:, r], jitter * signs)
    np.testing.assert_array_equal(sigma2, np.full(X.shape[1], noise, dtype=np.float32))
    np.testing.assert_array_equal(keys, jax.random.split(jax.random.PRNGKey(seed), 3)[:2])
    assert default["initialization_method"] == "principal_factor"
    explicit = GS_LH(X, K, seed=seed, init_method="principal_factor", init_seed=99, **SHORT_CHAIN)
    for key in ("B", "sigma2", "eta_last", "D_last"):
        np.testing.assert_array_equal(default[key], explicit[key])
