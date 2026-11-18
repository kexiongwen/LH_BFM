"""End-to-end precision contracts for opt-in double-precision CAVI.

These small tests check preserved data and actual recurrence precision, not
just the dtype of an exported estimate or a process-wide configuration flag.
"""

import importlib

import jax
import numpy as np
import pytest
from scipy.special import digamma, gammaln

from bfm import CaviBFM, GibbsBFM
from bfm._rsp.kernels.varimax import varimax_batch
from bfm._vi import CAVI
from bfm._vi.spectral_initialization import spectral_initialization


PRIOR = dict(a=10.123456789123, c1=1.312345678912, c2=0.712345678912,
             a_sigma=2.123456789123, b_sigma=0.812345678912)
ARRAYS = ("mu", "V", "W", "Phi", "beta", "rho", "m", "c")


@pytest.fixture(autouse=True)
def preserve_process_precision():
    previous = bool(jax.config.jax_enable_x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def data():
    return np.random.default_rng(73).normal(size=(12, 6)) * np.linspace(0.7, 1.7, 6)


def independent_elbo(X, state):
    """Uncollapsed likelihood/KL/entropy evaluation in NumPy float64.

    The residual uses observations directly, and the auxiliary normalizer
    is assembled from the prior rather than production ELBO byproducts.
    """
    mu, V, W, Phi, beta = [np.asarray(state[key], dtype=np.float64)
                           for key in ("mu", "V", "W", "Phi", "beta")]
    n, p = X.shape
    K = mu.shape[1]
    alpha = PRIOR["a_sigma"] + n / 2
    residual = np.zeros(p)
    for j in range(p):
        for i in range(n):
            residual[j] += ((X[i, j] - W[i] @ mu[j]) ** 2
                            + mu[j] @ Phi @ mu[j]
                            + W[i] @ V[j] @ W[i]
                            + np.trace(V[j] @ Phi))
    likelihood = (-0.5 * n * p * np.log(2 * np.pi)
                  - 0.5 * n * np.sum(np.log(beta) - digamma(alpha))
                  - 0.5 * np.sum(alpha * residual / beta))
    a, b = PRIOR["a_sigma"], PRIOR["b_sigma"]
    noise_kl = np.sum(a * np.log(beta / b) + gammaln(a) - gammaln(alpha)
                      + (alpha - a) * digamma(alpha) + alpha * (b - beta) / beta)
    score_kl = 0.5 * (np.sum(W ** 2) + n * np.trace(Phi) - n * K
                      - n * np.linalg.slogdet(Phi)[1])
    k = np.arange(1, K + 1, dtype=np.float64)
    prior_shape = PRIOR["a"] + k ** PRIOR["c1"]
    prior_rate = k ** (-PRIOR["c2"])
    second = mu ** 2 + np.diagonal(V, axis1=1, axis2=2)
    mass = np.sum(second ** 0.25, axis=0) + prior_rate
    auxiliary = np.sum(gammaln(2 * p + prior_shape) - gammaln(prior_shape)
                       + prior_shape * np.log(prior_rate) - p * np.log(4)
                       - (2 * p + prior_shape) * np.log(mass))
    entropy = 0.5 * (p * K * (1 + np.log(2 * np.pi))
                     + np.linalg.slogdet(V)[1].sum())
    return likelihood - noise_kl - score_kl + auxiliary + entropy


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("method", ["spectral", "spectral_varimax"])
def test_float64_recurrence_and_independent_elbo(jit, method):
    # A float64 request must work even if the caller has not enabled x64.
    jax.config.update("jax_enable_x64", False)
    X = data()
    state = CAVI(X, K=3, center=False, init_method=method, precision="float64",
                 max_iter=4, tol=1e-12, jit=jit, chunk_size=2, **PRIOR)
    assert jax.config.jax_enable_x64
    assert state["precision"] == state["dtype"] == "float64"
    for key in ARRAYS:
        array = np.asarray(state[key])
        assert array.dtype == np.float64, key
        assert np.isfinite(array).all(), key
    assert np.linalg.eigvalsh(state["V"]).min() > 0
    assert np.linalg.eigvalsh(state["Phi"]).min() > 0
    np.testing.assert_allclose(state["elbo"][-1], independent_elbo(X, state),
                               rtol=1e-12, atol=2e-9)
    assert np.min(np.diff(state["elbo"])) >= -1e-9


@pytest.mark.parametrize("center", [False, True])
def test_wrapper_preserves_unrounded_float64_data(center, monkeypatch):
    # Each uncentered column collapses to a constant if it passes through
    # float32. The values and their centered differences remain valid in f64.
    X = 1.0 + 2e-9 * np.random.default_rng(42).normal(size=(12, 6))
    assert np.all(np.ptp(X.astype(np.float32), axis=0) == 0)
    expected = X - X.mean(axis=0) if center else X
    module = importlib.import_module("bfm._vi.VI")
    original = module.spectral_initialization
    observed = []

    def inspect_prepared(prepared, *args, **kwargs):
        observed.append(np.asarray(prepared))
        return original(prepared, *args, **kwargs)

    monkeypatch.setattr(module, "spectral_initialization", inspect_prepared)
    model = CaviBFM(K=2, center=center, precision="float64",
                    init_method="spectral", max_iter=2, jit=False, **PRIOR).fit(X.tolist())
    assert model.precision_ == model.dtype_ == "float64"
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], expected)
    for key in ARRAYS:
        assert getattr(model, key + "_").dtype == np.float64


def test_double_precision_rank_resolves_small_spectral_direction():
    z = np.arange(-4.0, 5.0)
    w = z ** 2 - np.mean(z ** 2)
    X = z[:, None] * np.array([1.0, 2.0, -0.5, 3.0])
    X += 1e-8 * w[:, None] * np.array([2.0, -1.0, 1.0, 0.5])
    K = 2
    k = np.arange(1, K + 1, dtype=np.float64)
    shape = 2 * X.shape[1] + PRIOR["a"] + k ** PRIOR["c1"]
    rate = k ** (-PRIOR["c2"])
    single = spectral_initialization(X, K, PRIOR["a_sigma"], PRIOR["b_sigma"],
                                     shape, rate, precision="float32")
    double = spectral_initialization(X, K, PRIOR["a_sigma"], PRIOR["b_sigma"],
                                     shape, rate, precision="float64")
    assert single["initialization_info"]["initial_rank"] == 1
    assert single["initialization_info"]["tail_columns"] == 1
    assert double["initialization_info"]["initial_rank"] == 2
    assert double["initialization_info"]["tail_columns"] == 0
    assert double["initialization_info"]["rank_precision"] == "float64"
    for key in (*ARRAYS, "S", "elbo_var", "alpha_n"):
        assert np.asarray(double[key]).dtype == np.float64, key


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_varimax_rotates_only_the_signal_then_sorts_and_refreshes_scores(precision):
    # Unequal row scales distinguish raw varimax from Kaiser normalization.
    # The three signal columns change their prior order after rotation.
    rng = np.random.default_rng(7)
    n, p, rank, K = 13, 8, 3, 5
    X = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, p))
    X *= np.linspace(0.4, 2.0, p)
    X = X.astype(precision)
    k = np.arange(1, K + 1)
    args = (X, K, PRIOR["a_sigma"], PRIOR["b_sigma"],
            2 * p + PRIOR["a"] + k ** PRIOR["c1"], k ** (-PRIOR["c2"]))
    plain = spectral_initialization(*args, precision=precision)
    rotated = spectral_initialization(*args, precision=precision, varimax=True)
    info = rotated["initialization_info"]
    assert info["initial_rank"] == rank and info["tail_columns"] == K - rank
    plain_order = plain["initialization_info"]["column_permutation"]
    raw = np.asarray(plain["mu"])[:, np.argsort(plain_order)]
    signal, _, n_iter = varimax_batch(raw[None, :, :rank], polar="svd", return_n_iter=True)
    assert int(n_iter[0]) == info["varimax_n_iter"] > 0
    assert not np.allclose(raw[:, :rank], signal[0])
    expected = np.column_stack((signal[0], raw[:, rank:]))

    np.testing.assert_array_equal(rotated["V"], plain["V"])
    diag = np.asarray(plain["V"], dtype=np.float64)[:, 0, 0]
    mass = np.sum((expected.astype(np.float64) ** 2 + diag[:, None]) ** 0.25, axis=0)
    order = np.argsort(-mass, kind="stable")
    assert not np.array_equal(order, plain_order)
    np.testing.assert_array_equal(info["column_permutation"], order)
    unsorted = np.asarray(rotated["mu"])[:, np.argsort(order)]
    np.testing.assert_array_equal(unsorted[:, rank:], raw[:, rank:])
    tol = 12 * np.finfo(precision).eps
    np.testing.assert_allclose(unsorted[:, :rank], signal[0], rtol=tol, atol=tol)

    # The initial score update uses 1 / noise_seed; the IG update follows it.
    mu = expected[:, order].astype(np.float64)
    inverse_noise = 1.0 / (n * diag)
    weighted_mu = inverse_noise[:, None] * mu
    expected_phi = np.linalg.inv((1 + p / n) * np.eye(K) + mu.T @ weighted_mu)
    expected_w = X @ weighted_mu @ expected_phi
    np.testing.assert_allclose(rotated["Phi"], expected_phi, rtol=tol, atol=tol)
    np.testing.assert_allclose(rotated["W"], expected_w, rtol=tol, atol=tol)
    np.testing.assert_allclose(rotated["S"], expected_w.T @ expected_w + n * expected_phi,
                               rtol=tol, atol=tol)


@pytest.mark.parametrize("method", ["spectral", "spectral_varimax"])
def test_default_float32_survives_global_x64_enablement(method):
    X = data()
    options = dict(K=3, center=False, init_method=method,
                   max_iter=3, jit=False, **PRIOR)
    jax.config.update("jax_enable_x64", False)
    original = CAVI(X, **options)
    jax.config.update("jax_enable_x64", True)
    enabled = CAVI(X, **options)
    assert enabled["precision"] == enabled["dtype"] == "float32"
    for key in ARRAYS:
        assert np.asarray(enabled[key]).dtype == np.float32, key
        np.testing.assert_array_equal(enabled[key], original[key])
    np.testing.assert_array_equal(enabled["elbo"], original["elbo"])


def test_float64_warm_start_keeps_supplied_moments_before_first_sweep(monkeypatch):
    X = data()
    rng = np.random.default_rng(89)
    state = dict(mu=rng.normal(size=(6, 3)), W=0.1 * rng.normal(size=(12, 3)),
                 V=np.broadcast_to(0.1 * np.eye(3), (6, 3, 3)).copy(),
                 Phi=0.3 * np.eye(3))
    driver = importlib.import_module("bfm._vi.VI")
    make_carry = driver._make_carry
    observed = []

    def inspect_state(initial, *args, **kwargs):
        observed.append(initial)
        return make_carry(initial, *args, **kwargs)

    monkeypatch.setattr(driver, "_make_carry", inspect_state)
    result = CAVI(X, K=3, center=False, initial_state=state,
                  init_method="spectral", precision="float64", max_iter=2,
                  jit=True, chunk_size=2, **PRIOR)
    assert result["initialization_method"] == "provided"
    for key, value in state.items():
        np.testing.assert_array_equal(observed[0][key], value)
        assert np.asarray(result[key]).dtype == np.float64
    np.testing.assert_allclose(result["elbo"][-1], independent_elbo(X, result),
                               rtol=1e-12, atol=2e-9)


def test_gibbs_retains_float32_after_shared_preprocessing_change():
    jax.config.update("jax_enable_x64", True)
    model = GibbsBFM(K=2, n_iter=3, burn_in=1, seed=7).fit(data())
    assert model.B_samples_.dtype == np.float32
    assert model.sigma2_samples_.dtype == np.float32


@pytest.mark.parametrize("precision", ["float16", "FLOAT64", None, True, 64])
def test_invalid_precision_is_rejected(precision):
    with pytest.raises(ValueError, match="precision"):
        CaviBFM(K=2, precision=precision, max_iter=1).fit(data())
    with pytest.raises(ValueError, match="precision"):
        CAVI(data(), K=2, precision=precision, max_iter=1)
