"""Numerical contracts for the spectral and spectral-varimax CAVI starts.

Small cases test covariance geometry and a separately assembled five-term
ELBO; the demo comparisons are experiments, not part of this test suite.
"""

import importlib
import inspect
import itertools

import numpy as np
import pytest
from scipy.special import digamma, gammaln

from bfm import CaviBFM
from bfm._vi import CAVI
from bfm._vi.elbo import elbo_const
from bfm._vi.shrinkage import column_weights
from bfm._vi.spectral_initialization import spectral_initialization


PRIOR = dict(a=10.0, c1=1.3, c2=0.7, a_sigma=2.0, b_sigma=0.8)
STATE_ARRAYS = ("mu", "V", "W", "Phi", "beta", "rho", "m", "c", "S")


def _data(n=12, p=6, *, center=True):
    rng = np.random.default_rng(51)
    X = rng.normal(size=(n, p)) @ np.diag(np.linspace(0.7, 1.7, p))
    if center:
        X -= X.mean(axis=0)
    else:
        X += np.linspace(2.0, 5.0, p)
    return X.astype(np.float32)


def _start(X, K, *, seed=2026, varimax=False):
    shape, rate = column_weights(X.shape[1], K,
                                 PRIOR["a"], PRIOR["c1"], PRIOR["c2"])
    return spectral_initialization(
        X, K, PRIOR["a_sigma"], PRIOR["b_sigma"], shape, rate,
        seed=seed, varimax=varimax,
    )


def _five_term_elbo(X, state):
    """Evaluate expected likelihood, KLs and entropy in float64.

    Residual moments are summed observation by observation; neither the
    production residual helper nor its collapsed ELBO formula is used.
    The optimal auxiliary block uses the manuscript's integrated log Z.
    """
    X = np.asarray(X, dtype=np.float64)
    mu, V, W, Phi, beta = [
        np.asarray(state[name], dtype=np.float64)
        for name in ("mu", "V", "W", "Phi", "beta")
    ]
    n, p = X.shape
    K = mu.shape[1]
    alpha = PRIOR["a_sigma"] + n / 2
    residual = np.zeros(p)
    for j in range(p):
        for i in range(n):
            residual[j] += (
                (X[i, j] - W[i] @ mu[j]) ** 2
                + mu[j] @ Phi @ mu[j]
                + W[i] @ V[j] @ W[i]
                + np.trace(V[j] @ Phi)
            )

    log_likelihood = (
        -0.5 * n * p * np.log(2 * np.pi)
        -0.5 * n * np.sum(np.log(beta) - digamma(alpha))
        -0.5 * np.sum(alpha * residual / beta)
    )
    a_sigma, b_sigma = PRIOR["a_sigma"], PRIOR["b_sigma"]
    noise_kl = np.sum(
        a_sigma * np.log(beta / b_sigma)
        + gammaln(a_sigma) - gammaln(alpha)
        + (alpha - a_sigma) * digamma(alpha)
        + alpha * (b_sigma - beta) / beta
    )
    score_kl = 0.5 * (
        np.sum(W ** 2) + n * np.trace(Phi) - n * K
        - n * np.linalg.slogdet(Phi)[1]
    )
    indices = np.arange(1, K + 1, dtype=np.float64)
    prior_shape = PRIOR["a"] + indices ** PRIOR["c1"]
    prior_rate = indices ** (-PRIOR["c2"])
    second = mu ** 2 + np.diagonal(V, axis1=1, axis2=2)
    m = np.sum(second ** 0.25, axis=0) + prior_rate
    auxiliary = np.sum(
        gammaln(2 * p + prior_shape) - gammaln(prior_shape)
        + prior_shape * np.log(prior_rate) - p * np.log(4)
        - (2 * p + prior_shape) * np.log(m)
    )
    loading_entropy = 0.5 * (
        p * K * (1 + np.log(2 * np.pi))
        + np.linalg.slogdet(V)[1].sum()
    )
    return (log_likelihood - noise_kl - score_kl
            + auxiliary + loading_entropy), residual, m


@pytest.mark.parametrize("center", [True, False])
def test_spectral_geometry_noise_scale_and_no_hidden_centering(center):
    X = _data(center=center)
    n, p = X.shape
    K = 3
    state = _start(X, K)
    _, singular, vt = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    base = vt[:K].T * (singular[:K] / np.sqrt(n))
    second = np.mean(X.astype(np.float64) ** 2, axis=0)
    noise = np.maximum(second - np.sum(base ** 2, axis=1), 0.05 * second)
    mu, V = np.asarray(state["mu"]), np.asarray(state["V"])

    # Sign choices and the allowed column permutation cannot change this.
    np.testing.assert_allclose(mu @ mu.T, base @ base.T, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(
        V, noise[:, None, None] * np.eye(K)[None] / n,
        rtol=2e-6, atol=2e-7,
    )
    np.testing.assert_allclose(
        np.sum(V / noise[:, None, None], axis=0), p / n * np.eye(K),
        rtol=2e-6, atol=2e-7,
    )


def test_rank_deficient_tail_is_nonzero_and_seeded():
    z = np.arange(-4, 5, dtype=np.float32)
    X = z[:, None] * np.array([1, 2, -0.5, 3], dtype=np.float32)
    K = 4
    first, again, changed = _start(X, K), _start(X, K), _start(X, K, seed=17)
    for name in STATE_ARRAYS:
        np.testing.assert_array_equal(first[name], again[name])
    assert np.all(np.linalg.norm(first["mu"], axis=0) > 0)
    assert not np.array_equal(first["mu"], changed["mu"])

    # When all nonzero spectral directions are used, the stated 5% floor
    # determines every noise seed even after tail perturbations are added.
    expected = 0.05 * np.mean(X.astype(np.float64) ** 2, axis=0) / X.shape[0]
    np.testing.assert_allclose(np.asarray(first["V"])[:, 0, 0], expected,
                               rtol=2e-6, atol=1e-7)

    rotated = _start(X, K, varimax=True)
    assert rotated["initialization_info"]["initial_rank"] == 1
    assert rotated["initialization_info"]["varimax_n_iter"] == 0
    for name in (*STATE_ARRAYS, "elbo_var"):
        np.testing.assert_array_equal(rotated[name], first[name])


def test_seed_has_no_effect_when_no_extra_columns_are_needed():
    X = _data()
    first, second = _start(X, 3, seed=1), _start(X, 3, seed=9)
    for name in STATE_ARRAYS:
        np.testing.assert_array_equal(first[name], second[name])


@pytest.mark.parametrize("varimax", [False, True])
def test_initial_state_is_positive_and_matches_five_term_elbo(varimax):
    X = _data()
    K = 4
    state = _start(X, K, varimax=varimax)
    assert bool(state["ok"])
    for name in STATE_ARRAYS:
        array = np.asarray(state[name])
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    assert np.linalg.eigvalsh(state["V"]).min() > 0
    assert np.linalg.eigvalsh(state["Phi"]).min() > 0

    reference_elbo, residual, expected_m = _five_term_elbo(X, state)
    alpha = PRIOR["a_sigma"] + X.shape[0] / 2
    expected_beta = PRIOR["b_sigma"] + residual / 2
    np.testing.assert_allclose(state["beta"], expected_beta, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(state["c"], alpha / expected_beta, rtol=3e-6)
    np.testing.assert_allclose(state["m"], expected_m, rtol=3e-6)
    E2 = np.asarray(state["mu"], dtype=np.float64) ** 2 + np.diagonal(
        np.asarray(state["V"], dtype=np.float64), axis1=1, axis2=2,
    )
    indices = np.arange(1, K + 1, dtype=np.float64)
    expected_rho = (2 * X.shape[1] + PRIOR["a"] + indices ** PRIOR["c1"])[None] / (
        2 * E2 ** 0.75 * expected_m[None]
    )
    np.testing.assert_allclose(state["rho"], expected_rho, rtol=5e-6)
    absolute = float(state["elbo_var"]) + elbo_const(*X.shape, K, **PRIOR)
    np.testing.assert_allclose(absolute, reference_elbo, rtol=2e-6, atol=3e-3)


def test_column_order_minimizes_profiled_penalty_over_all_permutations():
    state = _start(_data(), 4)
    mu = np.asarray(state["mu"], dtype=np.float64)
    V = np.asarray(state["V"], dtype=np.float64)
    mass = np.sum((mu ** 2 + np.diagonal(V, axis1=1, axis2=2)) ** 0.25, axis=0)
    indices = np.arange(1, 5, dtype=np.float64)
    A = 2 * mu.shape[0] + PRIOR["a"] + indices ** PRIOR["c1"]
    b = indices ** (-PRIOR["c2"])
    ordered = np.sum(A * np.log(mass + b))
    assert np.all(np.diff(mass) <= 1e-6)
    for permutation in itertools.permutations(range(4)):
        alternative = np.sum(A * np.log(mass[list(permutation)] + b))
        assert ordered <= alternative + 1e-8


def test_removed_ppca_method_is_rejected_by_estimator_and_driver():
    X = _data()
    with pytest.raises(ValueError, match="init_method"):
        CaviBFM(K=3, init_method="ppca", max_iter=1).fit(X)
    with pytest.raises(ValueError, match="init_method"):
        CAVI(X, K=3, init_method="ppca", max_iter=1)


@pytest.mark.parametrize("api", [CaviBFM, CAVI])
def test_removed_initializer_parameters_cannot_shift_positional_arguments(api):
    params = list(inspect.signature(api).parameters.values())
    names = [p.name for p in params]
    assert "r" not in names and "v0" not in names
    assert params[names.index("max_iter") + 1].kind == inspect.Parameter.KEYWORD_ONLY


def test_default_driver_uses_spectral_initialization_in_float32():
    X = _data()
    options = dict(K=3, center=False, max_iter=3, tol=1e-6, jit=False, **PRIOR)
    implicit = CAVI(X, **options)
    explicit = CAVI(X, init_method="spectral", **options)
    assert implicit["initialization_method"] == "spectral"
    assert implicit["dtype"] == "float32"
    for name in ("mu", "V", "W", "Phi", "beta", "rho", "m", "c"):
        np.testing.assert_array_equal(implicit[name], explicit[name])
    np.testing.assert_array_equal(implicit["elbo"], explicit["elbo"])


@pytest.mark.parametrize("method", [None, "spectral_varimax"])
def test_estimator_spectral_fit_preserves_centering_and_matches_driver(method):
    X = _data(center=False).astype(np.float64)
    options = {} if method is None else {"init_method": method}
    model = CaviBFM(K=3, init_seed=2026,
                    center=True, max_iter=8, tol=1e-3, jit=False, **options, **PRIOR).fit(X)
    expected = method or "spectral"
    assert model.initialization_method_ == model.initialization_info_["method"] == expected
    if method == "spectral_varimax":
        assert model.initialization_info_["varimax_n_iter"] > 0
    else:
        assert "varimax_n_iter" not in model.initialization_info_
    assert model.dtype_ == "float32"
    centered = (X - X.mean(axis=0)).astype(np.float32)
    direct = CAVI(centered, K=3, center=False, init_method=expected, init_seed=2026,
                  max_iter=8, tol=1e-3, jit=False, **PRIOR)
    np.testing.assert_array_equal(model.x_offset_, X.mean(axis=0))
    np.testing.assert_array_equal(model.mu_, direct["mu"])
    assert np.isfinite(model.elbo_).all()
    assert model.n_iter_ >= 1
    assert np.min(np.diff(model.elbo_)) >= -0.5


@pytest.mark.parametrize("method", ["spectral", "spectral_varimax"])
def test_warm_start_takes_precedence_over_the_initializer(monkeypatch, method):
    X = _data()
    state = _start(X, 3)
    driver = importlib.import_module("bfm._vi.VI")

    def forbidden(*args, **kwargs):
        raise AssertionError("spectral initializer was called despite initial_state")

    monkeypatch.setattr(driver, "spectral_initialization", forbidden)
    result = CAVI(X, K=3, center=False, initial_state=state,
                  init_method=method, max_iter=2, jit=False, **PRIOR)
    assert result["initialization_method"] == "provided"
    assert np.isfinite(result["mu"]).all()
    assert result["n_iter"] >= 1


def test_spectral_rejects_zero_variance_columns_instead_of_dropping_them():
    X = _data()
    X[:, 1] = 0
    with pytest.raises(ValueError, match="(?i)(zero|variance|positive)"):
        _start(X, 3)


@pytest.mark.parametrize("seed", [-1, True, 0.5, "2026"])
def test_spectral_seed_requires_a_nonnegative_integer(seed):
    with pytest.raises(ValueError, match="seed"):
        _start(_data(), 3, seed=seed)
