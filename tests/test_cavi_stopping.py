import numpy as np
import pytest

from bfm import CaviBFM


@pytest.mark.parametrize("jit", [False, True])
def test_uncapped_cavi_stops_on_tolerance_across_chunks(jit):
    X = np.random.default_rng(123).normal(size=(16, 7))
    options = dict(K=2, chunk_size=4, tol=0.1, jit=jit, verbose=False)
    capped = CaviBFM(max_iter=7, **options).fit(X)
    reference = CaviBFM(max_iter=100, **options).fit(X)
    uncapped = CaviBFM(max_iter=None, **options).fit(X)

    assert capped.n_iter_ == 7 and capped.stop_reason_ == "max_iter"
    assert 7 < uncapped.n_iter_ == reference.n_iter_ < 100
    assert uncapped.converged_ and uncapped.stop_reason_ == "tol"
    assert len(uncapped.elbo_) == uncapped.n_iter_ + 1
    assert abs(uncapped.elbo_[-1] - uncapped.elbo_[-2]) < uncapped.tol
    np.testing.assert_array_equal(capped.elbo_, uncapped.elbo_[:8])
    for name in ("elbo_", "mu_", "V_", "W_", "Phi_", "beta_", "rho_", "m_", "c_"):
        np.testing.assert_array_equal(getattr(uncapped, name), getattr(reference, name))
