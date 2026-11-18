"""bfm: unified estimators for the L_{1/2}-shrunk Bayesian factor model.

sklearn-style interface to the two JAX inference algorithms of the
paper "Bayesian factorization via L_{1/2} shrinkage":

* :class:`GibbsBFM` -- the Gibbs sampler of Section 3 (exact MCMC),
  implemented in :mod:`bfm._gs`;
* :class:`CaviBFM` -- the coordinate ascent variational inference of
  Section 4 / Algorithm 1 (fast approximation, closed-form ELBO of
  Appendix B2), implemented in :mod:`bfm._vi`.

Both estimators share the hyperparameters of the prior (K, a, c1, c2,
a_sigma, b_sigma), the preprocessing (float64 column centering followed
by inference-dtype coercion) and the fitted accessors B_hat_ / sigma2_hat_ /
Sigma_ / n_factors().

Gibbs uses float32. CAVI defaults to float32 and supports explicit float64.

Runtime dependencies: JAX/jaxlib and NumPy (matplotlib only for
CaviBFM.plot_elbo, imported lazily).
"""

from .base import BaseBFM
from .cavi import CaviBFM
from .gibbs import GibbsBFM

__version__ = "0.1.0"

__all__ = ["BaseBFM", "GibbsBFM", "CaviBFM"]
