"""bfm._vi  (single precision).

JAX coordinate ascent variational inference (CAVI, Algorithm 1 of the
main text) for the Bayesian factor model
with the L_{1/2} shrinkage prior (paper: "Bayesian factorization via
L_{1/2} shrinkage"), with the closed-form ELBO of Appendix B2.

Single precision (float32) throughout; JAX's x64 mode is neither
required nor enabled by this package.  The sweep is deterministic --
no RNG anywhere; initialization is a deterministic thin SVD.

Public API
----------
CAVI           : full CAVI driver (chunked lax.while_loop; optional
                 XLA command buffers on GPU)
PPCA           : deterministic thin-SVD probabilistic-PCA initialization
elbo           : standalone closed-form ELBO at a saved state (diagnostic)
elbo_const     : constant term of the closed-form ELBO (host, once per run)
plot_elbo      : ELBO trajectory plot (lazy matplotlib import)
initialization : build the self-consistent initial variational state

Import note: ``elbo`` and ``initialization`` here are re-exported FUNCTIONS;
those package attributes shadow the submodules ``bfm._vi.elbo`` and
``bfm._vi.initialization``.  ``import bfm._vi.elbo as m`` therefore binds the
function, not the module.  Use
``importlib.import_module("bfm._vi.elbo")`` (likewise for ``initialization``)
or ``from bfm._vi.elbo import <name>`` to access the modules themselves;
``from bfm._vi import elbo`` remains the supported convenience import.

Runtime dependencies: JAX/jaxlib and NumPy only (the independent
float64 reference and cross-backend tests, which use PyTorch / SciPy,
are not included in this repository).  Design follows :mod:`bfm._gs`.
"""

from .VI import CAVI, PPCA, elbo, elbo_const, initialization, plot_elbo

__all__ = ["CAVI", "PPCA", "elbo", "elbo_const", "plot_elbo", "initialization"]
