"""bfm._vi: float32 by default, with explicit float64 support.

JAX coordinate ascent variational inference (CAVI, Algorithm 1 of the
main text) for the Bayesian factor model
with the L_{1/2} shrinkage prior (paper: "Bayesian factorization via
L_{1/2} shrinkage"), with the closed-form ELBO of Appendix B2.

Inference defaults to float32. Explicit ``precision="float64"`` enables
JAX's process-wide x64 flag; float32 calls still use explicit float32 arrays.
The sweep is deterministic. Both spectral starts use host float64 setup
and a fixed seed only when nonzero tail columns are needed.

Public API
----------
CAVI           : full CAVI driver (chunked lax.while_loop; optional
                 XLA command buffers on GPU)
elbo           : standalone closed-form ELBO at a saved state (diagnostic)
elbo_const     : constant term of the closed-form ELBO (host, once per run)
plot_elbo      : ELBO trajectory plot (lazy matplotlib import)
spectral_initialization : build the initial variational state, with optional varimax

``from bfm._vi import elbo, spectral_initialization`` imports the two
functions. Use ``importlib.import_module`` to access their same-named
submodules.

Runtime dependencies: JAX/jaxlib and NumPy only. Tests independently
check the ELBO with NumPy/SciPy. Design follows :mod:`bfm._gs`.
"""

from .VI import CAVI, elbo, elbo_const, plot_elbo
from .spectral_initialization import spectral_initialization

__all__ = ["CAVI", "elbo", "elbo_const", "plot_elbo", "spectral_initialization"]
