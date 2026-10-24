"""RSP alignment of posterior loadings in Bayesian factor models using JAX.

Rotation-sign-permutation (RSP) post-processing of factor loadings: a
per-draw varimax rotation followed by exact sign/permutation alignment
against a common reference matrix.  The method and its computational
strategies are described in ``Factor_Switching.md`` (Sections 3 and 4).

Default precision is float32; ``precision="float64"`` enables double
precision at the entry point.  optax (assignment problems with q > 7) and
scipy (the "scipy" assignment backend) are imported lazily inside the
functions that need them.
"""
from .api import rsp_exact
from .chains import transform_scores
from .io import from_flat, lambda_colnames, to_flat
from .types import RSPResult

__all__ = [
    "rsp_exact",
    "transform_scores",
    "from_flat",
    "to_flat",
    "lambda_colnames",
    "RSPResult",
]
