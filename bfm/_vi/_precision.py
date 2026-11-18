"""Precision selection for CAVI entry points.

Float64 is an explicit opt-in.  JAX's x64 switch is process-wide; enabling
it here is deliberate and is not reversed when a fit finishes.  Float32
calls still use explicit float32 arrays after x64 has been enabled.
"""

import jax
import numpy as np


def resolve_precision(precision: str) -> np.dtype:
    """Validate precision, enable JAX x64 if requested, and return the dtype."""
    if not isinstance(precision, str) or precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    return np.dtype(precision)
