"""
bfm._gs  (single precision)

JAX Gibbs sampler for the Bayesian factor model with the L_{1/2}
shrinkage prior (paper: "Bayesian factorization via L_{1/2}
shrinkage").

Single precision (float32) throughout; all sampler functions take an
explicit jax.random.PRNGKey as first argument.  The driver GS_LH returns
posterior draws as host-side numpy arrays.

Public API
----------
GS_LH          : full Gibbs sampler driver (chunked jax.lax.scan chain)
Initialization : principal-factor initialization for (B, sigma2)
shrinkage      : L_{1/2} shrinkage-parameter draw given B (lambda, tau -> w)
inv_gauss      : InvGauss(mu, 1) sampler (Michael-Schucany-Haas)
eta_sample     : latent-factor draw eta | B, Omega, X
B_sample       : loading-matrix draw B | eta, D, sigma2, X
sigma2_sample  : idiosyncratic-variance draw sigma2 | B, eta, X
gamma_sample   : Gamma draw (exact, or chi-square from a plan)
make_chi2_plan : bucket plan for per-element chi-square dof (lambda)
make_chi2_plan_scalar : row-chunk plan for a scalar chi-square dof
chi2_plan_cost : estimated normal draws of a chi-square plan

Import note: ``GS_LH`` here is the re-exported FUNCTION; the package attribute
shadows the submodule attribute ``bfm._gs.GS_LH``.  Consequently
``import bfm._gs.GS_LH as m`` binds the function rather than the module.  Use
``importlib.import_module("bfm._gs.GS_LH")`` or
``from bfm._gs.GS_LH import <name>`` to access the module itself;
``from bfm._gs import GS_LH`` remains the supported convenience import.

Implementation notes (vs the PyTorch reference): the CUDA-graph machinery
(graph.py) is replaced by native XLA mechanisms -- the chain is
jit+lax.scan compiled into one XLA While loop per chunk, optionally
recorded as a single CUDA graph via XLA command buffers (see GS_LH's
`graph` argument).  The chi-square Gamma approximation (Chi2.py) and
its _accept_chi2 decision are implemented as well, but with two changes:
jax.random.gamma is a pure XLA primitive (exact, capturable), so the
approximation is never mandatory -- on the GPU backend the driver
applies the same acceptance rule as torch and uses it as an
accuracy-for-speed tradeoff, subject to a chi2_max_normals cost gate;
and the dof are sampled from a host-side bucketed/chunked plan
instead of a dense (K, nu_max) mask, which bounds GPU memory and
removes the O(K * nu_max) padding.  Everywhere else Gamma draws are
exact.
"""

from .eta_sample import eta_sample
from .Chi2 import (chi2_plan_cost, gamma_sample, make_chi2_mask,
                   make_chi2_plan, make_chi2_plan_scalar)
from .Shrinkage import inv_gauss, shrinkage
from .Sigma2_sample import sigma2_sample
from .B_sample import B_sample
from .GS_LH import GS_LH, Initialization
from .ESS import ess, ess_summary

__all__ = [
    "GS_LH",
    "Initialization",
    "shrinkage",
    "inv_gauss",
    "eta_sample",
    "B_sample",
    "sigma2_sample",
    "gamma_sample",
    "make_chi2_mask",
    "make_chi2_plan",
    "make_chi2_plan_scalar",
    "chi2_plan_cost",
    "ess",
    "ess_summary",
]
