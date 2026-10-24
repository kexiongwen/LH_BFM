"""ESS.py  -- effective sample size diagnostics for the Gibbs chain.

Computes per-entry effective sample sizes (ESS) of the kept draws of
the loading matrix B and the noise variances sigma2 produced by
:func:`bfm._gs.GS_LH`, via ArviZ (:func:`arviz.ess`).

ArviZ is an OPTIONAL dependency of this package, imported lazily: every
module here works without arviz except the functions of this file,
which raise ImportError with an install hint when it is missing.

Conventions
-----------
* Input arrays are the kept draws: B with shape (S, p, K) and sigma2
  with shape (S, p), as returned by GS_LH (numpy float32 on the host).
  Multi-chain arrays (C, S, p, K) / (C, S, p) are accepted as-is.
* p == 1 disambiguation: sigma2 shape (C, S) is accepted as a multi-chain
  array when B is 4-D multi-chain (B's layout decides), and as (S, p) when
  B is 3-D single-chain.  Pass (C, S, p) explicitly to be unambiguous.
* ArviZ expects a (chain, draw, ...) layout; a single-chain input gets
  a chain axis prepended.  ESS then has the entry shape: (p, K) for B,
  (p,) for sigma2.
* ``method`` follows :func:`arviz.ess`: "bulk" (default; rank-normalized
  bulk ESS, the recommended summary), "tail", "quantile", "mean", "sd",
  "median", ....  ``prob`` is required by "tail" and "quantile" (e.g.
  prob=0.9 / 0.05) and is **not accepted** by the other methods; passing
  it where it has no effect raises a ValueError instead of reaching
  ArviZ's internal signature.  ``relative=True`` returns ESS per draw
  (ESS divided by the total number of draws).
* Constant entries: a draw entry that is exactly constant over all samples
  is a degenerate chain.  ArviZ reports ``ESS = total draws`` for such an
  entry; that is NOT evidence of good mixing and should not be read as an
  effective sample size.  This module forwards ArviZ's value unchanged (no
  NaN substitution); filter constant entries (e.g. ``entry.std(axis=draw)
  == 0``) before interpreting ESS when zero-initialized/shrunk columns are
  present.
* Rule of thumb (Vehtari et al., 2021, doi:10.1214/20-BA1221): bulk
  ESS above ~400 per entry makes the Monte Carlo error of posterior
  means negligible for most purposes.
* Scope: only ESS is computed here -- no R-hat and no MCSE.  GS_LH
  runs a single chain per call; for split-R-hat or MCSE, run several
  chains (separate GS_LH calls with different seeds), stack their
  draws with a leading chain axis (C, S, ...) and call
  :func:`arviz.rhat` / :func:`arviz.mcse` on the stacked arrays
  directly (this module already accepts that layout for ESS).

This module is an addition to :mod:`bfm._gs`: ESS diagnostics computed
on the host-side numpy draws; the sampler itself does not depend on it.
"""

import numpy as np

__all__ = ["ess", "ess_summary"]

#: ESS kinds that take a ``prob`` argument in ArviZ.
_TAIL_METHODS = ("tail", "quantile")


def _require_arviz():
    """Import arviz lazily, with an actionable error when missing."""
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "ESS diagnostics require arviz (e.g. pip install arviz, or "
            "pip install bfm-lh[ess])"
        ) from exc
    return az


def _validate_chain_draw(a: np.ndarray, name: str) -> np.ndarray:
    """Common checks once ``a`` is in (chain, draw, ...) layout."""
    if a.shape[1] < 4:
        raise ValueError(
            f"{name} has only {a.shape[1]} draws per chain; ESS needs at "
            "least 4 kept draws"
        )
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite draws; ESS is undefined")
    return a


def _as_chain_draw(a: np.ndarray, name: str, entry_ndim: int) -> np.ndarray:
    """Validate kept draws and return the (chain, draw, ...) layout."""
    a = np.asarray(a)
    if a.ndim == entry_ndim + 1:        # single chain: (S, ...) -> (1, S, ...)
        a = a[None]
    elif a.ndim != entry_ndim + 2:      # multi-chain: (C, S, ...)
        raise ValueError(
            f"{name} must have shape (S, ...) or (C, S, ...) with "
            f"{entry_ndim} entry dimension(s), got shape {a.shape}"
        )
    return _validate_chain_draw(a, name)


def _as_sigma2_draws(
    sigma2: np.ndarray, B_chain: np.ndarray, multi_chain: bool
) -> np.ndarray:
    """Resolve the p=1 (C, S) vs (S, p) ambiguity using B's layout (R3-1).

    For p == 1 the shape (C, S) is ambiguous: it is a valid multi-chain
    sigma2 draw array and, at the same time, a valid single-chain (S, p)
    array.  B's layout settles it: a 4-D multi-chain B requires a
    multi-chain sigma2 (explicit (C, S, p), or (C, S) when p == 1); a 3-D
    single-chain B requires (S,) / (S, p).
    """
    a = np.asarray(sigma2)
    C, S, p = B_chain.shape[0], B_chain.shape[1], B_chain.shape[2]
    if multi_chain:
        if a.ndim == 3 and a.shape == (C, S, p):
            pass
        elif p == 1 and a.ndim == 2 and a.shape == (C, S):
            a = a[..., None]
        else:
            raise ValueError(
                "sigma2 must share B's multi-chain (chain, draw) layout "
                f"{(C, S)} and have shape (C, S, p)"
                + (" or (C, S) when p == 1" if p == 1 else "")
                + f"; got {a.shape}. Note that for p=1 a 2-D (C,S) sigma2 "
                "is read as multi-chain because B is 4-D."
            )
    else:
        if a.ndim == 1 and p == 1 and a.shape == (S,):
            a = a[:, None]
        elif a.ndim == 2 and a.shape == (S, p):
            pass
        else:
            raise ValueError(
                "sigma2 must share B's single-chain layout: shape "
                f"(S, p) = ({S}, {p})"
                + (" or (S,) when p == 1" if p == 1 else "")
                + f"; got {a.shape}. Pass a leading chain axis to both "
                "arrays to use the multi-chain layout."
            )
        a = a[None]
    return _validate_chain_draw(a, "sigma2")


def _validate_method_prob(method, prob):
    """Validate the ESS kind and return the effective ``prob`` (or None)."""
    if not isinstance(method, str):
        raise ValueError(f"method must be a string, got {type(method).__name__}")
    if method in _TAIL_METHODS:
        if prob is None:
            raise ValueError(
                f"method={method!r} requires prob in (0, 1), e.g. prob=0.9"
            )
        if (
            isinstance(prob, (bool, np.bool_))
            or not np.isscalar(prob)
            or not np.isfinite(prob)
            or not 0.0 < float(prob) < 1.0
        ):
            raise ValueError(f"prob must be a finite number in (0, 1), got {prob!r}")
        return float(prob)
    if prob is not None:
        raise ValueError(
            f"prob is only used by method='tail' or 'quantile'; "
            f"got method={method!r}"
        )
    return None


def _ess_array(az, data, *, method, prob, relative):
    """Call ``az.ess`` with the method-appropriate keyword arguments."""
    p = _validate_method_prob(method, prob)
    kwargs = {"method": method, "relative": relative}
    if p is not None:
        kwargs["prob"] = p
    return np.asarray(az.ess(data, **kwargs))


def ess(B, sigma2=None, method: str = "bulk", prob=None, relative: bool = False):
    """Per-entry effective sample size of the kept B / sigma2 draws.

    Parameters
    ----------
    B : array-like, shape (S, p, K) or (C, S, p, K)
        Kept loading-matrix draws ("B" of the GS_LH output).
    sigma2 : array-like, shape (S, p) or (C, S, p), optional
        Kept noise-variance draws ("sigma2" of the GS_LH output).  When
        given, it must share the same (chain, draw) layout as B.
    method : str, default "bulk"
        ESS kind, passed to :func:`arviz.ess` ("bulk", "tail", "mean",
        "sd", "median", "quantile", ...).  "tail" and "quantile" require
        ``prob``; other methods reject it.
    prob : float, optional
        Probability for the "tail"/"quantile" methods (e.g. 0.9 / 0.05).
        Must be None for the other methods.
    relative : bool, default False
        If True, return ESS per draw (ESS / total draws).

    Returns
    -------
    dict
        {"B": (p, K) ndarray} plus {"sigma2": (p,) ndarray} when
        ``sigma2`` is given.

    Notes
    -----
    Constant entries are forwarded unchanged from ArviZ, which returns
    ESS equal to the total number of draws; this describes a degenerate
    (zero-variance) entry and is not evidence of good mixing.  See the
    module docstring for how to filter such entries.
    """
    if not isinstance(relative, (bool, np.bool_)):
        raise ValueError(f"relative must be a boolean, got {relative!r}")
    az = _require_arviz()
    out = {}
    B_raw = np.asarray(B)
    B = _as_chain_draw(B_raw, "B", entry_ndim=2)
    out["B"] = _ess_array(az, B, method=method, prob=prob,
                          relative=bool(relative))
    if sigma2 is not None:
        s2 = _as_sigma2_draws(sigma2, B, multi_chain=(B_raw.ndim == 4))
        out["sigma2"] = _ess_array(az, s2, method=method, prob=prob,
                                   relative=bool(relative))
    return out


def ess_summary(B, sigma2=None, method: str = "bulk", prob=None):
    """Compact ESS summary over the entries of B (and sigma2).

    The worst entries drive the reliability of the chain, so the
    minimum is the number to watch (see the module docstring for the
    ~400 rule of thumb).

    Parameters
    ----------
    B, sigma2 : array-like
        Kept draws, as in :func:`ess`.
    method : str, default "bulk"
        ESS kind (bulk ESS recommended).
    prob : float, optional
        Probability for "tail"/"quantile" methods; must be None
        otherwise (forwarded to :func:`ess`).

    Returns
    -------
    dict
        One entry per input array, each a dict with keys "min",
        "median", "mean", "max" (floats) and "n_draws" (total draws
        behind the ESS values).
    """
    e = ess(B, sigma2, method=method, prob=prob)
    Bc = _as_chain_draw(B, "B", entry_ndim=2)
    n_draws = int(Bc.shape[0] * Bc.shape[1])
    return {
        key: {
            "min": float(a.min()),
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "max": float(a.max()),
            "n_draws": int(n_draws),
        }
        for key, a in e.items()
    }
