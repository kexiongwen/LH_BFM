"""CAVI estimator for the L_{1/2}-shrunk Bayesian factor model.

sklearn-style wrapper of the JAX coordinate-ascent variational
inference driver :func:`bfm._vi.CAVI` (Section 4 / Algorithm 1 of the
paper "Bayesian factorization via L_{1/2} shrinkage", with the
closed-form ELBO of Appendix B2).  All numerical semantics (float32,
the variable-part ELBO stopping criterion, device-side diagnostics,
chunked while_loop, XLA command buffers) are those of the underlying
driver; this class only adds the estimator interface and the shared
preprocessing of :class:`bfm.base.BaseBFM`.
"""

from numbers import Real

import numpy as np

from ._posterior import cov_sigma_pairs, sigma_mean, sigma_variance
from ._vi import CAVI
from .base import BaseBFM

__all__ = ["CaviBFM"]


class CaviBFM(BaseBFM):
    """Coordinate ascent variational inference (fast posterior approximation).

    Parameters
    ----------
    K : int
        Truncation level (upper bound on the number of factors).
    a, c1, c2 : float
        Hyperparameters of the L_{1/2} shrinkage prior; the defaults
        a = 10, c1 = 2.0, c2 = 1.0 satisfy the paper's theory conditions.
    a_sigma, b_sigma : float
        Inverse-gamma hyperparameters of the noise variances.
    center : bool, default True
        Subtract the column means of X before fitting (in float64 on the
        host; the centered matrix is then cast to float32).
    tol : float or None, default 1e-1
        Absolute ELBO stopping tolerance.  After each complete sweep,
        stop when abs(Lvar_t - Lvar_prev) < tol, where Lvar excludes
        the invariant ELBO constant.  Must be positive and finite;
        None also resolves to 1e-1.
    max_iter : int, default 1000
        Maximum number of CAVI sweeps.
    r : int or None, default None
        Number of leading factors used by the PPCA initialization;
        defaults to min(K, min(n, p) - 2).  Ignored with
        ``initial_state``.
    v0 : float, default 1.0
        Initial loading covariance scale, V_j = v0 * I_K.  Ignored with
        ``initial_state``.
    initial_state : dict or None, default None
        Warm start with mu (p, K), V (p, K, K), W (n, K), Phi (K, K);
        validated and normalized by the underlying driver.
    jit : bool, default True
        Compile the sweep into chunked lax.while_loop runners; False
        runs eagerly for debugging.
    chunk_size : int, default 50
        Sweeps per compiled chunk (host-sync frequency; does not change
        the stopping point).
    graph : bool, default False
        On a GPU backend, compile the chunk runner with XLA command
        buffers (CUDA graphs); elsewhere a warning is issued and the
        plain runner is used.
    audit_every : int, default 50
        Host-side full-state finiteness audit threshold.  Audits happen only
        at compiled chunk boundaries (``t - last_audit >= audit_every``), so
        the effective cadence is the next chunk boundary at/after this many
        sweeps (about ``max(chunk_size, audit_every)``; set
        ``chunk_size <= audit_every`` for the requested cadence), not every
        sweep.  0 disables the periodic audit; the mandatory init/final
        audits and per-sweep device-side checks are unaffected.
    drop_budget : float or None, default 0.5
        Maximum permitted absolute ELBO decrease per sweep, independent
        of tol.  A larger decrease raises FloatingPointError,
        even if it satisfies the stopping threshold.  Must be finite
        and nonnegative; 0 requires non-decreasing ELBO.  None also
        resolves to the fixed default 0.5.
    verbose : bool, default False
        Print the ELBO at chunk boundaries.

    Attributes
    ----------
    mu_, V_ : (p, K) and (p, K, K) numpy arrays
        Variational means and covariances of the loading rows.
    W_ : (n, K) numpy array
        Variational means of the factor scores.
    Phi_ : (K, K) numpy array
        Common covariance of the factor scores.
    alpha_n_ : float
        Noise inverse-gamma shape a_sigma + n/2.
    beta_, c_, rho_, m_ : numpy arrays
        Noise inverse-gamma rates, E_q[1/sigma_j^2], shrinkage moments
        E_q[lambda_k^4/tau_jk^2], and column aggregates.
    elbo_ : list of float
        Absolute ELBO after initialization and each sweep.
    converged_, stop_reason_, n_iter_ : bool, str, int
        Stopping outcome ("tol" or "max_iter") and sweep count.
    execution_mode_, backend_, device_ : str
        Execution environment summary.
    compile_sec_, runtime_sec_ : float
        AOT compilation and sweep-loop wall times.
    B_hat_, sigma2_hat_, Sigma_ : numpy arrays
        Point estimates (see :class:`bfm.base.BaseBFM`); sigma2_hat_ is
        the inverse-gamma posterior mean beta / (alpha_n - 1).  These
        remain in the original fitted coordinates after alignment.
    Sigma_mean_, Sigma_var_, Sigma_sd_ : (p, p) numpy arrays
        Variational posterior moments of Sigma = B B^T + Omega, computed
        lazily from ``mu_``, ``V_``, ``beta_`` and ``alpha_n_``:
        ``Sigma_mean_ = E_q[Sigma] = Sigma_ + diag(tr V_1, ..., tr V_p)``,
        ``Sigma_var_[i, j] = Var_q(Sigma_ij)`` and ``Sigma_sd_`` its square
        root.  ``Sigma_var_``/``Sigma_sd_`` require ``alpha_n > 2`` (where
        ``alpha_n = a_sigma + n/2``).  These are moments of the mean-field
        approximation, not of the true posterior; see the module notes in
        :mod:`bfm._posterior` and :meth:`cov_Sigma` for entry-pair
        covariances.
    rsp_result_, B_aligned_
        Produced by :meth:`align_factors` from ``mu_[None, :, :]``.
        ``B_aligned_`` has shape (1, p, K); its single matrix is the
        transformed VI mean, not an MCMC draw.
    mu_aligned_, V_aligned_ : (p, K) and (p, K, K) numpy arrays
        Transformed loading means and covariances: mu R and R^T V_j R.
    W_aligned_, Phi_aligned_ : (n, K) and (K, K) numpy arrays
        Transformed score means and covariance: W R and R^T Phi R.
    eta_aligned_ : (K, n) numpy array
        Transpose view of W_aligned_: aligned score posterior MEANS,
        not random draws.  Together with mu_aligned_, preserves the
        fitted common component mu @ W.T up to numerical error.
    raw_output_ : dict
        The unmodified return value of the underlying driver (JAX
        device arrays).

    Notes
    -----
    Alignment outputs are separate from the fitted VI parameters,
    shrinkage moments, and ELBO; a successful refit clears them all.
    The column-dependent shrinkage prior is not invariant to general
    rotations or permutations.  Transformed moments are useful for
    interpretation and comparison, not an equivalent CAVI optimum in
    the original parameterization.  ``n_factors()`` (marginal credible
    intervals) uses the original ``mu_``/``V_`` by default;
    ``use_aligned=True`` switches to the transformed moments.  Repeated
    alignment starts from ``mu_``.  Alternatively,
    ``n_factors(method="linf", threshold=1e-1)`` counts columns whose
    maximum absolute variational mean exceeds the threshold.

    ``Sigma_`` is the plug-in estimate ``mu mu^T + diag(beta/(alpha_n-1))``;
    ``Sigma_mean_`` is the variational posterior mean and therefore adds
    ``diag(tr V_j)`` on the diagonal.  The entrywise posterior variance
    ``Sigma_var_`` is guarded by a default limit of 2e7 entries (p <= 4472);
    for larger p use :meth:`cov_Sigma` on selected entry pairs.

    Examples
    --------
    Align the VI means and covariances consistently::

        vi = CaviBFM(K=5).fit(X).align_factors()
        vi.mu_aligned_       # (p, K), also vi.B_aligned_[0]
        vi.eta_aligned_      # (K, n), posterior means
        vi.V_aligned_        # (p, K, K), loading covariances
        vi.Phi_aligned_      # (K, K), score covariance
    """

    _ALIGN_UPDATES_B_HAT = False
    _ALIGN_ATTRS = BaseBFM._ALIGN_ATTRS + (
        "mu_aligned_", "V_aligned_", "W_aligned_", "Phi_aligned_",
        "eta_aligned_",
    )

    def __init__(
        self,
        K,
        a=10.0,
        c1=2.0,
        c2=1.0,
        a_sigma=1.0,
        b_sigma=1.0,
        center=True,
        tol=1e-1,
        max_iter=1000,
        r=None,
        v0=1.0,
        initial_state=None,
        jit=True,
        chunk_size=50,
        graph=False,
        audit_every=50,
        drop_budget=0.5,
        verbose=False,
    ):
        super().__init__(
            K, a=a, c1=c1, c2=c2, a_sigma=a_sigma, b_sigma=b_sigma,
            center=center,
        )
        self.tol = tol
        self.max_iter = max_iter
        self.r = r
        self.v0 = v0
        self.initial_state = initial_state
        self.jit = jit
        self.chunk_size = chunk_size
        self.graph = graph
        self.audit_every = audit_every
        self.drop_budget = drop_budget
        self.verbose = verbose

    def _fit_arrays(self, Xc):
        # Posterior-moment cache (Sigma_mean_/Sigma_var_); a new fit must not
        # reuse moments of the previous fit.
        self._sigma_cache = {}
        out = CAVI(
            Xc,
            self.K,
            a=self.a,
            c1=self.c1,
            c2=self.c2,
            a_sigma=self.a_sigma,
            b_sigma=self.b_sigma,
            tol=self.tol,
            max_iter=self.max_iter,
            r=self.r,
            v0=self.v0,
            center=False,  # centering is done once by BaseBFM
            initial_state=self.initial_state,
            jit=self.jit,
            chunk_size=self.chunk_size,
            graph=self.graph,
            audit_every=self.audit_every,
            drop_budget=self.drop_budget,
            verbose=self.verbose,
            plot=False,
        )
        self.raw_output_ = out
        self.mu_ = np.asarray(out["mu"])
        self.V_ = np.asarray(out["V"])
        self.W_ = np.asarray(out["W"])
        self.Phi_ = np.asarray(out["Phi"])
        self.alpha_n_ = out["alpha_n"]
        self.beta_ = np.asarray(out["beta"])
        self.c_ = np.asarray(out["c"])
        self.rho_ = np.asarray(out["rho"])
        self.m_ = np.asarray(out["m"])
        self.elbo_ = list(out["elbo"])
        self.n_iter_ = out["n_iter"]
        self.converged_ = out["converged"]
        self.stop_reason_ = out["stop_reason"]
        self.execution_mode_ = out["execution_mode"]
        self.backend_ = out["backend"]
        self.device_ = out["device"]
        self.compile_sec_ = out["compile_sec"]
        self.runtime_sec_ = out["runtime_sec"]
        self._B_hat = self.mu_
        # posterior mean of IG(alpha_n, beta_j): beta_j / (alpha_n - 1);
        # requires alpha_n > 1 (same guard as bfm._posterior).  Raising
        # here lets BaseBFM.fit roll back to the previous fitted state.
        alpha_n = float(self.alpha_n_)
        if alpha_n <= 1.0:
            raise ValueError(
                "the noise-variance point estimate (inverse-gamma "
                "posterior mean beta / (alpha_n - 1)) requires "
                f"alpha_n = a_sigma + n/2 > 1, got alpha_n = {alpha_n} "
                f"(a_sigma = {self.a_sigma}, n = {Xc.shape[0]}); "
                "increase a_sigma or use more observations"
            )
        self._sigma2_hat = self.beta_ / (alpha_n - 1.0)

    def _alignment_loadings(self):
        self._check_fitted()
        return self.mu_[None, :, :]

    def _alignment_outputs(self, result):
        outputs = super()._alignment_outputs(result)
        p, K = self.mu_.shape
        transforms = np.asarray(result.transforms)
        if (transforms.shape != (1, K, K) or
                outputs["B_aligned_"].shape != (1, p, K)):
            raise ValueError("VI alignment requires one (p, K) loading matrix and (K, K) transform")
        R = transforms[0]
        with np.errstate(over="ignore", invalid="ignore"):
            outputs.update({
                "mu_aligned_": outputs["B_aligned_"][0],
                "W_aligned_": self.W_ @ R,
                "V_aligned_": R.T @ self.V_ @ R,
                "Phi_aligned_": R.T @ self.Phi_ @ R,
            })
        outputs["eta_aligned_"] = outputs["W_aligned_"].T
        if (not np.isfinite(R).all() or any(
            not np.isfinite(outputs[name]).all()
            for name in self._ALIGN_ATTRS if name != "rsp_result_"
        )):
            exc = FloatingPointError(
                "VI alignment produced non-finite moments; no estimator state was changed"
            )
            exc.rsp_result = result
            raise exc
        return outputs

    # ------------------------------------------------------------------
    # factor count: Gaussian credible intervals or mean-loading magnitude
    # ------------------------------------------------------------------
    def n_factors(
        self, level=0.95, *, interval="hpd", use_aligned=False,
        method="credible", threshold=1e-1,
    ):
        """Count active columns using a CAVI posterior summary.

        Parameters
        ----------
        level, interval
            Credible-interval settings, as in :class:`BaseBFM`.  Used
            only by ``method="credible"`` (the default).
        use_aligned : bool, default False
            Use the transformed means (and, for credible intervals,
            covariances) from a successful ``align_factors()`` call.
            General rotations can change either criterion's count.
        method : {"credible", "linf"}, default "credible"
            "credible" counts columns with at least one marginal
            credible interval excluding zero.  "linf" counts columns k
            with ``max_j abs(mu_[j, k]) > threshold``.  Equality leaves
            the column inactive.  The latter uses mean loadings only,
            without accounting for posterior uncertainty.
        threshold : float, default 1e-1
            Finite, nonnegative magnitude cutoff for "linf", in the
            data's units; ignored by "credible".  It is rounded to the
            selected means' dtype for comparison (normally float32).

        Returns
        -------
        int
            Number of active columns.
        """
        return int(self.active_entries(
            level=level, interval=interval, use_aligned=use_aligned,
            method=method, threshold=threshold,
        ).any(axis=0).sum())

    def active_entries(
        self, level=0.95, *, interval="hpd", use_aligned=False,
        method="credible", threshold=1e-1,
    ):
        """Return the (p, K) activity mask behind :meth:`n_factors`.

        With "credible", an entry is active when its marginal credible
        interval excludes zero.  With "linf", it is active when the
        absolute variational mean exceeds ``threshold``; a column is
        active iff any of its entries is active.  Parameters have the
        same meaning as in :meth:`n_factors`.
        """
        self._check_fitted()
        if not isinstance(method, str) or method not in ("credible", "linf"):
            raise ValueError(f"method must be 'credible' or 'linf', got {method!r}")
        if method == "credible":
            return super().active_entries(
                level=level, interval=interval, use_aligned=use_aligned,
            )

        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
            raise ValueError("threshold must be a finite nonnegative real number")
        try:
            threshold = float(threshold)
        except (OverflowError, ValueError) as exc:
            raise ValueError("threshold must be a finite nonnegative real number") from exc
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be a finite nonnegative real number")

        if use_aligned:
            if not hasattr(self, "mu_aligned_"):
                raise RuntimeError(
                    "use_aligned=True requires a successful "
                    "align_factors() call first"
                )
            mu = np.asarray(self.mu_aligned_)
        else:
            mu = np.asarray(self.mu_)
        # Compare at the stored precision so representational rounding
        # does not move values on the cutoff across the boundary.
        with np.errstate(over="ignore", under="ignore"):
            cutoff = np.asarray(threshold, dtype=mu.dtype)
        return np.abs(mu) > cutoff

    def _credible_active(self, level, interval, use_aligned):
        """CAVI path: closed-form Gaussian intervals from (mu, V).

        For q(B_jk) = N(mu_jk, V_jkk) the highest-density interval is the
        equal-tailed one, so ``interval`` has no effect here.  Active iff
        |mu_jk| > z_(1/2 + level/2) * sqrt(V_jkk).
        """
        if use_aligned:
            if not hasattr(self, "mu_aligned_"):
                raise RuntimeError(
                    "use_aligned=True requires a successful "
                    "align_factors() call first"
                )
            mu, V = self.mu_aligned_, self.V_aligned_
        else:
            mu, V = self.mu_, self.V_
        from statistics import NormalDist

        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        sd = np.sqrt(np.diagonal(np.asarray(V, dtype=np.float64),
                                 axis1=1, axis2=2))
        return np.abs(np.asarray(mu, dtype=np.float64)) > z * sd

    # ------------------------------------------------------------------
    # variational posterior covariance of Sigma = B B^T + Omega
    # ------------------------------------------------------------------
    @property
    def Sigma_mean_(self):
        """(p, p) variational posterior mean E_q[Sigma] of the covariance.

        Computed lazily from the fitted ``mu_``, ``V_``, ``beta_`` and
        ``alpha_n_`` as ``mu mu^T + diag(tr(V_j) + beta_j/(alpha_n-1))``,
        i.e. the plug-in :attr:`Sigma_` plus the diagonal correction
        ``diag(tr V_j)``.  float64; cached until the next successful fit.
        """
        self._check_fitted()
        cache = self._sigma_cache
        if "mean" not in cache:
            cache["mean"] = sigma_mean(
                self.mu_, self.V_, self.alpha_n_, self.beta_
            )
        return cache["mean"]

    @property
    def Sigma_var_(self):
        """(p, p) entrywise variational posterior variance of the covariance.

        ``Sigma_var_[i, j] = Var_q(Sigma_ij)`` with ``Sigma = B B^T + Omega``
        under the mean-field posterior; float64, computed lazily and cached
        until the next successful fit.  Requires ``alpha_n > 2``.  The
        returned matrix has p * p entries (guard: 2e7 entries by default);
        for larger p use :meth:`cov_Sigma` on selected entry pairs.
        """
        self._check_fitted()
        cache = self._sigma_cache
        if "var" not in cache:
            cache["var"] = sigma_variance(
                self.mu_, self.V_, self.alpha_n_, self.beta_
            )
        return cache["var"]

    @property
    def Sigma_sd_(self):
        """(p, p) entrywise posterior standard deviation sqrt(Sigma_var_)."""
        self._check_fitted()
        cache = self._sigma_cache
        if "sd" not in cache:
            cache["sd"] = np.sqrt(self.Sigma_var_)
        return cache["sd"]

    def cov_Sigma(self, pairs):
        """Cov_q(Sigma_ij, Sigma_kl) for selected entry pairs.

        Parameters
        ----------
        pairs : iterable of ((i, j), (k, l))
            Zero-based entry indices; order within each pair and between the
            two entries does not matter.

        Returns
        -------
        numpy.ndarray, shape (len(pairs),), float64
            The covariances in the order given.  Pairs with no shared row
            have covariance zero.  Requires ``alpha_n > 2``.

        Notes
        -----
        This is the mean-field variational posterior covariance, computed in
        closed form for the requested pairs; the full p^2 x p^2 covariance of
        ``vec(Sigma)`` is never formed.  See :mod:`bfm._posterior` for the
        case structure and the formulas.
        """
        self._check_fitted()
        return cov_sigma_pairs(
            self.mu_, self.V_, self.alpha_n_, self.beta_, pairs
        )

    def plot_elbo(self, **kwargs):
        """Plot the ELBO trajectory of this run (requires matplotlib).

        Keyword arguments are passed to :func:`bfm._vi.plot_elbo`
        (``ax``, ``logx``, ``show``, ``save``, ``title``).
        """
        self._check_fitted()
        from ._vi import plot_elbo

        return plot_elbo(self.elbo_, **kwargs)
