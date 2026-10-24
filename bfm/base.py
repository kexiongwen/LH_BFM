"""Shared base class for the BFM estimators.

:class:`BaseBFM` holds the hyperparameters shared by the Gibbs sampler
(:class:`bfm.GibbsBFM`) and the CAVI approximation (:class:`bfm.CaviBFM`),
and implements the common preprocessing (validation, float64 column
centering followed by the float32 coercion) and the common fitted
accessors (``B_hat_``, ``sigma2_hat_``, ``Sigma_``, ``n_factors``).

The model is the latent factor model of the paper "Bayesian
factorization via L_{1/2} shrinkage":

    x_i = B eta_i + eps_i,
    eta_i ~ N_K(0, I_K),
    eps_i ~ N_p(0, Omega),  Omega = diag(sigma_1^2, ..., sigma_p^2),

with the L_{1/2} shrinkage prior on the entries of B and inverse-gamma
priors on the sigma_j^2.
"""

import warnings

import numpy as np

__all__ = ["BaseBFM"]


class BaseBFM:
    """Shared hyperparameters and preprocessing for the BFM estimators.

    Parameters
    ----------
    K : int
        Truncation level of the loading matrix (number of columns).
    a, c1, c2 : float
        Hyperparameters of the L_{1/2} prior's lambda hyperprior
        Ga(a + k^{c1}, k^{-c2}); all must be > 0.  The defaults a = 10,
        c1 = 2.0, c2 = 1.0 satisfy the theory conditions a >= 4 and
        q = 4(c1 + c2) > 1.
    a_sigma, b_sigma : float
        Inverse-gamma hyperparameters of sigma_j^2 ~ IG(a_sigma,
        b_sigma); both must be > 0.
    center : bool, default True
        Subtract the column means of X before fitting (the model has
        zero mean).  The means are computed and subtracted in float64
        (host side) before the single-precision coercion, so large
        column offsets cannot quantize away the within-column variation.
        The column means are stored in ``x_offset_`` as a float64 array.

    Notes
    -----
    This class only stores parameters and provides shared machinery;
    use :class:`bfm.GibbsBFM` or :class:`bfm.CaviBFM` for fitting.
    Algorithm-specific validation is left to the underlying drivers so
    that the wrapper never drifts from their rules.
    """

    def __init__(
        self,
        K,
        a=10.0,
        c1=2.0,
        c2=1.0,
        a_sigma=1.0,
        b_sigma=1.0,
        center=True,
    ):
        self.K = K
        self.a = a
        self.c1 = c1
        self.c2 = c2
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.center = center

    # ------------------------------------------------------------------
    # fitting (template method)
    # ------------------------------------------------------------------
    # Alignment outputs produced by align_factors(); removed by fit() AFTER
    # the new fit succeeds, so a fit that fails (invalid X, numerical
    # error) keeps the previous state -- raw or aligned -- fully
    # consistent instead of half-cleared.
    _ALIGN_ATTRS = ("rsp_result_", "B_aligned_")
    # Gibbs summarizes aligned draws; VI keeps its fitted estimate and
    # exposes transformed moments separately.
    _ALIGN_UPDATES_B_HAT = True

    def fit(self, X):
        """Fit the model to the data matrix X and return ``self``.

        X is validated, coerced to float32 and column-centered (unless
        ``center=False``); the concrete algorithm then runs in
        :meth:`_fit_arrays`.  Results of a previous :meth:`align_factors`
        call are discarded only after the new fit succeeds: a fit that
        raises leaves the estimator exactly as it was.
        """
        # Fitting replaces attributes, so preserving their references is
        # enough to restore the previous fit without copying large draws.
        previous_state = self.__dict__.copy()
        try:
            Xc = self._prepare_X(X)
            self.n_samples_, self.n_features_in_ = Xc.shape
            self._fit_arrays(Xc)
            for attr in self._ALIGN_ATTRS:
                if hasattr(self, attr):
                    delattr(self, attr)
        except BaseException:
            # Restore preprocessing and partially assigned results even
            # when a long-running fit is interrupted, then re-raise.
            self.__dict__.clear()
            self.__dict__.update(previous_state)
            raise
        return self

    def _fit_arrays(self, Xc):
        """Run the concrete algorithm on the prepared data matrix.

        Replace fitted attributes rather than modifying existing arrays
        or containers in place, so :meth:`fit` can restore them on failure.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # shared preprocessing
    # ------------------------------------------------------------------

    def _prepare_X(self, X):
        """Validate, center in float64, then coerce the result to float32.

        The column means are computed and subtracted in float64 (on the
        host) *before* the float32 cast.  Casting an uncentered matrix
        first would round each entry at the scale of the column offset
        (e.g. ULP ~ 0.06 at 1e6 in float32), silently destroying the
        within-column variation that the model is supposed to explain.
        ``x_offset_`` is stored as a float64 array.
        """
        try:
            X64 = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"X must be convertible to a float64 array, got "
                f"{type(X).__name__}: {exc}"
            ) from exc
        if X64.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional, got shape {tuple(X64.shape)}"
            )
        if not np.isfinite(X64).all():
            raise ValueError(
                "X must contain only finite values (found NaN or inf)"
            )
        if self.center:
            offset = X64.mean(axis=0)
            Xc = X64 - offset
        else:
            offset = np.zeros(X64.shape[1], dtype=np.float64)
            Xc = X64
        with np.errstate(over="ignore", invalid="ignore"):
            X32 = Xc.astype(np.float32)
        if not np.isfinite(X32).all():
            raise ValueError(
                "the (centered) X matrix overflows the float32 coercion; "
                "rescale the data (the estimation model is single precision)"
            )
        self.x_offset_ = offset.astype(np.float64, copy=False)
        return X32

    # ------------------------------------------------------------------
    # fitted accessors
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not hasattr(self, "_B_hat"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; call fit(X) first"
            )

    @property
    def B_hat_(self):
        """(p, K) point estimate of the loading matrix.

        Posterior mean over the kept draws for :class:`GibbsBFM`; the
        variational mean ``mu`` for :class:`CaviBFM`.  After
        :meth:`align_factors` (GibbsBFM), this is the posterior mean of
        the ALIGNED draws; the mean of the raw draws remains available
        as ``B_samples_.mean(axis=0)``.  For CaviBFM it always remains
        ``mu_``; the transformed mean is available as ``mu_aligned_``.
        """
        self._check_fitted()
        return self._B_hat

    @property
    def sigma2_hat_(self):
        """(p,) point estimate of the idiosyncratic variances.

        Posterior mean over the kept draws for :class:`GibbsBFM`; the
        inverse-gamma posterior mean beta / (alpha_n - 1) for
        :class:`CaviBFM`.
        """
        self._check_fitted()
        return self._sigma2_hat

    @property
    def Sigma_(self):
        """(p, p) estimated covariance B_hat B_hat^T + diag(sigma2_hat_)."""
        self._check_fitted()
        return self._B_hat @ self._B_hat.T + np.diag(self._sigma2_hat)

    # ------------------------------------------------------------------
    # factor-dimension selection: marginal credible intervals
    # ------------------------------------------------------------------
    def n_factors(self, level=0.95, *, interval="hpd", use_aligned=False):
        """Estimate the number of active factor columns.

        Entry-wise rule: for every loading entry B_jk, compute its
        smallest (highest-density) marginal credible interval at
        credibility ``level``; if 0 lies inside the interval the entry is
        treated as zero.  A factor column k is ACTIVE iff at least one
        entry j in it is nonzero by this rule, and the estimated factor
        count is the number of active columns:

            r_hat = #{ k : exists j with 0 notin CI(B_jk) }.

        GibbsBFM builds the intervals empirically from the posterior
        draws (``B_samples_``, or ``B_aligned_`` with
        ``use_aligned=True``); CaviBFM uses the closed-form Gaussian
        interval ``|mu_jk| > z * sqrt(V_jkk)`` of its variational
        posterior (for a symmetric Gaussian the highest-density interval
        is the equal-tailed one).

        Parameters
        ----------
        level : float, default 0.95
            Credibility level in (0, 1); larger is more conservative
            (more entries called zero).
        interval : {"hpd", "quantile"}, default "hpd"
            Interval construction for the empirical (Gibbs) path:
            "hpd" is the shortest interval covering ``level`` of the
            draws (robust to multimodal marginals, e.g. residual sign
            switching in the chain); "quantile" is the equal-tailed
            interval.  The Gaussian (CAVI) path is identical under both.
        use_aligned : bool, default False
            Gibbs: use ``B_aligned_`` instead of the raw ``B_samples_``
            (requires a successful :meth:`align_factors` first).
            CAVI: use ``mu_aligned_``/``V_aligned_`` instead of
            ``mu_``/``V_``.  Note that the marginal intervals are
            invariant to sign flips and column permutations but NOT to
            the varimax rotation: RSP's default rotation redistributes
            mass across columns, so the two variants answer slightly
            different questions (raw = the prior's coordinate system,
            aligned = the displayed orientation).

        Returns
        -------
        int

        Notes
        -----
        Unlike a point-estimate rule this uses the full marginal
        posterior, so a weak residual column whose draws straddle zero is
        correctly called inactive.  The draws are autocorrelated; the
        interval quality depends on mixing (see ``GibbsBFM.ess``).
        """
        return int(
            self.active_entries(level=level, interval=interval,
                                use_aligned=use_aligned).any(axis=0).sum()
        )

    def active_entries(self, level=0.95, *, interval="hpd", use_aligned=False):
        """(p, K) bool mask of active loading entries (see :meth:`n_factors`).

        Entry (j, k) is active iff 0 lies outside its smallest marginal
        credible interval at credibility ``level``.
        """
        self._check_fitted()
        if not (isinstance(level, (int, float))
                and np.isfinite(level) and 0.0 < level < 1.0):
            raise ValueError(f"level must be in (0, 1), got {level}")
        if interval not in ("hpd", "quantile"):
            raise ValueError(
                f"interval must be 'hpd' or 'quantile', got {interval!r}"
            )
        return self._credible_active(level, interval, use_aligned)

    def _credible_draws(self, use_aligned):
        """Gibbs path: the draw array to build empirical intervals from."""
        self._check_fitted()
        draws = getattr(self, "B_samples_", None)
        if draws is None:
            raise RuntimeError(
                f"{type(self).__name__} has no posterior draws for the "
                "credible-interval rule"
            )
        if use_aligned:
            aligned = getattr(self, "B_aligned_", None)
            if aligned is None:
                raise RuntimeError(
                    "use_aligned=True requires a successful "
                    "align_factors() call first"
                )
            return aligned
        return draws

    def _credible_active(self, level, interval, use_aligned):
        """Gibbs path: empirical marginal intervals over the draws."""
        draws = np.asarray(self._credible_draws(use_aligned), dtype=np.float64)
        S = draws.shape[0]
        if S < 2:
            raise ValueError(
                f"the credible rule needs at least 2 draws, got S = {S}"
            )
        if interval == "quantile":
            a = 1.0 - level
            lo, hi = np.quantile(draws, [a / 2.0, 1.0 - a / 2.0], axis=0)
        else:
            # HPD: shortest window covering m = ceil(level * S) sorted
            # draws (a window of m consecutive points is s[i+m-1] - s[i])
            s = np.sort(draws, axis=0)
            m = int(np.ceil(level * S))
            if m >= S:
                lo, hi = s[0], s[-1]
            else:
                w = s[m - 1:] - s[: S - m + 1]             # (S - m + 1, p, K)
                i = np.argmin(w, axis=0)                   # (p, K)
                lo = np.take_along_axis(s, i[None], axis=0)[0]
                hi = np.take_along_axis(s, (i + m - 1)[None], axis=0)[0]
        return (lo > 0) | (hi < 0)

    # ------------------------------------------------------------------
    # factor switching (RSP post-processing)
    # ------------------------------------------------------------------
    def align_factors(
        self,
        *,
        rotate=True,
        sp_path="elim",
        init="auto",
        reference=None,
        threshold=1e-6,
        max_iter=100,
        polar="qdwh",
        assignment_backend="auto",
        assignment_q_enum=None,
        sp_chunk_elems=None,
        precision="float32",
        verbose=False,
    ):
        """Align Gibbs draws or the VI loading mean using RSP.

        Runs the Rotation-Sign-Permutation algorithm implemented in
        :mod:`bfm._rsp` (Rey-Saez & Revuelta 2026).

        GibbsBFM aligns the kept ``B_samples_`` and updates ``B_hat_``
        to their aligned mean.  Consequently the plug-in ``Sigma_``
        can change, while ``Sigma_mc_`` and the raw draws used by
        ``ess()`` remain unchanged.

        CaviBFM supplies ``mu_[None, :, :]`` as one loading matrix,
        not an MCMC draw.  It exposes the rotated Gaussian means and
        covariances separately; ``B_hat_``, the fitted VI parameters,
        shrinkage moments, and ELBO remain unchanged.  These are
        representations for interpretation/comparison, not another
        optimum under the original coordinate-dependent shrinkage prior.

        Repeated calls always start from the original draws or VI mean
        and replace the previous alignment outputs.  A successful new
        ``fit`` discards them; a failed fit or alignment preserves them.

        Parameters
        ----------
        rotate, sp_path, init, reference, threshold, max_iter, polar,
        assignment_backend, assignment_q_enum, sp_chunk_elems, precision,
        verbose
            Passed through to :func:`bfm._rsp.rsp_exact` (see its
            docstring / the Factor_Switching.md notes for details).
            ``return_transforms`` is forced on for consistent score
            and covariance transforms.  ``assignment_backend="auto"``
            needs optax for K > 7 (install with ``pip install -e
            ".[rsp]"``).  The "scipy" backend needs scipy, which is NOT
            part of the ``rsp`` extra; install it separately
            (``pip install scipy``).

            ``reference`` coordinate system: with the default ``rotate=True``
            the reference is interpreted in the **varimax-rotated** coordinate
            system (the same system as the aligned draws); passing a reference
            in the original loading coordinates silently targets the wrong
            matrix, so pass ``rotate=False`` for such references (a
            UserWarning is emitted when both are supplied).

            ``precision="float64"`` enables JAX's process-wide x64 flag at the
            entry point and does not restore it afterwards; use ``"float32"``
            (default) unless a process-wide precision change is acceptable.

        Returns
        -------
        self
            New attributes:

            - ``rsp_result_``: the full :class:`bfm._rsp.RSPResult`
              (aligned draws, signs, permutations, loss histories,
              diagnostics, fit metadata, per-draw transforms);
            - ``B_aligned_``: (S, p, K) aligned loadings, with S=1 for VI.

            Gibbs scores are returned on request by
            :meth:`GibbsBFM.sample_factors`.  VI also exposes
            ``mu_aligned_``, ``V_aligned_``, ``W_aligned_``,
            ``Phi_aligned_``, and ``eta_aligned_ = W_aligned_.T``;
            see :class:`CaviBFM` for shapes and semantics.

        Raises
        ------
        RuntimeError
            If the estimator is unfitted or has no supported alignment input.
        FloatingPointError
            If RSP ended with status_code 4 or the transformed VI
              moments are non-finite: NO estimator attribute changes, the
              previous estimates (raw or from a previous successful
              alignment) are kept, and the failed run is attached to
              the exception as ``rsp_result`` for diagnostics.
        UserWarning
            If the run ended with status_code 3 (the exact objective
              increased beyond the numerical tolerance); the finite
              aligned results are still accepted.
        """
        draws = self._alignment_loadings()
        try:
            from ._rsp import rsp_exact
            from ._rsp.kernels.assignment import DEFAULT_Q_ENUM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "the RSP module bfm._rsp failed to import: "
                f"{exc}"
            ) from exc

        try:
            result = rsp_exact(
                draws,
                rotate=rotate,
                sp_path=sp_path,
                init=init,
                reference=reference,
                threshold=threshold,
                max_iter=max_iter,
                polar=polar,
                assignment_backend=assignment_backend,
                assignment_q_enum=(
                    DEFAULT_Q_ENUM if assignment_q_enum is None else assignment_q_enum
                ),
                sp_chunk_elems=sp_chunk_elems,
                precision=precision,
                return_transforms=True,
                verbose=verbose,
                metadata=self._alignment_metadata(draws),
            )
        except ImportError as exc:
            raise ImportError(
                f"the RSP alignment backend is missing a dependency "
                f"({exc}); install optax with: pip install -e \".[rsp]\" "
                "(optax is needed for K > 7); the \"scipy\" assignment "
                "backend needs scipy instead, which is not part of the "
                "rsp extra: pip install scipy"
            ) from exc

        status = int(result.status_code)
        if status == 4:
            # Zero estimator state change on failure: the failed run is
            # attached to the exception (exc.rsp_result) for diagnostics
            # and NO attribute is written.  A failed run must not
            # create alignment outputs or overwrite a successful one.
            exc = FloatingPointError(
                "rsp_exact ended with status_code 4 (non-finite "
                "objective, intermediate, or returned state); no "
                "estimator state was changed and the previous estimates "
                "are kept (the failed run is attached as exc.rsp_result)"
            )
            exc.rsp_result = result
            raise exc
        # Build and validate every output before publishing any of them.
        updates = self._alignment_outputs(result)
        if status == 3:
            warnings.warn(
                "rsp_exact ended with status_code 3 (the exact alignment "
                "objective increased beyond the numerical tolerance); "
                "accepting the finite aligned results -- inspect "
                "rsp_result_.diagnostics",
                UserWarning,
                stacklevel=2,
            )

        self.__dict__.update(updates)
        return self

    def _alignment_loadings(self):
        """Return original loadings with a leading alignment-sample axis."""
        self._check_fitted()
        draws = getattr(self, "B_samples_", None)
        if draws is None:
            raise RuntimeError(f"{type(self).__name__} has no alignment input")
        return draws

    def _alignment_metadata(self, draws):
        """Hashable fit metadata recorded on ``rsp_result_.metadata``.

        Only scalars and short scalar sequences: ``RSPResult`` keeps metadata
        in its pytree auxiliary data (see :mod:`bfm._rsp.types`).
        """
        meta = {
            "estimator": type(self).__name__,
            "K": int(self.K),
            "p": int(draws.shape[1]),
            "n_draws": int(draws.shape[0]),
            "center": bool(self.center),
        }
        n_samples = getattr(self, "n_samples_", None)
        if n_samples is not None:
            meta["n_samples"] = int(n_samples)
        for name in ("n_iter", "burn_in", "thin", "seed", "tol", "drop_budget"):
            value = getattr(self, name, None)
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, (int, np.integer)):
                meta[name] = int(value)
            elif isinstance(value, (float, np.floating)):
                meta[name] = float(value)
        return meta

    def _alignment_outputs(self, result):
        """Prepare alignment attributes without changing estimator state."""
        outputs = {
            "rsp_result_": result,
            "B_aligned_": np.asarray(result.aligned_loadings),
        }
        if self._ALIGN_UPDATES_B_HAT:
            outputs["_B_hat"] = np.asarray(result.mean_loadings)
        return outputs
