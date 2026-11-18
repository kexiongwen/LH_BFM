"""Rerun examples/demo2.ipynb at n=500, p=5000.

Same design and estimator settings as the notebook:
sparse Bernoulli(1/3) * U(0,1) loadings, heteroscedastic noise
sigma_j^2 ~ U(0.1, 1), r=5, K=50, a=10, c1=2.3, c2=1,
spectral-varimax initialization, float32.

Gibbs: 5000 sweeps, burn-in 2500, thin 1, seed 2026.
CAVI: absolute ELBO tolerance 0.1, drop_budget 0.5, max_iter=None.

X is drawn from the factor model x = eta B_0^T + noise after the same
NumPy seed stream that builds B_0 and sigma^2. That is the model the
notebook's multivariate_normal call samples, without factoring a
5000 x 5000 covariance.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", False)

from bfm import CaviBFM, GibbsBFM
from bfm._posterior import (
    covariance_sparsity_metrics,
    montecarlo_sigma_mean,
    sigma_mean,
    sigma_variance,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DATA_SEED = 12
SAMPLER_SEED = 2026
CAVI_INIT_SEED = 2026
N, P, K0 = 500, 5000, 5
K = 50
LINF_THRESHOLD = 0.1
COV_SPARSITY_THRESHOLD = 1e-2
# All 2500 retained draws are (2500, 5000, 50) and RSP was killed at the
# 14.6 GB cgroup limit. 1600 draws were killed as well. 800 equidistant
# draws fit and are what gibbs_alignment.json records.
GIBBS_ALIGN_DRAWS = 800
C_GIBBS, C_CAVI, C_TRUE, C_SHRUNK = "#2f6fd0", "#e07b39", "#b03a2e", "#c9d3df"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#5b6472",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e6eaf0",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
})


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{stamp} UTC] {message}", flush=True)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def write_summary(payload: dict) -> None:
    path = ROOT / "summary.json"
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    log(f"wrote {path}")


def generate_data():
    np.random.seed(DATA_SEED)
    B_0 = np.random.binomial(1, 1 / 3, (P, K0)) * np.random.rand(P, K0)
    s2_true = uniform.rvs(loc=0.1, scale=0.9, size=P)
    eta = np.random.randn(N, K0)
    X = eta @ B_0.T + np.random.randn(N, P) * np.sqrt(s2_true)
    return X, B_0, s2_true


def frobenius(a, b) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(diff, ord="fro"))


def principal_cosines(B_hat, B_0) -> list[float]:
    U_true = np.linalg.qr(np.asarray(B_0, dtype=np.float64))[0][:, :K0]
    U_post = np.linalg.qr(np.asarray(B_hat, dtype=np.float64))[0][:, :K0]
    cosines = np.linalg.svd(U_post.T @ U_true, compute_uv=False)
    return [float(c) for c in cosines]


def active_columns_hpd(draws, level=0.95) -> np.ndarray:
    """Column activity by the same HPD rule as BaseBFM, one column at a time."""
    draws = np.asarray(draws, dtype=np.float64)
    S, _, K_dim = draws.shape
    if S < 2:
        raise ValueError(f"the credible rule needs at least 2 draws, got S = {S}")
    m = int(np.ceil(level * S))
    active = np.zeros(K_dim, dtype=bool)
    for k in range(K_dim):
        column = draws[:, :, k]
        s = np.sort(column, axis=0)
        if m >= S:
            lo, hi = s[0], s[-1]
        else:
            width = s[m - 1:] - s[: S - m + 1]
            i = np.argmin(width, axis=0)
            lo = s[i, np.arange(s.shape[1])]
            hi = s[i + m - 1, np.arange(s.shape[1])]
        active[k] = bool(np.any((lo > 0) | (hi < 0)))
    return active


def column_numbers(mask: np.ndarray) -> list[int]:
    return [int(k) + 1 for k in np.flatnonzero(mask)]


def equidistant_indices(n_available: int, n_keep: int) -> np.ndarray:
    """Indices of an equidistant subset, including the first and last draws."""
    n_available = int(n_available)
    n_keep = min(int(n_keep), n_available)
    if n_keep < 1:
        raise ValueError(f"n_keep must be positive, got {n_keep}")
    return np.unique(np.rint(np.linspace(0, n_available - 1, n_keep)).astype(np.int64))


def savefig(fig, name: str) -> str:
    path = FIG / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"figure {path.name}")
    return str(path.relative_to(ROOT))


def main():
    summary: dict = {
        "demo": "demo2",
        "description": (
            "Sparse random loadings, heteroscedastic noise, "
            "rerun of examples/demo2.ipynb at n=500, p=5000."
        ),
        "n": N,
        "p": P,
        "K": K,
        "true_factors": K0,
        "data_seed": DATA_SEED,
        "chain_seed": SAMPLER_SEED,
        "cavi_init_seed": CAVI_INIT_SEED,
        "backend": jax.default_backend(),
        "data_generation": (
            "After np.random.seed(12), B_0 and sigma^2 follow the notebook. "
            "X = eta @ B_0.T + eps with independent standard normals, which "
            "is an exact draw from N(0, B_0 B_0.T + diag(sigma^2))."
        ),
        "figures": [],
    }
    log(f"backend={summary['backend']} devices={jax.devices()}")

    t0 = time.perf_counter()
    X, B_0, s2_true = generate_data()
    summary["loading_sparsity_exact_zeros"] = float((B_0 == 0).mean())
    summary["data_generation_seconds"] = time.perf_counter() - t0
    log(
        f"X {X.shape} sparsity={summary['loading_sparsity_exact_zeros']:.4f} "
        f"gen={summary['data_generation_seconds']:.2f}s"
    )
    Sig_true = B_0 @ B_0.T
    Sig_true[np.diag_indices(P)] += s2_true
    norm_Sig_true = float(np.linalg.norm(Sig_true, ord="fro"))
    summary["sigma_true_frobenius"] = norm_Sig_true

    fig, ax = plt.subplots(figsize=(11, 2.3), constrained_layout=True)
    im = ax.imshow(B_0.T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(f"True loading matrix $B_0$ ({K0} factors x {P} variables)", loc="left")
    ax.set_xlabel("variable index j")
    ax.set_ylabel("factor k")
    ax.set_yticks(np.arange(K0))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("loading")
    cb.outline.set_visible(False)
    summary["figures"].append(savefig(fig, "true_loadings.png"))

    log("fitting GibbsBFM")
    t0 = time.perf_counter()
    gs = GibbsBFM(
        K=K, a=10.0, c1=2.3, c2=1.0, center=True,
        init_method="spectral_varimax", init_seed=2026,
        n_iter=5000, burn_in=2500, seed=SAMPLER_SEED, verbose=True,
    ).fit(X)
    wall_gs = time.perf_counter() - t0
    summary["gibbs"] = {
        "dtype": str(gs.B_samples_.dtype),
        "kept_draws": int(gs.B_samples_.shape[0]),
        "scan_seconds": float(gs.runtime_sec_),
        "wall_seconds": float(wall_gs),
        "gamma_mode": gs.gamma_mode_,
        "chi2_normals": int(gs.chi2_normals_),
        "seed": int(gs.seed_),
        "initialization_method": gs.initialization_method_,
        "initialization_info": gs.initialization_info_,
        "n_iter": gs.n_iter,
        "burn_in": gs.burn_in,
        "thin": gs.thin,
    }
    log(
        f"Gibbs kept={gs.B_samples_.shape[0]} scan={gs.runtime_sec_:.1f}s "
        f"wall={wall_gs:.1f}s"
    )
    write_summary(summary)

    log("fitting CaviBFM until ELBO tolerance")
    t0 = time.perf_counter()
    vi = CaviBFM(
        K=K, a=10.0, c1=2.3, c2=1.0, center=True,
        init_method="spectral_varimax", init_seed=CAVI_INIT_SEED,
        precision="float32", max_iter=None, tol=1e-1, verbose=True,
    ).fit(X)
    wall_vi = time.perf_counter() - t0
    summary["cavi"] = {
        "dtype": str(vi.dtype_),
        "precision": vi.precision_,
        "initialization_method": vi.initialization_method_,
        "initialization_info": vi.initialization_info_,
        "n_iter": int(vi.n_iter_),
        "stop_reason": vi.stop_reason_,
        "converged": bool(vi.converged_),
        "elbo_initial": float(vi.elbo_[0]),
        "elbo_final": float(vi.elbo_[-1]),
        "elbo_increase": float(vi.elbo_[-1] - vi.elbo_[0]),
        "initialization_seconds": float(vi.initialization_sec_),
        "compile_seconds": float(vi.compile_sec_),
        "loop_seconds": float(vi.runtime_sec_),
        "wall_seconds": float(wall_vi),
    }
    log(
        f"CAVI iter={vi.n_iter_} stop={vi.stop_reason_} "
        f"elbo {vi.elbo_[0]:.3f} -> {vi.elbo_[-1]:.3f} wall={wall_vi:.1f}s"
    )
    np.save(ROOT / "cavi_elbo.npy", np.asarray(vi.elbo_, dtype=np.float64))
    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    ax.plot(np.arange(len(vi.elbo_)), vi.elbo_, color=C_CAVI, linewidth=1.5)
    ax.set(xlabel="Completed CAVI sweeps", ylabel="ELBO",
           title="CAVI ELBO — spectral-varimax initialization")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    summary["figures"].append(savefig(fig, "cavi_elbo.png"))
    write_summary(summary)

    log("factor counts and column norms")
    gs_active = active_columns_hpd(gs.B_samples_)
    r_gs = int(gs_active.sum())
    r_vi = int(vi.n_factors(method="credible"))
    r_vi_linf = int(vi.n_factors(method="linf", threshold=LINF_THRESHOLD))
    vi_active = vi.active_entries(method="credible").any(axis=0)
    vi_linf_active = vi.active_entries(method="linf", threshold=LINF_THRESHOLD).any(axis=0)
    colnorm_gs = np.linalg.norm(gs.B_samples_, axis=1).mean(0)
    colnorm_vi = np.linalg.norm(vi.mu_, axis=0)
    summary["factor_counts_before_alignment"] = {
        "gibbs_credible": r_gs,
        "gibbs_credible_columns_1based": column_numbers(gs_active),
        "cavi_credible": r_vi,
        "cavi_credible_columns_1based": column_numbers(vi_active),
        "cavi_linf": r_vi_linf,
        "cavi_linf_threshold": LINF_THRESHOLD,
        "cavi_linf_columns_1based": column_numbers(vi_linf_active),
        "true_factors": K0,
    }
    summary["column_l2_norms"] = {
        "gibbs_posterior_mean": colnorm_gs.tolist(),
        "cavi_variational_mean": colnorm_vi.tolist(),
    }
    log(
        f"factors gibbs={r_gs} cavi_credible={r_vi} cavi_linf={r_vi_linf} true={K0}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), constrained_layout=True, sharey=True)
    ymax = max(float(colnorm_gs.max()), float(colnorm_vi.max())) * 1.15

    def draw_colnorms(ax, norms, count_label, color, title):
        xs = np.arange(1, len(norms) + 1)
        ax.bar(
            xs, norms, width=0.82,
            color=np.where(xs <= K0, color, C_SHRUNK),
            edgecolor="white", linewidth=0.4,
        )
        ax.axvline(K0 + 0.5, color=C_TRUE, ls="--", lw=1.0)
        ax.set_title(f"{title}: column norms\n{count_label}", loc="left")
        ax.set_xlabel("factor column k")
        ax.set_ylim(0, ymax)
        ax.set_xticks(np.arange(5, len(norms) + 1, 5))

    draw_colnorms(axes[0], colnorm_gs, f"credible r={r_gs}", C_GIBBS, "GibbsBFM")
    draw_colnorms(
        axes[1], colnorm_vi,
        f"credible r={r_vi}; linf r={r_vi_linf} (threshold={LINF_THRESHOLD:g})",
        C_CAVI, "CaviBFM",
    )
    summary["figures"].append(savefig(fig, "column_norms.png"))

    log("noise recovery")
    corr_gs = float(np.corrcoef(s2_true, gs.sigma2_hat_)[0, 1])
    corr_vi = float(np.corrcoef(s2_true, vi.sigma2_hat_)[0, 1])
    err_gs = float(np.abs(gs.sigma2_hat_ - s2_true).mean())
    err_vi = float(np.abs(vi.sigma2_hat_ - s2_true).mean())
    summary["noise"] = {
        "gibbs_correlation": corr_gs,
        "cavi_correlation": corr_vi,
        "gibbs_mean_absolute_error": err_gs,
        "cavi_mean_absolute_error": err_vi,
    }
    log(f"noise corr gibbs={corr_gs:.4f} cavi={corr_vi:.4f} mae {err_gs:.4f} {err_vi:.4f}")
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    ax.scatter(s2_true, gs.sigma2_hat_, s=8, alpha=0.35, color=C_GIBBS,
               edgecolors="none", label=f"GibbsBFM (r = {corr_gs:.3f})")
    ax.scatter(s2_true, vi.sigma2_hat_, s=8, alpha=0.35, color=C_CAVI,
               marker="x", linewidths=0.6, label=f"CaviBFM (r = {corr_vi:.3f})")
    lim = [0, 1.05 * max(float(s2_true.max()), float(gs.sigma2_hat_.max()),
                         float(vi.sigma2_hat_.max()))]
    ax.plot(lim, lim, color=C_TRUE, ls="--", lw=1.0, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"true $\sigma_j^2$")
    ax.set_ylabel(r"estimated $\sigma_j^2$")
    ax.set_title("Recovered noise variances", loc="left")
    ax.legend(loc="upper left")
    summary["figures"].append(savefig(fig, "noise_recovery.png"))

    log("covariance recovery")
    # p*p = 25e6 exceeds the estimator property cap of 20e6 entries.
    # The underlying moment functions accept max_entries=None.
    t0 = time.perf_counter()
    Sig_gs_plugin = gs.Sigma_
    Sig_vi_plugin = vi.Sigma_
    Sig_gs_mean = montecarlo_sigma_mean(
        gs.B_samples_, gs.sigma2_samples_, max_entries=None,
    )
    Sig_vi_mean = sigma_mean(
        vi.mu_, vi.V_, vi.alpha_n_, vi.beta_, max_entries=None,
    )
    labels = {
        "GibbsBFM plug-in": Sig_gs_plugin,
        "GibbsBFM Sigma_mean_": Sig_gs_mean,
        "CaviBFM plug-in": Sig_vi_plugin,
        "CaviBFM Sigma_mean_": Sig_vi_mean,
    }
    covariance = {}
    for name, est in labels.items():
        absolute = frobenius(est, Sig_true)
        covariance[name] = {
            "frobenius": absolute,
            "relative_frobenius": absolute / norm_Sig_true,
        }
        log(f"  {name}: rel={absolute / norm_Sig_true:.6f}")
    cos_gs = principal_cosines(gs.B_hat_, B_0)
    cos_vi = principal_cosines(vi.B_hat_, B_0)
    summary["covariance_before_alignment"] = covariance
    summary["principal_angle_cosines"] = {"GibbsBFM": cos_gs, "CaviBFM": cos_vi}
    summary["covariance_seconds"] = time.perf_counter() - t0
    log(f"principal cosines gibbs={np.round(cos_gs, 4)} cavi={np.round(cos_vi, 4)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    names = ["GibbsBFM", "CaviBFM"]
    plug = [covariance["GibbsBFM plug-in"]["relative_frobenius"],
            covariance["CaviBFM plug-in"]["relative_frobenius"]]
    post = [covariance["GibbsBFM Sigma_mean_"]["relative_frobenius"],
            covariance["CaviBFM Sigma_mean_"]["relative_frobenius"]]
    x = np.arange(2)
    width = 0.36
    axes[0].bar(x - width / 2, plug, width, color=[C_GIBBS, C_CAVI], label="plug-in")
    axes[0].bar(x + width / 2, post, width, color=[C_GIBBS, C_CAVI], alpha=0.45,
                hatch="//", label="posterior mean")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("relative Frobenius error")
    axes[0].set_title("Covariance recovery", loc="left")
    axes[0].legend(loc="upper left")
    axes[1].plot(np.arange(1, K0 + 1), cos_gs, "o-", color=C_GIBBS, label="GibbsBFM")
    axes[1].plot(np.arange(1, K0 + 1), cos_vi, "s--", color=C_CAVI, label="CaviBFM")
    axes[1].axhline(1.0, color=C_TRUE, lw=0.9, alpha=0.8)
    axes[1].set_xlabel("principal direction")
    axes[1].set_ylabel("principal-angle cosine")
    axes[1].set_title("Factor-subspace alignment", loc="left")
    axes[1].legend(loc="lower right")
    summary["figures"].append(savefig(fig, "covariance_and_subspace.png"))

    log("covariance sparsity and diagonal uncertainty")
    sparsity = {}
    for name, mean in (("GibbsBFM", Sig_gs_mean), ("CaviBFM", Sig_vi_mean)):
        metrics = covariance_sparsity_metrics(
            mean, Sig_true, threshold=COV_SPARSITY_THRESHOLD,
        )
        sparsity[name] = metrics
        log(
            f"  {name} FDR={metrics['fdr']:.6f} FNDR={metrics['fndr']:.6f} "
            f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}"
        )
    summary["covariance_sparsity"] = {
        "threshold": COV_SPARSITY_THRESHOLD,
        "metrics": sparsity,
        "note": (
            "Sigma_mean_ on the estimator raises for p=5000 because "
            "p*p exceeds the default 2e7-entry cap. Moments here were "
            "computed with max_entries=None."
        ),
    }
    gs_sd_diag = np.sqrt(gs.Sigma_diag_var_)
    vi_var = sigma_variance(
        vi.mu_, vi.V_, vi.alpha_n_, vi.beta_, max_entries=None,
    )
    vi_sd_diag = np.sqrt(np.diag(vi_var))
    summary["posterior_sd_diag"] = {
        "gibbs_mean": float(gs_sd_diag.mean()),
        "cavi_mean": float(vi_sd_diag.mean()),
    }
    pair_var = float(gs.cov_Sigma([((0, 1), (0, 1))])[0])
    pair_cov = float(gs.cov_Sigma([((0, 1), (2, 3))])[0])
    summary["gibbs_entry_covariance"] = {
        "var_sigma_01": pair_var,
        "cov_sigma_01_sigma_23": pair_cov,
        "draws": int(gs.B_samples_.shape[0]),
    }
    log(
        f"diag SD mean gibbs={gs_sd_diag.mean():.4f} cavi={vi_sd_diag.mean():.4f}"
    )
    # Free the large covariance matrices before ESS / alignment.
    del Sig_gs_plugin, Sig_vi_plugin, Sig_gs_mean, Sig_vi_mean, vi_var, Sig_true

    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    ax.plot(gs_sd_diag, color=C_GIBBS, lw=0.6, label="Gibbs")
    ax.plot(vi_sd_diag, color=C_CAVI, lw=0.6, alpha=0.85, label="CAVI")
    ax.set_xlabel("variable index j")
    ax.set_ylabel(r"posterior SD of $\Sigma_{jj}$")
    ax.set_title("Diagonal posterior uncertainty", loc="left")
    ax.legend(loc="upper right")
    summary["figures"].append(savefig(fig, "sigma_diag_sd.png"))
    write_summary(summary)

    log("ESS on sigma2 and a column-stratified loading subsample")
    try:
        rng = np.random.default_rng(0)
        var_idx = np.sort(rng.choice(P, size=40, replace=False))
        # Full per-entry ESS of B is 250000 series of length 2500.
        # Loading ESS uses a fixed 40-variable block. Noise ESS uses every
        # variable, paired with one loading column so the (S, p) layouts match.
        from bfm._gs.ESS import ess as ess_fn

        ess_b = ess_fn(gs.B_samples_[:, var_idx, :])
        ess_s = ess_fn(gs.B_samples_[:, :, :1], gs.sigma2_samples_)
        b = np.asarray(ess_b["B"], dtype=np.float64)
        s2 = np.asarray(ess_s["sigma2"], dtype=np.float64)

        def pack(a):
            return {
                "min": float(np.min(a)),
                "median": float(np.median(a)),
                "mean": float(np.mean(a)),
                "max": float(np.max(a)),
            }

        summary["ess"] = {
            "method": "bulk",
            "B_subsample_variables_0based": var_idx.tolist(),
            "B_subsample": pack(b),
            "sigma2_all_variables": pack(s2),
            "n_draws": int(gs.B_samples_.shape[0]),
            "note": (
                "Loading ESS is over 40 variables x 50 factors "
                "(2000 entries), stratified by a fixed random subset. "
                "Noise ESS uses every variable."
            ),
        }
        log(
            f"ESS B subsample min={summary['ess']['B_subsample']['min']:.1f} "
            f"median={summary['ess']['B_subsample']['median']:.1f}; "
            f"sigma2 min={summary['ess']['sigma2_all_variables']['min']:.1f} "
            f"median={summary['ess']['sigma2_all_variables']['median']:.1f}"
        )
    except Exception as exc:
        summary["ess_error"] = f"{type(exc).__name__}: {exc}"
        log(f"ESS failed: {summary['ess_error']}")
        traceback.print_exc()
    write_summary(summary)

    trace = gs.sigma2_samples_.mean(axis=1)
    window = max(1, len(trace) // 25)
    rolling = np.convolve(trace, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    ax.plot(trace, color=C_GIBBS, lw=0.6, alpha=0.55, label="per-draw mean")
    ax.plot(np.arange(window - 1, len(trace)), rolling, color="#123f7a",
            lw=1.6, label=f"rolling mean (window {window})")
    ax.axhline(float(s2_true.mean()), color=C_TRUE, ls="--", lw=1.0,
               label=f"truth mean = {float(s2_true.mean()):.2f}")
    ax.set_xlabel("kept sweep")
    ax.set_ylabel(r"mean $\sigma_j^2$")
    ax.set_title("Trace of the idiosyncratic variance level", loc="left")
    ax.legend(loc="upper right", ncol=3)
    summary["figures"].append(savefig(fig, "sigma2_trace.png"))

    log("CAVI RSP alignment")
    try:
        t0 = time.perf_counter()
        vi.align_factors(precision="float32")
        summary["cavi_alignment_seconds"] = time.perf_counter() - t0
        summary["cavi_alignment"] = {
            "status": int(vi.rsp_result_.status_code),
            "stop": vi.rsp_result_.diagnostics.get("stop_reason"),
            "iterations": int(vi.rsp_result_.n_iter),
            "loss_final": float(vi.rsp_result_.loss_final),
            "credible_aligned": int(vi.n_factors(method="credible", use_aligned=True)),
            "linf_aligned": int(vi.n_factors(
                method="linf", threshold=LINF_THRESHOLD, use_aligned=True,
            )),
        }
        log(f"CAVI alignment {summary['cavi_alignment']}")
    except Exception as exc:
        summary["cavi_alignment_error"] = f"{type(exc).__name__}: {exc}"
        log(f"CAVI alignment failed: {summary['cavi_alignment_error']}")
        traceback.print_exc()
    write_summary(summary)

    log("saving Gibbs draws and releasing them before RSP")
    draw_path = Path("/tmp/demo2_n500_p5000_B.npy")
    np.save(draw_path, np.ascontiguousarray(gs.B_samples_))
    del gs, vi, X, B_0, s2_true
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    log(f"saved {draw_path} ({draw_path.stat().st_size / 1e9:.2f} GB)")

    align_out = ROOT / "gibbs_alignment.json"
    log(f"Gibbs RSP alignment of {GIBBS_ALIGN_DRAWS} equidistant draws in a fresh process")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--align",
         str(draw_path), str(align_out), str(GIBBS_ALIGN_DRAWS)],
        check=False,
    )
    summary["gibbs_alignment_seconds"] = time.perf_counter() - t0
    summary["gibbs_alignment_returncode"] = int(proc.returncode)
    if align_out.is_file():
        summary["gibbs_alignment"] = json.loads(align_out.read_text())
        log(f"Gibbs alignment {summary['gibbs_alignment']}")
    else:
        summary["gibbs_alignment_error"] = (
            f"alignment process exited {proc.returncode} without a result"
        )
        log(summary["gibbs_alignment_error"])
    write_summary(summary)
    log("done")


def align_saved(src: str, dest: str, n_keep: int = GIBBS_ALIGN_DRAWS) -> None:
    """Align an equidistant subset of saved Gibbs loadings.

    The full retained array does not fit in the 14.6 GB cgroup. Only the
    subset is copied off the memory map before RSP runs.
    """
    from bfm._rsp import rsp_exact

    mapped = np.load(src, mmap_mode="r")
    n_available = int(mapped.shape[0])
    index = equidistant_indices(n_available, n_keep)
    draws = np.array(mapped[index], dtype=np.float32, copy=True)
    del mapped
    gc.collect()
    log(
        f"aligning {draws.shape[0]} of {n_available} draws "
        f"{draws.shape} {draws.dtype}"
    )
    t0 = time.perf_counter()
    result = rsp_exact(draws, precision="float32", return_transforms=True)
    aligned = np.asarray(result.aligned_loadings)
    active = active_columns_hpd(aligned)
    mean = np.asarray(result.mean_loadings)
    diagnostics = result.diagnostics
    stop = diagnostics.get("stop_reason") if isinstance(diagnostics, dict) else str(diagnostics)
    payload = {
        "status": int(result.status_code),
        "stop": stop,
        "iterations": int(result.n_iter),
        "loss_final": float(result.loss_final),
        "seconds": time.perf_counter() - t0,
        "n_draws_available": n_available,
        "n_draws_aligned": int(draws.shape[0]),
        "draw_index_rule": "equidistant linspace over the retained draws",
        "draw_index_first": int(index[0]),
        "draw_index_last": int(index[-1]),
        "reason_for_subset": (
            "Aligning all retained draws of shape (5000, 50) was killed at "
            "the 14.6 GB cgroup limit, as was a 1600-draw subset. "
            f"This uses {int(draws.shape[0])} equidistant retained draws."
        ),
        "credible_aligned": int(active.sum()),
        "credible_aligned_columns_1based": column_numbers(active),
        "column_linf_aligned": np.max(np.abs(mean), axis=0).tolist(),
    }
    Path(dest).write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    log(f"alignment wrote {dest}")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--align":
            keep = int(sys.argv[4]) if len(sys.argv) > 4 else GIBBS_ALIGN_DRAWS
            align_saved(sys.argv[2], sys.argv[3], keep)
        else:
            main()
    except Exception:
        traceback.print_exc()
        raise
