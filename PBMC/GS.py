"""Fit the fixed K=50 Gibbs workflow and its 20-column, 9-cluster representation."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import arviz as az
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bfm import GibbsBFM

if __package__:
    from .CAVI import _diagnostic_jsonable, load_baseline_input
    from .data_processing import _jsonable, _software_versions
    from .downstream import DownstreamConfig, run_downstream, write_top_markers
else:
    from CAVI import _diagnostic_jsonable, load_baseline_input
    from data_processing import _jsonable, _software_versions
    from downstream import DownstreamConfig, run_downstream, write_top_markers

__all__ = [
    "GSConfig",
    "GSResult",
    "conditional_factor_mean",
    "run_gs",
    "run_gs_from_baseline",
    "save_gs_result",
    "save_gs_figures",
    "save_top_markers",
]

_F32 = jnp.float32
N_SCORE_COLUMNS = 20


@dataclass(frozen=True)
class GSConfig:
    K: int = 50
    a: float = 10.0
    c1: float = 2.0
    c2: float = 1.0
    a_sigma: float = 1.0
    b_sigma: float = 1.0
    n_iter: int = 20000
    burn_in: int = 10000
    thin: int = 10
    seed: int = 2026
    batch_size: int = 32
    n_neighbors: int = 10
    leiden_resolution: float = 0.45
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.K < 1 or self.thin < 1 or self.batch_size < 1:
            raise ValueError("K, thin, and batch_size must be positive")
        if not 0 <= self.burn_in < self.n_iter:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_iter")
        if (self.n_iter - self.burn_in) // self.thin < 1:
            raise ValueError("the run would retain no draws")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.n_neighbors < 1 or self.leiden_resolution <= 0:
            raise ValueError("n_neighbors and leiden_resolution must be positive")


@dataclass
class GSResult:
    adata: ad.AnnData
    estimator: GibbsBFM
    config: GSConfig
    name: str
    diagnostics: dict
    batch_means: np.ndarray
    batch_indices: np.ndarray


@jax.jit
def _aligned_conditional_mean_batch(Xc, loadings, variances, transforms):
    def one_draw(args):
        B, sigma2, transform = args
        scale = jnp.sqrt(jnp.maximum(sigma2, jnp.finfo(_F32).tiny ** 0.5))
        C = B.T / scale
        precision = C @ C.T + jnp.eye(B.shape[1], dtype=_F32)
        rhs = C @ (Xc.T / scale[:, None])
        factor = jnp.linalg.cholesky(precision)
        conditional_mean = jsla.cho_solve((factor, True), rhs)
        return transform.T @ conditional_mean

    return jax.lax.map(one_draw, (loadings, variances, transforms))


def conditional_factor_mean(
    X,
    loadings,
    variances,
    offset,
    *,
    transforms=None,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    loadings = np.asarray(loadings)
    variances = np.asarray(variances)
    if loadings.ndim != 3 or min(loadings.shape) < 1:
        raise ValueError("loadings must have nonempty shape (S, p, K)")
    S, p, K = loadings.shape
    if variances.shape != (S, p):
        raise ValueError(f"variances must have shape {(S, p)}")
    if not np.isfinite(loadings).all() or not np.isfinite(variances).all():
        raise ValueError("loadings and variances must be finite")
    if np.any(variances <= 0):
        raise ValueError("variances must be strictly positive")
    X64 = np.asarray(X, dtype=np.float64)
    offset64 = np.asarray(offset, dtype=np.float64)
    if X64.ndim != 2 or X64.shape[0] < 1 or X64.shape[1] != p:
        raise ValueError(f"X must have shape (n, {p}), got {X64.shape}")
    if offset64.shape != (p,) or not np.isfinite(offset64).all():
        raise ValueError("offset must be finite and have the loading-row shape")
    if not np.isfinite(X64).all():
        raise ValueError("X must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        Xc = (X64 - offset64).astype(np.float32)
    if not np.isfinite(Xc).all():
        raise ValueError("centering X overflowed the float32 coercion")
    if transforms is None:
        transforms = np.broadcast_to(np.eye(K, dtype=np.float32), (S, K, K))
    else:
        transforms = np.asarray(transforms)
        if transforms.shape != (S, K, K):
            raise ValueError(f"transforms must have shape {(S, K, K)}")
        if not np.isfinite(transforms).all():
            raise ValueError("transforms must be finite")
    n = X64.shape[0]
    n_batches = (S + int(batch_size) - 1) // int(batch_size)
    batch_means = np.empty((n_batches, n, K), dtype=np.float64)
    batch_indices = np.empty((n_batches, 2), dtype=np.int64)
    total = np.zeros((K, n), dtype=np.float64)
    Xc_j = jnp.asarray(Xc)
    for batch_index, start in enumerate(range(0, S, int(batch_size))):
        stop = min(start + int(batch_size), S)
        aligned = np.asarray(_aligned_conditional_mean_batch(
            Xc_j,
            jnp.asarray(loadings[start:stop]),
            jnp.asarray(variances[start:stop]),
            jnp.asarray(transforms[start:stop]),
        ))
        if not np.isfinite(aligned).all():
            raise FloatingPointError(f"conditional means became non-finite in batch {batch_index}")
        batch_total = aligned.sum(axis=0, dtype=np.float64)
        total += batch_total
        batch_means[batch_index] = (batch_total / (stop - start)).T
        batch_indices[batch_index] = (start, stop)
    return (total / S).T, batch_means, batch_indices


def _rsp_summary(estimator: GibbsBFM, warning_messages: list[str]) -> dict:
    rsp = estimator.rsp_result_
    n_iter = int(rsp.n_iter)
    return {
        "applied": True,
        "status_code": int(rsp.status_code),
        "converged": bool(rsp.converged),
        "n_iter": n_iter,
        "stop_reason": str(rsp.diagnostics["stop_reason"]),
        "nonfinite_fields": list(rsp.diagnostics["nonfinite_fields"]),
        "loss_initial": float(rsp.loss_initial),
        "loss_final": float(rsp.loss_final),
        "loss_history": np.asarray(rsp.loss_history[:n_iter]).copy(),
        "loss_history_post": np.asarray(rsp.loss_history_post[:n_iter]).copy(),
        "varimax_n_iter": np.asarray(rsp.varimax_n_iter).copy(),
        "warnings": warning_messages,
    }


def _ess_summary(values: dict[str, np.ndarray], n_draws: int) -> dict:
    return {
        key: {
            "min": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "max": float(array.max()),
            "n_draws": int(n_draws),
        }
        for key, array in values.items()
    }


def _sampling_diagnostics(estimator: GibbsBFM) -> dict:
    B = estimator.B_samples_
    sigma2 = estimator.sigma2_samples_
    S = B.shape[0]
    bulk = estimator.ess(method="bulk")
    tail = estimator.ess(method="tail", prob=0.05)
    bulk_summary = _ess_summary(bulk, S)
    tail_summary = _ess_summary(tail, S)
    mcse_B = np.asarray(az.mcse(np.asarray(estimator.B_aligned_)[None], method="mean"))
    mcse_sigma2 = np.asarray(az.mcse(sigma2[None], method="mean"))
    covariance_diag = np.einsum("spk,spk->sp", B, B) + sigma2
    mcse_covariance = np.asarray(az.mcse(covariance_diag[None], method="mean"))
    near_constant = np.std(B, axis=0) <= np.finfo(np.float32).eps
    return {
        "bulk_summary": bulk_summary,
        "tail_summary": tail_summary,
        "bulk_B": bulk["B"],
        "bulk_sigma2": bulk["sigma2"],
        "tail_B": tail["B"],
        "tail_sigma2": tail["sigma2"],
        "mcse_B_aligned": mcse_B,
        "mcse_sigma2": mcse_sigma2,
        "mcse_covariance_diag": mcse_covariance,
        "near_constant_count": int(near_constant.sum()),
        "review_required": bool(
            min(bulk_summary["B"]["min"], bulk_summary["sigma2"]["min"]) < 400
        ),
    }


def run_gs(
    adata: ad.AnnData,
    X: np.ndarray,
    hvg_names: np.ndarray,
    *,
    config: GSConfig | None = None,
) -> GSResult:
    config = config or GSConfig()
    if X.shape != (adata.n_obs, len(hvg_names)):
        raise ValueError("X_scaled does not match the baseline cells and HVGs")
    estimator = GibbsBFM(
        K=config.K,
        a=config.a,
        c1=config.c1,
        c2=config.c2,
        a_sigma=config.a_sigma,
        b_sigma=config.b_sigma,
        center=False,
        n_iter=config.n_iter,
        burn_in=config.burn_in,
        thin=config.thin,
        seed=config.seed,
        graph=False,
        verbose=False,
    )
    start = time.perf_counter()
    estimator.fit(X)
    fit_wall_sec = time.perf_counter() - start
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        estimator.align_factors()
        rsp_wall_sec = time.perf_counter() - start
    rsp = _rsp_summary(estimator, [str(item.message) for item in caught])
    transforms = estimator.rsp_result_.transforms

    start = time.perf_counter()
    scores, batch_means, batch_indices = conditional_factor_mean(
        X,
        estimator.B_samples_,
        estimator.sigma2_samples_,
        estimator.x_offset_,
        transforms=transforms,
        batch_size=config.batch_size,
    )
    score_wall_sec = time.perf_counter() - start
    S = estimator.B_samples_.shape[0]
    fit = {
        "n_iter": int(config.n_iter),
        "burn_in": int(config.burn_in),
        "thin": int(config.thin),
        "n_kept": int(S),
        "retained_sweeps": config.burn_in + config.thin * np.arange(1, S + 1),
        "seed": int(estimator.seed_),
        "runtime_sec": float(estimator.runtime_sec_),
        "fit_wall_sec": float(fit_wall_sec),
        "rsp_wall_sec": float(rsp_wall_sec),
        "score_wall_sec": float(score_wall_sec),
        "gamma_mode": str(estimator.gamma_mode_),
        "chi2_normals": int(estimator.chi2_normals_),
        "backend": str(jax.default_backend()),
    }
    sampling = _sampling_diagnostics(estimator)
    selected_columns = np.arange(N_SCORE_COLUMNS)
    loading_column_norm = np.linalg.norm(estimator.B_hat_, axis=0)
    scores = scores * loading_column_norm
    batch_means = batch_means * loading_column_norm
    scores = scores[:, selected_columns]
    batch_means = batch_means[:, :, selected_columns]
    active_mask = np.asarray(
        estimator.active_entries(level=0.95, interval="hpd", use_aligned=True)
    )
    singular_values = np.linalg.svd(scores, compute_uv=False)
    name = "gibbs_scaled_norm_d20"
    representation = f"X_{name}"
    adata.obsm[representation] = scores
    run = run_downstream(
        adata,
        representation=representation,
        name=name,
        config=DownstreamConfig(
            n_pcs=N_SCORE_COLUMNS,
            n_neighbors=config.n_neighbors,
            leiden_resolution=config.leiden_resolution,
            random_state=config.random_state,
        ),
    )
    comparison = {}
    for label, key in (
        ("cavi", "leiden_cavi_scaled_norm_d20"),
        ("pca_scaled", "leiden_scaled"),
    ):
        if key in adata.obs:
            reference = adata.obs[key].astype(str)
            current = adata.obs[run["leiden"]].astype(str)
            comparison[label] = {
                "ari": float(adjusted_rand_score(reference, current)),
                "nmi": float(adjusted_mutual_info_score(reference, current)),
            }
    diagnostics = {
        "fit": fit,
        "sampling": sampling,
        "rsp": rsp,
        "active_mask": active_mask,
        "active_factor_count": int(active_mask.any(axis=0).sum()),
        "selected_active_factor_count": int(
            active_mask[:, selected_columns].any(axis=0).sum()
        ),
        "selected_columns": selected_columns,
        "score_singular_values": singular_values,
        "score_weighting": "loading_column_l2_norm",
        "loading_column_norm": loading_column_norm,
        "selected_loading_column_norm": loading_column_norm[selected_columns],
        "conditional_mean": {
            "batch_size": int(config.batch_size),
            "n_batches": int(batch_indices.shape[0]),
            "score_shape": list(scores.shape),
            "full_score_array_allocated": False,
            "aligned": True,
        },
        "comparison": comparison,
        "formal_complete": bool(rsp["status_code"] == 1),
    }
    adata.uns[name] = {
        "config_json": json.dumps(_jsonable(asdict(config)), sort_keys=True),
        "input_matrix": "X_scaled",
        "hvg_names": np.asarray(hvg_names, dtype=str),
        "diagnostics": diagnostics,
        "run": run,
        "software": _software_versions(),
    }
    return GSResult(adata, estimator, config, name, diagnostics, batch_means, batch_indices)


def run_gs_from_baseline(
    path: str | Path, *, config: GSConfig | None = None
) -> GSResult:
    adata, X, hvg_names = load_baseline_input(path)
    return run_gs(adata, X, hvg_names, config=config)


def save_gs_result(
    result: GSResult,
    output: str | Path,
    model_output: str | Path | None = None,
) -> tuple[Path, Path]:
    output = Path(output)
    if output.suffix.lower() != ".h5ad":
        raise ValueError(f"output must end in .h5ad, got {output}")
    model_output = (
        output.with_name(f"{output.stem}_model.npz")
        if model_output is None else Path(model_output)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    estimator = result.estimator
    sampling = result.diagnostics["sampling"]
    diagnostics_json = json.dumps(
        _diagnostic_jsonable({
            "config": asdict(result.config),
            "diagnostics": result.diagnostics,
        }),
        sort_keys=True,
    )
    rsp = estimator.rsp_result_
    arrays = {
        "B_samples": estimator.B_samples_,
        "sigma2_samples": estimator.sigma2_samples_,
        "x_offset": estimator.x_offset_,
        "B_posterior_mean": estimator.B_hat_,
        "D_last": estimator.raw_output_["D_last"],
        "score_mean": result.adata.obsm[f"X_{result.name}"],
        "batch_means": result.batch_means,
        "batch_indices": result.batch_indices,
        "active_mask": result.diagnostics["active_mask"],
        "selected_columns": result.diagnostics["selected_columns"],
        "selected_loading_column_norm": result.diagnostics["selected_loading_column_norm"],
        "loading_column_norm": result.diagnostics["loading_column_norm"],
        "score_singular_values": result.diagnostics["score_singular_values"],
        "bulk_B": sampling["bulk_B"],
        "bulk_sigma2": sampling["bulk_sigma2"],
        "tail_B": sampling["tail_B"],
        "tail_sigma2": sampling["tail_sigma2"],
        "mcse_B_aligned": sampling["mcse_B_aligned"],
        "mcse_sigma2": sampling["mcse_sigma2"],
        "mcse_covariance_diag": sampling["mcse_covariance_diag"],
        "hvg_names": result.adata.uns[result.name]["hvg_names"],
        "cell_names": result.adata.obs_names.to_numpy(dtype=str),
        "diagnostics_json": diagnostics_json,
        "B_aligned_mean": estimator.B_hat_,
        "transforms": rsp.transforms,
        "signs": rsp.signs,
        "permutations": rsp.permutations,
    }
    np.savez_compressed(model_output, **arrays)
    result.adata.uns[result.name]["model_npz"] = str(model_output.resolve())
    result.adata.write_h5ad(output, compression="gzip")
    return output, model_output


def save_top_markers(
    result: GSResult, output_prefix: str | Path, n: int = 5
) -> tuple[list[dict], Path, Path]:
    marker_key = result.adata.uns[result.name]["run"]["markers"]
    return write_top_markers(result.adata, marker_key, "GS", output_prefix, n)


def save_gs_figures(result: GSResult, output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    name = result.name
    diagnostics = result.diagnostics
    singular = diagnostics["score_singular_values"]
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(np.arange(1, len(singular) + 1), singular, marker=".", linewidth=1)
    ax.set_xlabel("factor column")
    ax.set_ylabel("singular value")
    ax.set_title(f"Score spectrum: {name}")
    path = output_dir / f"score_spectrum_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    raw_norm = np.linalg.norm(result.estimator.B_samples_.mean(axis=0), axis=0)
    aligned_norm = np.linalg.norm(result.estimator.B_aligned_.mean(axis=0), axis=0)
    columns = np.arange(1, len(raw_norm) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(columns, raw_norm, marker=".", linewidth=1, label="Before RSP")
    ax.plot(columns, aligned_norm, marker=".", linewidth=1, label="After RSP")
    ax.set_xlabel("factor column")
    ax.set_ylabel("loading column L2 norm")
    ax.set_title(f"Loading column norms: {name}")
    ax.legend()
    path = output_dir / f"loading_column_norms_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    run = result.adata.uns[name]["run"]
    fig = sc.pl.embedding(
        result.adata, basis=run["umap"], color=run["leiden"], show=False, return_fig=True
    )
    path = output_dir / f"umap_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    sampling = diagnostics["sampling"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].hist(sampling["bulk_B"].ravel(), bins=40)
    axes[0].set_title("Bulk ESS: loadings")
    axes[1].hist(sampling["bulk_sigma2"].ravel(), bins=40)
    axes[1].set_title("Bulk ESS: noise variances")
    for ax in axes:
        ax.axvline(400, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("ESS")
        ax.set_ylabel("entries")
    path = output_dir / f"ess_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", nargs="?", type=Path, default=Path("PBMC/data/notebook_scaled_cavi_norm_d20_k50.h5ad"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    result = run_gs_from_baseline(args.data_path)
    output = args.output or Path(f"PBMC/data/{result.name}_k{result.config.K}.h5ad")
    h5ad_path, model_path = save_gs_result(result, output, model_output=args.model_output)
    fit = result.diagnostics["fit"]
    sampling = result.diagnostics["sampling"]
    cluster_key = result.adata.uns[result.name]["run"]["leiden"]
    print(
        f"{result.name}: kept={fit['n_kept']}, gamma={fit['gamma_mode']}, "
        f"clusters={result.adata.obs[cluster_key].nunique()}"
    )
    print(f"RSP status={result.diagnostics['rsp']['status_code']}")
    print(
        f"min bulk ESS={min(sampling['bulk_summary']['B']['min'], sampling['bulk_summary']['sigma2']['min']):.3g}"
    )
    print(f"Saved H5AD to {h5ad_path}; model to {model_path}")
    if args.figure_dir is not None:
        paths = save_gs_figures(result, args.figure_dir)
        print(f"Saved {len(paths)} figures to {args.figure_dir}")
    _, csv_path, md_path = save_top_markers(
        result, h5ad_path.with_name(f"{h5ad_path.stem}_top5_markers")
    )
    print(f"Saved top-5 markers to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
