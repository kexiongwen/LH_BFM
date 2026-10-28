"""Fit the fixed K=50 CAVI workflow and its 20-column, 9-cluster representation."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bfm import CaviBFM

if __package__:
    from .data_processing import (
        PreprocessConfig,
        _jsonable,
        _software_versions,
        build_model_inputs,
    )
    from .downstream import DownstreamConfig, run_downstream, write_top_markers
else:
    from data_processing import (
        PreprocessConfig,
        _jsonable,
        _software_versions,
        build_model_inputs,
    )
    from downstream import DownstreamConfig, run_downstream, write_top_markers

__all__ = [
    "CAVIConfig",
    "CAVIResult",
    "load_baseline_input",
    "run_cavi",
    "run_cavi_from_baseline",
    "save_cavi_result",
    "save_cavi_figures",
    "save_top_markers",
]

N_SCORE_COLUMNS = 20


@dataclass(frozen=True)
class CAVIConfig:
    K: int = 50
    a: float = 10.0
    c1: float = 2.0
    c2: float = 1.0
    a_sigma: float = 1.0
    b_sigma: float = 1.0
    tol: float = 0.1
    max_iter: int = 1000
    drop_budget: float = 0.5
    chunk_size: int = 50
    audit_every: int = 50
    n_neighbors: int = 10
    leiden_resolution: float = 0.45
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.K < 1 or self.max_iter < 1:
            raise ValueError("K and max_iter must be positive")
        if self.tol <= 0 or self.drop_budget < 0:
            raise ValueError("tol must be positive and drop_budget nonnegative")
        if self.chunk_size < 1 or self.audit_every < 0:
            raise ValueError("chunk_size must be positive and audit_every nonnegative")
        if self.n_neighbors < 1 or self.leiden_resolution <= 0:
            raise ValueError("n_neighbors and leiden_resolution must be positive")


@dataclass
class CAVIResult:
    adata: ad.AnnData
    estimator: CaviBFM
    config: CAVIConfig
    name: str
    diagnostics: dict


def _diagnostic_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _diagnostic_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_jsonable(item) for item in value]
    return _jsonable(value)


def load_baseline_input(
    path: str | Path, *, expected_n_hvg: int = 2000
) -> tuple[ad.AnnData, np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"baseline does not exist: {path}")
    adata = sc.read_h5ad(path)
    if "lognorm" not in adata.layers or "highly_variable" not in adata.var:
        raise ValueError("baseline must contain lognorm and the HVG mask")
    X, hvg_names = build_model_inputs(
        adata, PreprocessConfig(n_top_genes=expected_n_hvg)
    )
    if not np.isfinite(X).all():
        raise ValueError("X_scaled contains non-finite values")
    return adata, X, hvg_names.to_numpy(dtype=str)


def _alignment_diagnostics(estimator: CaviBFM) -> dict[str, float]:
    transform = np.asarray(estimator.rsp_result_.transforms[0])
    orthogonality = float(
        np.linalg.norm(transform.T @ transform - np.eye(estimator.K))
    )
    score_error = float(
        np.max(np.abs(estimator.W_aligned_ - estimator.W_ @ transform))
    )
    max_error = 0.0
    max_scale = 0.0
    for start in range(0, estimator.W_.shape[0], 128):
        stop = min(start + 128, estimator.W_.shape[0])
        raw = estimator.mu_ @ estimator.W_[start:stop].T
        aligned = estimator.mu_aligned_ @ estimator.W_aligned_[start:stop].T
        max_error = max(max_error, float(np.max(np.abs(raw - aligned))))
        max_scale = max(max_scale, float(np.max(np.abs(raw))))
    return {
        "orthogonality_error": orthogonality,
        "score_transform_max_error": score_error,
        "common_component_max_error": max_error,
        "common_component_max_scale": max_scale,
        "common_component_relative_error": max_error / max(max_scale, np.finfo(np.float32).eps),
    }


def _fit_diagnostics(estimator: CaviBFM) -> tuple[dict, np.ndarray, np.ndarray]:
    elbo = np.asarray(estimator.elbo_, dtype=np.float64)
    delta = np.diff(elbo)
    fit = {
        "n_iter": int(estimator.n_iter_),
        "converged": bool(estimator.converged_),
        "stop_reason": str(estimator.stop_reason_),
        "final_delta": float(delta[-1]) if len(delta) else 0.0,
        "negative_delta_count": int((delta < 0).sum()),
        "max_drop": float(max(0.0, -delta.min())) if len(delta) else 0.0,
        "compile_sec": float(estimator.compile_sec_),
        "runtime_sec": float(estimator.runtime_sec_),
        "execution_mode": str(estimator.execution_mode_),
        "backend": str(estimator.backend_),
        "device": str(estimator.device_),
    }
    return fit, elbo, delta


def _rsp_diagnostics(estimator: CaviBFM, warning_messages: list[str]) -> dict:
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


def run_cavi(
    adata: ad.AnnData,
    X: np.ndarray,
    hvg_names: np.ndarray,
    *,
    config: CAVIConfig | None = None,
) -> CAVIResult:
    config = config or CAVIConfig()
    if X.shape != (adata.n_obs, len(hvg_names)):
        raise ValueError("X_scaled does not match the baseline cells and HVGs")
    estimator = CaviBFM(
        K=config.K,
        a=config.a,
        c1=config.c1,
        c2=config.c2,
        a_sigma=config.a_sigma,
        b_sigma=config.b_sigma,
        center=False,
        tol=config.tol,
        max_iter=config.max_iter,
        jit=True,
        chunk_size=config.chunk_size,
        graph=False,
        audit_every=config.audit_every,
        drop_budget=config.drop_budget,
        verbose=False,
    ).fit(X)
    fit, elbo, delta = _fit_diagnostics(estimator)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.align_factors()
    rsp = _rsp_diagnostics(estimator, [str(item.message) for item in caught])
    alignment = _alignment_diagnostics(estimator)

    selected_columns = np.arange(N_SCORE_COLUMNS)
    loading_column_norm = np.linalg.norm(estimator.mu_aligned_, axis=0)
    scores = np.asarray(estimator.W_aligned_) * loading_column_norm
    scores = scores[:, selected_columns]
    active_mask = np.asarray(
        estimator.active_entries(level=0.95, interval="hpd", use_aligned=True)
    )
    singular_values = np.linalg.svd(scores, compute_uv=False)
    name = "cavi_scaled_norm_d20"
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
    if "leiden_scaled" in adata.obs:
        reference = adata.obs["leiden_scaled"].astype(str)
        current = adata.obs[run["leiden"]].astype(str)
        comparison = {
            "reference": "leiden_scaled",
            "ari": float(adjusted_rand_score(reference, current)),
            "nmi": float(adjusted_mutual_info_score(reference, current)),
        }
    diagnostics = {
        "fit": fit,
        "elbo": elbo,
        "elbo_delta": delta,
        "active_mask": active_mask,
        "active_factor_count": int(active_mask.any(axis=0).sum()),
        "selected_active_factor_count": int(
            active_mask[:, selected_columns].any(axis=0).sum()
        ),
        "selected_columns": selected_columns,
        "selected_loading_column_norm": loading_column_norm[selected_columns],
        "score_singular_values": singular_values,
        "score_weighting": "loading_column_l2_norm",
        "loading_column_norm": loading_column_norm,
        "alignment": alignment,
        "rsp": rsp,
        "comparison_with_pca_scaled": comparison,
        "formal_complete": bool(fit["converged"] and rsp["status_code"] == 1),
    }
    adata.uns[name] = {
        "config_json": json.dumps(_jsonable(asdict(config)), sort_keys=True),
        "input_matrix": "X_scaled",
        "hvg_names": np.asarray(hvg_names, dtype=str),
        "diagnostics": diagnostics,
        "run": run,
        "software": _software_versions(),
    }
    return CAVIResult(adata, estimator, config, name, diagnostics)


def run_cavi_from_baseline(
    path: str | Path, *, config: CAVIConfig | None = None
) -> CAVIResult:
    adata, X, hvg_names = load_baseline_input(path)
    return run_cavi(adata, X, hvg_names, config=config)


def save_cavi_result(
    result: CAVIResult,
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
    diagnostics_json = json.dumps(
        _diagnostic_jsonable({
            "config": asdict(result.config),
            "diagnostics": result.diagnostics,
        }),
        sort_keys=True,
    )
    rsp = estimator.rsp_result_
    arrays = {
        "mu": estimator.mu_,
        "V": estimator.V_,
        "W": estimator.W_,
        "Phi": estimator.Phi_,
        "beta": estimator.beta_,
        "c": estimator.c_,
        "rho": estimator.rho_,
        "m": estimator.m_,
        "score_mean": result.adata.obsm[f"X_{result.name}"],
        "active_mask": result.diagnostics["active_mask"],
        "selected_columns": result.diagnostics["selected_columns"],
        "selected_loading_column_norm": result.diagnostics["selected_loading_column_norm"],
        "loading_column_norm": result.diagnostics["loading_column_norm"],
        "score_singular_values": result.diagnostics["score_singular_values"],
        "elbo": result.diagnostics["elbo"],
        "elbo_delta": result.diagnostics["elbo_delta"],
        "hvg_names": result.adata.uns[result.name]["hvg_names"],
        "cell_names": result.adata.obs_names.to_numpy(dtype=str),
        "diagnostics_json": diagnostics_json,
        "mu_aligned": estimator.mu_aligned_,
        "V_aligned": estimator.V_aligned_,
        "W_aligned": estimator.W_aligned_,
        "Phi_aligned": estimator.Phi_aligned_,
        "B_aligned": estimator.B_aligned_,
        "transforms": rsp.transforms,
        "signs": rsp.signs,
        "permutations": rsp.permutations,
    }
    np.savez_compressed(model_output, **arrays)
    result.adata.uns[result.name]["model_npz"] = str(model_output.resolve())
    result.adata.write_h5ad(output, compression="gzip")
    return output, model_output


def save_top_markers(
    result: CAVIResult, output_prefix: str | Path, n: int = 5
) -> tuple[list[dict], Path, Path]:
    marker_key = result.adata.uns[result.name]["run"]["markers"]
    return write_top_markers(result.adata, marker_key, "CAVI", output_prefix, n)


def save_cavi_figures(result: CAVIResult, output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    history = result.diagnostics["elbo"]
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(np.arange(len(history)), history, linewidth=1.2)
    ax.set_xlabel("sweep")
    ax.set_ylabel("ELBO")
    ax.set_title(f"CAVI ELBO: {result.name}")
    path = output_dir / f"elbo_{result.name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    singular = result.diagnostics["score_singular_values"]
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(np.arange(1, len(singular) + 1), singular, marker=".", linewidth=1)
    ax.set_xlabel("factor column")
    ax.set_ylabel("singular value")
    ax.set_title(f"Score spectrum: {result.name}")
    path = output_dir / f"score_spectrum_{result.name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    raw_norm = np.linalg.norm(result.estimator.mu_, axis=0)
    aligned_norm = np.linalg.norm(result.estimator.mu_aligned_, axis=0)
    columns = np.arange(1, len(raw_norm) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(columns, raw_norm, marker=".", linewidth=1, label="Before RSP")
    ax.plot(columns, aligned_norm, marker=".", linewidth=1, label="After RSP")
    ax.set_xlabel("factor column")
    ax.set_ylabel("loading column L2 norm")
    ax.set_title(f"Loading column norms: {result.name}")
    ax.legend()
    path = output_dir / f"loading_column_norms_{result.name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    run = result.adata.uns[result.name]["run"]
    fig = sc.pl.embedding(
        result.adata, basis=run["umap"], color=run["leiden"], show=False, return_fig=True
    )
    path = output_dir / f"umap_{result.name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", nargs="?", type=Path, default=Path("PBMC/data/notebook_scaled_pca_baseline.h5ad"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    result = run_cavi_from_baseline(args.data_path)
    output = args.output or Path(f"PBMC/data/{result.name}_k{result.config.K}.h5ad")
    h5ad_path, model_path = save_cavi_result(result, output, model_output=args.model_output)
    fit = result.diagnostics["fit"]
    cluster_key = result.adata.uns[result.name]["run"]["leiden"]
    print(
        f"{result.name}: sweeps={fit['n_iter']}, converged={fit['converged']}, "
        f"stop={fit['stop_reason']}, clusters={result.adata.obs[cluster_key].nunique()}"
    )
    print(f"RSP status: {result.diagnostics['rsp']['status_code']}")
    print(f"Saved H5AD to {h5ad_path}; model to {model_path}")
    if args.figure_dir is not None:
        paths = save_cavi_figures(result, args.figure_dir)
        print(f"Saved {len(paths)} figures to {args.figure_dir}")
    _, csv_path, md_path = save_top_markers(
        result, h5ad_path.with_name(f"{h5ad_path.stem}_top5_markers")
    )
    print(f"Saved top-5 markers to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
