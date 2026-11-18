"""Run spectral CAVI in float64 and compare with a saved float32 fit.

The saved baseline is loaded, never refitted. Raw observations and truth are
required to match the archived float32 experiment exactly. The full-precision
observations are passed to the new fit without an intermediate float32 cast.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from bfm import CaviBFM


_STATE_KEYS = ("mu", "V", "W", "Phi", "beta", "c", "rho", "m")


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value).tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _estimation_metrics(model, B_true, sigma2_true, Sigma_true, linf_threshold):
    """Orientation-invariant errors, plus the two explicit factor-count rules."""
    B_true = np.asarray(B_true, dtype=np.float64)
    sigma2_true = np.asarray(sigma2_true, dtype=np.float64)
    mu = np.asarray(model.mu_, dtype=np.float64)
    true_norm = float(np.linalg.norm(Sigma_true, ord="fro"))
    covariances = {
        "plugin": np.asarray(model.Sigma_, dtype=np.float64),
        "posterior_mean": np.asarray(model.Sigma_mean_, dtype=np.float64),
    }
    metrics = {}
    for name, covariance in covariances.items():
        error = float(np.linalg.norm(covariance - Sigma_true, ord="fro"))
        metrics[f"covariance_{name}_frobenius"] = error
        metrics[f"covariance_{name}_relative_frobenius"] = error / true_norm

    noise_error = np.asarray(model.sigma2_hat_, dtype=np.float64) - sigma2_true
    metrics.update({
        "noise_mae": float(np.abs(noise_error).mean()),
        "noise_rmse": float(np.sqrt(np.mean(noise_error ** 2))),
        "n_factors_credible": int(model.n_factors(method="credible")),
        "n_factors_linf": int(model.n_factors(
            method="linf", threshold=linf_threshold)),
        "loading_column_l2": np.linalg.norm(mu, axis=0).tolist(),
        "loading_column_linf": np.max(np.abs(mu), axis=0).tolist(),
    })

    # Use dominant left singular vectors rather than arbitrary QR directions
    # attached to nearly-zero loading columns.  Missing directions get cosine
    # zero, explicitly accounting for estimates with insufficient numerical rank.
    u_true, singular_true, _ = np.linalg.svd(B_true, full_matrices=False)
    true_rank = int(np.count_nonzero(
        singular_true > max(B_true.shape) * np.finfo(float).eps * singular_true[0]))
    u_fit, singular_fit, _ = np.linalg.svd(mu, full_matrices=False)
    cutoff = max(mu.shape) * np.finfo(np.asarray(model.mu_).dtype).eps * singular_fit[0]
    fit_rank = int(np.count_nonzero(singular_fit > cutoff))
    available = min(true_rank, fit_rank)
    cosines = np.linalg.svd(
        u_fit[:, :available].T @ u_true[:, :true_rank], compute_uv=False,
    ) if available else np.empty(0)
    cosines = np.clip(np.pad(cosines, (0, true_rank - available)), 0.0, 1.0)
    metrics.update({
        "estimated_loading_numerical_rank": fit_rank,
        "subspace_principal_cosines": cosines.tolist(),
        "subspace_min_cosine": float(cosines.min()),
        "subspace_mean_cosine": float(cosines.mean()),
    })

    # The truth is implicitly padded with zero columns to the fitted K.  This
    # is a descriptive loading-recovery error after optimal orthogonal matching,
    # not another optimum under the column-dependent prior.
    cross_singular = np.linalg.svd(mu.T @ B_true, compute_uv=False)
    squared_error = max(float(
        np.sum(mu ** 2) + np.sum(B_true ** 2) - 2 * cross_singular.sum()), 0.0)
    metrics["loading_procrustes_relative_frobenius"] = (
        float(np.sqrt(squared_error)) / float(np.linalg.norm(B_true, ord="fro")))
    return metrics


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_baseline_inputs(X, B_true, sigma2_true, *, baseline_dir):
    """Fail if the regenerated raw dataset differs from the saved experiment."""
    with np.load(Path(baseline_dir) / "inputs.npz") as saved:
        for key, value in (("X", X), ("B_true", B_true), ("sigma2_true", sigma2_true)):
            array = np.asarray(value)
            if array.dtype != saved[key].dtype:
                raise AssertionError(f"{key} dtype differs from saved baseline")
            np.testing.assert_array_equal(array, saved[key], err_msg=f"{key} differs from saved baseline")


def compare_precision(
    X, B_true, sigma2_true, Sigma_true, *, estimator_kwargs, baseline_dir,
    output_dir, demo_name, data_seed, init_seed=2026, linf_threshold=0.1,
):
    """Return saved float32 view, freshly fitted float64 model, and summary."""
    baseline_dir, output_dir = Path(baseline_dir), Path(output_dir)
    if baseline_dir.resolve() == output_dir.resolve():
        raise ValueError("The float64 output directory must differ from the saved baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    check_baseline_inputs(X, B_true, sigma2_true, baseline_dir=baseline_dir)
    for reserved in ("init_method", "init_seed", "initial_state", "precision"):
        if reserved in estimator_kwargs:
            raise ValueError(f"{reserved} is controlled by this precision comparison")
    X = np.asarray(X)
    if X.dtype != np.float64:
        raise AssertionError("The precision rerun requires the original float64 raw observations")
    old_summary = json.loads((baseline_dir / "summary.json").read_text())
    for key, value in estimator_kwargs.items():
        if old_summary["configuration"].get(key) != value:
            raise AssertionError(f"Fitting setting {key} differs from the saved baseline")
    for key, value in (("data_seed", data_seed), ("init_seed", init_seed),
                       ("linf_threshold", linf_threshold)):
        if old_summary["configuration"][key] != value:
            raise AssertionError(f"Comparison setting {key} differs from saved baseline")
    with np.load(baseline_dir / "spectral_state.npz") as saved:
        old_state = {key: np.array(saved[key]) for key in saved.files}
    old_dtypes = {key: str(old_state[key].dtype) for key in _STATE_KEYS}
    if set(old_dtypes.values()) != {"float32"}:
        raise AssertionError("The archived recurrence state is not entirely float32")
    baseline = SimpleNamespace(
        K=estimator_kwargs["K"], n_iter_=old_summary["results"]["spectral"]["n_iter"],
        elbo_=old_state["elbo"], sigma2_hat_=old_state["sigma2_hat"],
    )
    old = copy.deepcopy(old_summary["results"]["spectral"])
    old.update(precision="float32", source="saved run2 spectral fit; not rerun",
               state_dtypes=old_dtypes)
    baseline_files = [baseline_dir / name for name in
                      ("inputs.npz", "summary.json", "spectral_state.npz")]
    before = {str(path): _file_digest(path) for path in baseline_files}

    print(f"[{demo_name}] spectral CAVI precision=float64: fitting ...", flush=True)
    start = time.perf_counter()
    model = CaviBFM(**estimator_kwargs, init_method="spectral",
                    init_seed=init_seed, precision="float64").fit(X)
    wall = time.perf_counter() - start
    state = {key: np.asarray(getattr(model, key + "_")) for key in _STATE_KEYS}
    dtypes = {key: str(value.dtype) for key, value in state.items()}
    if set(dtypes.values()) != {"float64"} or not jax.config.jax_enable_x64:
        raise AssertionError(f"CAVI did not run entirely in float64: {dtypes}")
    for key in ("dtype", "input_dtype", "elbo_dtype", "precision"):
        if model.raw_output_[key] != "float64":
            raise AssertionError(f"Driver {key} is {model.raw_output_[key]!r}, expected float64")
    state.update(alpha_n=model.alpha_n_, elbo=np.asarray(model.elbo_, dtype=np.float64),
                 x_offset=model.x_offset_, sigma2_hat=model.sigma2_hat_)
    np.savez_compressed(output_dir / "spectral_float64_state.npz", **state)
    np.savez_compressed(output_dir / "inputs.npz", X=X, B_true=B_true,
                        sigma2_true=sigma2_true)
    new = {
        "init_method": "spectral", "precision": "float64",
        "source": "fresh single-start float64 fit",
        "initialization_method": model.initialization_method_,
        "initialization_info": model.initialization_info_,
        "state_dtypes": dtypes, "input_dtype": model.raw_output_["input_dtype"],
        "elbo_dtype": model.raw_output_["elbo_dtype"],
        "converged": bool(model.converged_), "stop_reason": model.stop_reason_,
        "n_iter": int(model.n_iter_), "initial_elbo": float(model.elbo_[0]),
        "final_elbo": float(model.elbo_[-1]),
        "final_elbo_increment": float(model.elbo_[-1] - model.elbo_[-2]),
        "initialization_sec": float(model.initialization_sec_),
        "compile_sec": float(model.compile_sec_), "runtime_sec": float(model.runtime_sec_),
        "wall_sec": float(wall), "ms_per_sweep": float(1000 * model.runtime_sec_ / model.n_iter_),
        "metrics": _estimation_metrics(model, B_true, sigma2_true, Sigma_true, linf_threshold),
    }
    after = {str(path): _file_digest(path) for path in baseline_files}
    if before != after:
        raise AssertionError("Saved baseline evidence changed during the float64 fit")
    comparison = {
        "demo": demo_name,
        "assertion_tested": "Run the new spectral initialization in actual float64 CAVI on the original raw observations, using unchanged model and stopping settings; compare with the saved float32 spectral fit.",
        "non_claims": ["Only one dataset and one fit per demo/precision are compared.",
                       "The ELBO stopping flag is not a stationarity or global-optimality certificate.",
                       "Historical and fresh timings are not randomized repeated benchmarks.",
                       "Floating-point preprocessing, spectral rank thresholds, and CAVI arithmetic all follow their selected precision."],
        "configuration": {**dict(estimator_kwargs), "init_method": "spectral",
                          "precision": "float64", "init_seed": int(init_seed),
                          "data_seed": int(data_seed), "linf_threshold": float(linf_threshold),
                          "n": int(X.shape[0]), "p": int(X.shape[1]),
                          "true_factors": int(np.asarray(B_true).shape[1])},
        "timing_protocol": "The float64 fit runs once after the float32 Gibbs fit. The archived float32 fit has its own timing protocol, recorded below. Iteration-loop times exclude initialization and explicit AOT compilation. The two runs can have different compilation-cache and system states.",
        "baseline_timing_protocol": old_summary["timing_protocol"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "jax": jax.__version__, "backend": jax.default_backend(),
                        "devices": [str(device) for device in jax.devices()],
                        "platform": platform.platform(), "jax_enable_x64": bool(jax.config.jax_enable_x64)},
        "raw_input_dtype": str(X.dtype),
        "input_hashes": {"X": _array_digest(X), "B_true": _array_digest(B_true),
                         "sigma2_true": _array_digest(sigma2_true)},
        "baseline_file_sha256_before": before, "baseline_file_sha256_after": after,
        "results": {"float32": old, "float64": new},
        "float64_minus_float32": {
            "n_iter": new["n_iter"] - old["n_iter"],
            "final_elbo": new["final_elbo"] - old["final_elbo"],
            "runtime_sec": new["runtime_sec"] - old["runtime_sec"],
            "noise_mae": new["metrics"]["noise_mae"] - old["metrics"]["noise_mae"],
            "covariance_posterior_mean_relative_frobenius": new["metrics"]["covariance_posterior_mean_relative_frobenius"] - old["metrics"]["covariance_posterior_mean_relative_frobenius"],
        },
    }
    comparison = _json_value(comparison)
    (output_dir / "summary.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    print_precision_comparison(comparison)
    return baseline, model, comparison


def print_precision_comparison(comparison):
    print("\nSpectral CAVI: saved float32 versus fresh float64 (before RSP)")
    print(f"{'Precision':<10} {'Sweeps':>7} {'Stop':>10} {'Initial ELBO':>17} {'Final ELBO':>17} {'Last delta':>12} {'Init s':>8} {'Compile s':>10} {'Loop s':>9} {'Wall s':>9}")
    for name, item in comparison["results"].items():
        print(f"{name:<10} {item['n_iter']:7d} {item['stop_reason']:>10} {item['initial_elbo']:17.6f} {item['final_elbo']:17.6f} {item['final_elbo_increment']:12.6g} {item['initialization_sec']:8.3f} {item['compile_sec']:10.3f} {item['runtime_sec']:9.3f} {item['wall_sec']:9.3f}")
    print(f"\n{'Precision':<10} {'r CI':>6} {'r linf':>7} {'Noise MAE':>12} {'Noise RMSE':>12} {'Cov relF':>12} {'Min cosine':>12} {'B relF*':>12}")
    for name, item in comparison["results"].items():
        m = item["metrics"]
        print(f"{name:<10} {m['n_factors_credible']:6d} {m['n_factors_linf']:7d} {m['noise_mae']:12.6f} {m['noise_rmse']:12.6f} {m['covariance_posterior_mean_relative_frobenius']:12.6f} {m['subspace_min_cosine']:12.6f} {m['loading_procrustes_relative_frobenius']:12.6f}")
        print(f"  {name} recurrence dtypes:", item["state_dtypes"])
    print("float64 driver input/ELBO dtype:", comparison["results"]["float64"]["input_dtype"],
          comparison["results"]["float64"]["elbo_dtype"],
          "| JAX x64:", comparison["environment"]["jax_enable_x64"])
    print("Cov relF uses the posterior-mean covariance; B relF* uses orthogonal matching.")
    print("The float32 baseline is historical; it was not refitted. Compare single-run timings with their cache/order caveat.")


def plot_precision_comparison(vi_float32, vi, comparison, *, output_dir, sigma2_true, show=True):
    """Save ELBO and recovery comparisons, labeling the historical baseline."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    colors = {"float32": "#4666a8", "float64": "#d8792f"}
    models = {"float32": vi_float32, "float64": vi}
    labels = {"float32": "float32 (saved)", "float64": "float64 (new)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for name, model in models.items():
        values = np.asarray(model.elbo_)
        axes[0].plot(np.arange(values.size), values, color=colors[name], lw=1.6,
                     label=f"{labels[name]}: {model.n_iter_} sweeps")
        axes[0].scatter([model.n_iter_], [values[-1]], color=colors[name], s=25)
        tail = min(150, model.n_iter_)
        axes[1].plot(np.arange(-tail, 1), values[-tail-1:], color=colors[name],
                     lw=1.6, label=labels[name])
    axes[0].set(title="Spectral CAVI: complete ELBO trajectories", xlabel="Completed sweeps", ylabel="ELBO")
    axes[1].set(title="Each run's final 150 sweeps", xlabel="Sweeps relative to stopping", ylabel="ELBO")
    for ax in axes:
        ax.legend()
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.savefig(output_dir / "elbo_comparison.png", dpi=150)

    estimate_fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(sigma2_true, color="#454545", lw=.8, alpha=.6, label="truth")
    cov_errors = []
    for name, model in models.items():
        metrics = comparison["results"][name]["metrics"]
        axes[0, 0].plot(model.sigma2_hat_, color=colors[name], lw=.9, label=labels[name])
        axes[0, 1].plot(np.arange(1, model.K + 1), metrics["loading_column_l2"],
                        color=colors[name], marker=".", lw=1, label=labels[name])
        cosines = metrics["subspace_principal_cosines"]
        axes[1, 1].plot(np.arange(1, len(cosines) + 1), cosines,
                        color=colors[name], marker="o", lw=1.5, label=labels[name])
        cov_errors.append(metrics["covariance_posterior_mean_relative_frobenius"])
    axes[0, 0].set(title="Noise-variance estimates", xlabel="Variable j", ylabel="Variance")
    axes[0, 1].set(title="Loading-mean column norms", xlabel="Column k", ylabel="L2 norm")
    bars = axes[1, 0].bar([labels[name] for name in models], cov_errors, color=list(colors.values()))
    axes[1, 0].bar_label(bars, fmt="%.5f", padding=3)
    axes[1, 0].set(title="Posterior-mean covariance recovery", ylabel="Relative Frobenius error",
                   ylim=(0, 1.2 * max(cov_errors)))
    axes[1, 1].set(title="Leading factor-subspace recovery", xlabel="Principal direction",
                   ylabel="Principal-angle cosine", ylim=(0, 1.015))
    for position in ((0, 0), (0, 1), (1, 1)):
        axes[position].legend()
    estimate_fig.savefig(output_dir / "estimation_comparison.png", dpi=150)
    if show:
        plt.show()
    return fig, estimate_fig
