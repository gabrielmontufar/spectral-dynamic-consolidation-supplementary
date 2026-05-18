"""Run bounded validation from the existing Oso consistency CSV files.

This script is intentionally data-conservative: it reads the consistency files
already present in the repository and derives residual metrics from those
observed/predicted retained-pressure values. It does not download, synthesize,
or impute validation observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_INPUTS = {
    "records": "oso_ring_shear_normalized_records.csv",
    "metrics": "oso_ring_shear_consistency_metrics.csv",
    "predictions": "oso_ring_shear_consistency_predictions.csv",
}

REQUIRED_COLUMNS = {
    "records": {"time_s", "retained_observed", "thickness_m", "source_file"},
    "metrics": {
        "model",
        "cv_fit_m2_s",
        "thickness_m",
        "n_train",
        "n_heldout",
        "rmse_train",
        "rmse_heldout",
        "mae_heldout",
        "bias_heldout",
        "p95_abs_residual",
        "r2_heldout",
        "source_file",
        "best_model_for_record",
    },
    "predictions": {
        "source_file",
        "model",
        "time_s",
        "retained_observed",
        "retained_predicted",
    },
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def require_inputs(root: Path) -> dict[str, Path]:
    paths = {key: root / name for key, name in REQUIRED_INPUTS.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Required validation input file(s) missing:\n  {joined}")
    return paths


def read_checked_csv(path: Path, key: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS[key].difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    return frame


def r2_score(observed: pd.Series, predicted: pd.Series) -> float:
    obs = observed.to_numpy(dtype=float)
    pred = predicted.to_numpy(dtype=float)
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    ss_res = float(np.sum((pred - obs) ** 2))
    return 1.0 - ss_res / ss_tot


def survival_double(pi: np.ndarray | float, terms: int = 120) -> np.ndarray | float:
    pi_arr = np.asarray(pi, dtype=float)
    out = np.zeros_like(pi_arr, dtype=float)
    m = np.arange(1, 2 * terms, 2, dtype=float)
    weights = 8.0 / (m**2 * np.pi**2)
    for idx, value in np.ndenumerate(pi_arr):
        out[idx] = np.sum(weights * np.exp(-(m**2) * np.pi**2 * value))
    return float(out) if np.isscalar(pi) else out


def survival_single(pi: np.ndarray | float, terms: int = 120) -> np.ndarray | float:
    pi_arr = np.asarray(pi, dtype=float)
    out = np.zeros_like(pi_arr, dtype=float)
    n = np.arange(terms, dtype=float)
    alpha = (n + 0.5) * np.pi
    weights = 2.0 / (alpha**2)
    for idx, value in np.ndenumerate(pi_arr):
        out[idx] = np.sum(weights * np.exp(-(alpha**2) * value))
    return float(out) if np.isscalar(pi) else out


def residual_summary(frame: pd.DataFrame) -> pd.Series:
    residual = frame["retained_predicted"] - frame["retained_observed"]
    abs_residual = residual.abs()
    return pd.Series(
        {
            "prediction_points": int(len(frame)),
            "prediction_rmse": float(np.sqrt(np.mean(residual**2))),
            "prediction_mae": float(abs_residual.mean()),
            "prediction_bias": float(residual.mean()),
            "prediction_median_abs_residual": float(abs_residual.median()),
            "prediction_p95_abs_residual": float(abs_residual.quantile(0.95)),
            "prediction_r2": r2_score(frame["retained_observed"], frame["retained_predicted"]),
        }
    )


def build_master_table(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.groupby(["source_file", "model"], sort=True)
        .apply(residual_summary, include_groups=False)
        .reset_index()
    )
    observed_ranges = (
        predictions.groupby(["source_file", "model"], sort=True)
        .agg(
            validation_time_min_s=("time_s", "min"),
            validation_time_max_s=("time_s", "max"),
            observed_retained_min=("retained_observed", "min"),
            observed_retained_max=("retained_observed", "max"),
            predicted_retained_min=("retained_predicted", "min"),
            predicted_retained_max=("retained_predicted", "max"),
        )
        .reset_index()
    )
    grouped = grouped.merge(observed_ranges, on=["source_file", "model"], how="left")
    carry_columns = [
        "source_file",
        "model",
        "best_model_for_record",
        "cv_fit_m2_s",
        "thickness_m",
        "n_train",
        "n_heldout",
        "rmse_train",
        "rmse_heldout",
        "mae_heldout",
        "bias_heldout",
        "p95_abs_residual",
        "r2_heldout",
    ]
    master = metrics[carry_columns].merge(grouped, on=["source_file", "model"], how="left")
    for column in ["n_train", "n_heldout", "prediction_points"]:
        master[column] = master[column].astype("Int64")
    return master


def build_model_comparison(master: pd.DataFrame) -> pd.DataFrame:
    comparison = (
        master.groupby("model", sort=True)
        .agg(
            records=("source_file", "nunique"),
            prediction_points=("prediction_points", "sum"),
            median_heldout_rmse=("rmse_heldout", "median"),
            mean_heldout_rmse=("rmse_heldout", "mean"),
            median_heldout_mae=("mae_heldout", "median"),
            median_heldout_bias=("bias_heldout", "median"),
            median_heldout_r2=("r2_heldout", "median"),
            median_prediction_rmse=("prediction_rmse", "median"),
            mean_prediction_rmse=("prediction_rmse", "mean"),
            median_prediction_mae=("prediction_mae", "median"),
            median_prediction_bias=("prediction_bias", "median"),
            median_prediction_r2=("prediction_r2", "median"),
        )
        .reset_index()
    )
    for column in ["records", "prediction_points"]:
        comparison[column] = comparison[column].astype("Int64")
    return comparison


def build_metrics_summary(
    records: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    master: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    best_counts = metrics.drop_duplicates("source_file")["best_model_for_record"].value_counts()
    best_by_prediction = comparison.sort_values("median_prediction_rmse").iloc[0]
    best_by_heldout = comparison.sort_values("median_heldout_rmse").iloc[0]
    rows = [
        ("input_records_file", REQUIRED_INPUTS["records"], "existing Oso consistency dataset"),
        ("input_metrics_file", REQUIRED_INPUTS["metrics"], "existing per-record held-out metrics"),
        ("input_predictions_file", REQUIRED_INPUTS["predictions"], "existing sampled observed/predicted curves"),
        ("normalized_record_rows", len(records), "rows in available normalized Oso records"),
        ("unique_source_records", records["source_file"].nunique(), "unique Oso source files"),
        ("prediction_rows_used", len(predictions), "sampled prediction rows used for residual metrics"),
        ("models_compared", ",".join(sorted(metrics["model"].unique())), "models present in consistency metrics"),
        ("best_model_count_double", int(best_counts.get("double", 0)), "records where double is listed as best"),
        ("best_model_count_single", int(best_counts.get("single", 0)), "records where single is listed as best"),
        (
            "best_model_by_median_prediction_rmse",
            best_by_prediction["model"],
            "lower median residual RMSE on existing sampled predictions",
        ),
        (
            "best_model_by_median_heldout_rmse",
            best_by_heldout["model"],
            "lower median held-out RMSE from existing metrics file",
        ),
        (
            "lowest_median_prediction_rmse",
            best_by_prediction["median_prediction_rmse"],
            "normalized retained-pressure units",
        ),
        (
            "lowest_median_heldout_rmse",
            best_by_heldout["median_heldout_rmse"],
            "normalized retained-pressure units",
        ),
        (
            "master_table_rows",
            len(master),
            "one row per source file and model combination in the consistency metrics",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


def plot_validation(predictions: pd.DataFrame, metrics: pd.DataFrame, figure_path: Path) -> None:
    best_pairs = metrics[["source_file", "best_model_for_record"]].drop_duplicates()
    best_predictions = predictions.merge(best_pairs, on="source_file", how="inner")
    best_predictions = best_predictions[
        best_predictions["model"] == best_predictions["best_model_for_record"]
    ].copy()
    if best_predictions.empty:
        raise ValueError("No prediction rows match the best model labels in the metrics file.")

    residual = best_predictions["retained_predicted"] - best_predictions["retained_observed"]
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(residual.abs().mean())

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=600, constrained_layout=True)
    ax = axes[0]
    for _, frame in best_predictions.groupby("source_file", sort=True):
        ax.plot(
            frame["retained_observed"],
            frame["retained_predicted"],
            ".",
            ms=2.8,
            alpha=0.55,
        )
    low = float(min(best_predictions["retained_observed"].min(), best_predictions["retained_predicted"].min()))
    high = float(max(best_predictions["retained_observed"].max(), best_predictions["retained_predicted"].max()))
    pad = 0.04 * max(high - low, 1e-6)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="black", lw=1.0)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Observed retained pressure")
    ax.set_ylabel("Predicted retained pressure")
    ax.set_title(f"Best-model sampled predictions\nRMSE={rmse:.3f}, MAE={mae:.3f}")
    ax.grid(alpha=0.25)

    ax = axes[1]
    comparison_data = [
        predictions.loc[predictions["model"] == model, "retained_predicted"]
        - predictions.loc[predictions["model"] == model, "retained_observed"]
        for model in sorted(predictions["model"].unique())
    ]
    labels = sorted(predictions["model"].unique())
    ax.boxplot(comparison_data, tick_labels=labels, showfliers=True)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xlabel("Drainage model")
    ax.set_ylabel("Predicted - observed")
    ax.set_title("Residuals from existing Oso consistency predictions")
    ax.grid(axis="y", alpha=0.25)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fit_decay_record(time_s: np.ndarray, retained: np.ndarray, h_m: float, model: str) -> dict[str, float | np.ndarray]:
    keep = np.isfinite(time_s) & np.isfinite(retained)
    time_s = time_s[keep]
    retained = np.clip(retained[keep], -0.25, 1.25)
    order = np.argsort(time_s)
    time_s = time_s[order] - float(np.min(time_s[order]))
    retained = retained[order]
    split = max(10, int(0.4 * len(time_s)))
    train_t = time_s[:split]
    train_y = retained[:split]
    test_t = time_s[split:]
    test_y = retained[split:]
    fn = survival_double if model == "double" else survival_single
    cv_grid = np.logspace(-10, -3, 141)
    h2 = max(float(h_m) ** 2, 1e-8)
    losses = []
    for cv in cv_grid:
        pred = fn(cv * train_t / h2)
        losses.append(float(np.sqrt(np.mean((pred - train_y) ** 2))))
    cv = float(cv_grid[int(np.argmin(losses))])
    pred_test = fn(cv * test_t / h2)
    residual = pred_test - test_y
    return {
        "cv_fit_m2_s": cv,
        "n_train": int(len(train_t)),
        "n_heldout": int(len(test_t)),
        "rmse_heldout": float(np.sqrt(np.mean(residual**2))),
        "mae_heldout": float(np.mean(np.abs(residual))),
        "bias_heldout": float(np.mean(residual)),
        "r2_heldout": r2_score(pd.Series(test_y), pd.Series(pred_test)),
        "test_t": test_t,
        "test_y": test_y,
        "pred_test": pred_test,
    }


def run_mount_kabasan(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = root / "external_data" / "mount_kabasan" / "Japan_exp_failure_period_data.csv"
    if not source.exists():
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.read_csv(source, skiprows=4)
    rows = []
    pred_rows = []
    for sensor in ["Piezo.4_(kPa)", "Piezo.6_(kPa)", "Piezo.7_(kPa)"]:
        frame = raw[["TIMER_(sec)", sensor]].dropna().copy()
        if len(frame) < 100:
            continue
        time = frame["TIMER_(sec)"].to_numpy(dtype=float)
        pressure = frame[sensor].to_numpy(dtype=float)
        # Use a decay-style retained-pressure normalization analogous to Oso:
        # early high pressure against late residual pressure.
        early = pressure[: max(20, len(pressure) // 10)]
        late = pressure[int(0.9 * len(pressure)) :]
        p_start = float(np.nanpercentile(early, 90))
        p_residual = float(np.nanmedian(late))
        amplitude = p_start - p_residual
        if abs(amplitude) < 0.25:
            continue
        retained = (pressure - p_residual) / amplitude
        h_m = 0.5
        for model in ["double", "single"]:
            fit = fit_decay_record(time, retained, h_m, model)
            rows.append(
                {
                    "dataset_id": "mount_kabasan_2003",
                    "source_file": source.name,
                    "sensor_id": sensor.replace("_(kPa)", ""),
                    "model": model,
                    "h_m": h_m,
                    "p_start_kpa": p_start,
                    "p_residual_kpa": p_residual,
                    "Pu_kpa": amplitude,
                    "cv_fit_m2_s": fit["cv_fit_m2_s"],
                    "n_train": fit["n_train"],
                    "n_heldout": fit["n_heldout"],
                    "rmse_heldout": fit["rmse_heldout"],
                    "mae_heldout": fit["mae_heldout"],
                    "bias_heldout": fit["bias_heldout"],
                    "r2_heldout": fit["r2_heldout"],
                    "interpretation": "field-scale pore-pressure dissipation consistency; no slope FS calibration",
                }
            )
            for t, obs, pred in zip(fit["test_t"], fit["test_y"], fit["pred_test"]):
                pred_rows.append(
                    {
                        "dataset_id": "mount_kabasan_2003",
                        "sensor_id": sensor.replace("_(kPa)", ""),
                        "model": model,
                        "time_s": float(t),
                        "retained_observed": float(obs),
                        "retained_predicted": float(pred),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def run_flume_inventory(root: Path) -> pd.DataFrame:
    folder = root / "external_data" / "usgs_flume_2016"
    if not folder.exists():
        return pd.DataFrame()
    rows = []
    for path in sorted(folder.glob("*_archive_corrected.csv")):
        header = pd.read_csv(path, nrows=0, encoding="latin1")
        cols = list(header.columns)
        stations = sorted(
            {
                col.split("_", 1)[1].replace("m(kPa)", "").replace(".0m(kPa)", "m")
                for col in cols
                if col.startswith("Nstress_")
            }
        )
        for station in stations:
            n_col = next((c for c in cols if c.startswith("Nstress_") and station in c), None)
            pp_cols = [c for c in cols if c.startswith("PP") and station.replace("m", "") in c]
            if not n_col or not pp_cols:
                continue
            use_cols = ["Time(s)", n_col] + pp_cols[:4]
            sample = pd.read_csv(path, usecols=use_cols, encoding="latin1")
            sample = sample[sample["Time(s)"] >= 0].replace([np.inf, -np.inf], np.nan).dropna()
            if sample.empty:
                continue
            nstress = sample[n_col].astype(float)
            pp = sample[pp_cols[:4]].astype(float).mean(axis=1)
            ru = (pp / nstress.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
            if ru.empty:
                continue
            rows.append(
                {
                    "dataset_id": "usgs_flume_2016",
                    "experiment_file": path.name,
                    "station": station,
                    "n_points": int(len(ru)),
                    "ru_median": float(ru.median()),
                    "ru_p95": float(ru.quantile(0.95)),
                    "regime_obs": (
                        "nearly_undrained"
                        if ru.median() >= 0.75
                        else "partly_drained"
                        if ru.median() > 0.25
                        else "drained"
                    ),
                    "interpretation": "observed regime inventory only; cv/source calibration not inferred",
                }
            )
    return pd.DataFrame(rows)


def plot_multidataset(
    oso_predictions: pd.DataFrame,
    oso_metrics: pd.DataFrame,
    mount_predictions: pd.DataFrame,
    figure_path: Path,
) -> None:
    best_pairs = oso_metrics[["source_file", "best_model_for_record"]].drop_duplicates()
    oso_best = oso_predictions.merge(best_pairs, on="source_file", how="inner")
    oso_best = oso_best[oso_best["model"] == oso_best["best_model_for_record"]].copy()
    mount_best = pd.DataFrame()
    if not mount_predictions.empty:
        mount_best = mount_predictions[mount_predictions["model"] == "single"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=600, constrained_layout=True)
    ax = axes[0]
    ax.plot(oso_best["retained_observed"], oso_best["retained_predicted"], ".", ms=2.5, alpha=0.45, label="Oso")
    if not mount_best.empty:
        ax.plot(
            mount_best["retained_observed"],
            mount_best["retained_predicted"],
            ".",
            ms=2.5,
            alpha=0.35,
            label="Mount Kaba-san",
        )
    low, high = -0.15, 1.15
    ax.plot([low, high], [low, high], color="black", lw=1.0)
    ax.plot([low, high], [low + 0.1, high + 0.1], "--", color="0.6", lw=0.8)
    ax.plot([low, high], [low - 0.1, high - 0.1], "--", color="0.6", lw=0.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Observed retained pressure")
    ax.set_ylabel("Predicted retained pressure")
    ax.set_title("Independent pressure-dissipation records")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    frames = []
    labels = []
    if not oso_best.empty:
        frames.append(oso_best["retained_predicted"] - oso_best["retained_observed"])
        labels.append("Oso")
    if not mount_best.empty:
        frames.append(mount_best["retained_predicted"] - mount_best["retained_observed"])
        labels.append("Mount")
    ax.boxplot(frames, tick_labels=labels, showfliers=True)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_ylabel("Predicted - observed")
    ax.set_title("Held-out residuals")
    ax.grid(axis="y", alpha=0.25)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(root: Path) -> dict[str, Path]:
    paths = require_inputs(root)
    records = read_checked_csv(paths["records"], "records")
    metrics = read_checked_csv(paths["metrics"], "metrics")
    predictions = read_checked_csv(paths["predictions"], "predictions")

    outputs = root / "outputs"
    figures = root / "generated_figures"
    outputs.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    master = build_master_table(metrics, predictions)
    comparison = build_model_comparison(master)
    summary = build_metrics_summary(records, metrics, predictions, master, comparison)
    mount_metrics, mount_predictions = run_mount_kabasan(root)
    flume_inventory = run_flume_inventory(root)
    inventory = pd.DataFrame(
        [
            {
                "dataset_id": "oso_ring_shear",
                "scale": "laboratory",
                "primary_variables": "pore pressure, specimen thickness",
                "status": "pressure-dissipation prediction metrics available",
                "source": "USGS ScienceBase DOI 10.5066/F7KH0KSD",
            },
            {
                "dataset_id": "mount_kabasan_2003",
                "scale": "field experiment",
                "primary_variables": "pore pressure, displacement",
                "status": "pressure-dissipation consistency metrics available"
                if not mount_metrics.empty
                else "source not available locally",
                "source": "USGS ScienceBase DOI 10.5066/P18XMZPC",
            },
            {
                "dataset_id": "usgs_flume_2016",
                "scale": "physical flume",
                "primary_variables": "pore pressure, normal stress, flow depth",
                "status": "observed regime inventory available"
                if not flume_inventory.empty
                else "source not available locally",
                "source": "USGS ScienceBase DOI 10.5066/F7N58JKH",
            },
        ]
    )

    out_paths = {
        "summary": outputs / "validation_metrics_summary.csv",
        "comparison": outputs / "model_comparison_metrics.csv",
        "master": outputs / "validation_master_table.csv",
        "inventory": outputs / "external_dataset_inventory.csv",
        "mount_metrics": outputs / "mount_kabasan_consistency_metrics.csv",
        "mount_predictions": outputs / "mount_kabasan_consistency_predictions.csv",
        "flume_inventory": outputs / "flume_regime_observation_summary.csv",
        "figure": figures / "fig06_retained_pressure_validation.png",
        "multi_figure": figures / "fig07_multidataset_pressure_consistency.png",
    }
    summary.to_csv(out_paths["summary"], index=False)
    comparison.to_csv(out_paths["comparison"], index=False)
    master.to_csv(out_paths["master"], index=False)
    inventory.to_csv(out_paths["inventory"], index=False)
    mount_metrics.to_csv(out_paths["mount_metrics"], index=False)
    mount_predictions.to_csv(out_paths["mount_predictions"], index=False)
    flume_inventory.to_csv(out_paths["flume_inventory"], index=False)
    plot_validation(predictions, metrics, out_paths["figure"])
    plot_multidataset(predictions, metrics, mount_predictions, out_paths["multi_figure"])
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive external pressure-dissipation consistency metrics from available public datasets."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use locally available normalized/downloaded public data only. This workflow does not download files.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root_from_script(),
        help="Repository root containing the Oso consistency CSV files.",
    )
    args = parser.parse_args()
    out_paths = run(args.root.resolve())
    for label, path in out_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
