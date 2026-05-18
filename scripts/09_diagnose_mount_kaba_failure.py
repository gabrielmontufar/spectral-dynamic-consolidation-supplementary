from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
REGIMES = ["drained", "partly_drained", "nearly_undrained"]


def obs_col(df: pd.DataFrame) -> str:
    return "R_star_obs" if "R_star_obs" in df.columns else "R_obs"


def class_report(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    rows = []
    for label in REGIMES:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        support = int((y_true == label).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )
    macro = pd.DataFrame(rows)[["precision", "recall", "f1_score"]].mean()
    rows.append(
        {
            "label": "macro_avg",
            "precision": float(macro["precision"]),
            "recall": float(macro["recall"]),
            "f1_score": float(macro["f1_score"]),
            "support": int(len(y_true)),
        }
    )
    rows.append(
        {
            "label": "accuracy",
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": float((y_true == y_pred).mean()),
            "support": int(len(y_true)),
        }
    )
    return pd.DataFrame(rows)


def first_crossing_time(g: pd.DataFrame, column: str, threshold: float = 0.9) -> float:
    hit = g[g[column] >= threshold]
    if hit.empty:
        return np.nan
    return float(hit["t_s"].iloc[0])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    flume = pd.read_csv(OUT / "flume_leave_one_experiment_predictions.csv")
    y_true = flume["regime_obs"].astype(str)
    y_pred = flume["regime_pred"].astype(str)
    confusion = pd.crosstab(y_true, y_pred).reindex(index=REGIMES, columns=REGIMES, fill_value=0)
    confusion.index.name = "observed"
    confusion.columns.name = "predicted"
    confusion.to_csv(OUT / "flume_confusion_matrix.csv")
    report = class_report(y_true, y_pred)
    report.to_csv(OUT / "flume_regime_classification_report.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=300, constrained_layout=True)
    image = ax.imshow(confusion.values, cmap="Blues")
    ax.set_xticks(range(len(REGIMES)), REGIMES, rotation=25, ha="right")
    ax.set_yticks(range(len(REGIMES)), REGIMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Fig. 17. Flume regime confusion matrix")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(int(confusion.iloc[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.82, label="count")
    fig.savefig(FIG / "fig17_flume_confusion_matrix.png", bbox_inches="tight")
    fig.savefig(FIG / "fig19_flume_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    metrics = pd.read_csv(OUT / "mount_kabasan_predictive_metrics.csv")
    pred = pd.read_csv(OUT / "mount_kabasan_predictive_predictions.csv")
    observed_col = obs_col(pred)
    baseline = (
        metrics[metrics["model"].isin(["M0_drained", "M1_undrained"])]
        .groupby("sensor_id")["mae"]
        .min()
        .rename("best_baseline_mae")
    )
    joined = metrics.merge(baseline, on="sensor_id", how="left")
    joined["mae_skill_vs_best_baseline"] = 1.0 - joined["mae"] / joined["best_baseline_mae"]
    joined["failure_interpretation"] = np.where(
        joined["mae_skill_vs_best_baseline"] > 0,
        "improves_baseline",
        "negative_or_no_gain_against_baseline",
    )

    decomp_rows = []
    for (sensor, model), g in pred.groupby(["sensor_id", "model"]):
        test = g[g["split"] == "test"]
        if test.empty:
            test = g
        residual = test["R_pred"] - test[observed_col]
        obs_range = float(test[observed_col].max() - test[observed_col].min())
        pred_range = float(test["R_pred"].max() - test["R_pred"].min())
        decomp_rows.append(
            {
                "sensor_id": sensor,
                "model": model,
                "n_test_points": int(len(test)),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual * residual))),
                "bias": float(np.mean(residual)),
                "obs_range_R": obs_range,
                "pred_range_R": pred_range,
                "amplitude_ratio_pred_obs": pred_range / obs_range if obs_range else np.nan,
                "residual_p05": float(residual.quantile(0.05)),
                "residual_p95": float(residual.quantile(0.95)),
            }
        )
    decomp = pd.DataFrame(decomp_rows).merge(
        joined[["sensor_id", "model", "best_baseline_mae", "mae_skill_vs_best_baseline", "failure_interpretation"]],
        on=["sensor_id", "model"],
        how="left",
    )
    decomp.to_csv(OUT / "mount_kaba_failure_decomposition.csv", index=False)

    inversion = joined[
        ["sensor_id", "model", "train_fraction", "cv_train_m2s", "n", "mae", "rmse", "bias", "spearman", "best_baseline_mae", "mae_skill_vs_best_baseline", "failure_interpretation"]
    ].copy()
    inversion.to_csv(OUT / "mount_kaba_parameter_inversion.csv", index=False)

    timing_rows = []
    for (sensor, model), g in pred.groupby(["sensor_id", "model"]):
        test = g[g["split"] == "test"].sort_values("t_s")
        if test.empty:
            test = g.sort_values("t_s")
        obs_peak = test.loc[test[observed_col].idxmax()]
        pred_peak = test.loc[test["R_pred"].idxmax()]
        timing_rows.append(
            {
                "sensor_id": sensor,
                "model": model,
                "obs_peak_t_s": float(obs_peak["t_s"]),
                "pred_peak_t_s": float(pred_peak["t_s"]),
                "peak_lag_s": float(pred_peak["t_s"] - obs_peak["t_s"]),
                "obs_first_R_ge_0p9_t_s": first_crossing_time(test, observed_col, 0.9),
                "pred_first_R_ge_0p9_t_s": first_crossing_time(test, "R_pred", 0.9),
                "threshold_lag_s": first_crossing_time(test, "R_pred", 0.9) - first_crossing_time(test, observed_col, 0.9),
            }
        )
    timing = pd.DataFrame(timing_rows).merge(
        joined[["sensor_id", "model", "mae_skill_vs_best_baseline", "failure_interpretation"]],
        on=["sensor_id", "model"],
        how="left",
    )
    timing.to_csv(OUT / "mount_kaba_timing_validation.csv", index=False)

    plot = joined[joined["model"].isin(["M4_constant_source", "M5_convolution_qt"])].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300, constrained_layout=True)
    x = np.arange(len(plot))
    colors = np.where(plot["mae_skill_vs_best_baseline"] > 0, "#2ca02c", "#d62728")
    ax.bar(x, plot["mae_skill_vs_best_baseline"], color=colors)
    ax.axhline(0, color="0.15", lw=0.9)
    ax.set_xticks(x, [f"{r.sensor_id}\n{r.model.replace('_', ' ')}" for r in plot.itertuples()], fontsize=7)
    ax.set_ylabel("MAE skill vs best M0/M1 baseline")
    ax.set_title("Fig. 18. Mount Kaba-san failure decomposition")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG / "fig18_mount_kaba_failure_decomposition.png", bbox_inches="tight")
    fig.savefig(FIG / "fig16_mount_kaba_failure_decomposition.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300, constrained_layout=True)
    timing_plot = timing[timing["model"].isin(["M4_constant_source", "M5_convolution_qt"])].copy()
    ax.bar(np.arange(len(timing_plot)), timing_plot["peak_lag_s"], color="#9467bd")
    ax.axhline(0, color="0.15", lw=0.9)
    ax.set_xticks(
        np.arange(len(timing_plot)),
        [f"{r.sensor_id}\n{r.model.replace('_', ' ')}" for r in timing_plot.itertuples()],
        fontsize=7,
    )
    ax.set_ylabel("Predicted peak lag (s)")
    ax.set_title("Fig. 19. Mount Kaba-san timing validation")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG / "fig19_mount_kaba_timing_validation.png", bbox_inches="tight")
    fig.savefig(FIG / "fig17_mount_kaba_FS_timing.png", bbox_inches="tight")
    plt.close(fig)

    print("wrote", OUT / "flume_confusion_matrix.csv")
    print("wrote", OUT / "flume_regime_classification_report.csv")
    print("wrote", OUT / "mount_kaba_failure_decomposition.csv")
    print("wrote", OUT / "mount_kaba_parameter_inversion.csv")
    print("wrote", OUT / "mount_kaba_timing_validation.csv")
    print("wrote", FIG / "fig17_flume_confusion_matrix.png")
    print("wrote", FIG / "fig18_mount_kaba_failure_decomposition.png")
    print("wrote", FIG / "fig19_mount_kaba_timing_validation.png")


if __name__ == "__main__":
    main()

