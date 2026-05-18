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


def survival_double(pi: np.ndarray, terms: int = 120) -> np.ndarray:
    pi = np.maximum(np.asarray(pi, dtype=float), 0.0)
    m = np.arange(1, 2 * terms, 2, dtype=float)
    weights = 8.0 / (m * m * np.pi * np.pi)
    rate = m * m * np.pi * np.pi
    return np.exp(-pi[:, None] * rate[None, :]) @ weights


def survival_single(pi: np.ndarray, terms: int = 120) -> np.ndarray:
    pi = np.maximum(np.asarray(pi, dtype=float), 0.0)
    n = np.arange(terms, dtype=float)
    a = (n + 0.5) * np.pi
    weights = 2.0 / (a * a)
    rate = a * a
    return np.exp(-pi[:, None] * rate[None, :]) @ weights


def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask])))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    s = pd.Series(a).corr(pd.Series(b), method="spearman")
    return float(s) if pd.notna(s) else float("nan")


def build_oso_rows() -> pd.DataFrame:
    records = pd.read_csv(ROOT / "data_normalized" / "oso_ring_shear_normalized.csv")
    vm = pd.read_csv(ROOT / "data_normalized" / "validation_master_table.csv")
    cv = (
        vm.loc[vm["model"] == "double", ["source_file", "cv_fit_m2_s", "thickness_m"]]
        .drop_duplicates("source_file")
        .set_index("source_file")
    )
    rows = []
    for source, g in records.groupby("source_file"):
        if source not in cv.index:
            continue
        h = float(cv.loc[source, "thickness_m"])
        c = float(cv.loc[source, "cv_fit_m2_s"])
        gg = g.sort_values("time_s").iloc[:: max(1, len(g) // 250)].copy()
        t = (gg["time_s"] - gg["time_s"].min()).to_numpy(dtype=float)
        pi = c * t / max(h * h, 1e-12)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "Oso",
                    "record_id": source,
                    "scale": "laboratory",
                    "h_m": h,
                    "cv_m2_s": c,
                    "T_s": t,
                    "Pu_ref": np.nan,
                    "Pi": pi,
                    "R_value": gg["retained_observed"].to_numpy(dtype=float),
                    "response_type": "R_obs",
                    "domain_flag": "in_domain",
                    "boundary_source_class": "double-drainage laboratory consolidation",
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_mount_rows() -> pd.DataFrame:
    pred_path = OUT / "mount_kabasan_predictive_predictions.csv"
    met_path = OUT / "mount_kabasan_predictive_metrics.csv"
    if not pred_path.exists() or not met_path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(pred_path)
    metrics = pd.read_csv(met_path)
    obs_col = "R_star_obs" if "R_star_obs" in pred.columns else "R_obs"
    rows = []
    for sensor, g in pred[pred["model"] == "M4_constant_source"].groupby("sensor_id"):
        m = metrics[(metrics["sensor_id"] == sensor) & (metrics["model"] == "M4_constant_source")]
        if m.empty:
            continue
        cv = float(m["cv_train_m2s"].iloc[0])
        h = 0.5
        gg = g.sort_values("t_s").copy()
        t = (gg["t_s"] - gg["t_s"].min()).to_numpy(dtype=float)
        pi = cv * t / max(h * h, 1e-12)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "Mount Kaba-san",
                    "record_id": sensor,
                    "scale": "field-transfer",
                    "h_m": h,
                    "cv_m2_s": cv,
                    "T_s": t,
                    "Pu_ref": float(m["Pu_train"].iloc[0]) if "Pu_train" in m.columns else np.nan,
                    "Pi": pi,
                    "R_value": gg[obs_col].to_numpy(dtype=float),
                    "response_type": "R_star_obs",
                    "domain_flag": "field_transfer_boundary",
                    "boundary_source_class": "field transfer with unmodelled boundary impedance and source history",
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_flume_rows() -> pd.DataFrame:
    pred_path = OUT / "flume_leave_one_experiment_predictions.csv"
    if not pred_path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(pred_path)
    rows = []
    for (exp, station), g in pred.groupby(["experiment_file", "station"]):
        gg = g.sort_values("t_s").iloc[:: max(1, len(g) // 250)].copy()
        # The flume layer is a regime-screening check. Pi_proxy is used only for
        # negative-control ordering, not as site-calibrated consolidation Pi.
        t = (gg["t_s"] - gg["t_s"].min()).to_numpy(dtype=float)
        pi_proxy = (t + 1.0) / (np.nanmax(t) + 1.0)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "USGS flume",
                    "record_id": f"{exp}:{station}",
                    "scale": "physical-flume",
                    "h_m": np.nan,
                    "cv_m2_s": np.nan,
                    "T_s": t,
                    "Pu_ref": np.nan,
                    "Pi": pi_proxy,
                    "R_value": gg["ru_obs"].clip(0, 1).to_numpy(dtype=float),
                    "response_type": "ru_obs_clipped",
                    "domain_flag": "regime_screen_only",
                    "boundary_source_class": "regime-screening Pi proxy; not calibrated consolidation Pi",
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collapse_metrics(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1234)
    for name, g in matrix.groupby("dataset"):
        pi = g["Pi"].to_numpy(dtype=float)
        obs = np.clip(g["R_value"].to_numpy(dtype=float), -0.25, 1.25)
        physical = survival_double(pi)
        single = survival_single(pi)
        constant = np.full_like(obs, np.nanmedian(obs))
        shuffled_pi = survival_double(rng.permutation(pi))
        random_h_pi = pi * rng.lognormal(mean=0.0, sigma=1.0, size=len(pi))
        random_h = survival_double(random_h_pi)
        velocity_only = 1.0 - (pi - np.nanmin(pi)) / max(np.nanmax(pi) - np.nanmin(pi), 1e-12)
        for model, pred in [
            ("physical_double_drainage", physical),
            ("single_drainage_envelope", single),
            ("constant_R_baseline", constant),
            ("shuffled_Pi_control", shuffled_pi),
            ("random_h_control", random_h),
            ("velocity_only_proxy", velocity_only),
        ]:
            rows.append(
                {
                    "dataset": name,
                    "model_or_control": model,
                    "n_points": int(len(obs)),
                    "median_abs_error": mae(obs, pred),
                    "spearman_Pi_response": spearman(pi, obs),
                    "coverage_5_95_envelope": float(
                        np.mean(
                            (obs >= np.minimum(physical, single) - 0.05)
                            & (obs <= np.maximum(physical, single) + 0.05)
                        )
                    ),
                }
            )
    result = pd.DataFrame(rows)
    phys = result[result["model_or_control"] == "physical_double_drainage"][
        ["dataset", "median_abs_error"]
    ].rename(columns={"median_abs_error": "physical_mae"})
    result = result.merge(phys, on="dataset", how="left")
    result["skill_vs_physical"] = 1.0 - result["median_abs_error"] / result["physical_mae"]
    return result


def negative_report(metrics: pd.DataFrame) -> pd.DataFrame:
    controls = metrics[metrics["model_or_control"].str.contains("control|baseline|proxy")].copy()
    controls["expected_behavior"] = np.where(
        controls["skill_vs_physical"] < 0,
        "degrades_vs_physical",
        "does_not_degrade",
    )
    return controls[
        [
            "dataset",
            "model_or_control",
            "median_abs_error",
            "physical_mae",
            "skill_vs_physical",
            "expected_behavior",
        ]
    ]


def make_figures(matrix: pd.DataFrame, metrics: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    pi_grid = np.logspace(-4, 1, 300)
    dd = survival_double(pi_grid)
    sd = survival_single(pi_grid)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300, constrained_layout=True)
    ax.fill_between(pi_grid, np.minimum(dd, sd), np.maximum(dd, sd), color="#d9e8fb", alpha=0.8, label="DD-SD envelope")
    ax.plot(pi_grid, dd, color="#1f77b4", lw=1.0, label="double drainage")
    ax.plot(pi_grid, sd, color="#ff7f0e", lw=1.0, label="single drainage")
    for dataset, g in matrix.groupby("dataset"):
        ax.plot(g["Pi"], np.clip(g["R_value"], -0.25, 1.25), ".", ms=2.0, alpha=0.35, label=dataset)
    ax.set_xscale("log")
    ax.set_xlabel("Dynamic consolidation number or declared Pi proxy")
    ax.set_ylabel("Observed R or transfer-normalized R*")
    ax.set_title("Dimensionless-collapse envelope")
    ax.legend(fontsize=7, ncols=2)
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "dimensionless_collapse_envelope.png", bbox_inches="tight")
    plt.close(fig)

    neg = negative_report(metrics)
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=300, constrained_layout=True)
    plot = neg.pivot_table(index="model_or_control", values="skill_vs_physical", aggfunc="median").sort_values("skill_vs_physical")
    ax.barh(plot.index, plot["skill_vs_physical"], color="#b44b4b")
    ax.axvline(0, color="0.15", lw=0.9)
    ax.set_xlabel("Skill relative to physical Pi-R model (negative is worse)")
    ax.set_title("Negative-control skill degradation")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG / "negative_control_skill.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    parts = [build_oso_rows(), build_mount_rows(), build_flume_rows()]
    matrix = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    matrix.to_csv(OUT / "dimensionless_validation_matrix.csv", index=False)
    metrics = collapse_metrics(matrix)
    metrics.to_csv(OUT / "dimensionless_collapse_metrics.csv", index=False)
    negative_report(metrics).to_csv(OUT / "negative_control_report.csv", index=False)
    make_figures(matrix, metrics)
    print("wrote", OUT / "dimensionless_validation_matrix.csv")
    print("wrote", OUT / "dimensionless_collapse_metrics.csv")
    print("wrote", OUT / "negative_control_report.csv")
    print("wrote", FIG / "dimensionless_collapse_envelope.png")
    print("wrote", FIG / "negative_control_skill.png")


if __name__ == "__main__":
    main()
