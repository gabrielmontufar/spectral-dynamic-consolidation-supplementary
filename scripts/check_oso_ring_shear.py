"""Check the retained-pressure operator against USGS Oso ring-shear data.

The script downloads public ScienceBase files from the USGS data release
10.5066/F7KH0KSD and fits the pressure-dissipation part of the operator to
laboratory consolidation records. It is a held-out laboratory consistency
check, not a field-scale landslide calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
RAW = OUT / "external_data" / "oso_ring_shear"
FIG_OUT = OUT / "generated_figures"
SCIENCEBASE_JSON = (
    "https://www.sciencebase.gov/catalog/item/"
    "590b7111e4b0e541a0378c9b?format=json"
)
SCIENCEBASE_PAGE = (
    "https://www.usgs.gov/data/data-ring-shear-strength-testing-"
    "glaciolacustrine-silty-clay-2014-oso-washington-landslide"
)
NORMALIZED_OFFLINE = OUT / "oso_ring_shear_normalized_records.csv"


def survival_double(pi: np.ndarray | float, terms: int = 120) -> np.ndarray | float:
    """Mean survival fraction for double-drainage decay with no new source."""
    pi_arr = np.asarray(pi, dtype=float)
    out = np.zeros_like(pi_arr, dtype=float)
    m = np.arange(1, 2 * terms, 2, dtype=float)
    weights = 8.0 / (m**2 * math.pi**2)
    for idx, value in np.ndenumerate(pi_arr):
        out[idx] = np.sum(weights * np.exp(-(m**2) * math.pi**2 * value))
    return float(out) if np.isscalar(pi) else out


def survival_single(pi: np.ndarray | float, terms: int = 120) -> np.ndarray | float:
    """Mean survival fraction for one drained and one impermeable boundary."""
    pi_arr = np.asarray(pi, dtype=float)
    out = np.zeros_like(pi_arr, dtype=float)
    n = np.arange(terms, dtype=float)
    alpha = (n + 0.5) * math.pi
    weights = 2.0 / (alpha**2)
    for idx, value in np.ndenumerate(pi_arr):
        out[idx] = np.sum(weights * np.exp(-(alpha**2) * value))
    return float(out) if np.isscalar(pi) else out


def download_oso_files() -> list[Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    catalog_path = RAW / "sciencebase_item.json"
    if not catalog_path.exists():
        urllib.request.urlretrieve(SCIENCEBASE_JSON, catalog_path)
    item = json.loads(catalog_path.read_text(encoding="utf-8"))
    selected = [
        f for f in item["files"]
        if "consolidation" in f["name"].lower() and f.get("size", 0) < 900000
    ]
    # Keep the check reproducible and fast: use a balanced subset across
    # material labels and normal stresses, preferring compact files.
    selected = sorted(selected, key=lambda f: f.get("size", 0))[:12]
    paths = []
    for file_info in selected:
        dest = RAW / file_info["name"]
        if not dest.exists():
            urllib.request.urlretrieve(file_info["downloadUri"], dest)
        paths.append(dest)
    return paths


def load_record(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame = frame.rename(columns={
        "time (s)": "time_s",
        "pore-water pressure (kPa)": "pore_pressure_kpa",
        "specimen thickness (mm)": "thickness_mm",
    })
    frame = frame[["time_s", "pore_pressure_kpa", "thickness_mm"]].dropna()
    frame = frame.drop_duplicates("time_s")
    frame = frame.sort_values("time_s")
    return frame


def normalize_decay(frame: pd.DataFrame) -> pd.DataFrame | None:
    if len(frame) < 50:
        return None
    time = frame["time_s"].to_numpy(dtype=float)
    pressure = frame["pore_pressure_kpa"].to_numpy(dtype=float)
    thickness = frame["thickness_mm"].to_numpy(dtype=float)
    if time[-1] <= time[0]:
        return None
    # Robust endpoints: early peak and late residual median.
    p0 = float(np.nanpercentile(pressure[: max(10, len(pressure) // 20)], 90))
    p_inf = float(np.nanmedian(pressure[int(0.9 * len(pressure)):]))
    amplitude = p0 - p_inf
    if amplitude <= 3.0:
        return None
    retained = (pressure - p_inf) / amplitude
    keep = np.isfinite(retained) & (retained >= -0.2) & (retained <= 1.3)
    if keep.sum() < 50:
        return None
    result = pd.DataFrame({
        "time_s": time[keep] - time[keep][0],
        "retained_observed": retained[keep],
        "thickness_m": thickness[keep] / 1000.0,
    })
    return result[result["time_s"] >= 0.0].reset_index(drop=True)


def fit_model(record: pd.DataFrame, model_name: str) -> dict[str, float]:
    model = survival_double if model_name == "double" else survival_single
    time = record["time_s"].to_numpy(dtype=float)
    obs = record["retained_observed"].to_numpy(dtype=float)
    thickness = float(np.nanmedian(record["thickness_m"]))
    split = max(20, int(0.6 * len(record)))
    train_t, train_y = time[:split], obs[:split]
    val_t, val_y = time[split:], obs[split:]
    if len(val_t) < 20:
        val_t, val_y = train_t, train_y
    # Search cv over a broad laboratory range, then refine around the best value.
    cv_grid = np.logspace(-9, -3, 121)
    def rmse_for(cv: float, t: np.ndarray, y: np.ndarray) -> float:
        pi = cv * t / max(thickness**2, 1e-8)
        pred = model(pi)
        return float(np.sqrt(np.mean((pred - y) ** 2)))
    train_rmse = np.array([rmse_for(cv, train_t, train_y) for cv in cv_grid])
    best = cv_grid[int(np.argmin(train_rmse))]
    refined = np.logspace(math.log10(best) - 0.2, math.log10(best) + 0.2, 61)
    train_rmse_refined = np.array([rmse_for(cv, train_t, train_y) for cv in refined])
    cv = float(refined[int(np.argmin(train_rmse_refined))])
    pred_val = model(cv * val_t / max(thickness**2, 1e-8))
    residual = pred_val - val_y
    rmse_val = float(np.sqrt(np.mean((pred_val - val_y) ** 2)))
    ss_res = float(np.sum((pred_val - val_y) ** 2))
    ss_tot = float(np.sum((val_y - np.mean(val_y)) ** 2))
    r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "model": model_name,
        "cv_fit_m2_s": cv,
        "thickness_m": thickness,
        "n_train": int(len(train_t)),
        "n_heldout": int(len(val_t)),
        "rmse_train": float(np.min(train_rmse_refined)),
        "rmse_heldout": rmse_val,
        "mae_heldout": float(np.mean(np.abs(residual))),
        "bias_heldout": float(np.mean(residual)),
        "p95_abs_residual": float(np.percentile(np.abs(residual), 95)),
        "r2_heldout": r2_val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check retained-pressure operators against Oso ring-shear data.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the included normalized Oso records instead of downloading raw ScienceBase files.",
    )
    args = parser.parse_args()
    FIG_OUT.mkdir(exist_ok=True)
    if args.offline and NORMALIZED_OFFLINE.exists():
        normalized_records = [
            (name, frame.drop(columns=["source_file"]).reset_index(drop=True))
            for name, frame in pd.read_csv(NORMALIZED_OFFLINE).groupby("source_file", sort=False)
        ]
    else:
        paths = download_oso_files()
        normalized_records = []
        for path in paths:
            raw = load_record(path)
            record = normalize_decay(raw)
            if record is None:
                continue
            normalized_records.append((path.name, record))
        if normalized_records:
            pd.concat(
                [
                    frame.assign(source_file=name)
                    for name, frame in normalized_records
                ],
                ignore_index=True,
            ).to_csv(NORMALIZED_OFFLINE, index=False)
    summary_rows = []
    prediction_rows = []
    for source_name, record in normalized_records:
        if len(record) > 500:
            record = record.iloc[np.linspace(0, len(record) - 1, 500).astype(int)].reset_index(drop=True)
        fits = [fit_model(record, "double"), fit_model(record, "single")]
        best = min(fits, key=lambda row: row["rmse_heldout"])
        for fit in fits:
            fit["source_file"] = source_name
            fit["best_model_for_record"] = best["model"]
            fit["sciencebase_page"] = SCIENCEBASE_PAGE
            summary_rows.append(fit)
            model = survival_double if fit["model"] == "double" else survival_single
            time = record["time_s"].to_numpy(dtype=float)
            pi = fit["cv_fit_m2_s"] * time / max(fit["thickness_m"] ** 2, 1e-8)
            pred = model(pi)
            sample_idx = np.linspace(0, len(record) - 1, min(180, len(record))).astype(int)
            for idx in sample_idx:
                prediction_rows.append({
                    "source_file": source_name,
                    "model": fit["model"],
                    "time_s": float(time[idx]),
                    "retained_observed": float(record["retained_observed"].iloc[idx]),
                    "retained_predicted": float(pred[idx]),
                })
    summary = pd.DataFrame(summary_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary.to_csv(OUT / "oso_ring_shear_consistency_metrics.csv", index=False)
    predictions.to_csv(OUT / "oso_ring_shear_consistency_predictions.csv", index=False)
    if not summary.empty:
        agg = summary.groupby("model").agg(
            records=("source_file", "count"),
            median_rmse_heldout=("rmse_heldout", "median"),
            median_mae_heldout=("mae_heldout", "median"),
            median_bias_heldout=("bias_heldout", "median"),
            median_p95_abs_residual=("p95_abs_residual", "median"),
            median_r2_heldout=("r2_heldout", "median"),
        ).reset_index()
        agg.to_csv(OUT / "oso_ring_shear_consistency_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
        data = [summary.loc[summary["model"] == model, "rmse_heldout"] for model in ["double", "single"]]
        ax.boxplot(data, tick_labels=["double drainage", "single drainage"], showfliers=True)
        ax.set_ylabel("Held-out RMSE in normalized retained pressure")
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(FIG_OUT / "figure_10_oso_consistency_rmse.png", bbox_inches="tight", facecolor="white")
        fig.savefig(FIG_OUT / "Fig10.tif", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        best_file = summary.sort_values("rmse_heldout").iloc[0]["source_file"]
        subset = predictions[predictions["source_file"] == best_file]
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
        obs = subset[subset["model"] == "double"]
        ax.plot(obs["time_s"], obs["retained_observed"], "k.", ms=2.5, label="observed")
        for model, color in [("double", "#084081"), ("single", "#b2182b")]:
            frame = subset[subset["model"] == model]
            ax.plot(frame["time_s"], frame["retained_predicted"], color=color, lw=1.8, label=f"{model} fit")
        ax.set_xlabel("Time since start of consolidation (s)")
        ax.set_ylabel("Normalized retained pore pressure")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(FIG_OUT / "figure_11_oso_consistency_example.png", bbox_inches="tight", facecolor="white")
        fig.savefig(FIG_OUT / "Fig11.tif", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"Checked {summary['source_file'].nunique() if not summary.empty else 0} Oso consolidation records")
    print(f"Metrics written to {OUT}")


if __name__ == "__main__":
    main()
