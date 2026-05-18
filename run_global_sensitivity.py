"""Global sensitivity benchmark for the retained-pressure screening criterion."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproduce_article_123 import spectral_retention


OUT = Path(__file__).resolve().parent
FIG_OUT = OUT / "generated_figures"


def lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty((n, d), dtype=float)
    base = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        values[:, j] = rng.permutation(base[:, j])
    return values


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def prcc(inputs: pd.DataFrame, output: pd.Series) -> dict[str, float]:
    """Partial rank correlation by rank residualization."""
    ranked_x = inputs.rank(method="average").to_numpy(dtype=float)
    ranked_y = output.rank(method="average").to_numpy(dtype=float)
    ranked_y = (ranked_y - ranked_y.mean()) / ranked_y.std()
    result = {}
    for j, name in enumerate(inputs.columns):
        xj = ranked_x[:, j]
        xj = (xj - xj.mean()) / xj.std()
        others = np.delete(ranked_x, j, axis=1)
        others = (others - others.mean(axis=0)) / others.std(axis=0)
        design = np.column_stack([np.ones(len(others)), others])
        beta_x, *_ = np.linalg.lstsq(design, xj, rcond=None)
        beta_y, *_ = np.linalg.lstsq(design, ranked_y, rcond=None)
        rx = xj - design @ beta_x
        ry = ranked_y - design @ beta_y
        result[name] = float(np.corrcoef(rx, ry)[0, 1])
    return result


def main() -> None:
    FIG_OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(12345)
    n = 6000
    u = lhs_unit(n, 7, rng)
    # Conservative screening ranges around the benchmark cases.
    h = 10 ** np.interp(u[:, 0], [0, 1], [math.log10(0.02), math.log10(0.12)])
    cv = 10 ** np.interp(u[:, 1], [0, 1], [math.log10(1e-6), math.log10(2e-4)])
    T = 10 ** np.interp(u[:, 2], [0, 1], [math.log10(2.0), math.log10(180.0)])
    Pu = np.interp(u[:, 3], [0, 1], [5.0, 35.0])
    phi = np.interp(u[:, 4], [0, 1], [24.0, 38.0])
    tau = np.interp(u[:, 5], [0, 1], [12.0, 42.0])
    sigma_eff0 = np.interp(u[:, 6], [0, 1], [25.0, 90.0])
    pi = cv * T / h**2
    retained = spectral_retention(pi)
    fs_pd = (sigma_eff0 - Pu * retained) * np.tan(np.deg2rad(phi)) / tau
    psi = fs_pd - 1.0
    data = pd.DataFrame({
        "h_m": h,
        "cv_m2_s": cv,
        "T_s": T,
        "Pu_kPa": Pu,
        "phi_deg": phi,
        "tau_kPa": tau,
        "sigma_eff0_kPa": sigma_eff0,
        "Pi": pi,
        "R_Pi": retained,
        "FS_PD": fs_pd,
        "Psi": psi,
    })
    data.to_csv(OUT / "global_sensitivity_lhs_samples.csv", index=False)
    inputs = ["h_m", "cv_m2_s", "T_s", "Pu_kPa", "phi_deg", "tau_kPa", "sigma_eff0_kPa"]
    rows = []
    for variable in inputs:
        rows.append({
            "input": variable,
            "spearman_RPi": spearman_corr(data[variable].to_numpy(), data["R_Pi"].to_numpy()),
            "spearman_FS_PD": spearman_corr(data[variable].to_numpy(), data["FS_PD"].to_numpy()),
            "spearman_Psi": spearman_corr(data[variable].to_numpy(), data["Psi"].to_numpy()),
        })
    rank = pd.DataFrame(rows)
    rank["abs_spearman_FS_PD"] = rank["spearman_FS_PD"].abs()
    rank = rank.sort_values("abs_spearman_FS_PD", ascending=False)
    rank.to_csv(OUT / "global_sensitivity_spearman.csv", index=False)
    input_frame = data[inputs]
    prcc_rows = []
    prcc_r = prcc(input_frame, data["R_Pi"])
    prcc_fs = prcc(input_frame, data["FS_PD"])
    prcc_psi = prcc(input_frame, data["Psi"])
    for variable in inputs:
        prcc_rows.append({
            "input": variable,
            "prcc_RPi": prcc_r[variable],
            "prcc_FS_PD": prcc_fs[variable],
            "prcc_Psi": prcc_psi[variable],
            "abs_prcc_FS_PD": abs(prcc_fs[variable]),
        })
    prcc_frame = pd.DataFrame(prcc_rows).sort_values("abs_prcc_FS_PD", ascending=False)
    prcc_frame.to_csv(OUT / "global_sensitivity_prcc.csv", index=False)
    summary = pd.DataFrame([
        ["n_samples", n],
        ["failure_fraction_FS_lt_1", float(np.mean(fs_pd < 1.0))],
        ["median_R_Pi", float(np.median(retained))],
        ["median_FS_PD", float(np.median(fs_pd))],
        ["p05_FS_PD", float(np.percentile(fs_pd, 5))],
        ["p95_FS_PD", float(np.percentile(fs_pd, 95))],
    ], columns=["metric", "value"])
    summary.to_csv(OUT / "global_sensitivity_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    ordered = rank.sort_values("abs_spearman_FS_PD")
    colors = ["#b2182b" if v < 0 else "#2166ac" for v in ordered["spearman_FS_PD"]]
    ax.barh(ordered["input"], ordered["spearman_FS_PD"], color=colors)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel("Spearman rank correlation with FS_PD")
    ax.set_ylabel("Input variable")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG_OUT / "figure_12_global_sensitivity_spearman.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig12.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    ordered = prcc_frame.sort_values("abs_prcc_FS_PD")
    colors = ["#b2182b" if v < 0 else "#2166ac" for v in ordered["prcc_FS_PD"]]
    ax.barh(ordered["input"], ordered["prcc_FS_PD"], color=colors)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel("PRCC with FS_PD")
    ax.set_ylabel("Input variable")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG_OUT / "figure_13_global_sensitivity_prcc.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig13.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    ax.scatter(np.log10(data["Pi"]), data["FS_PD"], s=4, alpha=0.22, color="#2b8cbe")
    ax.axhline(1.0, color="black", ls="--", lw=0.9)
    ax.set_xlabel("log10(Pi)")
    ax.set_ylabel("Partly drained factor of safety")
    ax.grid(alpha=0.25)
    fig.savefig(FIG_OUT / "figure_14_global_sensitivity_pi_fs.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig14.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Global sensitivity samples written: {n}")


if __name__ == "__main__":
    main()
