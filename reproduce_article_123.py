"""Reproduce the numerical evidence for the article 123 supplementary package.

Manuscript:
    A spectral dynamic-consolidation criterion for drainage transition in rapid
    saturated landslides.

The script regenerates the CSV data and figures used for the synthetic
benchmark. It uses only numpy, pandas and matplotlib.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle


OUT = Path(__file__).resolve().parent
FIG_OUT = OUT / "generated_figures"


def spectral_retention(pi: np.ndarray | float, terms: int = 500) -> np.ndarray | float:
    """Mean retained pressure fraction for double drainage and uniform source."""
    pi_arr = np.asarray(pi, dtype=float)
    out = np.zeros_like(pi_arr, dtype=float)
    m = np.arange(1, 2 * terms, 2, dtype=float)
    for idx, value in np.ndenumerate(pi_arr):
        out[idx] = np.sum(
            8.0
            * (1.0 - np.exp(-(m**2) * math.pi**2 * value))
            / (m**4 * math.pi**4 * value)
        )
    return float(out) if np.isscalar(pi) else out


def solve_threshold(target: float) -> float:
    lo, hi = 1e-6, 100.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if spectral_retention(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fd_retention(pi: float, n: int = 120, steps: int = 800) -> float:
    dx = 1.0 / (n + 1)
    dt = pi / steps
    r = dt / dx**2
    main = np.full(n, 1.0 + 2.0 * r)
    off = np.full(n - 1, -r)
    a = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    u = np.zeros(n)
    rhs_add = np.full(n, dt)
    for _ in range(steps):
        u = np.linalg.solve(a, u + rhs_add)
    return float(dx * np.sum(u) / pi)


def fem_retention(pi: float, elements: int = 90, steps: int = 700) -> float:
    n = elements - 1
    h = 1.0 / elements
    dt = pi / steps
    m_main = np.full(n, 2.0 * h / 3.0)
    m_off = np.full(n - 1, h / 6.0)
    k_main = np.full(n, 2.0 / h)
    k_off = np.full(n - 1, -1.0 / h)
    mass = np.diag(m_main) + np.diag(m_off, 1) + np.diag(m_off, -1)
    stiff = np.diag(k_main) + np.diag(k_off, 1) + np.diag(k_off, -1)
    load = np.full(n, h)
    lhs = mass + dt * stiff
    u = np.zeros(n)
    for _ in range(steps):
        u = np.linalg.solve(lhs, mass @ u + dt * load)
    return float((load @ u) / pi)


def regime_from_r(r_value: float) -> str:
    if r_value >= 0.9:
        return "nearly undrained"
    if r_value <= 0.1:
        return "drained"
    return "partly drained"


def build_data() -> dict[str, pd.DataFrame]:
    thresholds = {target: solve_threshold(target) for target in [0.9, 0.5, 0.1]}

    pi_curve = np.logspace(-4, 2, 360)
    retention_curve = pd.DataFrame(
        {"Pi": pi_curve, "R_Pi": spectral_retention(pi_curve, terms=700)}
    )

    validation_pi = np.array(
        [0.004, 0.02, thresholds[0.5], 0.12, 0.5, thresholds[0.1], 2.4]
    )
    validation_rows = []
    for value in validation_pi:
        spectral = spectral_retention(value, terms=1200)
        fd = fd_retention(float(value))
        fem = fem_retention(float(value))
        validation_rows.append(
            {
                "Pi": value,
                "R_spectral": spectral,
                "R_FD": fd,
                "R_FEM": fem,
                "FD_abs_error": abs(fd - spectral),
                "FEM_abs_error": abs(fem - spectral),
            }
        )
    validation = pd.DataFrame(validation_rows)

    reference_pi = np.logspace(-3, 1, 80)
    ref = spectral_retention(reference_pi, terms=5000)
    convergence_rows = []
    for terms in [5, 10, 20, 50, 100]:
        err = np.max(np.abs(spectral_retention(reference_pi, terms=terms) - ref))
        convergence_rows.append(
            {
                "Check": "spectral truncation",
                "Resolution": f"{terms} odd terms",
                "Maximum absolute error": err,
                "Comment": "series truncation relative to 5000-term reference",
            }
        )
    for n, steps in [(40, 300), (80, 600), (120, 800)]:
        err = max(abs(fd_retention(float(v), n=n, steps=steps) - spectral_retention(v)) for v in validation_pi)
        convergence_rows.append(
            {
                "Check": "finite difference",
                "Resolution": f"{n} interior nodes, {steps} time steps",
                "Maximum absolute error": err,
                "Comment": "implicit time integration",
            }
        )
    for elements, steps in [(30, 300), (60, 600), (90, 700)]:
        err = max(abs(fem_retention(float(v), elements=elements, steps=steps) - spectral_retention(v)) for v in validation_pi)
        convergence_rows.append(
            {
                "Check": "finite element",
                "Resolution": f"{elements} linear elements, {steps} time steps",
                "Maximum absolute error": err,
                "Comment": "consistent-mass Galerkin solution",
            }
        )
    convergence = pd.DataFrame(convergence_rows)

    cases = pd.DataFrame(
        [
            ["A", 5e-6, 2.0, 0.05, 50.0, 20.0, 30.0, 25.0, "short saturated shear pulse below a road cut"],
            ["B", 5e-6, 60.0, 0.05, 50.0, 20.0, 30.0, 25.0, "prolonged rapid movement affecting a transport corridor"],
            ["C", 1e-4, 60.0, 0.05, 50.0, 20.0, 30.0, 25.0, "more permeable drainage path or engineered relief drain"],
            ["D", 2e-5, 20.0, 0.03, 65.0, 18.0, 32.0, 31.0, "thin sheared zone behind a retaining structure"],
            ["E", 1e-6, 120.0, 0.08, 45.0, 16.0, 28.0, 23.0, "low-permeability colluvium supporting a rural road"],
        ],
        columns=[
            "Case",
            "cv_m2_s",
            "T_s",
            "h_m",
            "sigma_eff0_kPa",
            "Pu_kPa",
            "phi_deg",
            "tau_kPa",
            "Infrastructure reading",
        ],
    )
    cases["Pi"] = cases["cv_m2_s"] * cases["T_s"] / cases["h_m"] ** 2
    cases["R_Pi"] = spectral_retention(cases["Pi"].to_numpy())
    cases["retained_pressure_kPa"] = cases["Pu_kPa"] * cases["R_Pi"]
    cases["FS_PD"] = (
        (cases["sigma_eff0_kPa"] - cases["retained_pressure_kPa"])
        * np.tan(np.deg2rad(cases["phi_deg"]))
        / cases["tau_kPa"]
    )
    cases["Regime"] = [regime_from_r(v) for v in cases["R_Pi"]]

    literature = pd.DataFrame(
        [
            ["Velocity-based drainage choice", "movement rate or loading rate", "quick practical decision", "does not include h or cv explicitly", "replaces binary label with retained pressure R(Pi)"],
            ["Conventional drained/undrained stability", "selected strength envelope", "clear design bounds", "no intermediate state unless analyst brackets cases", "computes partly drained effective stress continuously"],
            ["Classical consolidation", "time factor", "well established diffusion physics", "not directly linked to landslide FS in screening workflows", "couples spectral pressure retention to FS"],
            ["Transient seepage analysis", "boundary flux and hydraulic gradients", "captures rainfall or seepage forcing", "may not represent rapid internal pressure generation", "adds a source-driven rapid-event operator"],
            ["Fully coupled hydromechanical FEM", "field equations, constitutive law and geometry", "most general numerical route", "costly and opaque for screening inventories", "provides a reproducible pre-analysis and check on regime choice"],
        ],
        columns=["Approach", "Primary control variable", "Strength", "Limitation for rapid saturated sliding", "Contribution of the present criterion"],
    )

    external = pd.DataFrame(
        [
            ["Iverson pore-pressure feedback model", "Motion styles depend on coupling between pressure generation and drainage", "The proposed Pi and R(Pi) isolate the dissipation part of that feedback", "External conceptual consistency, not direct calibration"],
            ["USGS Oso ring-shear data release", "Undrained tests used closed chamber water lines; naturally drained tests used open lines", "The limiting cases match the model endpoints R approximately 1 for closed drainage and lower R for open drainage", "A future calibration can ingest released time series"],
            ["Risk frameworks for infrastructure slopes", "Road vulnerability depends on physical response, not only hazard presence", "The retained-pressure variable supplies a transparent descriptor for road cuts, embankments and retaining-system back-slopes", "Supports engineering interpretation but does not replace site-specific risk modelling"],
        ],
        columns=["External anchor", "Published or released observation", "How the criterion is checked", "Limit of the check"],
    )

    base = cases.loc[cases["Case"] == "B"].iloc[0].to_dict()
    sensitivity_rows = []
    for par in ["tau_kPa", "Pu_kPa", "h_m", "cv_m2_s", "T_s", "phi_deg", "sigma_eff0_kPa"]:
        pert = dict(base)
        pert[par] *= 1.05
        pi = pert["cv_m2_s"] * pert["T_s"] / pert["h_m"] ** 2
        r_value = spectral_retention(pi)
        fs = (
            (pert["sigma_eff0_kPa"] - pert["Pu_kPa"] * r_value)
            * math.tan(math.radians(pert["phi_deg"]))
            / pert["tau_kPa"]
        )
        sensitivity_rows.append(
            {
                "Parameter": par.replace("_kPa", "").replace("_m2_s", "").replace("_m", ""),
                "Base value": base[par],
                "Normalized sensitivity of FS_PD": (fs - base["FS_PD"]) / (0.05 * base["FS_PD"]),
                "Interpretation": "stabilizing when increased" if fs > base["FS_PD"] else "destabilizing when increased",
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)

    thresholds_df = pd.DataFrame(
        [
            ["nearly undrained / partly drained", "R(Pi)=0.9", thresholds[0.9]],
            ["central transition", "R(Pi)=0.5", thresholds[0.5]],
            ["partly drained / drained", "R(Pi)=0.1", thresholds[0.1]],
        ],
        columns=["Boundary", "Criterion", "Pi"],
    )

    return {
        "retention_curve": retention_curve,
        "validation_fd_fem": validation,
        "convergence_checks": convergence,
        "benchmark_cases": cases,
        "literature_comparison": literature,
        "external_consistency_check": external,
        "sensitivity": sensitivity,
        "thresholds": thresholds_df,
    }


def save_csv(data: dict[str, pd.DataFrame]) -> None:
    for name, frame in data.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)


def make_figures(data: dict[str, pd.DataFrame]) -> None:
    FIG_OUT.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220, constrained_layout=True)
    slope = Polygon([[0.5, 0.8], [6.3, 0.8], [6.3, 2.9], [0.5, 1.45]], closed=True, facecolor="#d9cfbb", edgecolor="black", lw=1.0)
    ax.add_patch(slope)
    ax.plot([0.8, 5.6], [1.60, 2.65], color="#2b8cbe", lw=4)
    ax.plot([0.6, 6.2], [1.48, 2.95], color="#6b6b6b", lw=2)
    ax.add_patch(Rectangle((4.8, 3.25), 1.0, 0.12, facecolor="none", edgecolor="black", lw=1.2))
    ax.add_patch(Rectangle((4.8, 3.37), 1.0, 0.12, facecolor="none", edgecolor="black", lw=1.2))
    ax.annotate("upslope drained\nboundary", xy=(0.85, 1.58), xytext=(1.05, 3.55), ha="center", arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("rapid loading\nduration T", xy=(4.55, 2.55), xytext=(3.85, 3.75), ha="center", arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("road / civil corridor", xy=(5.3, 3.43), xytext=(5.45, 3.85), ha="center", arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(2.8, 2.20, "saturated shear band\nthickness h", rotation=14, ha="center", va="center")
    ax.text(3.4, 0.45, "Output: retained pore pressure R(Pi)Pu and partly drained stability", ha="center")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    fig.savefig(FIG_OUT / "figure_1_conceptual_slope.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    curve = data["retention_curve"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220, constrained_layout=True)
    ax.semilogx(curve["Pi"], curve["R_Pi"], color="#084081", lw=2)
    for target in [0.9, 0.5, 0.1]:
        pi = float(data["thresholds"].loc[data["thresholds"]["Criterion"] == f"R(Pi)={target}", "Pi"].iloc[0])
        ax.axhline(target, color="0.35", ls="--", lw=0.8)
        ax.axvline(pi, color="#d7301f", ls=":", lw=1.0)
        ax.text(pi * 1.08, target + 0.025, f"R={target}, Pi={pi:.4g}", fontsize=8)
    ax.set_xlabel("Dynamic consolidation number, Pi = cvT/h^2")
    ax.set_ylabel("Retained pore-pressure fraction, R(Pi)")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(FIG_OUT / "figure_2_retention_curve.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    validation = data["validation_fd_fem"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220, constrained_layout=True)
    ax.loglog(validation["Pi"], validation["FD_abs_error"], "o-", label="implicit finite differences")
    ax.loglog(validation["Pi"], validation["FEM_abs_error"], "s-", label="linear FEM")
    ax.set_xlabel("Pi")
    ax.set_ylabel("Absolute error in R(Pi)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIG_OUT / "figure_3_numerical_validation_errors.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pi = np.logspace(-4, 2, 300)
    r = spectral_retention(pi)
    sigma0, pu, phi, tau = 50.0, 20.0, math.radians(30.0), 25.0
    fs = (sigma0 - pu * r) * math.tan(phi) / tau
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220, constrained_layout=True)
    ax.semilogx(pi, fs, color="#238b45", lw=2)
    ax.axhline(1.0, color="black", ls="--", lw=0.9)
    ax.text(1.8e-4, 1.015, "FS = 1", fontsize=9)
    ax.set_xlabel("Pi")
    ax.set_ylabel("Partly drained factor of safety")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(FIG_OUT / "figure_4_stability_shift.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    h_vals = np.logspace(-2, -0.3, 240)
    t_vals = np.logspace(0, 3, 240)
    h_grid, t_grid = np.meshgrid(h_vals, t_vals)
    r_grid = spectral_retention(5e-6 * t_grid / h_grid**2)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=220, constrained_layout=True)
    ax.contourf(h_grid, t_grid, r_grid, levels=[0, 0.1, 0.9, 1.01], colors=["#d9f0d3", "#fff7bc", "#fdd49e"], alpha=0.95)
    ax.contour(h_grid, t_grid, r_grid, levels=[0.1, 0.5, 0.9], colors="black", linewidths=[1.0, 1.3, 1.0], linestyles=["solid", "dashed", "dashdot"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Shear-band thickness h (m)")
    ax.set_ylabel("Rapid-loading duration T (s)")
    ax.text(0.012, 750, "drained", fontsize=10)
    ax.text(0.052, 24, "partly drained", fontsize=10)
    ax.text(0.18, 2.2, "nearly undrained", fontsize=10)
    handles = [
        Line2D([0], [0], color="black", lw=1.0, linestyle="solid", label="R=0.1"),
        Line2D([0], [0], color="black", lw=1.3, linestyle="dashed", label="R=0.5"),
        Line2D([0], [0], color="black", lw=1.0, linestyle="dashdot", label="R=0.9"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True, edgecolor="black", title="Retention contours")
    ax.set_title("Regime map for cv = 5e-6 m2/s")
    fig.savefig(FIG_OUT / "figure_5_regime_map.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sens = data["sensitivity"].copy()
    sens["abs"] = sens["Normalized sensitivity of FS_PD"].abs()
    sens = sens.sort_values("abs", ascending=True)
    colors = ["#b2182b" if v < 0 else "#2166ac" for v in sens["Normalized sensitivity of FS_PD"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220, constrained_layout=True)
    ax.barh(sens["Parameter"], sens["Normalized sensitivity of FS_PD"], color=colors)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("Normalized log-sensitivity of partly drained FS")
    ax.set_ylabel("Parameter")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG_OUT / "figure_6_sensitivity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    data = build_data()
    save_csv(data)
    make_figures(data)
    print(f"CSV files written to {OUT}")
    print(f"Figures written to {FIG_OUT}")


if __name__ == "__main__":
    main()
