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


def single_drainage_retention(pi: np.ndarray | float, terms: int = 500) -> np.ndarray | float:
    """Mean retained pressure fraction for one drained and one impermeable boundary."""
    pi_arr = np.asarray(pi, dtype=float)
    out = np.zeros_like(pi_arr, dtype=float)
    n = np.arange(terms, dtype=float)
    alpha = (n + 0.5) * math.pi
    for idx, value in np.ndenumerate(pi_arr):
        out[idx] = np.sum(2.0 * (1.0 - np.exp(-(alpha**2) * value)) / (alpha**4 * value))
    return float(out) if np.isscalar(pi) else out


def source_shape_weights(shape: str, s: np.ndarray) -> np.ndarray:
    """Normalized pressure-generation histories over 0 <= s <= 1."""
    if shape == "constant":
        return np.ones_like(s)
    if shape == "front_loaded":
        return 2.0 * (1.0 - s)
    if shape == "back_loaded":
        return 2.0 * s
    if shape == "middle_pulse":
        return 6.0 * s * (1.0 - s)
    raise ValueError(f"Unknown source shape: {shape}")


def temporal_source_retention(
    pi: np.ndarray | float,
    shape: str,
    boundary: str = "double",
    terms: int = 500,
    quadrature_points: int = 400,
) -> np.ndarray | float:
    """Retained fraction for a normalized temporal source history.

    The pressure-generation history is q(t)=Pu/T*g(t/T), where integral_0^1
    g(s) ds = 1. The constant-source case reproduces spectral_retention.
    """
    pi_arr = np.asarray(pi, dtype=float)
    out = np.zeros_like(pi_arr, dtype=float)
    s = np.linspace(0.0, 1.0, quadrature_points)
    g = source_shape_weights(shape, s)
    if boundary == "double":
        m = np.arange(1, 2 * terms, 2, dtype=float)
        alpha = m * math.pi
        modal_weight = 8.0 / (alpha**2)
    elif boundary == "single":
        n = np.arange(terms, dtype=float)
        alpha = (n + 0.5) * math.pi
        modal_weight = 2.0 / (alpha**2)
    else:
        raise ValueError(f"Unknown boundary: {boundary}")
    for idx, value in np.ndenumerate(pi_arr):
        kernel = np.exp(-np.outer(alpha**2 * value, 1.0 - s))
        modal_integral = np.trapezoid(kernel * g, s, axis=1)
        out[idx] = np.sum(modal_weight * modal_integral)
    return float(out) if np.isscalar(pi) else out


def spectral_tail_bound(pi: float, retained_terms: int, max_terms: int = 200000) -> float:
    """Upper bound for the positive tail after retained odd modes."""
    start = 2 * retained_terms + 1
    m = np.arange(start, 2 * max_terms, 2, dtype=float)
    term_bound = np.minimum(8.0 / (math.pi**2 * m**2), 8.0 / (math.pi**4 * pi * m**4))
    return float(np.sum(term_bound))


def solve_threshold(target: float) -> float:
    lo, hi = 1e-6, 100.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if spectral_retention(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_threshold_model(target: float, model) -> float:
    lo, hi = 1e-6, 100.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if model(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for the repeated one-dimensional implicit solves."""
    n = len(diag)
    cp = upper.astype(float).copy()
    dp = rhs.astype(float).copy()
    bp = diag.astype(float).copy()
    for i in range(1, n):
        w = lower[i - 1] / bp[i - 1]
        bp[i] -= w * cp[i - 1]
        dp[i] -= w * dp[i - 1]
    x = np.empty(n, dtype=float)
    x[-1] = dp[-1] / bp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i + 1]) / bp[i]
    return x


def fd_retention(pi: float, n: int = 120, steps: int = 800) -> float:
    dx = 1.0 / (n + 1)
    dt = pi / steps
    r = dt / dx**2
    main = np.full(n, 1.0 + 2.0 * r)
    off = np.full(n - 1, -r)
    u = np.zeros(n)
    rhs_add = np.full(n, dt)
    for _ in range(steps):
        u = solve_tridiagonal(off, main, off, u + rhs_add)
    return float(dx * np.sum(u) / pi)


def fem_retention(pi: float, elements: int = 90, steps: int = 700) -> float:
    n = elements - 1
    h = 1.0 / elements
    dt = pi / steps
    m_main = np.full(n, 2.0 * h / 3.0)
    m_off = np.full(n - 1, h / 6.0)
    k_main = np.full(n, 2.0 / h)
    k_off = np.full(n - 1, -1.0 / h)
    load = np.full(n, h)
    lhs_main = m_main + dt * k_main
    lhs_off = m_off + dt * k_off
    u = np.zeros(n)
    for _ in range(steps):
        rhs = m_main * u + dt * load
        rhs[:-1] += m_off * u[1:]
        rhs[1:] += m_off * u[:-1]
        u = solve_tridiagonal(lhs_off, lhs_main, lhs_off, rhs)
    return float((load @ u) / pi)


def regime_from_r(r_value: float) -> str:
    if r_value >= 0.9:
        return "nearly undrained"
    if r_value <= 0.1:
        return "drained"
    return "partly drained"


def build_data() -> dict[str, pd.DataFrame]:
    thresholds = {target: solve_threshold(target) for target in [0.9, 0.5, 0.1]}
    single_thresholds = {
        target: solve_threshold_model(target, single_drainage_retention)
        for target in [0.9, 0.5, 0.1]
    }

    pi_curve = np.logspace(-4, 2, 360)
    retention_curve = pd.DataFrame(
        {"Pi": pi_curve, "R_Pi": spectral_retention(pi_curve, terms=700)}
    )

    verification_pi = np.array(
        [0.004, 0.02, thresholds[0.5], 0.12, 0.5, thresholds[0.1], 2.4]
    )
    verification_rows = []
    for value in verification_pi:
        spectral = spectral_retention(value, terms=1200)
        fd = fd_retention(float(value))
        fem = fem_retention(float(value))
        verification_rows.append(
            {
                "Pi": value,
                "R_spectral": spectral,
                "R_FD": fd,
                "R_FEM": fem,
                "FD_abs_error": abs(fd - spectral),
                "FEM_abs_error": abs(fem - spectral),
            }
        )
    verification = pd.DataFrame(verification_rows)

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
        err = max(abs(fd_retention(float(v), n=n, steps=steps) - spectral_retention(v)) for v in verification_pi)
        convergence_rows.append(
            {
                "Check": "finite difference",
                "Resolution": f"{n} interior nodes, {steps} time steps",
                "Maximum absolute error": err,
                "Comment": "implicit time integration",
            }
        )
    for elements, steps in [(30, 300), (60, 600), (90, 700)]:
        err = max(abs(fem_retention(float(v), elements=elements, steps=steps) - spectral_retention(v)) for v in verification_pi)
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

    boundary_condition_rows = []
    for target in [0.9, 0.5, 0.1]:
        dd = thresholds[target]
        sd = single_thresholds[target]
        boundary_condition_rows.append(
            {
                "Retained fraction boundary": f"R(Pi)={target}",
                "Pi_double_drainage": dd,
                "Pi_single_drainage": sd,
                "single_to_double_ratio": sd / dd,
                "Interpretation": "single drainage requires larger hydraulic time to dissipate the same fraction",
            }
        )
    boundary_conditions = pd.DataFrame(boundary_condition_rows)

    source_pi = np.array([0.004418, 0.02, 0.1126, 0.5, 0.8331, 2.4])
    temporal_rows = []
    for boundary in ["double", "single"]:
        for shape in ["front_loaded", "constant", "middle_pulse", "back_loaded"]:
            values = temporal_source_retention(source_pi, shape, boundary=boundary, terms=400)
            for pi_value, retained in zip(source_pi, values):
                temporal_rows.append(
                    {
                        "boundary_condition": boundary,
                        "source_shape": shape,
                        "Pi": pi_value,
                        "R_shape": retained,
                        "R_constant_double": spectral_retention(pi_value),
                    }
                )
    temporal_source = pd.DataFrame(temporal_rows)

    truncation_rows = []
    bound_pi = np.array([thresholds[0.9], thresholds[0.5], thresholds[0.1], 2.4])
    reference_terms = 5000
    for retained_terms in [5, 10, 20, 50, 100]:
        for pi_value in bound_pi:
            approx = spectral_retention(pi_value, terms=retained_terms)
            reference = spectral_retention(pi_value, terms=reference_terms)
            actual_error = abs(reference - approx)
            truncation_rows.append(
                {
                    "Pi": pi_value,
                    "retained_odd_terms": retained_terms,
                    "actual_error_vs_5000_terms": actual_error,
                    "positive_tail_bound": spectral_tail_bound(pi_value, retained_terms),
                    "bound_to_actual_ratio": spectral_tail_bound(pi_value, retained_terms) / actual_error if actual_error > 0 else np.nan,
                    "monotonicity_basis": "each positive modal term decreases with Pi; thresholds are unique",
                }
            )
    truncation_bounds = pd.DataFrame(truncation_rows)

    return {
        "retention_curve": retention_curve,
        "verification_fd_fem": verification,
        "convergence_checks": convergence,
        "benchmark_cases": cases,
        "literature_comparison": literature,
        "external_consistency_check": external,
        "sensitivity": sensitivity,
        "thresholds": thresholds_df,
        "boundary_condition_comparison": boundary_conditions,
        "temporal_source_retention": temporal_source,
        "truncation_bound_check": truncation_bounds,
    }


def save_csv(data: dict[str, pd.DataFrame]) -> None:
    for name, frame in data.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)


def make_figures(data: dict[str, pd.DataFrame]) -> None:
    FIG_OUT.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.9, 5.0), dpi=600, constrained_layout=True)
    slope = Polygon(
        [[0.35, 0.70], [8.45, 0.70], [8.45, 3.15], [0.35, 1.25]],
        closed=True,
        facecolor="#ddd2bd",
        edgecolor="black",
        lw=1.2,
    )
    ax.add_patch(slope)
    ax.plot([1.10, 7.90], [1.55, 3.08], color="#0b79bd", lw=5.0, solid_capstyle="butt")
    ax.plot([1.10, 8.45], [1.55, 3.15], color="#666666", lw=2.4, solid_capstyle="butt")
    ax.add_patch(Rectangle((6.20, 3.60), 1.90, 0.16, facecolor="white", edgecolor="black", lw=1.2))
    ax.add_patch(Rectangle((6.20, 3.76), 1.90, 0.16, facecolor="white", edgecolor="black", lw=1.2))
    ax.plot([6.20, 8.10], [3.72, 3.72], color="black", lw=4.0, solid_capstyle="butt")
    ax.annotate("upslope drained\nboundary", xy=(1.15, 1.58), xytext=(0.75, 3.70), ha="center", arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.annotate("rapid loading\nduration T", xy=(5.02, 2.60), xytext=(4.18, 4.28), ha="center", arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.annotate("road / civil corridor", xy=(7.15, 3.92), xytext=(7.15, 4.55), ha="center", arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.annotate("", xy=(3.55, 2.18), xytext=(3.42, 1.84), arrowprops=dict(arrowstyle="<->", lw=1.3))
    ax.text(3.24, 2.38, "saturated shear band\nthickness h", ha="center", va="bottom")
    ax.text(4.45, 0.30, r"Output: retained pore pressure R($\Pi$)P$_u$ and partly drained stability", ha="center")
    ax.set_xlim(0, 8.9)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    fig.savefig(FIG_OUT / "figure_1_conceptual_slope.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig1.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    curve = data["retention_curve"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
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
    fig.savefig(FIG_OUT / "Fig2.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    verification = data["verification_fd_fem"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    ax.loglog(verification["Pi"], verification["FD_abs_error"], "o-", label="implicit finite differences")
    ax.loglog(verification["Pi"], verification["FEM_abs_error"], "s-", label="linear FEM")
    ax.set_xlabel("Pi")
    ax.set_ylabel("Absolute error in R(Pi)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIG_OUT / "figure_3_numerical_verification_errors.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig3.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pi = np.logspace(-4, 2, 300)
    r = spectral_retention(pi)
    sigma0, pu, phi, tau = 50.0, 20.0, math.radians(30.0), 25.0
    fs = (sigma0 - pu * r) * math.tan(phi) / tau
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    ax.semilogx(pi, fs, color="#238b45", lw=2)
    ax.axhline(1.0, color="black", ls="--", lw=0.9)
    ax.text(1.8e-4, 1.015, "FS = 1", fontsize=9)
    ax.set_xlabel("Pi")
    ax.set_ylabel("Partly drained factor of safety")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(FIG_OUT / "figure_4_stability_shift.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig4.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    h_vals = np.logspace(-2, -0.3, 240)
    t_vals = np.logspace(0, 3, 240)
    h_grid, t_grid = np.meshgrid(h_vals, t_vals)
    r_grid = spectral_retention(5e-6 * t_grid / h_grid**2)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=600, constrained_layout=True)
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
    fig.savefig(FIG_OUT / "Fig5.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sens = data["sensitivity"].copy()
    sens["abs"] = sens["Normalized sensitivity of FS_PD"].abs()
    sens = sens.sort_values("abs", ascending=True)
    colors = ["#b2182b" if v < 0 else "#2166ac" for v in sens["Normalized sensitivity of FS_PD"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    ax.barh(sens["Parameter"], sens["Normalized sensitivity of FS_PD"], color=colors)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("Normalized log-sensitivity of partly drained FS")
    ax.set_ylabel("Parameter")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG_OUT / "figure_6_sensitivity.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig6.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pi_curve = np.logspace(-4, 2, 260)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    ax.semilogx(pi_curve, spectral_retention(pi_curve), label="double drainage", color="#084081", lw=2.0)
    ax.semilogx(pi_curve, single_drainage_retention(pi_curve), label="single drainage", color="#b2182b", lw=2.0)
    ax.axhline(0.5, color="0.35", ls="--", lw=0.9)
    ax.set_xlabel("Dynamic consolidation number, Pi = cvT/h^2")
    ax.set_ylabel("Retained pore-pressure fraction, R(Pi)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIG_OUT / "figure_7_boundary_condition_comparison.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig7.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    source = data["temporal_source_retention"]
    subset = source[(source["boundary_condition"] == "double") & (source["Pi"].isin([0.1126, 0.8331]))]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    labels = {"front_loaded": "front loaded", "constant": "constant", "middle_pulse": "middle pulse", "back_loaded": "back loaded"}
    for pi_value, group in subset.groupby("Pi"):
        ordered = group.set_index("source_shape").loc[["front_loaded", "constant", "middle_pulse", "back_loaded"]]
        ax.plot(
            [labels[v] for v in ordered.index],
            ordered["R_shape"],
            marker="o",
            label=f"Pi={pi_value:g}",
        )
    ax.set_ylabel("Retained fraction at event end")
    ax.set_xlabel("Normalized pressure-generation history")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIG_OUT / "figure_8_temporal_source_retention.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig8.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    bounds = data["truncation_bound_check"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    for pi_value, group in bounds.groupby("Pi"):
        label = f"Pi={pi_value:.4g}"
        ax.loglog(group["retained_odd_terms"], group["actual_error_vs_5000_terms"], "o-", label=label)
        ax.loglog(group["retained_odd_terms"], group["positive_tail_bound"], "--", color=ax.lines[-1].get_color(), alpha=0.65)
    ax.set_xlabel("Retained odd terms")
    ax.set_ylabel("Absolute truncation error / positive tail bound")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG_OUT / "figure_9_truncation_bound_check.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig9.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    data = build_data()
    save_csv(data)
    make_figures(data)
    print(f"CSV files written to {OUT}")
    print(f"Figures written to {FIG_OUT}")


if __name__ == "__main__":
    main()
