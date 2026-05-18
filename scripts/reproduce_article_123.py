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

    claim_passport = pd.DataFrame(
        [
            [
                "Closed-form retained-pressure operator",
                "R(Pi) is computed from a positive modal series and checked against finite-difference and finite-element solutions.",
                "Applies to the stated one-dimensional saturated shear-band, boundary-condition and source assumptions.",
                "A coupled or monitored case with nonuniform source, anisotropy or three-dimensional drainage should be analysed with a site-specific model.",
                "Escalate to coupled hydro-mechanical analysis rather than treating R(Pi) as a universal constitutive law.",
            ],
            [
                "Continuous drainage-regime classification",
                "Thresholds R(Pi)=0.9, 0.5 and 0.1 are computed from the same operator and audited with truncation-error bounds.",
                "Use as a screening boundary between nearly undrained, partly drained and drained response.",
                "Do not use the thresholds as calibrated field limits without independent hydraulic and geometric data.",
                "Report h, cv, T, boundary condition and source history whenever the classification is used.",
            ],
            [
                "Partly drained stability diagnostic",
                "The retained pressure is inserted into an effective-stress factor of safety for synthetic infrastructure-slope cases.",
                "Supports transparent comparison of drainage assumptions in screening inventories.",
                "Does not replace site-specific strength selection, progressive-failure analysis or calibrated pore-pressure modelling.",
                "Use the diagnostic as a pre-analysis check before full design calculations.",
            ],
            [
                "External pressure-dissipation consistency",
                "Oso ring-shear records, Mount Kaba-san pore-pressure records and flume observations are treated as bounded consistency checks.",
                "Supports plausibility of the pressure-dissipation component, not field-scale landslide prediction.",
                "A field validation claim would require observed geometry, hydraulic properties, pore-pressure forcing and slope response for the same event.",
                "Label these outputs as consistency checks unless a full field calibration is added.",
            ],
        ],
        columns=["Claim", "Evidence provided", "Allowed scope", "Main falsifier or limitation", "Required escalation"],
    )

    evidence_hierarchy = pd.DataFrame(
        [
            ["Analytical limits", "Pi -> 0 and Pi -> infinity", "Operator tends to undrained and drained limits.", "Theory check"],
            ["Numerical verification", "FD and FEM benchmarks", "Independent discretisations reproduce the spectral operator.", "Verification"],
            ["Convergence audit", "Modal truncation, grid refinement and tail bounds", "Thresholds are numerically stable and unique.", "Verification"],
            ["Synthetic benchmark", "Five infrastructure-slope cases", "Shows how h, cv and T change retained pressure and FS_PD.", "Controlled demonstration"],
            ["Global sensitivity", "Latin-hypercube and PRCC checks", "Separates hydraulic controls of Pi from mechanical controls of FS_PD.", "Robustness check"],
            ["External records", "Oso, Mount Kaba-san and flume public datasets", "Pressure-dissipation behaviour is directionally compatible with bounded observations.", "Consistency check"],
            ["Field-scale calibration", "Not claimed in this package", "Requires geometry, hydraulic properties and observed slope response for the same event.", "Future validation"],
        ],
        columns=["Evidence tier", "Material used", "What it supports", "Evidence type"],
    )

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
        "claim_passport": claim_passport,
        "evidence_hierarchy": evidence_hierarchy,
    }


def save_csv(data: dict[str, pd.DataFrame]) -> None:
    for name, frame in data.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)


def make_figures(data: dict[str, pd.DataFrame]) -> None:
    FIG_OUT.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.9, 4.35), dpi=600, constrained_layout=True)
    ground_surface = np.array([[0.35, 1.25], [8.45, 3.05]])
    slope = Polygon(
        [[0.35, 0.70], [8.45, 0.70], ground_surface[1], ground_surface[0]],
        closed=True,
        facecolor="#e6ddca",
        edgecolor="#2f2f2f",
        lw=1.1,
    )
    ax.add_patch(slope)
    ax.plot(ground_surface[:, 0], ground_surface[:, 1], color="#111111", lw=1.4, solid_capstyle="butt")
    surface_tangent = ground_surface[1] - ground_surface[0]
    tangent = surface_tangent / np.linalg.norm(surface_tangent)
    normal_up = np.array([-tangent[1], tangent[0]])
    normal_down = -normal_up
    cover_thickness = 0.16
    band_upper = np.array([ground_surface[0] + tangent * 0.62 + normal_down * cover_thickness,
                           ground_surface[1] - tangent * 0.62 + normal_down * cover_thickness])
    tangent = tangent / np.linalg.norm(tangent)
    band_thickness = 0.14
    band_lower = band_upper + normal_down * band_thickness
    shear_band = Polygon(
        [band_lower[0], band_lower[1], band_upper[1], band_upper[0]],
        closed=True,
        facecolor="#dbeef7",
        edgecolor="#2f5f7f",
        lw=1.1,
        hatch="///",
        alpha=0.95,
    )
    ax.add_patch(shear_band)
    ax.plot(band_lower[:, 0], band_lower[:, 1], color="#2f2f2f", lw=0.9, solid_capstyle="butt")
    ax.plot(band_upper[:, 0], band_upper[:, 1], color="#2f2f2f", lw=0.9, solid_capstyle="butt")
    ax.add_patch(Rectangle((6.35, 3.72), 1.75, 0.14, facecolor="white", edgecolor="#222222", lw=0.9))
    ax.add_patch(Rectangle((6.35, 3.86), 1.75, 0.14, facecolor="white", edgecolor="#222222", lw=0.9))
    ax.plot([6.35, 8.10], [3.83, 3.83], color="#222222", lw=1.6, solid_capstyle="butt")
    arrow = dict(arrowstyle="->", lw=0.75, mutation_scale=8, shrinkA=1, shrinkB=1, color="#1f1f1f")
    ax.annotate("upper drained\nboundary", xy=band_upper[0] + tangent * 0.35, xytext=(1.45, 2.45), ha="center", fontsize=9.5, arrowprops=arrow)
    ax.annotate("lower drained\nboundary", xy=band_lower[0] + tangent * 0.55, xytext=(1.02, 0.34), ha="center", fontsize=9.5, arrowprops=arrow)
    load_x = 5.25
    y_surface = ground_surface[0, 1] + (ground_surface[1, 1] - ground_surface[0, 1]) * ((load_x - ground_surface[0, 0]) / (ground_surface[1, 0] - ground_surface[0, 0]))
    ax.annotate("loading interval\n$T$", xy=(load_x, y_surface + 0.04), xytext=(load_x, y_surface + 0.78), ha="center", fontsize=9.5, arrowprops=arrow)
    ax.annotate("road / civil corridor", xy=(7.22, 4.00), xytext=(7.22, 4.50), ha="center", fontsize=9.5, arrowprops=arrow)
    band_mid = (band_lower[0] + band_upper[0]) / 2 + tangent * 3.20
    ax.annotate("saturated\nshear band", xy=band_mid, xytext=(3.95, 2.38), ha="center", fontsize=9.0, arrowprops=arrow)

    # Local cross-section inset: h is a normal distance between the two drainage boundaries.
    inset_x, inset_y, inset_w, inset_h = 1.45, 3.76, 2.45, 0.72
    ax.add_patch(Rectangle((inset_x, inset_y), inset_w, inset_h, facecolor="#fbfbf8", edgecolor="0.35", lw=0.75))
    ax.text(inset_x + inset_w / 2, inset_y + inset_h - 0.10, "normal section", ha="center", va="top", fontsize=8.5)
    y_low, y_up = inset_y + 0.18, inset_y + 0.42
    x0, x1 = inset_x + 0.44, inset_x + inset_w - 0.46
    ax.add_patch(Rectangle((x0, y_low), x1 - x0, y_up - y_low, facecolor="#dbeef7", edgecolor="#2f5f7f", hatch="///", lw=0.7, alpha=0.95))
    dim_x = x1 + 0.12
    ax.plot([dim_x - 0.055, dim_x + 0.055], [y_low, y_low], color="black", lw=0.75)
    ax.plot([dim_x - 0.055, dim_x + 0.055], [y_up, y_up], color="black", lw=0.75)
    ax.annotate("", xy=(dim_x, y_up), xytext=(dim_x, y_low), arrowprops=dict(arrowstyle="<->", lw=0.8, mutation_scale=9, shrinkA=0, shrinkB=0))
    ax.text(x1 + 0.23, (y_low + y_up) / 2, "$h$", ha="left", va="center", fontsize=9.5)
    ax.text(x0 - 0.08, y_up, "upper", ha="right", va="center", fontsize=7.5)
    ax.text(x0 - 0.08, y_low, "lower", ha="right", va="center", fontsize=7.5)

    ax.annotate(r"$u_r = R(\Pi)P_u$", xy=band_upper[0] + tangent * 5.1, xytext=(5.95, 1.34), ha="center", fontsize=9.5, arrowprops=arrow)
    ax.text(6.95, 1.00, r"$F_{S,PD}$ from retained pressure", ha="center", fontsize=9.5)
    ax.set_xlim(0, 8.9)
    ax.set_ylim(-0.18, 4.45)
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
        ax.text(pi * 1.08, target + 0.025, rf"$R={target}$, $\Pi={pi:.4g}$", fontsize=8)
    ax.set_xlabel(r"Dynamic consolidation number, $\Pi = c_vT/h^2$")
    ax.set_ylabel(r"Retained pore-pressure fraction, $R(\Pi)$")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(FIG_OUT / "figure_2_retention_curve.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig2.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    verification = data["verification_fd_fem"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    ax.loglog(verification["Pi"], verification["FD_abs_error"], "o-", label="implicit finite differences")
    ax.loglog(verification["Pi"], verification["FEM_abs_error"], "s-", label="linear FEM")
    ax.set_xlabel(r"$\Pi$")
    ax.set_ylabel(r"Absolute error in $R(\Pi)$")
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
    ax.text(1.8e-4, 1.015, r"$FS = 1$", fontsize=9)
    ax.set_xlabel(r"$\Pi$")
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
    ax.set_xlabel(r"Shear-band thickness $h$ (m)")
    ax.set_ylabel(r"Rapid-loading duration $T$ (s)")
    ax.text(0.012, 750, "drained", fontsize=10)
    ax.text(0.052, 24, "partly drained", fontsize=10)
    ax.text(0.18, 2.2, "nearly undrained", fontsize=10)
    handles = [
        Line2D([0], [0], color="black", lw=1.0, linestyle="solid", label=r"$R=0.1$"),
        Line2D([0], [0], color="black", lw=1.3, linestyle="dashed", label=r"$R=0.5$"),
        Line2D([0], [0], color="black", lw=1.0, linestyle="dashdot", label=r"$R=0.9$"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True, edgecolor="black", title="Retention contours")
    ax.set_title(r"Regime map for $c_v = 5 \times 10^{-6}$ m$^2$/s")
    fig.savefig(FIG_OUT / "figure_5_regime_map.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "Fig5.tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sens = data["sensitivity"].copy()
    sens["abs"] = sens["Normalized sensitivity of FS_PD"].abs()
    sens = sens.sort_values("abs", ascending=True)
    parameter_labels = {
        "tau": r"$\tau$",
        "Pu": r"$P_u$",
        "Pu_kPa": r"$P_u$",
        "h": r"$h$",
        "h_m": r"$h$",
        "T_s": r"$T$",
        "cv": r"$c_v$",
        "cv_m2_s": r"$c_v$",
        "phi_deg": r"$\phi'$",
        "sigma_eff0": r"$\sigma'_{n0}$",
    }
    sens["Parameter label"] = sens["Parameter"].map(parameter_labels).fillna(sens["Parameter"])
    colors = ["#b2182b" if v < 0 else "#2166ac" for v in sens["Normalized sensitivity of FS_PD"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600, constrained_layout=True)
    ax.barh(sens["Parameter label"], sens["Normalized sensitivity of FS_PD"], color=colors)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel(r"Normalized log-sensitivity of partly drained $FS$")
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
    ax.set_xlabel(r"Dynamic consolidation number, $\Pi = c_vT/h^2$")
    ax.set_ylabel(r"Retained pore-pressure fraction, $R(\Pi)$")
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
            label=rf"$\Pi={pi_value:g}$",
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
        label = rf"$\Pi={pi_value:.4g}$"
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
