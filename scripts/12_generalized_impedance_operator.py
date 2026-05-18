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


def matrix_robin(n: int, b0: float, b1: float) -> tuple[np.ndarray, np.ndarray]:
    dx = 1.0 / (n - 1)
    a = np.zeros((n, n), dtype=float)
    a[0, 0] = -2.0 / dx**2 - 2.0 * b0 / dx
    a[0, 1] = 2.0 / dx**2
    for i in range(1, n - 1):
        a[i, i - 1] = 1.0 / dx**2
        a[i, i] = -2.0 / dx**2
        a[i, i + 1] = 1.0 / dx**2
    a[-1, -2] = 2.0 / dx**2
    a[-1, -1] = -2.0 / dx**2 - 2.0 * b1 / dx
    weights = np.full(n, dx)
    weights[0] = weights[-1] = 0.5 * dx
    return a, weights


def retained_uniform_initial(pi_values: np.ndarray, b0: float, b1: float, n: int = 80) -> np.ndarray:
    a, weights = matrix_robin(n, b0, b1)
    vals, vecs = np.linalg.eig(a)
    inv = np.linalg.inv(vecs)
    initial = np.ones(n)
    coeff = inv @ initial
    out = []
    for pi in np.asarray(pi_values, dtype=float):
        state = vecs @ (np.exp(vals * max(pi, 0.0)) * coeff)
        out.append(float(np.real(weights @ state)))
    return np.asarray(out)


def source_history_centroid(time: np.ndarray, q: np.ndarray) -> float:
    time = np.asarray(time, dtype=float)
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    denom = np.trapezoid(q, time)
    if denom <= 0:
        return float("nan")
    return float(np.trapezoid(time * q, time) / (time[-1] * denom))


def source_history_retention(pi_total: float, b0: float, b1: float, kind: str) -> tuple[float, float]:
    tau = np.linspace(0.0, 1.0, 240)
    if kind == "early":
        q = np.exp(-8.0 * tau)
    elif kind == "late":
        q = np.exp(-8.0 * (1.0 - tau))
    elif kind == "mid":
        q = np.exp(-0.5 * ((tau - 0.5) / 0.12) ** 2)
    else:
        q = np.ones_like(tau)
    q = q / np.trapezoid(q, tau)
    chi = source_history_centroid(tau, q)
    # Linear superposition: pressure generated at tau has only (1-tau) of the
    # event duration left to dissipate.
    survival = retained_uniform_initial(pi_total * (1.0 - tau), b0, b1, n=60)
    retained = float(np.trapezoid(q * survival, tau))
    return retained, chi


def retained_source(pi_values: np.ndarray, b0: float, b1: float, kind: str = "uniform") -> tuple[np.ndarray, float]:
    values = []
    chi = float("nan")
    for pi in np.asarray(pi_values, dtype=float):
        retained, chi = source_history_retention(float(pi), b0, b1, kind)
        values.append(retained)
    return np.asarray(values, dtype=float), chi


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    pi = np.logspace(-4, 1, 140)
    b_values = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 1000.0])
    rows = []
    for b in b_values:
        r, _ = retained_source(pi, b, b, "uniform")
        for p, rv in zip(pi, r):
            rows.append({"Pi": p, "B0": b, "B1": b, "chi_q": 0.5, "R_g": rv, "case": "symmetric Robin uniform source"})
    atlas = pd.DataFrame(rows)
    atlas.to_csv(OUT / "generalized_impedance_operator_atlas.csv", index=False)

    lim_rows = []
    cases = [
        ("undrained_boundary", 0.0, 0.0),
        ("partial_B_0p3", 0.3, 0.3),
        ("partial_B_1", 1.0, 1.0),
        ("partial_B_10", 10.0, 10.0),
        ("double_drainage_limit", 1000.0, 1000.0),
        ("single_drainage_proxy", 1000.0, 0.0),
    ]
    for name, b0, b1 in cases:
        r, _ = retained_source(pi, b0, b1, "uniform")
        for target in [0.9, 0.5, 0.1]:
            if np.nanmin(r) > target:
                threshold_pi = np.nan
                status = "not_reached_in_Pi_grid"
            else:
                idx = int(np.nanargmin(np.abs(r - target)))
                threshold_pi = float(pi[idx])
                status = "reached"
            lim_rows.append({"case": name, "B0": b0, "B1": b1, "R_threshold": target, "Pi_at_threshold": threshold_pi, "status": status})
    pd.DataFrame(lim_rows).to_csv(OUT / "generalized_impedance_thresholds.csv", index=False)

    source_rows = []
    for b0, b1, bname in [(1000.0, 1000.0, "double_drainage_limit"), (1.0, 1.0, "partial_B_1")]:
        for kind in ["early", "uniform", "mid", "late"]:
            for p in [0.01, 0.05, 0.1, 0.5, 1.0]:
                retained, chi = source_history_retention(p, b0, b1, kind)
                source_rows.append({"boundary_case": bname, "source_history": kind, "Pi_total": p, "chi_q": chi, "R_g": retained})
    pd.DataFrame(source_rows).to_csv(OUT / "generalized_source_history_index.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300, constrained_layout=True)
    for b in b_values:
        label = "B -> inf" if b >= 999 else f"B={b:g}"
        g = atlas[atlas["B0"] == b]
        ax.plot(g["Pi"], g["R_g"], lw=1.1, label=label)
    for target in [0.9, 0.5, 0.1]:
        ax.axhline(target, color="0.25", lw=0.6, ls="--")
        ax.text(1.15e-4, target + 0.015, f"R={target}", fontsize=7)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Dynamic consolidation number, Pi")
    ax.set_ylabel("Generalized retained-pressure fraction, R_g")
    ax.set_title("Drainage-transition atlas in Pi-B space")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    fig.savefig(FIG / "fig19_generalized_impedance_drainage_atlas.png", bbox_inches="tight")
    plt.close(fig)

    src = pd.DataFrame(source_rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=300, constrained_layout=True)
    sub = src[(src["boundary_case"] == "double_drainage_limit") & (src["Pi_total"] == 0.1)]
    ax.bar(sub["source_history"], sub["R_g"], color="#557a95")
    for i, row in enumerate(sub.itertuples()):
        ax.text(i, row.R_g + 0.015, f"chi={row.chi_q:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R_g at Pi=0.1")
    ax.set_title("Source-history centroid shifts retained pressure")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG / "fig20_source_history_centroid_operator.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote generalized impedance/source-history operator outputs")


if __name__ == "__main__":
    main()
