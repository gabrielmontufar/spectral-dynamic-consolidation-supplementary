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


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    metrics = pd.read_csv(OUT / "oso_loro_metrics.csv")
    records = sorted(metrics["held_out_record"].unique())
    rows = []
    rng = np.random.default_rng(123)

    wide = metrics.pivot_table(index="held_out_record", columns="model", values="mae", aggfunc="median")
    for model in ["M4_constant_source", "M5_single_boundary_proxy"]:
        if model not in wide:
            continue
        for baseline in ["M0_drained", "M1_undrained"]:
            skill = 1.0 - wide[model] / wide[baseline]
            estimate, lo, hi = bootstrap_ci(skill.to_numpy(float), rng)
            rows.append(
                {
                    "dataset": "Oso ring-shear",
                    "validation": "leave-one-record-out",
                    "metric": "mae_skill_vs_" + baseline,
                    "model": model,
                    "baseline": baseline,
                    "n_records": len(records),
                    "estimate": estimate,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )

    for model in ["M0_drained", "M1_undrained", "M4_constant_source", "M5_single_boundary_proxy"]:
        vals = metrics.loc[metrics["model"] == model, "mae"].to_numpy(float)
        estimate, lo, hi = bootstrap_ci(vals, rng)
        rows.append(
            {
                "dataset": "Oso ring-shear",
                "validation": "leave-one-record-out",
                "metric": "mae",
                "model": model,
                "baseline": "",
                "n_records": len(records),
                "estimate": estimate,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "oso_loro_bootstrap_skill.csv", index=False)

    skill_plot = result[result["metric"].str.startswith("mae_skill_vs_M0")].copy()
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300, constrained_layout=True)
    y = np.arange(len(skill_plot))
    ax.errorbar(
        skill_plot["estimate"],
        y,
        xerr=[skill_plot["estimate"] - skill_plot["ci95_low"], skill_plot["ci95_high"] - skill_plot["estimate"]],
        fmt="o",
        capsize=4,
        color="#1f77b4",
    )
    ax.axvline(0, color="0.2", lw=0.9)
    ax.set_yticks(y, [m.replace("_", " ") for m in skill_plot["model"]])
    ax.set_xlabel("MAE skill relative to drained baseline")
    ax.set_title("Fig. 16. Oso LORO bootstrap skill")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG / "fig16_oso_loro_bootstrap_skill.png", bbox_inches="tight")
    fig.savefig(FIG / "fig18_oso_loro_skill_bootstrap.png", bbox_inches="tight")
    plt.close(fig)

    print("wrote", OUT / "oso_loro_bootstrap_skill.csv")
    print("wrote", FIG / "fig16_oso_loro_bootstrap_skill.png")


if __name__ == "__main__":
    main()
