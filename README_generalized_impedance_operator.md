# Generalized Impedance- and Source-History-Aware Operator

This script supports the revised manuscript framing:

`Impedance- and source-history-aware spectral drainage-transition operator`.

The original double-drainage retained-pressure curve is treated as a benchmark
limit of a broader one-dimensional diffusion-source operator. The generalized
operator introduces:

- `B0`, `B1`: dimensionless hydraulic boundary admittances.
- `chi_q`: temporal centroid of pressure generation,
  `integral(t q(t) dt) / [T integral(q(t) dt)]`.
- `R_g(Pi, B0, B1, chi_q)`: generalized retained-pressure fraction.

Run:

```bash
python scripts/12_generalized_impedance_operator.py
```

Generated files:

- `outputs/generalized_impedance_operator_atlas.csv`
- `outputs/generalized_impedance_thresholds.csv`
- `outputs/generalized_source_history_index.csv`
- `figures/fig19_generalized_impedance_drainage_atlas.png`
- `figures/fig20_source_history_centroid_operator.png`

Interpretation:

The atlas is a reproducible methodological extension. It shows how imperfect
drainage boundaries and early/late pressure generation shift the retained
pressure relative to the double-drainage benchmark. It does not replace a
site-specific coupled hydro-mechanical analysis.

