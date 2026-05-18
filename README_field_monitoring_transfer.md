# Independent Public Field-Monitoring Transfer Check

This check implements the Cleveland Corral validation layer proposed in
`validacion_campo_bases_datos_articulo.docx`.

Data source:

- Reid, M.E., Brien, D.L., and LaHusen, R.G., USGS data release,
  `https://doi.org/10.5066/P1P9DMFX`.

The script uses the public daily monitoring files for the Cleveland Corral
landslide near U.S. Highway 50. The available variables include rainfall,
groundwater pressure head, piezometer depths and extensometer displacement.

Run:

```bash
python scripts/11_field_monitoring_cleveland_corral.py
```

or regenerate it with the full validation package:

```bash
python scripts/run_all_validation.py --offline --fast
python scripts/run_all_validation.py --offline --full
```

Generated files:

- `outputs/cleveland_corral_field_monitoring_matrix.csv`
- `outputs/cleveland_corral_transfer_metrics.csv`
- `outputs/cleveland_corral_transfer_predictions.csv`
- `outputs/cleveland_corral_pressure_displacement_events.csv`
- `outputs/cleveland_corral_pressure_displacement_screen.csv`
- `outputs/field_monitoring_transfer_summary.csv`
- `figures/fig17_cleveland_corral_field_monitoring_transfer.png`
- `figures/fig18_cleveland_corral_displacement_screen.png`

Interpretation:

This is an independent public field-monitoring consistency check. It tests
whether the Pi-R retained-pressure operator improves held-out storm-window
pressure-head retention relative to drained and undrained baselines. It does
not calibrate a full landslide mobility model, and the displacement result is
reported only as an auxiliary pressure-displacement screen.

