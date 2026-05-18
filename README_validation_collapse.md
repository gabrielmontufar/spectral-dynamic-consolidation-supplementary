# Dimensionless-Collapse and Negative-Control Validation

This validation layer projects the Oso laboratory records, the Mount Kaba-san
field-transfer diagnostics and the USGS flume regime-screening records into a
common Pi-R diagnostic space.

The objective is falsifiability and external consistency, not field-scale
calibration. The Oso records provide the direct retained-pressure collapse
check because their drainage length and fitted consolidation coefficients are
available. The USGS flume records are retained only as a regime-screening
proxy, and Mount Kaba-san is retained as a field-transfer boundary case because
the observed transfer-normalized ratio R*_obs is not constrained to [0, 1].

Regenerate the files with:

```bash
python scripts/validation_dimensionless_collapse.py
```

or as part of the complete package:

```bash
python scripts/run_all_validation.py --offline --fast
python scripts/run_all_validation.py --offline --full
```

Generated files:

- `outputs/dimensionless_validation_matrix.csv`
- `outputs/dimensionless_collapse_metrics.csv`
- `outputs/negative_control_report.csv`
- `figures/dimensionless_collapse_envelope.png`
- `figures/negative_control_skill.png`

