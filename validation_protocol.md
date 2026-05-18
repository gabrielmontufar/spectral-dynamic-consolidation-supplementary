# Validation Protocol for MRNB Corrections

This protocol documents an intentionally bounded external validation layer for
the supplementary spectral-consolidation package. It does not create or infer
new experimental observations. Available public records are used only where
the required measured variables exist and the split between calibration and
held-out evaluation can be reproduced.

## Available Data

The validation workflow uses these repository files and downloaded public data:

- `oso_ring_shear_normalized_records.csv`: normalized retained-pressure records
  derived from the public USGS Oso ring-shear consolidation data release.
- `oso_ring_shear_consistency_metrics.csv`: per-record fitted double- and
  single-drainage held-out metrics.
- `oso_ring_shear_consistency_predictions.csv`: sampled observed and fitted
  retained-pressure curves for the fitted models.
- `external_data/mount_kabasan/Japan_exp_failure_period_data.csv`: public
  USGS Mount Kaba-san field experiment record, used for field-scale
  pore-pressure dissipation consistency.
- `external_data/usgs_flume_2016/*_archive_corrected.csv`: public USGS
  debris-flow flume records, used as an observed pore-pressure ratio regime
  inventory. The flume records are not converted into full retained-pressure
  predictions unless dataset-specific `cv`, source history and boundary
  conditions are calibrated.

The Oso files are laboratory-scale pressure-dissipation consistency checks.
Mount Kaba-san is treated as a field-scale pressure-dissipation consistency
check, not as a calibrated slope-stability back-analysis. The flume files are
used to document observed pore-pressure regimes only.

## Validation Questions

1. Are the retained-pressure predictions consistent with the available Oso
   normalized retained-pressure observations?
2. Which drainage idealization has lower observed prediction residuals on the
   available consistency records?
3. Does Mount Kaba-san provide a field-scale pressure-dissipation check under
   the same held-out protocol?
4. What drainage-regime observations are present in the USGS flume records
   before any site-specific retained-pressure prediction is claimed?
5. Do the per-record held-out metrics and sampled prediction residuals support
   the same bounded interpretation: dissipation consistency rather than direct
   landslide calibration?

## Reproducible Command

Run the validation workflow from the repository root:

```powershell
python validation\run_all_validation.py
```

The script reads only the Oso consistency CSV files listed above and writes:

- `outputs/validation_metrics_summary.csv`
- `outputs/model_comparison_metrics.csv`
- `outputs/validation_master_table.csv`
- `outputs/external_dataset_inventory.csv`
- `outputs/mount_kabasan_consistency_metrics.csv`
- `outputs/mount_kabasan_consistency_predictions.csv`
- `outputs/flume_regime_observation_summary.csv`
- `generated_figures/fig06_retained_pressure_validation.png`
- `generated_figures/fig07_multidataset_pressure_consistency.png`

## Reported Metrics

The validation script reports observed residual metrics computed from the
existing sampled prediction table:

- RMSE of normalized retained pressure.
- MAE of normalized retained pressure.
- Mean signed bias.
- Median absolute residual.
- 95th percentile absolute residual.
- Sampled-prediction R2 where the observed variance is nonzero.

It also carries forward the original per-record held-out metrics from
`oso_ring_shear_consistency_metrics.csv`, including `rmse_heldout`,
`mae_heldout`, `bias_heldout`, `p95_abs_residual`, and `r2_heldout`.

## Guardrails

- The workflow fails if a required input CSV is missing.
- The workflow fails if required columns are absent.
- The workflow does not download raw data.
- The workflow does not create synthetic validation observations.
- The workflow does not modify the original Oso consistency CSV files.
- The workflow does not compute field-scale factor of safety for Mount
  Kaba-san because a defensible shear strength and stress path calibration is
  not contained in the time-series CSV alone.
- The workflow does not claim retained-pressure prediction for the flume
  records until dataset-specific hydraulic calibration is added.
