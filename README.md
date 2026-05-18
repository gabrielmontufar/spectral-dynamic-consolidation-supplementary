# Supplementary Material

Manuscript: A spectral dynamic-consolidation criterion for drainage transition in rapid saturated landslides.

This package contains synthetic benchmark data and reproducible Python scripts used to regenerate the retained-pressure curve, numerical verification, convergence checks, local and global sensitivity analyses, boundary-condition comparison, temporal source-history comparison, truncation-error bound check, and bounded external pressure-dissipation consistency checks against public USGS Oso ring-shear, Mount Kaba-san and debris-flow flume records.

## Runtime

Python 3 with:

- numpy
- pandas
- matplotlib

The script does not require SciPy.

The scripts generate all figures from code. Figure 1 is a technical schematic drawn with Matplotlib patches and annotations; it does not rely on external or generative-AI image files. High-resolution TIFF versions named `Fig1.tif` through `Fig11.tif` are generated for journal production, while PNG copies are kept for review and repository viewing.

## Reproduction

Run:

```bash
python reproduce_article_123.py
python check_oso_ring_shear.py
python check_oso_ring_shear.py --offline
python run_global_sensitivity.py
python validation/run_all_validation.py --offline
```

The first script writes the synthetic benchmark CSV files in the same folder and writes regenerated figures to `generated_figures/`. The second script downloads selected public USGS/ScienceBase Oso ring-shear consolidation files, fits double- and single-drainage pressure-dissipation operators using a temporal split, and writes held-out consistency metrics and figures. The `--offline` option uses the included normalized Oso records. The third script runs a Latin-hypercube global sensitivity check. The final script runs the bounded external consistency workflow and regenerates the output metrics and multi-dataset figures from available public data.

## Files

- `retention_curve.csv`: spectral retained-pressure curve.
- `benchmark_cases.csv`: synthetic infrastructure-slope benchmark cases.
- `verification_fd_fem.csv`: spectral, finite-difference and finite-element comparison.
- `convergence_checks.csv`: grid, mesh and spectral-series convergence checks.
- `literature_comparison.csv`: comparison with existing approaches.
- `external_consistency_check.csv`: external consistency anchors.
- `sensitivity.csv`: normalized sensitivity of the partly drained factor of safety.
- `thresholds.csv`: retained-pressure regime boundaries.
- `boundary_condition_comparison.csv`: double-drainage versus single-drainage retained-pressure thresholds.
- `temporal_source_retention.csv`: retained fractions for front-loaded, constant, middle-pulse and back-loaded pressure-generation histories.
- `truncation_bound_check.csv`: spectral truncation errors and positive tail bounds used to audit convergence and uniqueness of the thresholds.
- `reproduce_article_123.py`: script for regenerating the numerical data and figures.
- `check_oso_ring_shear.py`: script for downloading public USGS Oso ring-shear consolidation data and fitting pressure-dissipation operators.
- `oso_ring_shear_consistency_summary.csv`: aggregate laboratory consistency metrics.
- `oso_ring_shear_consistency_metrics.csv`: per-record fitted hydraulic parameters and held-out errors.
- `oso_ring_shear_consistency_predictions.csv`: sampled observed and fitted retained-pressure curves.
- `oso_ring_shear_normalized_records.csv`: normalized Oso pressure-dissipation records used by the offline consistency-check option.
- `global_sensitivity_lhs_samples.csv`: Latin-hypercube samples for h, cv, T, Pu, phi, tau and sigma_eff0.
- `global_sensitivity_spearman.csv`: Spearman/PRCC-style rank correlations for R(Pi), FS_PD and Psi.
- `global_sensitivity_prcc.csv`: partial rank correlations after removing rank-linear dependence on the remaining inputs.
- `global_sensitivity_summary.csv`: summary statistics from the global sensitivity benchmark.
- `run_global_sensitivity.py`: script for regenerating the global sensitivity benchmark.
- `validation_protocol.md`: predeclared external consistency/validation protocol and guardrails.
- `data_sources.md`: official USGS/ScienceBase source list and DOI traceability.
- `validation/run_all_validation.py`: bounded external consistency workflow for Oso, Mount Kaba-san and available USGS flume records.
- `outputs/external_dataset_inventory.csv`: inventory of the public validation sources and the role each source can support.
- `outputs/mount_kabasan_consistency_metrics.csv`: field-scale pressure-dissipation consistency metrics for the Mount Kaba-san failure-period pore-pressure records.
- `outputs/flume_regime_observation_summary.csv`: observed pore-pressure regime inventory from available USGS flume corrected CSV files.
- `requirements.txt`: minimal Python package list.
- `generated_figures/`: regenerated figures used for manuscript review and journal production.

The benchmark data are synthetic and support reproducibility of the mathematical operator. The Oso consistency-check files are derived from the public USGS data release cited in the manuscript and are used only as laboratory-scale dissipation consistency checks. Mount Kaba-san and the USGS flume records are official public USGS data used for bounded external consistency and regime-inventory checks, not as field-scale design calibration.

