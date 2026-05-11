# Supplementary material

Manuscript: A spectral dynamic-consolidation criterion for drainage transition in rapid saturated landslides

This package contains synthetic benchmark data and scripts used to reproduce the retention curve, numerical verification, sensitivity analysis, tables and figures.

Runtime used for generation: Python 3 with numpy, pandas and matplotlib. The script is self-contained and does not require SciPy.

Files:

- retention_curve.csv: spectral R(Pi) curve.
- benchmark_cases.csv: synthetic infrastructure slope benchmark.
- validation_fd_fem.csv: spectral, finite-difference and finite-element comparison.
- convergence_checks.csv: grid, mesh and series convergence checks.
- literature_comparison.csv: comparison with existing approaches.
- external_consistency_check.csv: external consistency anchors.
- sensitivity.csv: normalized sensitivity of FS_PD.
- thresholds.csv: drainage-regime boundaries.
- reproduce_article_123.py: numerical script for reproducing data and figures.

The data are synthetic and are intended for reproducibility of the mathematical benchmark. They are not field measurements.
