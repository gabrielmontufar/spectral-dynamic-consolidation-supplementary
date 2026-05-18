# Supplementary Material

Manuscript: A spectral dynamic-consolidation criterion for drainage transition in rapid saturated landslides.

This package contains synthetic benchmark data and a reproducible Python script used to regenerate the retained-pressure curve, numerical verification, convergence checks, sensitivity analysis and figures.

## Runtime

Python 3 with:

- numpy
- pandas
- matplotlib

The script does not require SciPy.

The script generates all six figures from code. Figure 1 is a technical schematic drawn with Matplotlib patches and annotations; it does not rely on external or generative-AI image files. High-resolution TIFF versions named `Fig1.tif` through `Fig6.tif` are generated for journal production, while PNG copies are kept for review and repository viewing.

## Reproduction

Run:

```bash
python reproduce_article_123.py
```

The script writes CSV files in the same folder and writes regenerated figures to `generated_figures/`.

## Files

- `retention_curve.csv`: spectral retained-pressure curve.
- `benchmark_cases.csv`: synthetic infrastructure-slope benchmark cases.
- `validation_fd_fem.csv`: spectral, finite-difference and finite-element comparison.
- `convergence_checks.csv`: grid, mesh and spectral-series convergence checks.
- `literature_comparison.csv`: comparison with existing approaches.
- `external_consistency_check.csv`: external consistency anchors.
- `sensitivity.csv`: normalized sensitivity of the partly drained factor of safety.
- `thresholds.csv`: retained-pressure regime boundaries.
- `reproduce_article_123.py`: script for regenerating the numerical data and figures.
- `requirements.txt`: minimal Python package list.
- `generated_figures/`: regenerated figures used for manuscript review and journal production.

The data are synthetic and are intended to support reproducibility of the mathematical benchmark. They are not field measurements.
