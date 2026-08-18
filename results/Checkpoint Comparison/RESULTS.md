# Results in text form

## Checkpoint comparison

Checkpoint 5, used by the current ranking script:

- Original validation, 10,000 samples: Pearson 0.8967, Spearman 0.8924, direction accuracy 96.20%, MSE 94.04, RMSE 9.70.
- New 2026 validation, 12,191 samples: Pearson 0.8573, Spearman 0.8915, direction accuracy 98.70%, MSE 147.39, RMSE 12.14.

Metric-specific winners:

- Original validation correlation: checkpoint 7, Pearson 0.9189.
- Original validation RMSE: checkpoint 6, RMSE 8.4881.
- New 2026 validation RMSE: checkpoint 4, RMSE 11.9970.
- New 2026 direction accuracy: checkpoint 5, 98.70%.

The newer holdout retained strong rank correlation and direction accuracy but had worse magnitude error. Checkpoint 5 is a design choice, not a universal numerical winner.

## Interval calibration

The 2026 completed-positive-peak audit used 5,410 cases:

- Lower-bound achieved coverage: 94.25% against a nominal 90% target.
- Upper-bound achieved coverage: 92.50% against a nominal 90% target.
- Central interval achieved coverage: 86.75% against a nominal 80% target.

All three intervals were conservative on this audit.

## Training-cost graph in text form

`training_cost.html` contains 21 recorded checkpoint MSE values.

- First recorded MSE: 89.79.
- Final recorded MSE: 59.23.
- Lowest recorded MSE: 59.23.
- Largest outlier: 1,524.07 at the fifth zero-based graph position.
- Linear fit shown by the graph: 328.05 at the beginning to 39.95 at the end.

The fitted decline is visually strong, but the single 1,524 outlier heavily affects that line. The raw checkpoints fluctuate, so the final lower error is more defensible evidence than the fitted slope alone.

## Preserved prediction output

The saved 2026-06-23 run evaluated 503 S&P 500 constituents, marked 265 as eligible current trough cases, and excluded 238. These rows are historical output and must not be read as current investment recommendations.
