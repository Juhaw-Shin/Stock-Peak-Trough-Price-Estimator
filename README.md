# Market Prediction AI

A research system that predicts the return from an identified stock trough to its next local peak, then calibrates uncertainty bounds and ranks current S&P 500 candidates.

## Project in one glance

```text
Daily stock, volume, benchmark, and macroeconomic history
        |
        v
191 normalized market and company features
        |
        v
Feed-forward return predictor
        |
        v
Local residual calibration using similar historical cases
        |
        v
Lower, expected, and upper peak-return estimates
        |
        v
Confidence-aware candidate ranking
```

## Honest validation result

The production prediction script uses checkpoint 5.

- Original 10,000-sample validation: Pearson 0.8967, Spearman 0.8924, direction accuracy 96.20%, RMSE 9.70 percentage points.
- New 2026 validation with 12,191 samples across 202 tickers: Pearson 0.8573, Spearman 0.8915, direction accuracy 98.70%, RMSE 12.14 percentage points.
- The calibrated interval achieved 94.25% lower-bound coverage, 92.50% upper-bound coverage, and 86.75% central coverage. These exceed the nominal 90%, 90%, and 80% targets, so the intervals were conservative on this holdout.

Checkpoint 5 was not the winner on every metric: checkpoint 4 had the lowest new-2026 RMSE, while checkpoints 6 and 7 were strongest on the original validation. This is analyzed to be a overfitting.

## Repository layout

```text
src/       Current prediction, validation, model, and data-processing code
NPZ/       Small learned weights, biases, normalization, and macro coefficients
results/   Checkpoint comparason and recent stock prediction
data/      Macroeconomical data with stock-related data.
```

## Main commands

```powershell
python src/Validate_New_Data.py
python src/Stock_Prediction_Better.py
```
