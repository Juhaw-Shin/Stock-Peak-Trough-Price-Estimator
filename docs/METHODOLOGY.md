# Methodology

## Target

The model predicts the percentage return from a detected trough to the next detected peak. It does not directly predict tomorrow's price.

## Features and Predictor

Each sample contains 191 normalized inputs derived from stock price, volume, market benchmark behavior, company-relative extrema, and macroeconomic series. `Ai_Back.py` implements the compact neural-network operations; saved `Weight_N.npz` and `Bias_N.npz` files represent seven checkpoints.

The production ranking uses checkpoint 5 for its point estimate. The originally trained weights are not modified by the newer ranking code.

## Uncertainty Bounds

The predictor finds completed historical examples near the current feature state and locally calibrates prediction residuals. Residual q10 and q90 values define a lower one-sided 90% bound, an upper one-sided 90% bound, and their combined central 80% interval.

## Ranking

Candidates are ranked based on lower remaining upside, expected remaining upside, interval width, and confidence. Short-duration cases receive wider intervals because early trough estimates have less supporting evidence.

## Validation Construction

The newer evaluation uses stock data collected from 2026 after the original training pipeline was developed. For each detected extremum, multiple samples are constructed at different observation durations before the next extremum. These samples therefore provide different temporal perspectives of the same eventual transition and are not statistically independent.

The prediction target is the percentage return from the current detected trough to the subsequent detected peak.
