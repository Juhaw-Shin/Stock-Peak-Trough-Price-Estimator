# Version history

## Original model

The first pipeline downloaded stock and macroeconomic history, constructed extrema-based samples, normalized 191 features, and trained a small feed-forward network with manually implemented NumPy operations.

## Seven saved checkpoints

Weights and biases from seven training stages were retained. Later validation compared all checkpoints on the original set and a separately constructed 2026 holdout. Checkpoint 5 is curruntly the best at generalized prediction.

## Recent-data validation

`Validate_New_Data.py` rebuilt the market and macroeconomic feature pipeline for 2026 without changing the training implementation. It added checkpoint comparison, correlation, direction, error, and temporal-coverage reports.

## Calibrated current prediction

`Stock_Prediction_Recent.py` added local residual bounds around checkpoint 5 predictions. `Stock_Prediction_Better.py` isolated the current workflow, tightened freshness checks, added provisional recent lows, adjusted short-duration uncertainty, and produced ranked S&P 500 output.
