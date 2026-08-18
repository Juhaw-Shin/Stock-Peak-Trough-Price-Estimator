"""Calibrated next-peak bounds for current S&P 500 constituents.

The point estimate is always the existing Weight_5 network.  This script does
not train or modify that network.  It calibrates Weight_5 residuals from the
existing and 2026 validation sets, conditional on features related to forecast
uncertainty, and applies those bounds to companies whose latest confirmed
extremum is a trough.

This copy is biased toward the most recent market state.  By default it
redownloads current constituent bars and only reuses cache that reaches the
latest benchmark date, so an intraday daily bar for today is included when
yfinance exposes one.

The lower bound is one-sided 90%: among locally similar completed peak cases,
approximately 90% of actual peak returns were above the lower bound.  The
upper bound is defined analogously.  Together q10/q90 form a central 80%
interval.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
from scipy.spatial import cKDTree
import yfinance as yf


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "src" / "Validate_New_Data.py").exists() and (path / "NPZ").exists():
            return path
    raise RuntimeError("Could not locate repository root containing src/Validate_New_Data.py and NPZ")


ROOT = find_project_root(SCRIPT_DIR)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import Validate_New_Data as validation


NPZ = ROOT / "NPZ"
NEW = ROOT / "data" / "generated"
OUTPUT = ROOT / "results" / "generated"
CACHE = ROOT / "data" / "cache" / "SP500_Current"
CONSTITUENTS_CACHE = ROOT / "data" / "cache" / "SP500_Constituents.csv"
WEIGHT_VERSION = 5
LOWER_QUANTILE = 0.10
UPPER_QUANTILE = 0.90
MIN_HISTORY_DAYS = 756  # roughly three trading years
MIN_DURATION = 4        # same minimum used by the training feature builder
RECENT_OUTPUT_PREFIX = "Stock_Predictions_Better_Weight5"


def atomic_savez(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as file:
        np.savez(file, **values)
    os.replace(temp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def weighted_quantile(values: np.ndarray, quantiles: list[float], weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    cumulative = np.cumsum(weights) - 0.5 * weights
    if cumulative[-1] <= 0:
        return np.quantile(values, quantiles)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)


def duration_class(duration: float) -> str:
    if duration <= 7:
        return "very_short_4_7"
    if duration <= 14:
        return "short_8_14"
    if duration <= 28:
        return "medium_15_28"
    return "long_29_plus"


def beta_class(beta: float) -> str:
    if beta < 0.75:
        return "low_beta"
    if beta <= 1.25:
        return "market_beta"
    return "high_beta"


def rsi_class(rsi: float) -> str:
    if rsi < 30:
        return "oversold"
    if rsi <= 70:
        return "neutral"
    return "overbought"


def descriptors(raw_features: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Features used only to select comparable calibration residuals.

    These do not alter Weight 5's expected value.  They describe information
    quantity and instability: duration, beta, RSI, prior swing duration,
    typical extremum spacing, recent return volatility/momentum, volume level
    and curvature, and the magnitude of the model prediction.
    """
    raw = np.atleast_2d(raw_features)
    point = np.asarray(point).reshape(-1)
    return_window = raw[:, 17:80]
    return np.column_stack([
        np.log1p(np.maximum(raw[:, 14], 0.0)),                 # duration/info
        raw[:, 0],                                            # beta
        raw[:, 9] / 100.0,                                    # RSI
        np.log1p(np.maximum(raw[:, 15], 0.0)),                # prior swing
        np.log1p(np.maximum(raw[:, 16], 0.0)),                # avg peak spacing
        np.std(return_window, axis=1),                         # realized vol
        np.mean(return_window, axis=1),                        # momentum
        np.log1p(np.maximum(np.abs(raw[:, 10]), 0.0)),        # fall volume peak
        np.log1p(np.maximum(np.abs(raw[:, 11]), 0.0)),        # fall peakiness
        np.log1p(np.maximum(np.abs(raw[:, 12]), 0.0)),        # rise peakiness
        np.log1p(np.maximum(np.abs(raw[:, 13]), 0.0)),        # rise volume peak
        np.sign(point) * np.log1p(np.abs(point)),             # prediction scale
    ])


@dataclass
class BoundResult:
    lower: float
    expected: float
    upper: float
    neighbors: int
    lower_coverage: float
    upper_coverage: float
    duration_group: str
    nearest_distance: float


class PeakBoundCalibrator:
    """Local residual quantiles calibrated only on completed positive peaks."""

    def __init__(self) -> None:
        mean_std = validation.load_npz(NPZ / "Mean_Std.npz")
        self.feature_mean = mean_std["mean"]
        self.feature_std = mean_std["std"]

        old_x = np.load(NPZ / "Training Data" / "Valid_Sample.npy")
        old_y = np.load(NPZ / "Training Data" / "Valid_Result.npy").ravel()
        new_x = np.load(NEW / "Valid_Sample_New.npy")
        new_y = np.load(NEW / "Valid_Result_New.npy").ravel()
        sample = np.vstack([old_x, new_x])
        actual = np.concatenate([old_y, new_y])
        source_weight = np.concatenate([np.ones(len(old_y)), np.full(len(new_y), 2.0)])
        point = validation.predict(sample, WEIGHT_VERSION).ravel()
        raw = sample * self.feature_std + self.feature_mean

        # Positive completed transitions are the historical analogues of
        # current trough -> next peak cases.
        keep = (actual > 0) & np.isfinite(point) & np.isfinite(raw).all(axis=1)
        self.raw = raw[keep]
        self.actual = actual[keep]
        self.point = point[keep]
        self.residual = self.actual - self.point
        self.source_weight = source_weight[keep]
        desc = descriptors(self.raw, self.point)
        self.center = np.median(desc, axis=0)
        q25, q75 = np.quantile(desc, [0.25, 0.75], axis=0)
        self.scale = np.where(q75 > q25, q75 - q25, 1.0)
        self.scaled = np.clip((desc - self.center) / self.scale, -8.0, 8.0)

        self.groups: dict[str, tuple[np.ndarray, cKDTree]] = {}
        durations = self.raw[:, 14]
        for group in ("very_short_4_7", "short_8_14", "medium_15_28", "long_29_plus"):
            indices = np.where(np.array([duration_class(value) == group for value in durations]))[0]
            self.groups[group] = (indices, cKDTree(self.scaled[indices]))

    @property
    def calibration_size(self) -> int:
        return len(self.actual)

    def bounds(self, raw_feature: np.ndarray, point: float) -> BoundResult:
        group = duration_class(float(raw_feature[14]))
        indices, tree = self.groups[group]
        query = np.clip((descriptors(raw_feature, np.array([point]))[0] - self.center) / self.scale, -8.0, 8.0)
        count = min(800, len(indices))
        distance, local = tree.query(query, k=count)
        distance = np.atleast_1d(distance)
        local = np.atleast_1d(local)
        selected = indices[local]
        weights = self.source_weight[selected] / (0.25 + distance)
        q10, q50, q90 = weighted_quantile(
            self.residual[selected], [LOWER_QUANTILE, 0.50, UPPER_QUANTILE], weights,
        )

        # Short observations contain less information.  Conditional residuals
        # already capture most of this; this explicit finite-information factor
        # prevents a spuriously narrow interval in the smallest-duration bins.
        duration = float(raw_feature[14])
        information_factor = math.sqrt(14.0 / max(duration, 5.0)) if duration < 14 else 1.0
        q10 = q50 + (q10 - q50) * information_factor
        q90 = q50 + (q90 - q50) * information_factor
        lower = point + q10
        expected = point + q50
        upper = point + q90
        lower_coverage = float(np.average(self.residual[selected] >= q10, weights=weights))
        upper_coverage = float(np.average(self.residual[selected] <= q90, weights=weights))
        return BoundResult(
            lower=float(lower), expected=float(expected), upper=float(upper),
            neighbors=count, lower_coverage=lower_coverage,
            upper_coverage=upper_coverage, duration_group=group,
            nearest_distance=float(distance[0]),
        )


def temporal_coverage_audit() -> dict:
    """Calibrate on the older validation set and score untouched 2026 peaks."""
    mean_std = validation.load_npz(NPZ / "Mean_Std.npz")
    mean, std = mean_std["mean"], mean_std["std"]
    old_x = np.load(NPZ / "Training Data" / "Valid_Sample.npy")
    old_y = np.load(NPZ / "Training Data" / "Valid_Result.npy").ravel()
    new_x = np.load(NEW / "Valid_Sample_New.npy")
    new_y = np.load(NEW / "Valid_Result_New.npy").ravel()
    old_point = validation.predict(old_x, WEIGHT_VERSION).ravel()
    new_point = validation.predict(new_x, WEIGHT_VERSION).ravel()
    old_raw, new_raw = old_x * std + mean, new_x * std + mean
    old_keep, new_keep = old_y > 0, new_y > 0
    old_desc = descriptors(old_raw[old_keep], old_point[old_keep])
    center = np.median(old_desc, axis=0)
    q25, q75 = np.quantile(old_desc, [0.25, 0.75], axis=0)
    scale = np.where(q75 > q25, q75 - q25, 1.0)
    scaled = np.clip((old_desc - center) / scale, -8.0, 8.0)
    residual = old_y[old_keep] - old_point[old_keep]
    old_duration = old_raw[old_keep, 14]
    groups = {}
    names = ("very_short_4_7", "short_8_14", "medium_15_28", "long_29_plus")
    for name in names:
        indices = np.where(np.array([duration_class(value) == name for value in old_duration]))[0]
        groups[name] = (indices, cKDTree(scaled[indices]))

    actual = new_y[new_keep]
    lower, upper = [], []
    for raw, point in zip(new_raw[new_keep], new_point[new_keep]):
        indices, tree = groups[duration_class(raw[14])]
        query = np.clip((descriptors(raw, np.array([point]))[0] - center) / scale, -8.0, 8.0)
        distance, local = tree.query(query, k=min(800, len(indices)))
        selected = indices[np.atleast_1d(local)]
        weights = 1.0 / (0.25 + np.atleast_1d(distance))
        q10, q50, q90 = weighted_quantile(residual[selected], [0.10, 0.50, 0.90], weights)
        factor = math.sqrt(14.0 / max(raw[14], 5.0)) if raw[14] < 14 else 1.0
        lower.append(point + q50 + (q10 - q50) * factor)
        upper.append(point + q50 + (q90 - q50) * factor)
    lower, upper = np.asarray(lower), np.asarray(upper)
    return {
        "test_set": "2026 completed positive-peak samples",
        "test_samples": int(len(actual)),
        "lower_bound_achieved_coverage": float(np.mean(actual >= lower)),
        "upper_bound_achieved_coverage": float(np.mean(actual <= upper)),
        "central_interval_achieved_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
    }


def current_constituents() -> pd.DataFrame:
    # Public mirror of the current S&P 500 constituent table.
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    try:
        frame = pd.read_csv(url)
        atomic_text(CONSTITUENTS_CACHE, frame.to_csv(index=False))
    except Exception:
        if not CONSTITUENTS_CACHE.exists():
            raise
        frame = pd.read_csv(CONSTITUENTS_CACHE)
    frame["YahooSymbol"] = frame["Symbol"].str.replace(".", "-", regex=False)
    return frame[["YahooSymbol", "Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]


def benchmark(end_date: str, refresh: bool, include_recent: bool) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE / "_SP500.npz"
    if path.exists() and not refresh:
        saved = validation.load_npz(path)
        cached_request = str(saved.get("requested_end", np.array([""]))[0])
        saved_dates = saved["dates"]
        reaches_latest_request = len(saved_dates) and cached_request >= end_date
        if include_recent:
            expected_latest = np.datetime64((pd.Timestamp(end_date) - pd.Timedelta(days=1)).date())
            reaches_latest_request = reaches_latest_request and saved_dates[-1] >= expected_latest
        if reaches_latest_request:
            return saved["dates"], saved["price"]
    data = yf.download(
        "^GSPC", start="2005-01-01", end=end_date, interval="1d",
        auto_adjust=True, progress=False, threads=False,
    )
    dates = data.index.tz_localize(None).normalize().to_numpy(dtype="datetime64[D]")
    price = validation.ohlc_average(data, "^GSPC").to_numpy(dtype=np.float64)
    atomic_savez(path, {"dates": dates, "price": price, "requested_end": np.array([end_date])})
    return dates, price


def seed_trained_tickers(benchmark_dates: np.ndarray) -> set[str]:
    """Reuse the project's exact adjusted histories for its 212 model tickers."""
    old_price = validation.load_npz(NPZ / "Ticker" / "Price" / "Price.npz")
    old_volume = validation.load_npz(NPZ / "Ticker" / "Volume" / "Volume.npz")
    new_price = validation.load_npz(NEW / "Price_New.npz")
    new_volume = validation.load_npz(NEW / "Volume_New.npz")
    seeded: set[str] = set()
    for ticker in new_price:
        path = CACHE / f"{ticker}.npz"
        values = np.concatenate([old_price[ticker + "/Price"], new_price[ticker]])
        volumes = np.concatenate([old_volume[ticker + "/Volume"], new_volume[ticker]])
        if len(values) != len(benchmark_dates):
            continue
        atomic_savez(path, {"dates": benchmark_dates, "price": values, "volume": volumes})
        seeded.add(ticker)
    return seeded


def extract_ticker(data: pd.DataFrame, ticker: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        price = validation.ohlc_average(data, ticker)
        volume = validation.volume_series(data, ticker)
    except KeyError:
        return None
    good = price.notna() & volume.notna() & (volume > 0)
    if not good.any():
        return None
    dates = price.index[good].tz_localize(None).normalize().to_numpy(dtype="datetime64[D]")
    return dates, price[good].to_numpy(dtype=np.float64), volume[good].to_numpy(dtype=np.float64)


def collect_constituents(
    frame: pd.DataFrame,
    end_date: str,
    benchmark_dates: np.ndarray,
    refresh: bool,
    include_recent: bool,
) -> dict[str, str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    seeded = seed_trained_tickers(benchmark_dates) if not refresh and not include_recent else set()
    status: dict[str, str] = {}
    pending: list[str] = []
    latest_benchmark_date = benchmark_dates[-1]
    for ticker in frame["YahooSymbol"]:
        path = CACHE / f"{ticker}.npz"
        if ticker in seeded:
            status[ticker] = "project_history"
        elif path.exists() and not refresh:
            saved = validation.load_npz(path)
            lag_days = int((benchmark_dates[-1] - saved["dates"][-1]) / np.timedelta64(1, "D"))
            current_enough = saved["dates"][-1] >= latest_benchmark_date if include_recent else lag_days <= 7
            if len(saved["price"]) >= MIN_HISTORY_DAYS and current_enough:
                status[ticker] = "cached_yfinance"
            else:
                pending.append(ticker)
        else:
            pending.append(ticker)

    for offset in range(0, len(pending), 24):
        chunk = pending[offset:offset + 24]
        data = None
        for attempt in range(3):
            data = yf.download(
                chunk, start="2005-01-01", end=end_date, interval="1d",
                auto_adjust=True, progress=False, threads=True, group_by="column",
            )
            if not data.empty:
                break
            time.sleep(2 ** attempt)
        if data is None or data.empty:
            for ticker in chunk:
                status[ticker] = "download_failed"
            continue
        for ticker in chunk:
            extracted = extract_ticker(data, ticker)
            if extracted is None:
                status[ticker] = "download_failed"
                continue
            dates, price, volume = extracted
            atomic_savez(CACHE / f"{ticker}.npz", {"dates": dates, "price": price, "volume": volume})
            status[ticker] = "yfinance"
        print(f"Collected {min(offset + len(chunk), len(pending))}/{len(pending)} additional constituents")
    return status


def labeled_extrema(price: np.ndarray) -> tuple[np.ndarray, set[int]]:
    raw_42 = validation.get_rel(price, 42)
    raw_48 = validation.get_rel(price, 48)
    raw = np.union1d(raw_42, raw_48).astype(int)
    filtered = validation.filter_extrema(price, raw)
    troughs: set[int] = set()
    for split in (42, 48):
        position = 0
        for part in np.array_split(price, split):
            prominence = part.mean() * 0.05
            found = ss.find_peaks(-part, prominence=prominence, distance=5)[0] + position
            troughs.update(int(value) for value in found)
            position += len(part)
    return filtered, troughs


def select_active_trough(price: np.ndarray, extrema: np.ndarray, troughs: set[int]) -> tuple[int, int, str] | tuple[None, None, str]:
    latest = int(extrema[-1])
    if latest in troughs:
        return latest, int(extrema[-2]), "confirmed_trough"

    # The historical extrema algorithm confirms turns late.  For current
    # predictions, use the lowest point after the latest confirmed peak once
    # there are enough observations for the same feature shape the model saw
    # during training.
    previous_peak = latest
    search_start = previous_peak + 1
    if len(price) - search_start < MIN_DURATION + 1:
        return None, None, "no_recent_low_after_latest_peak"
    trough = search_start + int(np.argmin(price[search_start:]))
    if trough <= previous_peak:
        return None, None, "no_recent_low_after_latest_peak"
    if len(price) - 1 - trough < MIN_DURATION:
        return None, None, "provisional_trough_has_fewer_than_five_observations"
    if trough - previous_peak < MIN_DURATION:
        return None, None, "provisional_fall_too_short"
    return trough, previous_peak, "provisional_recent_low"


def macro_for_trough(trough_date: np.datetime64) -> np.ndarray | None:
    macro = validation.load_npz(NEW / "Macro_Coef_New.npz")
    windows = validation.macro_windows(macro)
    date = pd.Timestamp(trough_date)
    month_index = (date.year - 2005) * 12 + date.month - 1
    return windows.get(str(month_index - 1))


def current_feature(
    ticker: str,
    dates: np.ndarray, price: np.ndarray, volume: np.ndarray,
    benchmark_dates: np.ndarray, benchmark_price: np.ndarray,
    trained_ticker: bool,
) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    if len(price) < MIN_HISTORY_DAYS:
        return None, {"reason": "less_than_three_years_history"}
    extrema, troughs = labeled_extrema(price)
    if len(extrema) < 2:
        return None, {"reason": "too_few_confirmed_extrema"}
    trough, previous, trough_status = select_active_trough(price, extrema, troughs)
    if trough is None or previous is None:
        return None, {"reason": trough_status}
    duration = len(price) - 1 - trough
    if duration < MIN_DURATION:
        return None, {"reason": "trough_has_fewer_than_five_observations"}
    if price[-1] < price[trough]:
        return None, {"reason": "current_price_has_broken_below_confirmed_trough"}

    # Match the benchmark to this ticker's actual trading dates.
    bench = pd.Series(benchmark_price, index=pd.to_datetime(benchmark_dates))
    aligned = bench.reindex(pd.to_datetime(dates)).ffill().bfill().to_numpy(dtype=np.float64)
    if len(aligned) != len(price):
        return None, {"reason": "benchmark_alignment_failed"}

    ps = ss.savgol_filter(price, window_length=21, polyorder=3)
    vs = ss.savgol_filter(volume, window_length=21, polyorder=3)
    ps2 = ps[:len(ps) // 2 * 2].reshape(-1, 2).mean(axis=1)
    vs2 = vs[:len(vs) // 2 * 2].reshape(-1, 2).mean(axis=1)
    price_return = 100.0 * np.diff(ps2) / ps2[:-1]
    volume_rate = 100.0 * np.diff(vs2) / vs2[:-1]
    divided = (trough + duration) // 2
    return_window = price_return[divided - 64:divided - 1]
    volume_window = volume_rate[divided - 64:divided - 1]
    if len(return_window) != 63 or len(volume_window) != 63:
        return None, {"reason": "insufficient_smoothed_return_history"}

    raw_union = np.union1d(validation.get_rel(price, 42), validation.get_rel(price, 48))
    if trained_ticker:
        old_org = validation.load_npz(NPZ / "Ticker" / "Ext_Org.npz")
        peak_duration = 5283.0 / len(old_org[ticker])
    else:
        peak_duration = len(price) / max(len(raw_union), 1)

    end = len(price) - 1
    ticker_values = price[trough:end + 1]
    fall_values = price[previous:trough + 1]
    volume_values = volume[trough:end + 1]
    volume_fall = volume[previous:trough + 1]
    normalizer = np.mean(volume[end - 125:end + 1])
    volume_values = volume_values / normalizer
    volume_fall = volume_fall / normalizer
    rise_delta = np.diff(ticker_values) / price[trough]
    fall_delta = np.diff(fall_values) / price[previous]
    ticker_ret = np.diff(price[end - 125:end + 1]) / price[end - 125:end]
    bench_ret = np.diff(aligned[end - 125:end + 1]) / aligned[end - 125:end]
    beta = np.cov(ticker_ret, bench_ret)[0, 1] / np.var(bench_ret, ddof=1)
    fall_duration = trough - previous
    rises = [part.mean() / duration for part in np.array_split(rise_delta * 100.0, 4)]
    falls = [part.mean() / fall_duration for part in np.array_split(fall_delta * 100.0, 4)]

    smooth = ss.savgol_filter(
        volume_values, 2 * round((len(volume_values) - 3) / 8) + 3,
        polyorder=min(round(len(volume_values) / 3) - 2, 2),
    )
    volume_peakiness = np.max(np.abs(np.diff(smooth, 2)))
    volume_peak = np.max(volume_values)
    smooth_fall = ss.savgol_filter(
        volume_fall, 2 * round((len(volume_fall) - 3) / 8) + 3,
        polyorder=min(round(len(volume_fall) / 3) - 2, 2),
    )
    volume_fall_peakiness = np.max(np.abs(np.diff(smooth_fall, 2)))
    volume_fall_peak = np.max(volume_fall)

    if not np.isfinite(peak_duration):
        peak_duration = len(price) / max(len(raw_union), 1)
    pattern = np.asarray([
        beta, *rises, *falls, validation.rsi(price, end), volume_fall_peak,
        volume_fall_peakiness, volume_peakiness, volume_peak, duration,
        fall_duration, peak_duration,
    ])
    macro = macro_for_trough(dates[trough])
    if macro is None:
        return None, {"reason": "macro_window_unavailable"}
    raw = np.concatenate([pattern, return_window, volume_window, macro])
    if raw.shape != (191,) or not np.isfinite(raw).all():
        return None, {"reason": "nonfinite_feature"}
    return raw, {
        "reason": "eligible", "trough_index": trough, "trough_date": str(dates[trough])[:10],
        "trough_status": trough_status,
        "current_date": str(dates[-1])[:10],
        "duration": duration, "beta": beta, "rsi": pattern[9],
        "current_price": float(price[-1]), "trough_price": float(price[trough]),
        "current_return_from_trough": float(100.0 * (price[-1] / price[trough] - 1.0)),
        "history_days": len(price),
    }


def reliability_label(nearest_distance: float, history_days: int, trained_ticker: bool) -> str:
    if history_days < 1260 or nearest_distance > 4.0:
        return "low"
    if not trained_ticker or nearest_distance > 2.5:
        return "medium"
    return "high"


def reliability_score(nearest_distance: float, history_days: int, trained_ticker: bool, trough_status: str) -> float:
    distance_score = 1.0 / (1.0 + max(nearest_distance, 0.0))
    history_score = min(history_days / 2520.0, 1.0)
    trained_score = 1.0 if trained_ticker else 0.85
    trough_score = 1.0 if trough_status == "confirmed_trough" else 0.78
    return float(distance_score * history_score * trained_score * trough_score)


def ensure_prediction_data(latest_market_date: np.datetime64, refresh: bool) -> None:
    required = [
        NPZ / "Mean_Std.npz",
        NPZ / f"Weight_{WEIGHT_VERSION}.npz",
        NPZ / f"Bias_{WEIGHT_VERSION}.npz",
        NPZ / "Training Data" / "Valid_Sample.npy",
        NPZ / "Training Data" / "Valid_Result.npy",
        NEW / "Valid_Sample_New.npy",
        NEW / "Valid_Result_New.npy",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Validation Data is lacking: " + ", ".join(missing))

    # A trough in the current month consumes the previous completed macro
    # month.  Extend the macro cache automatically when this script is run on
    # a later date.
    latest = pd.Timestamp(latest_market_date)
    required_macro_month = (latest.to_period("M") - 1).to_timestamp()
    try:
        validation.build_macro(required_macro_month, refresh)
    except Exception as error:
        raise RuntimeError(f"Macro Data is lacking: {error}") from error


def predict_all_sp500(
    end_date: str,
    refresh: bool = False,
    top: int = 25,
    include_recent: bool = True,
) -> pd.DataFrame:
    try:
        constituents = current_constituents()
    except Exception as error:
        raise RuntimeError(f"S&P 500 Constituent Data is lacking: {error}") from error
    try:
        benchmark_dates, benchmark_price = benchmark(end_date, refresh, include_recent)
    except Exception as error:
        raise RuntimeError(f"S&P 500 Benchmark Data is lacking: {error}") from error
    ensure_prediction_data(benchmark_dates[-1], refresh)
    status = collect_constituents(constituents, end_date, benchmark_dates, refresh, include_recent)
    calibrator = PeakBoundCalibrator()
    trained = set(validation.ticker_list())
    info = constituents.set_index("YahooSymbol").to_dict("index")
    rows: list[dict] = []
    excluded: list[dict] = []

    for number, ticker in enumerate(constituents["YahooSymbol"], start=1):
        path = CACHE / f"{ticker}.npz"
        if not path.exists():
            excluded.append({"ticker": ticker, "reason": status.get(ticker, "missing_cache")})
            continue
        market = validation.load_npz(path)
        raw, metadata = current_feature(
            ticker,
            market["dates"], market["price"], market["volume"],
            benchmark_dates, benchmark_price, ticker in trained,
        )
        if raw is None:
            excluded.append({"ticker": ticker, **metadata})
            continue
        standardized = (raw - calibrator.feature_mean) / calibrator.feature_std
        point = float(validation.predict(standardized[None, :], WEIGHT_VERSION)[0, 0])
        bound = calibrator.bounds(raw, point)

        # A future peak cannot be below the current observed point while the
        # current trough state remains valid.  Enforce that path constraint.
        current_gain = metadata["current_return_from_trough"]
        lower_return = max(bound.lower, current_gain)
        expected_return = max(bound.expected, current_gain)
        upper_return = max(bound.upper, expected_return)
        trough_price = metadata["trough_price"]
        current_price = metadata["current_price"]
        lower_price = trough_price * (1.0 + lower_return / 100.0)
        expected_price = trough_price * (1.0 + expected_return / 100.0)
        upper_price = trough_price * (1.0 + upper_return / 100.0)
        remaining_lower = 100.0 * (lower_price / current_price - 1.0)
        remaining_expected = 100.0 * (expected_price / current_price - 1.0)
        remaining_upper = 100.0 * (upper_price / current_price - 1.0)
        interval_width = remaining_upper - remaining_lower
        confidence = reliability_score(
            bound.nearest_distance, metadata["history_days"], ticker in trained, metadata["trough_status"],
        )
        rank_score = (
            remaining_lower
            + 0.35 * remaining_expected
            - 0.18 * max(interval_width, 0.0)
            + 4.0 * confidence
        )
        company = info[ticker]
        rows.append({
            "ticker": company["Symbol"], "yahoo_ticker": ticker,
            "company": company["Security"], "sector": company["GICS Sector"],
            "data_date": metadata["current_date"],
            "trough_status": metadata["trough_status"],
            "current_price": current_price, "trough_date": metadata["trough_date"],
            "trough_price": trough_price, "duration_days": metadata["duration"],
            "beta": metadata["beta"], "beta_class": beta_class(metadata["beta"]),
            "rsi": metadata["rsi"], "rsi_class": rsi_class(metadata["rsi"]),
            "duration_class": bound.duration_group,
            "weight5_point_return_from_trough_pct": point,
            "lower_peak_return_from_trough_pct": lower_return,
            "expected_peak_return_from_trough_pct": expected_return,
            "upper_peak_return_from_trough_pct": upper_return,
            "lower_peak_price": lower_price, "expected_peak_price": expected_price,
            "upper_peak_price": upper_price,
            "lower_remaining_upside_pct": remaining_lower,
            "expected_remaining_upside_pct": remaining_expected,
            "upper_remaining_upside_pct": remaining_upper,
            "interval_width_pct": interval_width,
            "confidence_score": confidence,
            "rank_score": rank_score,
            "local_lower_coverage": bound.lower_coverage,
            "local_upper_coverage": bound.upper_coverage,
            "calibration_neighbors": bound.neighbors,
            "reliability": reliability_label(bound.nearest_distance, metadata["history_days"], ticker in trained),
            "trained_constituent": ticker in trained,
            "history_days": metadata["history_days"],
        })
        if number % 50 == 0:
            print(f"Analyzed {number}/{len(constituents)} constituents")

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(
            ["rank_score", "lower_remaining_upside_pct", "expected_remaining_upside_pct"],
            ascending=False,
        ).reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
    else:
        result.insert(0, "rank", [])
    excluded_frame = pd.DataFrame(excluded)
    atomic_text(OUTPUT / f"{RECENT_OUTPUT_PREFIX}.csv", result.to_csv(index=False))
    atomic_text(OUTPUT / f"{RECENT_OUTPUT_PREFIX}_Top.json", result.head(top).to_json(orient="records", indent=2))
    atomic_text(OUTPUT / f"{RECENT_OUTPUT_PREFIX}_Excluded.csv", excluded_frame.to_csv(index=False))
    summary = {
        "as_of": str(benchmark_dates[-1])[:10], "weight": WEIGHT_VERSION,
        "model_contract": "point estimates are produced only by NPZ/Weight_5.npz and NPZ/Bias_5.npz",
        "recent_value_policy": (
            "requires ticker cache to reach the latest benchmark date; default downloads include "
            "today's partially revealed daily bar when yfinance returns it"
        ),
        "current_setup_policy": "confirmed troughs plus provisional recent lows after latest confirmed peaks",
        "constituents": int(len(constituents)), "eligible_current_troughs": int(len(result)),
        "excluded": int(len(excluded_frame)), "calibration_peak_samples": calibrator.calibration_size,
        "lower_bound_definition": "one-sided 90% local residual bound (q10)",
        "upper_bound_definition": "one-sided 90% local residual bound (q90)",
        "joint_interval": "central 80% (q10 to q90)",
        "ranking": "rank_score = lower upside + 0.35 expected upside - 0.18 interval width + confidence bonus",
        "short_duration_policy": "conditional duration group plus sqrt(14/duration) interval widening below 14 days",
        "important_limitation": "Extrema are confirmed retrospectively by the original algorithm; this is not guaranteed real-time profit.",
        "temporal_coverage_audit": temporal_coverage_audit(),
    }
    atomic_text(OUTPUT / f"{RECENT_OUTPUT_PREFIX}_Summary.json", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(excluded_frame):
        lacking_messages = {
            "download_failed": "Market Data is lacking",
            "missing_cache": "Market Data is lacking",
            "less_than_three_years_history": "Historical Data is lacking",
            "macro_window_unavailable": "Macro Data is lacking",
            "benchmark_alignment_failed": "Benchmark Data is lacking",
            "insufficient_smoothed_return_history": "Return History Data is lacking",
            "nonfinite_feature": "Usable Feature Data is lacking",
            "no_recent_low_after_latest_peak": "Recent trough setup is lacking",
            "provisional_trough_has_fewer_than_five_observations": "Recent trough setup is too new",
            "provisional_fall_too_short": "Recent trough setup is too shallow in time",
        }
        for reason, message in lacking_messages.items():
            affected = excluded_frame.loc[excluded_frame.get("reason", pd.Series(dtype=str)) == reason, "ticker"].tolist()
            if affected:
                print(f"{', '.join(affected)}: {message}")
    if len(result):
        columns = [
            "rank", "ticker", "data_date", "trough_status", "current_price", "expected_peak_price",
            "expected_remaining_upside_pct", "rank_score", "duration_days", "reliability",
        ]
        print(result.head(top)[columns].to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="redownload current constituent histories")
    parser.add_argument(
        "--no-recent",
        action="store_true",
        help="allow older cached rows instead of requiring the latest benchmark date",
    )
    parser.add_argument("--top", type=int, default=25, help="number of ranked selections to print")
    parser.add_argument("--end", help="exclusive yfinance end date; default is tomorrow")
    args = parser.parse_args()
    end_date = args.end or (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        predict_all_sp500(end_date=end_date, refresh=args.refresh, top=args.top, include_recent=not args.no_recent)
    except RuntimeError as error:
        print(str(error))


if __name__ == "__main__":
    main()
