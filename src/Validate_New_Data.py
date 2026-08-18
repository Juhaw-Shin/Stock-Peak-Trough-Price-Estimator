"""Build and evaluate the 2026 holdout without modifying the training code.

This file intentionally mirrors the feature and forward-pass implementations in
Stock Data_Collect, Stock_Related.py, Ai_Back.py, and Ai_Structure.py.  Outputs
are written under NPZ/New.  Re-running is safe: completed downloads are cached,
and every generated file is replaced atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
from scipy.stats import spearmanr
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "NPZ"
NEW = ROOT / "data" / "generated"
TRAIN = NPZ / "Training Data"
START_DATE = "2026-01-01"
HISTORICAL_DAYS = 5283
MACRO_NAMES = [
    "Inflation_Rate", "Real_Rate", "Unemployment_Rate", "Credit_GDP",
    "Money_Supply", "Real_GDP", "VIX",
]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as file:
        np.save(file, array)
    os.replace(temp, path)


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


def ticker_list() -> list[str]:
    price = load_npz(NPZ / "Ticker" / "Price" / "Price.npz")
    volume = load_npz(NPZ / "Ticker" / "Volume" / "Volume.npz")
    # This reproduces the 212-name list used by Stock_Related.py.  FISV and
    # PSKY remain in the raw archive but contain legacy NaNs and were excluded
    # from the tensors on which these checkpoints were trained.
    return [
        key.removesuffix("/Price") for key, values in price.items()
        if key.removesuffix("/Price") + "/Volume" in volume
        and np.isfinite(values).all()
        and np.isfinite(volume[key.removesuffix("/Price") + "/Volume"]).all()
    ]


def ohlc_average(data: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(data.columns, pd.MultiIndex):
        parts = [data[(field, ticker)] for field in ("Open", "High", "Low", "Close")]
    else:
        parts = [data[field] for field in ("Open", "High", "Low", "Close")]
    return sum(parts) / 4.0


def volume_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(data.columns, pd.MultiIndex):
        return data[("Volume", ticker)]
    return data["Volume"]


def download_market(end_date: str, refresh: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    tickers = ticker_list()
    dates_path = NEW / "Dates_New.npy"
    price_path = NEW / "Price_New.npz"
    volume_path = NEW / "Volume_New.npz"

    if not refresh and dates_path.exists() and price_path.exists() and volume_path.exists():
        dates = np.load(dates_path)
        price = load_npz(price_path)
        volume = load_npz(volume_path)
        if set(price) >= set(tickers) and set(volume) >= set(tickers):
            had_extra = set(price) != set(tickers) or set(volume) != set(tickers)
            price = {ticker: price[ticker] for ticker in tickers}
            volume = {ticker: volume[ticker] for ticker in tickers}
            sizes = {len(price[t]) for t in tickers} | {len(volume[t]) for t in tickers}
            requested_end = np.datetime64(pd.Timestamp(end_date).date())
            cache_covers_request = len(dates) > 0 and requested_end <= dates[-1] + np.timedelta64(1, "D")
            if (cache_covers_request and sizes == {len(dates)}
                    and all(np.isfinite(price[t]).all() and np.isfinite(volume[t]).all() for t in tickers)):
                if had_extra:
                    atomic_savez(price_path, price)
                    atomic_savez(volume_path, volume)
                print(f"Using cached 2026 market data through {str(dates[-1])[:10]} ({len(dates)} rows)")
                return price, volume, dates

    benchmark = yf.download(
        "^GSPC", start=START_DATE, end=end_date, interval="1d",
        auto_adjust=True, progress=False, threads=False,
    )
    if benchmark.empty:
        raise RuntimeError("No 2026 market dates were returned by yfinance")
    index = benchmark.index.tz_localize(None).normalize()
    prices: dict[str, np.ndarray] = {}
    volumes: dict[str, np.ndarray] = {}

    for offset in range(0, len(tickers), 32):
        chunk = tickers[offset:offset + 32]
        data = None
        for attempt in range(3):
            data = yf.download(
                chunk, start=START_DATE, end=end_date, interval="1d",
                auto_adjust=True, progress=False, threads=True, group_by="column",
            )
            if not data.empty:
                break
            time.sleep(2 ** attempt)
        if data is None or data.empty:
            raise RuntimeError(f"Download failed for ticker chunk beginning with {chunk[0]}")

        for ticker in chunk:
            try:
                p = ohlc_average(data, ticker).reindex(index)
                v = volume_series(data, ticker).reindex(index)
            except KeyError as error:
                raise RuntimeError(f"yfinance omitted {ticker} from its response") from error
            if p.isna().any() or v.isna().any():
                missing = int((p.isna() | v.isna()).sum())
                raise RuntimeError(f"{ticker} has {missing} missing 2026 trading rows; no values were fabricated")
            prices[ticker] = p.to_numpy(dtype=np.float64)
            volumes[ticker] = v.to_numpy(dtype=np.float64)
        print(f"Downloaded {min(offset + len(chunk), len(tickers))}/{len(tickers)} tickers")

    dates = index.to_numpy(dtype="datetime64[D]")
    atomic_savez(price_path, prices)
    atomic_savez(volume_path, volumes)
    atomic_save(dates_path, dates)
    return prices, volumes, dates


def download_benchmark(end_date: str, refresh: bool) -> np.ndarray:
    cache = NEW / "SP500_2005_New.npy"
    if cache.exists() and not refresh:
        values = np.load(cache)
        if len(values) > HISTORICAL_DAYS and np.isfinite(values).all():
            return values
    data = yf.download(
        "^GSPC", start="2005-01-01", end=end_date, interval="1d",
        auto_adjust=True, progress=False, threads=False,
    )
    values = ohlc_average(data, "^GSPC").to_numpy(dtype=np.float64)
    if len(values) <= HISTORICAL_DAYS or not np.isfinite(values).all():
        raise RuntimeError("The S&P 500 benchmark does not align with the expected 2005-2026 history")
    atomic_save(cache, values)
    return values


def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2004-01-01"
    frame = pd.read_csv(url, parse_dates=["observation_date"])
    values = pd.to_numeric(frame[series_id], errors="coerce")
    return pd.Series(values.to_numpy(), index=frame["observation_date"]).dropna()


def savgol(values: np.ndarray, polyorder: int) -> np.ndarray:
    if len(values) < 15:
        raise ValueError("The original macro smoother requires at least 15 values")
    return ss.savgol_filter(values.astype(np.float64), window_length=15, polyorder=polyorder)


def value_to_rate(values: np.ndarray) -> np.ndarray:
    # Exact formula used in Data/Data Collection.
    return 100.0 * (values[1:] - values[:-1]) / values[1:]


def build_macro(end_month: pd.Timestamp, refresh: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    cache = NEW / "Macro_Coef_New.npz"
    manifest_path = NEW / "Macro_Manifest_New.json"
    required = (end_month.year - 2005) * 12 + end_month.month
    if cache.exists() and manifest_path.exists() and not refresh:
        macro = load_npz(cache)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(name in macro and len(macro[name]) >= required for name in MACRO_NAMES):
            return macro, macro_windows(macro), manifest

    workbook = ROOT / "data" / "reference" / "Data.xlsx"
    monthly_index = pd.date_range("2005-01-01", end_month, freq="MS")
    original = load_npz(NPZ / "Macro_Coef_2.npz")

    cpi = fred_series("CPIAUCSL")
    inflation_live = cpi.pct_change(12, fill_method=None) * 100.0
    fedfunds = fred_series("FEDFUNDS")
    unemployment = fred_series("UNRATE")
    m2_live = fred_series("M2SL")
    gdp_live = fred_series("GDPC1")

    inflation = pd.Series(index=monthly_index, dtype=float)
    interest = pd.Series(index=monthly_index, dtype=float)
    unemployment_monthly = pd.Series(index=monthly_index, dtype=float)
    inflation.iloc[:252] = original["Inflation_Rate"]
    # Recover the original interest rate from the project's exact Real_Rate formula.
    interest.iloc[:252] = original["Inflation_Rate"] - original["Real_Rate"]
    unemployment_monthly.iloc[:252] = original["Unemployment_Rate"]
    # Keep every original 2005-2025 feature byte-for-byte; only extend it.
    extension = monthly_index[252:]
    inflation.loc[extension] = inflation_live.reindex(extension)
    interest.loc[extension] = fedfunds.reindex(extension)
    unemployment_monthly.loc[extension] = unemployment.reindex(extension)
    inflation = inflation.ffill()
    interest = interest.ffill()
    unemployment_monthly = unemployment_monthly.ffill()

    money_raw = pd.read_excel(workbook, sheet_name="Money Supply", index_col=0)["Value"]
    money_raw.index = pd.to_datetime(money_raw.index)
    money_raw.update(m2_live)
    money_raw = money_raw.reindex(monthly_index).ffill()
    money_feature = savgol(value_to_rate(money_raw.to_numpy()), 1)
    # The rate has one fewer item; reproduce the project's inserted first value.
    money_feature = np.insert(money_feature, 0, 0.09806)

    credit_raw = pd.read_excel(workbook, sheet_name="CreditGDP", index_col=0)["Value"]
    credit_raw.index = pd.to_datetime(credit_raw.index)
    quarter_index = pd.date_range("2005-01-01", end_month, freq="QS")
    credit_raw = credit_raw.reindex(quarter_index).ffill()
    credit_quarterly = np.insert(value_to_rate(credit_raw.to_numpy()), 0, 0.455)
    credit_feature = savgol(np.repeat(credit_quarterly, 3)[:len(monthly_index)], 1)

    gdp_raw = pd.read_excel(workbook, sheet_name="Real GDP", index_col=0)["Value"]
    gdp_raw.index = pd.to_datetime(gdp_raw.index)
    gdp_raw.update(gdp_live)
    gdp_raw = gdp_raw.reindex(quarter_index).ffill()
    gdp_quarterly = np.insert(value_to_rate(gdp_raw.to_numpy()), 0, 1.02021)
    gdp_feature = savgol(np.repeat(gdp_quarterly, 3)[:len(monthly_index)], 1)

    vix_frame = yf.download(
        "^VIX", start="2005-01-01", end=(end_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=True, progress=False, threads=False,
    )
    vix_daily = ohlc_average(vix_frame, "^VIX")
    vix_monthly = vix_daily.resample("ME").mean().to_numpy(dtype=np.float64)
    if len(vix_monthly) < len(monthly_index):
        raise RuntimeError("VIX download ended before the required macro month")
    vix_feature = savgol(vix_monthly[:len(monthly_index)], 2)

    macro = {
        "Inflation_Rate": inflation.to_numpy(dtype=np.float64),
        "Real_Rate": (inflation - interest).to_numpy(dtype=np.float64),
        "Unemployment_Rate": unemployment_monthly.to_numpy(dtype=np.float64),
        "Credit_GDP": credit_feature,
        "Money_Supply": money_feature[:len(monthly_index)],
        "Real_GDP": gdp_feature,
        "VIX": vix_feature,
    }
    if not all(len(value) == len(monthly_index) and np.isfinite(value).all() for value in macro.values()):
        raise RuntimeError("Macro construction produced an incomplete or non-finite feature")

    manifest = {
        "months": [monthly_index[0].strftime("%Y-%m"), monthly_index[-1].strftime("%Y-%m")],
        "method": "Original transforms and Savitzky-Golay settings from Data/Data Collection",
        "sources": {
            "Inflation_Rate": "FRED CPIAUCSL, 12-month percent change",
            "Interest_Rate": "FRED FEDFUNDS; Real_Rate remains Inflation_Rate - Interest_Rate",
            "Unemployment_Rate": "FRED UNRATE",
            "Money_Supply": "FRED M2SL",
            "Real_GDP": "FRED GDPC1",
            "Credit_GDP": "Old/Data.xlsx; latest quarterly observation carried forward when unreleased",
            "VIX": "yfinance ^VIX monthly OHLC average",
        },
        "short_term_carry_forward": [
            "Quarterly Credit_GDP/GDP values between releases",
            "M2SL for a requested month newer than the latest published observation",
        ],
        "future_target_prediction_used": False,
    }
    atomic_savez(cache, macro)
    atomic_text(manifest_path, json.dumps(manifest, indent=2))
    windows = macro_windows(macro)
    atomic_savez(NEW / "Mo_Last_New.npz", windows)
    return macro, windows, manifest


def macro_windows(macro: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = min(len(macro[name]) for name in MACRO_NAMES)
    result: dict[str, np.ndarray] = {}
    for month in range(5, count):
        rows = []
        for index in range(month - 5, month + 1):
            rows.extend([macro[name][index] for name in MACRO_NAMES])
            rows.append(float(index))
        result[str(month)] = np.asarray(rows, dtype=np.float64)
    return result


def get_rel(data: np.ndarray, split: int) -> np.ndarray:
    # Exact Ai_Back.Get_Rel algorithm, without its extra one-item list wrapper.
    peaks: list[np.ndarray] = []
    troughs: list[np.ndarray] = []
    position = 0
    for part in np.array_split(data, split):
        prominence = part.mean() * 0.05
        peaks.append(ss.find_peaks(part, prominence=prominence, distance=5)[0] + position)
        troughs.append(ss.find_peaks(-part, prominence=prominence, distance=5)[0] + position)
        position += len(part)
    return np.sort(np.concatenate(peaks + troughs)).astype(int)


def filter_extrema(values: np.ndarray, extrema: np.ndarray) -> np.ndarray:
    # Exact Stock Data_Collect Filter/Distance_Filter behavior.
    extrema = extrema[extrema > 126]
    near = np.where(np.diff(extrema) < 5)[0]
    delete = []
    for index in near:
        item = extrema[index:index + 1].astype(int)
        delete.append(np.argmin(values[item]) + np.where(extrema == item[0])[0][0])
    return np.delete(extrema, delete).astype(int)


def build_derived(new_price: dict[str, np.ndarray], new_volume: dict[str, np.ndarray]) -> dict:
    old_price = load_npz(NPZ / "Ticker" / "Price" / "Price.npz")
    old_volume = load_npz(NPZ / "Ticker" / "Volume" / "Volume.npz")
    old_ext_org = load_npz(NPZ / "Ticker" / "Ext_Org.npz")
    combined_price: dict[str, np.ndarray] = {}
    combined_volume: dict[str, np.ndarray] = {}
    price_smooth_new: dict[str, np.ndarray] = {}
    volume_smooth_new: dict[str, np.ndarray] = {}
    price_smooth_2: dict[str, np.ndarray] = {}
    volume_smooth_2: dict[str, np.ndarray] = {}
    return_smooth: dict[str, np.ndarray] = {}
    volume_rate_smooth: dict[str, np.ndarray] = {}
    ext_org_all: dict[str, np.ndarray] = {}
    ext_all: dict[str, np.ndarray] = {}
    ext_org_new: dict[str, np.ndarray] = {}
    ext_new: dict[str, np.ndarray] = {}

    for ticker in new_price:
        price = np.concatenate([old_price[ticker + "/Price"], new_price[ticker]])
        volume = np.concatenate([old_volume[ticker + "/Volume"], new_volume[ticker]])
        if len(old_price[ticker + "/Price"]) != HISTORICAL_DAYS:
            raise RuntimeError(f"Unexpected historical length for {ticker}")
        combined_price[ticker] = price
        combined_volume[ticker] = volume
        ps = ss.savgol_filter(price, window_length=21, polyorder=3)
        vs = ss.savgol_filter(volume, window_length=21, polyorder=3)
        price_smooth_new[ticker] = ps[HISTORICAL_DAYS:]
        volume_smooth_new[ticker] = vs[HISTORICAL_DAYS:]
        ps2 = ps[:len(ps) // 2 * 2].reshape(-1, 2).mean(axis=1)
        vs2 = vs[:len(vs) // 2 * 2].reshape(-1, 2).mean(axis=1)
        price_smooth_2[ticker] = ps2
        volume_smooth_2[ticker] = vs2
        return_smooth[ticker] = 100.0 * np.diff(ps2) / ps2[:-1]
        volume_rate_smooth[ticker] = 100.0 * np.diff(vs2) / vs2[:-1]

        raw = np.union1d(get_rel(price, 42), get_rel(price, 48)).astype(int)
        filtered = filter_extrema(price, raw)
        ext_org_all[ticker] = raw
        ext_all[ticker] = filtered
        ext_org_new[ticker] = raw[raw >= HISTORICAL_DAYS] - HISTORICAL_DAYS
        ext_new[ticker] = filtered[filtered >= HISTORICAL_DAYS] - HISTORICAL_DAYS

    atomic_savez(NEW / "Price_Smoothed_New.npz", price_smooth_new)
    atomic_savez(NEW / "Volume_Smoothed_New.npz", volume_smooth_new)
    atomic_savez(NEW / "Price_Smoothed_New_2.npz", {
        key: value[HISTORICAL_DAYS // 2:] for key, value in price_smooth_2.items()
    })
    atomic_savez(NEW / "Volume_Smoothed_New_2.npz", {
        key: value[HISTORICAL_DAYS // 2:] for key, value in volume_smooth_2.items()
    })
    atomic_savez(NEW / "Ext_New.npz", ext_org_new)
    atomic_savez(NEW / "Ext_Fixed_New.npz", ext_new)
    return {
        "price": combined_price, "volume": combined_volume,
        "price_return": return_smooth, "volume_rate": volume_rate_smooth,
        "ext_org": ext_org_all, "ext": ext_all, "old_ext_org": old_ext_org,
    }


def rsi(prices: np.ndarray, end: int) -> float:
    delta = np.diff(prices[end - 13:end + 1])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = np.mean(gain[:14])
    avg_loss = np.mean(loss[:14])
    if avg_loss == 0:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))


def recent_data(
    price: np.ndarray, volume: np.ndarray, extrema: np.ndarray,
    old_ext_org_count: int, ext_index: int, duration: int, benchmark: np.ndarray,
) -> np.ndarray:
    # Mirrors Stock_Related.Recent_Data, including its duration scaling.
    end = extrema[ext_index] + duration
    ticker_values = price[extrema[ext_index]:end + 1]
    fall_values = price[extrema[ext_index - 1]:extrema[ext_index] + 1]
    volume_values = volume[extrema[ext_index]:end + 1]
    volume_fall = volume[extrema[ext_index - 1]:extrema[ext_index] + 1]
    normalizer = np.mean(volume[end - 125:end + 1])
    volume_values = volume_values / normalizer
    volume_fall = volume_fall / normalizer
    ticker_delta = np.diff(ticker_values) / price[extrema[ext_index]]
    fall_delta = np.diff(fall_values) / price[extrema[ext_index - 1]]

    sp_return = np.diff(benchmark[end - 125:end + 1]) / benchmark[end - 125:end][...]
    ticker_return = np.diff(price[end - 125:end + 1]) / price[end - 125:end]
    beta = np.cov(ticker_return, sp_return)[0, 1] / np.var(sp_return, ddof=1)
    fall_duration = extrema[ext_index] - extrema[ext_index - 1]
    rises = [part.mean() / duration for part in np.array_split(ticker_delta * 100.0, 4)]
    falls = [part.mean() / fall_duration for part in np.array_split(fall_delta * 100.0, 4)]

    smooth = ss.savgol_filter(
        volume_values,
        window_length=2 * round((len(volume_values) - 3) / 8) + 3,
        polyorder=min(round(len(volume_values) / 3) - 2, 2),
    )
    volume_peakiness = np.max(np.abs(np.diff(smooth, 2)))
    volume_peak = np.max(volume_values)
    # Preserve the original expression's effective window-length calculation.
    smooth_fall = ss.savgol_filter(
        volume_fall,
        window_length=2 * round(len(volume_fall - 3) / 8) + 3,
        polyorder=min(round(len(volume_fall) / 3) - 2, 2),
    )
    volume_fall_peakiness = np.max(np.abs(np.diff(smooth_fall, 2)))
    volume_fall_peak = np.max(volume_fall)
    peak_duration = HISTORICAL_DAYS / old_ext_org_count
    return np.asarray(
        [beta, *rises, *falls, rsi(price, end), volume_fall_peak,
         volume_fall_peakiness, volume_peakiness, volume_peak,
         duration, fall_duration, peak_duration],
        dtype=np.float64,
    )


def build_validation(derived: dict, macro: dict[str, np.ndarray], benchmark: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mean_std = load_npz(NPZ / "Mean_Std.npz")
    mean, std = mean_std["mean"], mean_std["std"]
    if mean.shape != (191,) or std.shape != (191,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Mean_Std.npz must contain finite 191-element mean and std arrays")
    if np.any(std <= 0):
        raise RuntimeError("Mean_Std.npz contains a non-positive standard deviation")
    macro_last = macro_windows(macro)
    features: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict] = []

    for ticker in derived["price"]:
        price = derived["price"][ticker]
        volume = derived["volume"][ticker]
        extrema = derived["ext"][ticker]
        return_smooth = derived["price_return"][ticker]
        volume_smooth = derived["volume_rate"][ticker]
        old_count = len(derived["old_ext_org"][ticker])
        for ext_index in range(1, len(extrema) - 1):
            ext_position = int(extrema[ext_index])
            next_position = int(extrema[ext_index + 1])
            if ext_position < HISTORICAL_DAYS or next_position >= len(price):
                continue
            gap = next_position - ext_position
            # The target extremum must never be part of the observed input.
            # This matches Stock Data_Collect, whose range excludes ``gap``.
            for duration in range(4, gap):
                end = ext_position + duration
                divided = end // 2
                return_window = return_smooth[divided - 64:divided - 1]
                volume_window = volume_smooth[divided - 64:divided - 1]
                if len(return_window) != 63 or len(volume_window) != 63:
                    continue
                # New observations have real dates, so use their calendar month.
                # The historical 21-day approximation drifts around holidays.
                extremum_date = pd.Timestamp(dates[ext_position - HISTORICAL_DAYS])
                month = (extremum_date.year - 2005) * 12 + extremum_date.month - 1
                macro_key = str(month - 1)
                if macro_key not in macro_last:
                    continue
                pattern = recent_data(
                    price, volume, extrema, old_count, ext_index, duration, benchmark,
                )
                combined = np.concatenate([pattern, return_window, volume_window, macro_last[macro_key]])
                if combined.shape != (191,) or not np.isfinite(combined).all():
                    continue
                target = 100.0 * (price[next_position] - price[ext_position]) / price[ext_position]
                features.append((combined - mean) / std)
                targets.append(float(target))
                metadata.append({
                    "ticker": ticker,
                    "extremum_date": str(dates[ext_position - HISTORICAL_DAYS])[:10],
                    "sample_end_date": str(dates[end - HISTORICAL_DAYS])[:10],
                    "next_extremum_date": str(dates[next_position - HISTORICAL_DAYS])[:10],
                    "duration": duration,
                    "target_percent": float(target),
                })

    sample = np.vstack(features) if features else np.empty((0, 191), dtype=np.float64)
    result = np.asarray(targets, dtype=np.float64)[:, None]
    frame = pd.DataFrame(metadata)
    atomic_save(NEW / "Valid_Sample_New.npy", sample)
    atomic_save(NEW / "Valid_Result_New.npy", result)
    atomic_text(NEW / "Valid_Metadata_New.csv", frame.to_csv(index=False))
    return sample, result, frame


def leaky_relu(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0, values, 0.01 * values)


def predict(sample: np.ndarray, checkpoint: int) -> np.ndarray:
    weights = load_npz(NPZ / f"Weight_{checkpoint}.npz")
    biases = load_npz(NPZ / f"Bias_{checkpoint}.npz")
    value = sample
    for layer in range(1, 6):
        value = value @ weights[f"Weight_Layer_{layer}"].T + biases[f"Bias_Layer_{layer}"]
        if layer < 5:
            value = leaky_relu(value)
    return value


def metrics(
    actual: np.ndarray, predicted: np.ndarray, group_ids: np.ndarray | None = None,
) -> dict[str, float | int]:
    actual = actual.ravel()
    predicted = predicted.ravel()
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("actual and predicted must be non-empty arrays of equal length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics received a non-finite actual or prediction")
    residual = predicted - actual
    actual_positive = actual > 0
    predicted_positive = predicted > 0
    positive_recall = float(np.mean(predicted_positive[actual_positive])) if actual_positive.any() else float("nan")
    negative_recall = float(np.mean(~predicted_positive[~actual_positive])) if (~actual_positive).any() else float("nan")
    pearson = float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 else float("nan")
    spear = float(spearmanr(actual, predicted).statistic) if len(actual) > 1 else float("nan")
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    baseline = actual - actual.mean()
    result: dict[str, float | int] = {
        "samples": int(len(actual)),
        "independent_transitions": int(len(np.unique(group_ids))) if group_ids is not None else int(len(actual)),
        "positive_target_rate": float(np.mean(actual_positive)),
        "positive_prediction_rate": float(np.mean(predicted_positive)),
        "pearson_correlation": pearson,
        "spearman_correlation": spear,
        "direction_correctness": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "balanced_direction_correctness": float(np.nanmean([positive_recall, negative_recall])),
        "positive_direction_recall": positive_recall,
        "negative_direction_recall": negative_recall,
        "mse": float(np.mean(residual ** 2)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_baseline_rmse": float(np.sqrt(np.mean(baseline ** 2))),
        "r_squared": float(1.0 - np.sum(residual ** 2) / denominator) if denominator else float("nan"),
        "actual_mean": float(np.mean(actual)),
        "prediction_mean": float(np.mean(predicted)),
    }
    if group_ids is not None:
        codes, _ = pd.factorize(group_ids, sort=False)
        counts = np.bincount(codes)
        weights = 1.0 / counts[codes]
        weights /= weights.sum()
        weighted_mean = float(np.sum(weights * actual))
        weighted_mse = float(np.sum(weights * residual ** 2))
        weighted_denominator = float(np.sum(weights * (actual - weighted_mean) ** 2))
        result.update({
            "transition_weighted_rmse": float(np.sqrt(weighted_mse)),
            "transition_weighted_mae": float(np.sum(weights * np.abs(residual))),
            "transition_weighted_direction_correctness": float(
                np.sum(weights * (np.sign(actual) == np.sign(predicted)))
            ),
            "transition_weighted_r_squared": (
                float(1.0 - weighted_mse / weighted_denominator) if weighted_denominator else float("nan")
            ),
        })
    return result


def evaluate(new_sample: np.ndarray | None = None, new_result: np.ndarray | None = None) -> pd.DataFrame:
    old_sample = np.load(TRAIN / "Valid_Sample.npy")
    old_result = np.load(TRAIN / "Valid_Result.npy")
    if new_sample is None:
        new_sample = np.load(NEW / "Valid_Sample_New.npy")
    if new_result is None:
        new_result = np.load(NEW / "Valid_Result_New.npy")
    if len(new_sample) == 0:
        raise RuntimeError("No fully observed new validation samples were produced")
    metadata_path = NEW / "Valid_Metadata_New.csv"
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    if len(metadata) != len(new_sample):
        raise RuntimeError("Valid_Metadata_New.csv does not align with Valid_Sample_New.npy")
    transition_ids = (
        metadata["ticker"].astype(str) + "|" + metadata["extremum_date"].astype(str)
        + "|" + metadata["next_extremum_date"].astype(str)
    ).to_numpy()

    rows = []
    predictions: dict[str, np.ndarray] = {}
    for checkpoint in range(1, 8):
        for dataset, sample, result in (
            ("current", old_sample, old_result), ("new_2026", new_sample, new_result),
        ):
            predicted = predict(sample, checkpoint)
            predictions[f"{dataset}_weight_{checkpoint}"] = predicted
            row = {"checkpoint": checkpoint, "dataset": dataset}
            row.update(metrics(result, predicted, transition_ids if dataset == "new_2026" else None))
            rows.append(row)
            print(
                f"weight {checkpoint} {dataset}: correlation={row['pearson_correlation']:.4f}, "
                f"direction={row['direction_correctness']:.4f}, RMSE={row['rmse']:.4f}"
            )
    frame = pd.DataFrame(rows)
    atomic_text(NEW / "Checkpoint_Comparison.csv", frame.to_csv(index=False))
    atomic_text(NEW / "Checkpoint_Comparison.json", json.dumps(rows, indent=2, allow_nan=True))
    atomic_savez(NEW / "Checkpoint_Predictions.npz", predictions)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("all", "collect", "evaluate"), default="all")
    parser.add_argument("--refresh", action="store_true", help="redownload even when a complete cache exists")
    parser.add_argument("--end", help="exclusive yfinance end date; default is tomorrow")
    args = parser.parse_args()
    NEW.mkdir(parents=True, exist_ok=True)
    end_date = args.end or (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if args.action == "evaluate":
        evaluate()
        return

    price, volume, dates = download_market(end_date, args.refresh)
    benchmark = download_benchmark(end_date, args.refresh)
    if len(benchmark) != HISTORICAL_DAYS + len(dates):
        raise RuntimeError(
            f"Benchmark has {len(benchmark)} rows but expected {HISTORICAL_DAYS + len(dates)}; "
            "the daily arrays are not aligned"
        )
    derived = build_derived(price, volume)
    # Only the month before an extremum is consumed, so the latest completed market month is enough.
    latest = pd.Timestamp(dates[-1].astype("datetime64[ns]"))
    macro_end = (latest.to_period("M") - 1).to_timestamp()
    macro, _, manifest = build_macro(macro_end, args.refresh)
    sample, result, metadata = build_validation(derived, macro, benchmark, dates)
    summary = {
        "market_dates": [str(dates[0])[:10], str(dates[-1])[:10]],
        "market_rows": int(len(dates)),
        "validation_samples": int(len(sample)),
        "validation_tickers": int(metadata["ticker"].nunique()) if len(metadata) else 0,
        "target_range": [float(result.min()), float(result.max())] if len(result) else [],
        "macro": manifest,
    }
    atomic_text(NEW / "Validation_Summary_New.json", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.action == "all":
        evaluate(sample, result)


if __name__ == "__main__":
    main()
