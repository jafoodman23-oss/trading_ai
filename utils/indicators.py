"""
Pure numpy/pandas implementations of technical indicators.
Used as fallback if pandas-ta is unavailable, and also as standalone utilities.
All functions accept pandas Series or numpy arrays and return pandas Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing method.
    Returns values in [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD indicator.
    Returns (macd_line, signal_line, histogram).
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns (upper_band, middle_band, lower_band).
    """
    middle = sma(series, period)
    rolling_std = series.rolling(window=period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range.
    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, min_periods=period).mean()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator %K and %D.
    Returns (stoch_k, stoch_d) both in [0, 100].
    """
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100.0 * (close - lowest_low) / denom
    stoch_d = sma(stoch_k, d)
    return stoch_k, stoch_d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume.
    Cumulative volume adds when price closes up, subtracts when down.
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Commodity Channel Index.
    CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Absolute Deviation)
    """
    typical_price = (high + low + close) / 3.0
    tp_sma = sma(typical_price, period)

    def mean_abs_dev(x: pd.Series) -> float:
        return (x - x.mean()).abs().mean()

    mad = typical_price.rolling(window=period).apply(mean_abs_dev, raw=False)
    return (typical_price - tp_sma) / (0.015 * mad.replace(0, np.nan))


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Williams %R oscillator. Values in [-100, 0].
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return -100.0 * (highest_high - close) / denom


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average Directional Index (trend strength, 0-100).
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional movements
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_series = pd.Series(plus_dm, index=high.index)
    minus_dm_series = pd.Series(minus_dm, index=high.index)

    # Smoothed using Wilder's method
    atr_series = true_range.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100.0 * plus_dm_series.ewm(com=period - 1, min_periods=period).mean() / atr_series.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_series.ewm(com=period - 1, min_periods=period).mean() / atr_series.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx_series


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Normalize a series using rolling z-score."""
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std = series.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    return (series - roll_mean) / roll_std
