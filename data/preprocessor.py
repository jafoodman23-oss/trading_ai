"""
Feature engineering pipeline.
Computes technical indicators from OHLCV data and normalizes them
using rolling z-scores. Returns numpy arrays ready for the RL environment.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Try ta library; fall back to our own implementations
try:
    import ta as _ta_lib
    TA_LIB = True
    PANDAS_TA = False
except ImportError:
    TA_LIB = False
    PANDAS_TA = False
    logger.warning("ta library not available — using built-in indicator implementations")

from utils.indicators import (
    ema, sma, rsi as _rsi, macd as _macd, bollinger_bands as _bb,
    atr as _atr, stochastic as _stoch, obv as _obv, adx as _adx,
    cci as _cci, williams_r as _wr, rolling_zscore,
)

ZSCORE_WINDOW = 252  # rolling normalization window


class FeatureEngineer:
    """
    Transforms raw OHLCV DataFrames into normalized feature matrices
    ready for the trading environment and RL model.

    Features computed (27 total):
        rsi_7, rsi_14, rsi_21
        macd_line, macd_signal, macd_hist
        bb_upper_pct, bb_lower_pct, bb_bandwidth   (% distance from price)
        ema_9_dist, ema_21_dist, ema_50_dist, ema_200_dist
        atr_14_norm
        stoch_k, stoch_d
        cci_20
        williams_r
        adx_14
        obv_zscore
        volume_ratio
        return_1, return_5, return_10, return_20
        hl_range_norm
        gap
    """

    def __init__(self, feature_list: Optional[List[str]] = None):
        self.feature_list = feature_list  # if None, compute all and return all
        self._n_features: Optional[int] = None

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("Call compute_features at least once to determine n_features")
        return self._n_features

    def compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute all features for a single-symbol OHLCV DataFrame.

        Parameters
        ----------
        df : DataFrame with columns open, high, low, close, volume (and optionally vwap).
             Index should be a DatetimeIndex.

        Returns
        -------
        np.ndarray of shape (n_valid_bars, n_features), dtype float32.
        """
        if df.empty:
            raise ValueError("Empty DataFrame passed to FeatureEngineer")

        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        open_ = df["open"]

        feat = pd.DataFrame(index=df.index)

        # ---- RSI --------------------------------------------------------
        feat["rsi_7"] = rolling_zscore(_compute_rsi(close, 7), ZSCORE_WINDOW)
        feat["rsi_14"] = rolling_zscore(_compute_rsi(close, 14), ZSCORE_WINDOW)
        feat["rsi_21"] = rolling_zscore(_compute_rsi(close, 21), ZSCORE_WINDOW)

        # ---- MACD -------------------------------------------------------
        macd_line, macd_sig, macd_hist = _compute_macd(close, 12, 26, 9)
        feat["macd_line"] = rolling_zscore(macd_line, ZSCORE_WINDOW)
        feat["macd_signal"] = rolling_zscore(macd_sig, ZSCORE_WINDOW)
        feat["macd_hist"] = rolling_zscore(macd_hist, ZSCORE_WINDOW)

        # ---- Bollinger Bands --------------------------------------------
        bb_upper, bb_mid, bb_lower = _compute_bb(close, 20, 2.0)
        price_range = (bb_upper - bb_lower).replace(0, np.nan)
        feat["bb_upper_pct"] = (bb_upper - close) / close
        feat["bb_lower_pct"] = (close - bb_lower) / close
        feat["bb_bandwidth"] = price_range / bb_mid.replace(0, np.nan)

        # ---- EMA distances (% from close) --------------------------------
        for period, col in [(9, "ema_9_dist"), (21, "ema_21_dist"), (50, "ema_50_dist"), (200, "ema_200_dist")]:
            ema_val = ema(close, period)
            feat[col] = (close - ema_val) / close

        # ---- ATR (normalized by close) ----------------------------------
        atr_val = _compute_atr(high, low, close, 14)
        feat["atr_14_norm"] = atr_val / close

        # ---- Stochastic --------------------------------------------------
        stoch_k, stoch_d = _compute_stoch(high, low, close, 14, 3)
        feat["stoch_k"] = rolling_zscore(stoch_k, ZSCORE_WINDOW)
        feat["stoch_d"] = rolling_zscore(stoch_d, ZSCORE_WINDOW)

        # ---- CCI --------------------------------------------------------
        feat["cci_20"] = rolling_zscore(_compute_cci(high, low, close, 20), ZSCORE_WINDOW)

        # ---- Williams %R ------------------------------------------------
        feat["williams_r"] = rolling_zscore(_compute_wr(high, low, close, 14), ZSCORE_WINDOW)

        # ---- ADX --------------------------------------------------------
        feat["adx_14"] = rolling_zscore(_compute_adx(high, low, close, 14), ZSCORE_WINDOW)

        # ---- OBV z-score ------------------------------------------------
        obv_raw = _compute_obv(close, volume)
        feat["obv_zscore"] = rolling_zscore(obv_raw, ZSCORE_WINDOW)

        # ---- Volume ratio (to 20-bar SMA) --------------------------------
        vol_sma = sma(volume, 20)
        feat["volume_ratio"] = volume / vol_sma.replace(0, np.nan)
        feat["volume_ratio"] = rolling_zscore(feat["volume_ratio"], ZSCORE_WINDOW)

        # ---- Price returns ----------------------------------------------
        for n, col in [(1, "return_1"), (5, "return_5"), (10, "return_10"), (20, "return_20")]:
            feat[col] = close.pct_change(n)

        # ---- High-low range normalized ----------------------------------
        feat["hl_range_norm"] = (high - low) / close

        # ---- Gap from previous close ------------------------------------
        feat["gap"] = (open_ - close.shift(1)) / close.shift(1)

        # ---- Apply feature selection if specified -----------------------
        if self.feature_list:
            missing = [f for f in self.feature_list if f not in feat.columns]
            if missing:
                logger.warning(f"Requested features not available: {missing}")
            feat = feat[[f for f in self.feature_list if f in feat.columns]]

        # ---- Handle NaN -------------------------------------------------
        feat = feat.ffill()
        feat = feat.dropna()

        # Clip extreme values to [-10, 10] to avoid exploding gradients
        feat = feat.clip(-10, 10)

        # Align close prices to the same rows that survived dropna
        aligned_close = close.reindex(feat.index)

        arr = feat.values.astype(np.float32)
        prices_arr = aligned_close.values.astype(np.float32)
        self._n_features = arr.shape[1]
        return arr, prices_arr

    def compute_features_multi(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Compute features for multiple symbols."""
        result: Dict[str, np.ndarray] = {}
        for symbol, df in data.items():
            try:
                result[symbol] = self.compute_features(df)
                logger.debug(f"Features for {symbol}: {result[symbol].shape}")
            except Exception as e:
                logger.error(f"Feature computation failed for {symbol}: {e}")
        return result

    def align_arrays(
        self, arrays: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Trim arrays to the same length (minimum length across symbols)."""
        if not arrays:
            return arrays
        min_len = min(len(v) for v in arrays.values())
        return {k: v[-min_len:] for k, v in arrays.items()}


# ------------------------------------------------------------------ #
#  Helper dispatch functions: prefer pandas-ta, fall back to custom   #
# ------------------------------------------------------------------ #

def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    if PANDAS_TA:
        result = ta.rsi(close, length=period)
        return result if result is not None else _rsi(close, period)
    return _rsi(close, period)


def _compute_macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if PANDAS_TA:
        result = ta.macd(close, fast=fast, slow=slow, signal=signal)
        if result is not None and not result.empty:
            cols = result.columns.tolist()
            return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]
    return _macd(close, fast, slow, signal)


def _compute_bb(
    close: pd.Series, period: int, std: float
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if PANDAS_TA:
        result = ta.bbands(close, length=period, std=std)
        if result is not None and not result.empty:
            cols = result.columns.tolist()
            upper = result.iloc[:, 0]
            mid = result.iloc[:, 1]
            lower = result.iloc[:, 2]
            return upper, mid, lower
    return _bb(close, period, std)


def _compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    if PANDAS_TA:
        result = ta.atr(high, low, close, length=period)
        return result if result is not None else _atr(high, low, close, period)
    return _atr(high, low, close, period)


def _compute_stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int, d: int
) -> Tuple[pd.Series, pd.Series]:
    if PANDAS_TA:
        result = ta.stoch(high, low, close, k=k, d=d)
        if result is not None and not result.empty:
            return result.iloc[:, 0], result.iloc[:, 1]
    return _stoch(high, low, close, k, d)


def _compute_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    if PANDAS_TA:
        result = ta.cci(high, low, close, length=period)
        return result if result is not None else _cci(high, low, close, period)
    return _cci(high, low, close, period)


def _compute_wr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    if PANDAS_TA:
        result = ta.willr(high, low, close, length=period)
        return result if result is not None else _wr(high, low, close, period)
    return _wr(high, low, close, period)


def _compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    if PANDAS_TA:
        result = ta.adx(high, low, close, length=period)
        if result is not None and not result.empty:
            return result.iloc[:, 0]
    return _adx(high, low, close, period)


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if PANDAS_TA:
        result = ta.obv(close, volume)
        return result if result is not None else _obv(close, volume)
    return _obv(close, volume)
