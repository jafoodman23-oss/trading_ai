"""
Historical market data fetcher using Alpaca's StockHistoricalDataClient.
Caches results to SQLite and only fetches missing date ranges.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — historical data fetching disabled")

from .storage import DataStorage


def _parse_timeframe(tf_str: str):
    """Convert a timeframe string like '1Min' or '1Day' to an Alpaca TimeFrame."""
    if not ALPACA_AVAILABLE:
        return None
    mapping = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    tf = mapping.get(tf_str)
    if tf is None:
        raise ValueError(f"Unknown timeframe '{tf_str}'. Valid: {list(mapping.keys())}")
    return tf


class HistoricalDataFetcher:
    """
    Fetches historical OHLCV bars from Alpaca with SQLite caching.

    Usage:
        fetcher = HistoricalDataFetcher(settings, storage)
        df = fetcher.fetch_bars("AAPL", start=datetime(2023,1,1), end=datetime(2024,1,1))
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds

    def __init__(self, settings, storage: DataStorage):
        self.settings = settings
        self.storage = storage
        self.timeframe_str = settings.data.timeframe

        if ALPACA_AVAILABLE:
            api_key = settings.alpaca.get_api_key(settings.mode)
            api_secret = settings.alpaca.get_api_secret(settings.mode)
            self.client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=api_secret,
            )
        else:
            self.client = None

    def fetch_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a single symbol.
        Checks SQLite cache first and only fetches uncached date ranges.
        Returns DataFrame with columns: open, high, low, close, volume, vwap.
        """
        tf_str = timeframe or self.timeframe_str

        if use_cache:
            cached = self.storage.get_bars(symbol, start=start, end=end)
            if not cached.empty:
                # Check if cached data covers the full range
                cached_start = cached.index.min().to_pydatetime()
                cached_end = cached.index.max().to_pydatetime()

                if cached_start <= start + timedelta(days=1) and cached_end >= end - timedelta(days=1):
                    logger.debug(f"Cache hit for {symbol} ({len(cached)} bars)")
                    return cached

                # Partial cache — only fetch missing tail
                fetch_start = cached_end + timedelta(seconds=1)
                logger.debug(f"Partial cache for {symbol}, fetching from {fetch_start}")
            else:
                fetch_start = start
        else:
            fetch_start = start

        if fetch_start >= end:
            return self.storage.get_bars(symbol, start=start, end=end)

        # Fetch from Alpaca
        new_data = self._fetch_from_alpaca(symbol, fetch_start, end, tf_str)

        if not new_data.empty:
            inserted = self.storage.insert_bars(new_data.reset_index(), symbol)
            logger.info(f"Stored {inserted} new bars for {symbol}")

        return self.storage.get_bars(symbol, start=start, end=end)

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bars for multiple symbols.
        Returns dict mapping symbol -> DataFrame.
        """
        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = self.fetch_bars(symbol, start=start, end=end, timeframe=timeframe)
                if not df.empty:
                    results[symbol] = df
                    logger.info(f"Fetched {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        return results

    def _fetch_from_alpaca(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        tf_str: str,
    ) -> pd.DataFrame:
        """Internal method: fetch bars directly from Alpaca API with retry logic."""
        if self.client is None:
            logger.warning("Alpaca client not initialized — returning empty DataFrame")
            return pd.DataFrame()

        tf = _parse_timeframe(tf_str)

        for attempt in range(self.MAX_RETRIES):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    end=end,
                    adjustment="all",  # corporate action adjusted
                )
                bars = self.client.get_stock_bars(request)
                df = bars.df

                if df.empty:
                    return pd.DataFrame()

                # Normalize columns
                df = df.reset_index()

                # Handle multi-index (symbol, timestamp) returned for single symbol
                if "symbol" in df.columns:
                    df = df[df["symbol"] == symbol].drop(columns=["symbol"], errors="ignore")

                # Rename to canonical column names
                rename_map = {
                    "t": "timestamp", "o": "open", "h": "high",
                    "l": "low", "c": "close", "v": "volume", "vw": "vwap",
                }
                df = df.rename(columns=rename_map)

                # Ensure required columns exist
                required = ["open", "high", "low", "close", "volume"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns from Alpaca response: {missing}")

                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                    df = df.set_index("timestamp")
                elif df.index.name in ("timestamp", "t"):
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df.index.name = "timestamp"

                if "vwap" not in df.columns:
                    df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

                return df[["open", "high", "low", "close", "volume", "vwap"]]

            except Exception as e:
                logger.warning(f"Alpaca fetch attempt {attempt+1}/{self.MAX_RETRIES} failed for {symbol}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (2 ** attempt))

        logger.error(f"All retries exhausted for {symbol}")
        return pd.DataFrame()

    def get_latest_bars(self, symbol: str, n: int = 300) -> pd.DataFrame:
        """Return the most recent N bars from cache."""
        end = datetime.utcnow()
        start = end - timedelta(days=max(n // 390 + 5, 7))  # rough estimate
        df = self.fetch_bars(symbol, start=start, end=end)
        return df.tail(n)
