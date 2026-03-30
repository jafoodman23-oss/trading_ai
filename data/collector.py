"""
Real-time market data collection via Alpaca WebSocket streams.
Normalizes incoming bars and dispatches them to registered callbacks.
Includes automatic reconnection with exponential backoff.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable, Dict, List, Optional

from loguru import logger

try:
    from alpaca.data.live import StockDataStream
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — real-time data collection disabled")


class BarData:
    """Normalized bar from the WebSocket stream."""
    __slots__ = ("symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap")

    def __init__(self, symbol: str, timestamp: datetime, open: float, high: float,
                 low: float, close: float, volume: float, vwap: float):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap = vwap

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
        }


BarCallback = Callable[[BarData], None]


class DataCollector:
    """
    Subscribes to Alpaca minute bars for a list of symbols via WebSocket.
    Calls registered callbacks whenever a new bar arrives.

    Usage:
        def my_callback(bar: BarData):
            print(bar.symbol, bar.close)

        collector = DataCollector(settings, callbacks=[my_callback])
        await collector.start()
        # ... runs forever, or call await collector.stop() to shut down
    """

    MAX_RECONNECT_ATTEMPTS = 10
    BASE_RECONNECT_DELAY = 1.0   # seconds
    MAX_RECONNECT_DELAY = 60.0   # seconds

    def __init__(self, settings, callbacks: Optional[List[BarCallback]] = None):
        self.settings = settings
        self.symbols = settings.symbols
        self.callbacks: List[BarCallback] = callbacks or []

        self._stream: Optional[StockDataStream] = None
        self._running = False
        self._stop_event = asyncio.Event()
        self._reconnect_count = 0

        # Per-symbol statistics for monitoring
        self._bar_counts: Dict[str, int] = {s: 0 for s in self.symbols}
        self._last_bar_time: Dict[str, Optional[datetime]] = {s: None for s in self.symbols}

    def add_callback(self, cb: BarCallback):
        """Register an additional callback."""
        self.callbacks.append(cb)

    def remove_callback(self, cb: BarCallback):
        """Unregister a callback."""
        self.callbacks.remove(cb)

    async def start(self):
        """Start the WebSocket stream. Runs until stop() is called."""
        if not ALPACA_AVAILABLE:
            logger.error("alpaca-py not installed — cannot start data collection")
            return

        self._running = True
        self._stop_event.clear()
        logger.info(f"Starting DataCollector for symbols: {self.symbols}")
        await self._reconnect_loop()

    async def stop(self):
        """Gracefully stop the collector."""
        logger.info("Stopping DataCollector...")
        self._running = False
        self._stop_event.set()
        if self._stream is not None:
            try:
                await self._stream.stop_ws()
            except Exception as e:
                logger.warning(f"Error stopping WebSocket: {e}")
            self._stream = None

    async def _on_bar(self, bar):
        """
        Handler called by the Alpaca SDK for each incoming bar.
        Normalizes the bar data and dispatches to all callbacks.
        """
        try:
            # Alpaca bar objects have: symbol, timestamp, open, high, low, close, volume, vwap
            symbol = bar.symbol
            ts = bar.timestamp
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()
            if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)  # store as UTC-naive

            bar_data = BarData(
                symbol=symbol,
                timestamp=ts,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                vwap=float(bar.vwap) if hasattr(bar, 'vwap') and bar.vwap else 0.0,
            )

            # Update stats
            self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1
            self._last_bar_time[symbol] = ts

            logger.debug(f"Bar: {symbol} @ {ts} close={bar_data.close:.4f}")

            # Dispatch to all callbacks (sync or async)
            for cb in self.callbacks:
                try:
                    result = cb(bar_data)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Callback error for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error processing bar: {e}", exc_info=True)

    async def _reconnect_loop(self):
        """
        Main loop that creates and maintains the WebSocket connection.
        Uses exponential backoff on failures.
        """
        delay = self.BASE_RECONNECT_DELAY

        while self._running and not self._stop_event.is_set():
            try:
                await self._connect_and_run()
                # If we get here without exception, stream ended cleanly
                if not self._running:
                    break
                logger.warning("Stream ended unexpectedly, reconnecting...")
            except asyncio.CancelledError:
                logger.info("DataCollector cancelled")
                break
            except Exception as e:
                self._reconnect_count += 1
                if self._reconnect_count > self.MAX_RECONNECT_ATTEMPTS:
                    logger.error(f"Max reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) reached. Giving up.")
                    break

                logger.warning(
                    f"WebSocket error (attempt {self._reconnect_count}/{self.MAX_RECONNECT_ATTEMPTS}): {e}. "
                    f"Reconnecting in {delay:.1f}s..."
                )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=delay,
                    )
                    break  # stop event was set
                except asyncio.TimeoutError:
                    pass  # normal timeout, continue reconnect

                # Exponential backoff
                delay = min(delay * 2, self.MAX_RECONNECT_DELAY)

        logger.info("DataCollector stopped")

    async def _connect_and_run(self):
        """Create a fresh WebSocket connection and subscribe to bars."""
        api_key = self.settings.alpaca.get_api_key(self.settings.mode)
        api_secret = self.settings.alpaca.get_api_secret(self.settings.mode)

        self._stream = StockDataStream(
            api_key=api_key,
            secret_key=api_secret,
        )

        # Subscribe to minute bars for all symbols
        self._stream.subscribe_bars(self._on_bar, *self.symbols)

        # Reset reconnect counter on successful connection
        self._reconnect_count = 0
        logger.info(f"WebSocket connected, subscribed to {len(self.symbols)} symbols")

        # Run the stream (blocks until disconnected)
        await self._stream._run_forever()

    def get_stats(self) -> Dict:
        """Return connection and throughput statistics."""
        return {
            "symbols": self.symbols,
            "bar_counts": dict(self._bar_counts),
            "last_bar_times": {k: str(v) for k, v in self._last_bar_time.items()},
            "reconnect_count": self._reconnect_count,
            "is_running": self._running,
        }
