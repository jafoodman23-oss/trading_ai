"""
Logging configuration using loguru.
Provides a main application logger and a separate structured trade logger.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger as _logger


def setup_logger(log_dir: str, level: str = "INFO"):
    """
    Configure the main application logger.
    Logs to stderr with color formatting and to a daily rotating log file.
    Returns the configured logger.
    """
    _logger.remove()

    # Stderr handler with color
    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler with rotation
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    _logger.add(
        str(log_path / "trading_{time:YYYY-MM-DD}.log"),
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        ),
        enqueue=True,  # Thread-safe async logging
    )

    return _logger


class TradeLogger:
    """
    Structured JSON logger for trade events.
    Writes one JSON object per line (JSONL format) for easy parsing.
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date: Optional[str] = None
        self._file = None
        self._ensure_file()

    def _ensure_file(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._file:
                self._file.close()
            self._current_date = today
            filepath = self.log_dir / f"trades_{today}.jsonl"
            self._file = open(filepath, "a", encoding="utf-8")

    def log_trade(self, event_type: str, **kwargs):
        """Log a trade event as a JSON line."""
        self._ensure_file()
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            **kwargs,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_order(self, order):
        self.log_trade(
            "order",
            order_id=str(order.id),
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            order_type=order.type,
            status=order.status,
        )

    def log_fill(self, order, fill_price: float, pnl: Optional[float] = None):
        self.log_trade(
            "fill",
            order_id=str(order.id),
            symbol=order.symbol,
            side=order.side,
            qty=order.filled_qty,
            fill_price=fill_price,
            pnl=pnl,
        )

    def log_risk_block(self, symbol: str, reason: str):
        self.log_trade("risk_block", symbol=symbol, reason=reason)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


# Module-level instances (initialized by setup functions)
trade_logger: Optional[TradeLogger] = None


def setup_trade_logger(log_dir: str) -> TradeLogger:
    """Initialize the module-level trade logger and return it."""
    global trade_logger
    trade_logger = TradeLogger(log_dir)
    return trade_logger


def get_logger():
    """Return the module-level loguru logger."""
    return _logger
