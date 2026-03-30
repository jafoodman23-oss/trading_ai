"""
Abstract broker interface.
All broker implementations (paper, Alpaca paper, Alpaca live) share this contract.
Switching from paper to live is just a config change — no code changes needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: str          # "buy" or "sell"
    qty: float
    type: str          # "market", "limit", "stop", "stop_limit"
    status: str        # "pending", "filled", "partial", "cancelled", "rejected"
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    client_order_id: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    @property
    def is_open(self) -> bool:
        return self.status in ("pending", "partial", "accepted", "new")

    @property
    def value(self) -> float:
        return self.filled_qty * self.filled_avg_price


@dataclass
class Position:
    """Represents an open position in a symbol."""
    symbol: str
    qty: float                  # positive = long, negative = short
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    market_value: float

    @property
    def side(self) -> str:
        return "long" if self.qty > 0 else "short"

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class Account:
    """Represents broker account state."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    initial_capital: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def total_return(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.portfolio_value / self.initial_capital) - 1.0

    @property
    def total_return_pct(self) -> float:
        return self.total_return * 100.0


class Broker(ABC):
    """
    Abstract base class for all broker implementations.

    Implementations:
        PaperBroker   — internal simulation, no API needed
        AlpacaBroker  — Alpaca paper or live via alpaca-py

    The mode (paper vs live) is determined by the settings object,
    so client code never needs to change when switching modes.
    """

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """Place an order. Returns an Order object."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successful."""
        ...

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns number of orders cancelled."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol. Returns None if no position."""
        ...

    @abstractmethod
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        ...

    @abstractmethod
    def get_account(self) -> Account:
        """Get current account state."""
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get a specific order by ID."""
        ...

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Get all open (pending) orders."""
        ...

    @abstractmethod
    def is_market_open(self) -> bool:
        """Return True if the market is currently open for trading."""
        ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Get the most recent price for a symbol."""
        ...

    def get_latest_prices(self, symbols: List[str]) -> dict:
        """Get latest prices for multiple symbols. Default: calls get_latest_price in a loop."""
        return {sym: self.get_latest_price(sym) for sym in symbols}
