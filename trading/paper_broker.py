"""
Internal paper trading broker.
Simulates market fills with realistic slippage and commission.
No external API required — operates entirely on fed price data.
"""
from __future__ import annotations

import uuid
from datetime import datetime, time
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from .broker import Account, Broker, Order, Position


class PaperBroker(Broker):
    """
    Simulated paper trading broker for backtesting and paper trading
    without an Alpaca account.

    Fills are simulated at current_price ± slippage.
    Commission is deducted on each trade.

    Usage:
        broker = PaperBroker(initial_capital=100_000)
        broker.update_prices({"SPY": 450.0, "AAPL": 185.0})
        order = broker.place_order("SPY", "buy", 10)
        account = broker.get_account()
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.0001,  # 1 bps
        slippage_rate: float = 0.0001,    # 1 bps
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # Internal state
        self._cash: float = initial_capital
        self._positions: Dict[str, dict] = {}  # symbol -> {qty, avg_entry}
        self._orders: Dict[str, Order] = {}
        self._prices: Dict[str, float] = {}
        self._realized_pnl: float = 0.0
        self._equity_history: List[float] = [initial_capital]
        self._trade_history: List[dict] = []

        # Market hours simulation (NYSE: 9:30-16:00 ET weekdays)
        self._force_market_open: bool = True  # set False for realistic sim

    def update_prices(self, prices: Dict[str, float]):
        """
        Update current market prices. Must be called before place_order/get_account.

        Parameters
        ----------
        prices : dict mapping symbol -> latest price
        """
        self._prices.update(prices)
        # Update position market values
        current_equity = self._calculate_equity()
        self._equity_history.append(current_equity)
        # Keep only last 10k entries
        if len(self._equity_history) > 10_000:
            self._equity_history = self._equity_history[-10_000:]

    def update_price(self, symbol: str, price: float):
        """Update price for a single symbol."""
        self._prices[symbol] = price

    # ------------------------------------------------------------------ #
    #  Broker interface                                                    #
    # ------------------------------------------------------------------ #

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
        """
        Place and immediately simulate a market fill.
        Limit orders are accepted but filled at the limit price if favorable.
        """
        order_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side.lower(),
            qty=qty,
            type=order_type,
            status="pending",
            limit_price=limit_price,
            stop_price=stop_price,
            timestamp=timestamp,
            client_order_id=client_order_id,
        )
        self._orders[order_id] = order

        # Simulate fill
        fill_price = self._simulate_fill(symbol, side, order_type, limit_price)
        if fill_price is None:
            order.status = "pending"
            logger.warning(f"Order {order_id} pending — no price available for {symbol}")
            return order

        self._execute_fill(order, fill_price)
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.is_open:
            order.status = "cancelled"
            return True
        return False

    def cancel_all_orders(self) -> int:
        count = 0
        for order in self._orders.values():
            if order.is_open:
                order.status = "cancelled"
                count += 1
        return count

    def get_position(self, symbol: str) -> Optional[Position]:
        pos_data = self._positions.get(symbol)
        if not pos_data or pos_data["qty"] == 0:
            return None

        current_price = self._prices.get(symbol, pos_data["avg_entry"])
        qty = pos_data["qty"]
        avg_entry = pos_data["avg_entry"]
        unrealized = (current_price - avg_entry) * qty
        market_value = qty * current_price

        return Position(
            symbol=symbol,
            qty=qty,
            avg_entry_price=avg_entry,
            current_price=current_price,
            unrealized_pnl=unrealized,
            market_value=market_value,
        )

    def get_all_positions(self) -> List[Position]:
        positions = []
        for symbol in self._positions:
            pos = self.get_position(symbol)
            if pos:
                positions.append(pos)
        return positions

    def get_account(self) -> Account:
        equity = self._calculate_equity()
        unrealized = sum(
            (self._prices.get(sym, data["avg_entry"]) - data["avg_entry"]) * data["qty"]
            for sym, data in self._positions.items()
            if data["qty"] > 0
        )
        portfolio_value = self._cash + sum(
            self._prices.get(sym, data["avg_entry"]) * data["qty"]
            for sym, data in self._positions.items()
            if data["qty"] > 0
        )

        return Account(
            equity=equity,
            cash=self._cash,
            buying_power=self._cash,  # simplified: no margin
            portfolio_value=portfolio_value,
            initial_capital=self.initial_capital,
            unrealized_pnl=unrealized,
            realized_pnl=self._realized_pnl,
        )

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.is_open]

    def is_market_open(self) -> bool:
        if self._force_market_open:
            return True
        now = datetime.utcnow()
        # Simple check: NYSE is open Mon-Fri 14:30-21:00 UTC
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        market_open = time(14, 30)
        market_close = time(21, 0)
        current_time = now.time()
        return market_open <= current_time <= market_close

    def get_latest_price(self, symbol: str) -> float:
        price = self._prices.get(symbol)
        if price is None:
            raise ValueError(f"No price available for {symbol}")
        return price

    def get_latest_prices(self, symbols: List[str]) -> dict:
        return {sym: self._prices.get(sym, 0.0) for sym in symbols}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _simulate_fill(
        self,
        symbol: str,
        side: str,
        order_type: str,
        limit_price: Optional[float],
    ) -> Optional[float]:
        """Calculate the simulated fill price."""
        current_price = self._prices.get(symbol)
        if current_price is None:
            return None

        if order_type == "market":
            # Market order: fill at current price ± slippage
            if side == "buy":
                return current_price * (1 + self.slippage_rate)
            else:
                return current_price * (1 - self.slippage_rate)

        elif order_type == "limit":
            if limit_price is None:
                return None
            # Fill only if price is favorable
            if side == "buy" and current_price <= limit_price:
                return min(current_price, limit_price)
            elif side == "sell" and current_price >= limit_price:
                return max(current_price, limit_price)
            return None  # Not filled yet

        elif order_type == "stop":
            if stop_price := limit_price:
                if side == "buy" and current_price >= stop_price:
                    return current_price * (1 + self.slippage_rate)
                elif side == "sell" and current_price <= stop_price:
                    return current_price * (1 - self.slippage_rate)
            return None

        return current_price

    def _execute_fill(self, order: Order, fill_price: float):
        """Apply a simulated fill to internal state."""
        symbol = order.symbol
        side = order.side
        qty = order.qty

        commission = fill_price * qty * self.commission_rate
        total_cost = fill_price * qty + commission

        if side == "buy":
            if total_cost > self._cash:
                # Adjust qty to what we can afford
                affordable_qty = (self._cash - commission) / fill_price
                if affordable_qty < 0.01:
                    order.status = "rejected"
                    logger.warning(f"Order rejected: insufficient funds for {symbol}")
                    return
                qty = affordable_qty
                total_cost = fill_price * qty + fill_price * qty * self.commission_rate

            self._cash -= total_cost
            self._update_position_buy(symbol, qty, fill_price)

        elif side == "sell":
            pos_qty = self._positions.get(symbol, {}).get("qty", 0.0)
            if pos_qty <= 0:
                order.status = "rejected"
                logger.warning(f"Order rejected: no position in {symbol} to sell")
                return

            sell_qty = min(qty, pos_qty)
            proceeds = fill_price * sell_qty
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission

            avg_entry = self._positions[symbol]["avg_entry"]
            pnl = (fill_price - avg_entry) * sell_qty - commission
            self._realized_pnl += pnl
            self._cash += net_proceeds

            self._update_position_sell(symbol, sell_qty)
            qty = sell_qty

            # Record trade
            self._trade_history.append({
                "symbol": symbol,
                "side": side,
                "qty": sell_qty,
                "price": fill_price,
                "commission": commission,
                "pnl": pnl,
                "timestamp": order.timestamp,
            })
        else:
            order.status = "rejected"
            logger.warning(f"Unknown order side: {side}")
            return

        # Update order
        order.filled_qty = qty
        order.filled_avg_price = fill_price
        order.status = "filled"

        logger.debug(
            f"Fill: {side.upper()} {qty:.4f} {symbol} @ ${fill_price:.4f} "
            f"(commission: ${commission:.2f})"
        )

    def _update_position_buy(self, symbol: str, qty: float, price: float):
        """Update position tracking after a buy fill."""
        if symbol in self._positions and self._positions[symbol]["qty"] > 0:
            pos = self._positions[symbol]
            old_qty = pos["qty"]
            old_avg = pos["avg_entry"]
            new_qty = old_qty + qty
            pos["avg_entry"] = (old_qty * old_avg + qty * price) / new_qty
            pos["qty"] = new_qty
        else:
            self._positions[symbol] = {"qty": qty, "avg_entry": price}

    def _update_position_sell(self, symbol: str, qty: float):
        """Update position tracking after a sell fill."""
        if symbol in self._positions:
            self._positions[symbol]["qty"] -= qty
            if self._positions[symbol]["qty"] <= 0.001:
                del self._positions[symbol]

    def _calculate_equity(self) -> float:
        """Calculate total portfolio equity."""
        position_value = sum(
            self._prices.get(sym, data["avg_entry"]) * data["qty"]
            for sym, data in self._positions.items()
            if data["qty"] > 0
        )
        return self._cash + position_value

    # ------------------------------------------------------------------ #
    #  Reporting                                                           #
    # ------------------------------------------------------------------ #

    def get_trade_history(self) -> List[dict]:
        return list(self._trade_history)

    def get_equity_curve(self) -> List[float]:
        return list(self._equity_history)

    def reset(self):
        """Reset broker state (for new training episodes)."""
        self._cash = self.initial_capital
        self._positions.clear()
        self._orders.clear()
        self._realized_pnl = 0.0
        self._equity_history = [self.initial_capital]
        self._trade_history.clear()
