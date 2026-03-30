"""
Alpaca Markets broker implementation.
Supports both paper trading (https://paper-api.alpaca.markets)
and live trading (https://api.alpaca.markets) — controlled by settings.mode.

Switching from paper to live: change mode: "paper" -> mode: "live" in config.yaml.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestBarRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.error("alpaca-py not installed — AlpacaBroker will not function")

from .broker import Account, Broker, Order, Position


class AlpacaBroker(Broker):
    """
    Alpaca-backed broker for paper and live trading.

    The same code works for both modes — the only difference is which
    API endpoint is used, determined by settings.mode.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    ORDER_POLL_TIMEOUT = 30.0   # seconds to wait for fill
    ORDER_POLL_INTERVAL = 0.5   # seconds between status checks

    def __init__(self, settings):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py required for AlpacaBroker")

        self.settings = settings
        api_key = settings.alpaca.get_api_key(settings.mode)
        api_secret = settings.alpaca.get_api_secret(settings.mode)
        paper = settings.mode == "paper"

        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

        self._initial_capital: Optional[float] = None
        self._price_cache: Dict[str, float] = {}

        logger.info(
            f"AlpacaBroker initialized: mode={settings.mode}, "
            f"paper={paper}"
        )

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
        """Place an order via Alpaca API."""
        alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        coid = client_order_id or str(uuid.uuid4())

        for attempt in range(self.MAX_RETRIES):
            try:
                if order_type == "market":
                    request = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                        client_order_id=coid,
                    )
                elif order_type == "limit" and limit_price:
                    request = LimitOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                        limit_price=limit_price,
                        client_order_id=coid,
                    )
                elif order_type == "stop" and stop_price:
                    request = StopOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                        stop_price=stop_price,
                        client_order_id=coid,
                    )
                else:
                    raise ValueError(f"Unsupported order type: {order_type}")

                alpaca_order = self.trading_client.submit_order(request)
                order = self._convert_order(alpaca_order)
                logger.info(
                    f"Order placed: {side.upper()} {qty} {symbol} "
                    f"({order_type}) -> ID={order.id}"
                )
                return order

            except Exception as e:
                logger.warning(f"Order attempt {attempt+1}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (2 ** attempt))

        raise RuntimeError(f"Failed to place order after {self.MAX_RETRIES} attempts")

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.warning(f"Cancel order {order_id} failed: {e}")
            return False

    def cancel_all_orders(self) -> int:
        try:
            cancelled = self.trading_client.cancel_orders()
            return len(cancelled) if cancelled else 0
        except Exception as e:
            logger.error(f"Cancel all orders failed: {e}")
            return 0

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            alpaca_pos = self.trading_client.get_open_position(symbol)
            return self._convert_position(alpaca_pos)
        except Exception:
            return None

    def get_all_positions(self) -> List[Position]:
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            return [self._convert_position(p) for p in alpaca_positions]
        except Exception as e:
            logger.error(f"get_all_positions failed: {e}")
            return []

    def get_account(self) -> Account:
        alpaca_account = self._retry(self.trading_client.get_account)
        equity = float(alpaca_account.equity)
        cash = float(alpaca_account.cash)
        buying_power = float(alpaca_account.buying_power)
        portfolio_value = float(alpaca_account.portfolio_value)

        if self._initial_capital is None:
            self._initial_capital = portfolio_value

        return Account(
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            initial_capital=self._initial_capital,
        )

    def get_order(self, order_id: str) -> Optional[Order]:
        try:
            alpaca_order = self.trading_client.get_order_by_id(order_id)
            return self._convert_order(alpaca_order)
        except Exception as e:
            logger.warning(f"get_order {order_id} failed: {e}")
            return None

    def get_open_orders(self) -> List[Order]:
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            alpaca_orders = self.trading_client.get_orders(request)
            return [self._convert_order(o) for o in alpaca_orders]
        except Exception as e:
            logger.error(f"get_open_orders failed: {e}")
            return []

    def is_market_open(self) -> bool:
        try:
            clock = self.trading_client.get_clock()
            return bool(clock.is_open)
        except Exception as e:
            logger.warning(f"is_market_open check failed: {e}")
            return False

    def get_latest_price(self, symbol: str) -> float:
        try:
            request = StockLatestBarRequest(symbol_or_symbols=symbol)
            bars = self.data_client.get_stock_latest_bar(request)
            if symbol in bars:
                price = float(bars[symbol].close)
                self._price_cache[symbol] = price
                return price
        except Exception as e:
            logger.warning(f"get_latest_price({symbol}) failed: {e}")

        # Fall back to cache
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        raise ValueError(f"No price available for {symbol}")

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Batch price fetch (more efficient than individual calls)."""
        try:
            request = StockLatestBarRequest(symbol_or_symbols=symbols)
            bars = self.data_client.get_stock_latest_bar(request)
            prices = {}
            for sym in symbols:
                if sym in bars:
                    prices[sym] = float(bars[sym].close)
                    self._price_cache[sym] = prices[sym]
                elif sym in self._price_cache:
                    prices[sym] = self._price_cache[sym]
            return prices
        except Exception as e:
            logger.warning(f"Batch price fetch failed: {e}")
            return {sym: self._price_cache.get(sym, 0.0) for sym in symbols}

    def wait_for_fill(self, order_id: str, timeout: float = None) -> Order:
        """
        Poll for an order to be filled.
        Returns the updated Order once filled or timeout occurs.
        """
        timeout = timeout or self.ORDER_POLL_TIMEOUT
        start = time.time()

        while time.time() - start < timeout:
            order = self.get_order(order_id)
            if order and order.is_filled:
                return order
            if order and order.status in ("cancelled", "rejected", "expired"):
                return order
            time.sleep(self.ORDER_POLL_INTERVAL)

        order = self.get_order(order_id)
        logger.warning(f"Order {order_id} not filled within {timeout}s")
        return order or Order(
            id=order_id, symbol="?", side="?", qty=0, type="?", status="timeout"
        )

    # ------------------------------------------------------------------ #
    #  Conversion helpers                                                  #
    # ------------------------------------------------------------------ #

    def _convert_order(self, alpaca_order) -> Order:
        """Convert Alpaca order object to our Order dataclass."""
        return Order(
            id=str(alpaca_order.id),
            symbol=str(alpaca_order.symbol),
            side="buy" if str(alpaca_order.side).lower() == "buy" else "sell",
            qty=float(alpaca_order.qty or 0),
            type=str(alpaca_order.order_type).lower().replace("ordertype.", ""),
            status=str(alpaca_order.status).lower().replace("orderstatus.", ""),
            filled_qty=float(alpaca_order.filled_qty or 0),
            filled_avg_price=float(alpaca_order.filled_avg_price or 0),
            limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            timestamp=alpaca_order.created_at or datetime.utcnow(),
            client_order_id=str(alpaca_order.client_order_id) if alpaca_order.client_order_id else None,
        )

    def _convert_position(self, alpaca_pos) -> Position:
        """Convert Alpaca position object to our Position dataclass."""
        qty = float(alpaca_pos.qty)
        avg_entry = float(alpaca_pos.avg_entry_price)
        current_price = float(alpaca_pos.current_price or avg_entry)
        unrealized = float(alpaca_pos.unrealized_pl or 0)
        market_value = float(alpaca_pos.market_value or qty * current_price)

        return Position(
            symbol=str(alpaca_pos.symbol),
            qty=qty,
            avg_entry_price=avg_entry,
            current_price=current_price,
            unrealized_pnl=unrealized,
            market_value=market_value,
        )

    def _retry(self, fn, *args, **kwargs):
        """Execute a function with retry logic."""
        last_err = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (2 ** attempt))
        raise RuntimeError(f"Operation failed after {self.MAX_RETRIES} retries: {last_err}")
