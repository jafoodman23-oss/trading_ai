"""
Risk management system.
Enforces position sizing, drawdown limits, daily loss limits, and stop losses.
All trading decisions must pass through RiskManager.check_trade().
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from .broker import Account, Broker


class RiskManager:
    """
    Centralized risk management for the trading system.

    Checks performed before each trade:
        1. Daily loss limit: block if today's P&L < -daily_loss_limit * initial_capital
        2. Max drawdown halt: block all trading if drawdown > max_drawdown_halt
        3. Position size limit: cap qty to max_position_pct of portfolio
        4. Max open positions: block new positions if at limit

    Position sizing:
        Uses a fractional Kelly criterion capped at max_position_pct.

    Stop loss calculation:
        Uses ATR-based stops with configurable multiplier.
    """

    def __init__(self, settings, broker: Optional[Broker] = None):
        self.settings = settings
        self.broker = broker

        # From settings
        self.max_position_pct = settings.trading.max_position_pct
        self.stop_loss_pct = settings.trading.stop_loss_pct
        self.take_profit_pct = settings.trading.take_profit_pct
        self.max_drawdown_halt = settings.risk.max_drawdown_halt
        self.daily_loss_limit = settings.risk.daily_loss_limit
        self.max_open_positions = settings.risk.max_open_positions
        self.kelly_fraction = settings.risk.kelly_fraction
        self.atr_mult_stop = settings.risk.atr_multiplier_stop
        self.atr_mult_take = settings.risk.atr_multiplier_take
        self.initial_capital = settings.trading.initial_capital

        # State tracking
        self._peak_equity: float = self.initial_capital
        self._day_start_equity: float = self.initial_capital
        self._current_date: date = datetime.utcnow().date()
        self._is_halted: bool = False
        self._halt_reason: str = ""

        # Per-symbol stop and take-profit prices
        self._stop_prices: Dict[str, float] = {}
        self._take_profit_prices: Dict[str, float] = {}

        logger.info(
            f"RiskManager initialized: "
            f"max_drawdown={self.max_drawdown_halt:.1%}, "
            f"daily_limit={self.daily_loss_limit:.1%}, "
            f"max_positions={self.max_open_positions}"
        )

    # ------------------------------------------------------------------ #
    #  Primary interface                                                   #
    # ------------------------------------------------------------------ #

    def check_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        account: Account,
        n_open_positions: int,
    ) -> Tuple[bool, float, str]:
        """
        Validate a proposed trade against all risk rules.

        Parameters
        ----------
        symbol : stock symbol
        side : "buy" or "sell"
        qty : proposed quantity
        price : current price
        account : current Account state
        n_open_positions : number of currently open positions

        Returns
        -------
        (approved, adjusted_qty, reason)
            approved : whether the trade is allowed
            adjusted_qty : possibly reduced quantity
            reason : explanation string (empty if approved without changes)
        """
        # ---- Hard halt check ------------------------------------------
        if self._is_halted:
            if side == "sell":
                return True, qty, "Halt override: closing position"
            return False, 0.0, f"Trading halted: {self._halt_reason}"

        # ---- Drawdown check -------------------------------------------
        current_drawdown = self._calculate_drawdown(account.portfolio_value)
        if current_drawdown <= -self.max_drawdown_halt:
            self._is_halted = True
            self._halt_reason = f"Max drawdown exceeded: {current_drawdown:.2%}"
            logger.critical(f"TRADING HALTED: {self._halt_reason}")
            if side == "sell":
                return True, qty, "Closing on halt"
            return False, 0.0, self._halt_reason

        # ---- Daily loss limit check -----------------------------------
        self._maybe_reset_daily(account.portfolio_value)
        daily_pnl_pct = (account.portfolio_value - self._day_start_equity) / self._day_start_equity
        if daily_pnl_pct <= -self.daily_loss_limit:
            if side == "sell":
                return True, qty, "Closing on daily loss limit"
            return False, 0.0, f"Daily loss limit reached: {daily_pnl_pct:.2%}"

        # ---- Max open positions (buy side only) -----------------------
        if side == "buy" and n_open_positions >= self.max_open_positions:
            return False, 0.0, f"Max open positions ({self.max_open_positions}) reached"

        # ---- Position size check (buy side) ----------------------------
        if side == "buy":
            max_value = account.portfolio_value * self.max_position_pct
            max_qty = max_value / price if price > 0 else 0.0
            if qty > max_qty:
                reason = f"Position size capped: {qty:.2f} -> {max_qty:.2f}"
                qty = max_qty
                logger.debug(reason)
                return True, qty, reason

        return True, qty, ""

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        atr: Optional[float],
        account: Account,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
    ) -> float:
        """
        Calculate optimal position size using fractional Kelly criterion,
        capped at max_position_pct of portfolio value.

        Kelly fraction: f = (W * R - (1 - W)) / R
            W = win rate
            R = avg_win / avg_loss ratio

        Parameters
        ----------
        symbol : stock symbol
        price : entry price
        atr : Average True Range (used for risk-per-share estimate)
        account : current account state
        win_rate : historical win rate (defaults to 0.5 if unknown)
        avg_win_loss_ratio : ratio of avg win to avg loss

        Returns
        -------
        qty : number of shares to buy
        """
        portfolio_value = account.portfolio_value
        if portfolio_value <= 0 or price <= 0:
            return 0.0

        # Kelly formula
        if avg_win_loss_ratio > 0:
            kelly_f = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        else:
            kelly_f = 0.0

        # Apply kelly fraction (conservative scaling)
        kelly_f = max(0.0, kelly_f * self.kelly_fraction)

        # Cap at max position percentage
        effective_pct = min(kelly_f, self.max_position_pct)

        # If ATR available, use risk-based sizing
        if atr and atr > 0:
            risk_per_share = atr * self.atr_mult_stop
            max_risk_amount = portfolio_value * 0.02  # max 2% of portfolio at risk
            atr_based_qty = max_risk_amount / risk_per_share
            kelly_based_qty = (portfolio_value * effective_pct) / price
            qty = min(atr_based_qty, kelly_based_qty)
        else:
            qty = (portfolio_value * effective_pct) / price

        # Ensure we don't exceed available cash
        max_affordable = account.cash / price
        qty = min(qty, max_affordable)

        return max(0.0, qty)

    def update(self, account: Account):
        """
        Update risk state with current account values.
        Must be called after each bar / account update.
        """
        # Update peak equity for drawdown calculation
        if account.portfolio_value > self._peak_equity:
            self._peak_equity = account.portfolio_value

        # Update day start equity on new trading day
        self._maybe_reset_daily(account.portfolio_value)

    def is_halted(self) -> bool:
        return self._is_halted

    def resume_trading(self):
        """Manually resume trading after halt (use with caution)."""
        self._is_halted = False
        self._halt_reason = ""
        logger.warning("Trading resumed manually")

    # ------------------------------------------------------------------ #
    #  Stop loss / take profit                                            #
    # ------------------------------------------------------------------ #

    def get_stop_loss_price(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate stop loss price for a position.
        If ATR is available, uses ATR-based stop; otherwise falls back to
        a fixed percentage stop.
        """
        if atr and atr > 0:
            stop_distance = atr * self.atr_mult_stop
        else:
            stop_distance = entry_price * self.stop_loss_pct

        if side == "buy":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def get_take_profit_price(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
    ) -> float:
        """Calculate take profit price for a position."""
        if atr and atr > 0:
            take_distance = atr * self.atr_mult_take
        else:
            take_distance = entry_price * self.take_profit_pct

        if side == "buy":
            return entry_price + take_distance
        else:
            return entry_price - take_distance

    def set_stop_take(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
    ):
        """Register stop loss and take profit prices for a symbol."""
        self._stop_prices[symbol] = self.get_stop_loss_price(entry_price, side, atr)
        self._take_profit_prices[symbol] = self.get_take_profit_price(entry_price, side, atr)
        logger.debug(
            f"Stop/Take set for {symbol}: "
            f"stop={self._stop_prices[symbol]:.4f}, "
            f"take={self._take_profit_prices[symbol]:.4f}"
        )

    def clear_stop_take(self, symbol: str):
        """Remove stop/take levels for a closed position."""
        self._stop_prices.pop(symbol, None)
        self._take_profit_prices.pop(symbol, None)

    def check_stop_take(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if current price has hit stop loss or take profit.

        Returns
        -------
        "stop" if stop loss triggered
        "take" if take profit triggered
        None if neither triggered
        """
        stop = self._stop_prices.get(symbol)
        take = self._take_profit_prices.get(symbol)

        if stop and current_price <= stop:
            logger.warning(f"STOP LOSS hit: {symbol} @ {current_price:.4f} (stop={stop:.4f})")
            return "stop"
        if take and current_price >= take:
            logger.info(f"TAKE PROFIT hit: {symbol} @ {current_price:.4f} (take={take:.4f})")
            return "take"
        return None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _calculate_drawdown(self, portfolio_value: float) -> float:
        """Current drawdown from peak (negative number)."""
        if self._peak_equity <= 0:
            return 0.0
        return (portfolio_value - self._peak_equity) / self._peak_equity

    def _maybe_reset_daily(self, portfolio_value: float):
        """Reset daily tracking at market open."""
        today = datetime.utcnow().date()
        if today != self._current_date:
            self._current_date = today
            self._day_start_equity = portfolio_value
            logger.info(f"New trading day: day_start_equity={portfolio_value:,.2f}")

    def get_risk_summary(self) -> dict:
        """Return current risk state summary."""
        return {
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "peak_equity": self._peak_equity,
            "day_start_equity": self._day_start_equity,
            "max_drawdown_limit": self.max_drawdown_halt,
            "daily_loss_limit": self.daily_loss_limit,
            "active_stops": dict(self._stop_prices),
            "active_takes": dict(self._take_profit_prices),
        }
