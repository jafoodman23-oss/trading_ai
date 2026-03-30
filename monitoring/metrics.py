"""
Financial metrics calculator.
All metrics computed with pure numpy/pandas — no external dependencies.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd


class MetricsCalculator:
    """Compute standard trading/portfolio performance metrics."""

    TRADING_DAYS_PER_YEAR = 252
    MINUTES_PER_DAY = 390  # NYSE regular session

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate_annual: float = 0.05,
        periods_per_year: int = 252,
    ) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        rf_per_period = risk_free_rate_annual / periods_per_year
        excess = returns - rf_per_period
        std = np.std(excess, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate_annual: float = 0.05,
        periods_per_year: int = 252,
    ) -> float:
        """Annualized Sortino ratio (uses downside deviation)."""
        if len(returns) < 2:
            return 0.0
        rf_per_period = risk_free_rate_annual / periods_per_year
        excess = returns - rf_per_period
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = np.std(downside, ddof=1)
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Maximum drawdown as a negative fraction.
        Returns value in [-1, 0].
        """
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / np.where(peak > 0, peak, 1)
        return float(np.min(drawdown))

    @staticmethod
    def calmar_ratio(
        returns: np.ndarray,
        equity_curve: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """Annualized return divided by max drawdown magnitude."""
        mdd = MetricsCalculator.max_drawdown(equity_curve)
        if mdd == 0:
            return 0.0
        annual_return = float(np.mean(returns) * periods_per_year)
        return annual_return / abs(mdd)

    @staticmethod
    def win_rate(trades: List[dict]) -> float:
        """Fraction of closed trades that were profitable."""
        closed = [t for t in trades if "pnl" in t]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t["pnl"] > 0)
        return wins / len(closed)

    @staticmethod
    def profit_factor(trades: List[dict]) -> float:
        """Gross profit / gross loss. > 1 means profitable."""
        closed = [t for t in trades if "pnl" in t]
        gross_profit = sum(t["pnl"] for t in closed if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in closed if t["pnl"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def expectancy(trades: List[dict]) -> float:
        """Average P&L per closed trade."""
        closed = [t for t in trades if "pnl" in t]
        if not closed:
            return 0.0
        return float(np.mean([t["pnl"] for t in closed]))

    @staticmethod
    def average_win(trades: List[dict]) -> float:
        wins = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
        return float(np.mean(wins)) if wins else 0.0

    @staticmethod
    def average_loss(trades: List[dict]) -> float:
        losses = [t["pnl"] for t in trades if t.get("pnl", 0) < 0]
        return float(np.mean(losses)) if losses else 0.0

    @staticmethod
    def total_return(equity_curve: np.ndarray) -> float:
        """Simple total return as a fraction."""
        if len(equity_curve) < 2 or equity_curve[0] == 0:
            return 0.0
        return float((equity_curve[-1] - equity_curve[0]) / equity_curve[0])

    @staticmethod
    def annualized_return(
        equity_curve: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """CAGR assuming equal bar spacing."""
        tr = MetricsCalculator.total_return(equity_curve)
        n = len(equity_curve) - 1
        if n <= 0:
            return 0.0
        years = n / periods_per_year
        if years <= 0:
            return 0.0
        return float((1 + tr) ** (1 / years) - 1)

    @staticmethod
    def rolling_sharpe(
        returns: np.ndarray,
        window: int = 252,
        risk_free_rate_annual: float = 0.05,
    ) -> np.ndarray:
        """Rolling Sharpe ratio over a sliding window."""
        result = np.full(len(returns), np.nan)
        rf = risk_free_rate_annual / window
        for i in range(window, len(returns) + 1):
            chunk = returns[i - window:i]
            excess = chunk - rf
            std = np.std(excess, ddof=1)
            if std > 0:
                result[i - 1] = np.mean(excess) / std * np.sqrt(window)
        return result

    @staticmethod
    def rolling_win_rate(trades: List[dict], window: int = 20) -> List[float]:
        """Rolling win rate over last `window` trades."""
        closed = [t for t in trades if "pnl" in t]
        result = []
        for i in range(len(closed)):
            chunk = closed[max(0, i - window + 1):i + 1]
            wins = sum(1 for t in chunk if t["pnl"] > 0)
            result.append(wins / len(chunk))
        return result

    @classmethod
    def full_report(
        cls,
        equity_curve: np.ndarray,
        trades: List[dict],
        periods_per_year: int = 252,
    ) -> dict:
        """Compute all metrics and return as a dictionary."""
        returns = np.diff(equity_curve) / np.where(equity_curve[:-1] > 0, equity_curve[:-1], 1)
        return {
            "total_return": cls.total_return(equity_curve),
            "annualized_return": cls.annualized_return(equity_curve, periods_per_year),
            "sharpe_ratio": cls.sharpe_ratio(returns, periods_per_year=periods_per_year),
            "sortino_ratio": cls.sortino_ratio(returns, periods_per_year=periods_per_year),
            "max_drawdown": cls.max_drawdown(equity_curve),
            "calmar_ratio": cls.calmar_ratio(returns, equity_curve, periods_per_year),
            "win_rate": cls.win_rate(trades),
            "profit_factor": cls.profit_factor(trades),
            "expectancy": cls.expectancy(trades),
            "avg_win": cls.average_win(trades),
            "avg_loss": cls.average_loss(trades),
            "n_trades": len([t for t in trades if "pnl" in t]),
        }
