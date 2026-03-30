"""
Backtesting evaluator.
Runs a trained agent through historical data and computes performance metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

from monitoring.metrics import MetricsCalculator


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    trades: List[dict]
    metrics: dict
    symbol: str = ""
    start_step: int = 0
    end_step: int = 0

    @property
    def total_return(self) -> float:
        return self.metrics.get("total_return", 0.0)

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe_ratio", 0.0)

    @property
    def max_drawdown(self) -> float:
        return self.metrics.get("max_drawdown", 0.0)

    @property
    def win_rate(self) -> float:
        return self.metrics.get("win_rate", 0.0)


class Evaluator:
    """
    Runs backtests and computes performance metrics.

    Usage:
        evaluator = Evaluator()
        result = evaluator.run_backtest(agent, env)
        evaluator.print_report(result)
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.calc = MetricsCalculator()

    def run_backtest(
        self,
        agent,
        env,
        n_episodes: int = 1,
        deterministic: bool = True,
    ) -> BacktestResult:
        """
        Run one or more evaluation episodes and average the results.

        Parameters
        ----------
        agent : any agent with a .predict(obs) method
        env   : Gymnasium-compatible trading environment
        n_episodes : how many episodes to run (averaged)
        deterministic : use greedy policy
        """
        all_equity_curves = []
        all_trades = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            terminated = truncated = False
            episode_trades = []

            while not (terminated or truncated):
                if hasattr(agent, "predict"):
                    action, _ = agent.predict(obs)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, step_info = env.step(int(action))

            # Collect episode results
            equity = np.array(env.equity_curve)
            all_equity_curves.append(equity)
            all_trades.extend(env.trades)

            summary = env.episode_summary()
            logger.debug(
                f"Backtest ep {ep+1}/{n_episodes}: "
                f"return={summary['total_return']*100:.2f}% "
                f"sharpe={summary['sharpe']:.3f} "
                f"trades={summary['n_trades']}"
            )

        # Use the first episode's equity curve for the result (or average if multiple)
        primary_equity = all_equity_curves[0]
        if len(all_equity_curves) > 1:
            # Pad to same length and average
            max_len = max(len(e) for e in all_equity_curves)
            padded = [
                np.pad(e, (0, max_len - len(e)), mode="edge")
                for e in all_equity_curves
            ]
            primary_equity = np.mean(padded, axis=0)

        metrics = self.calc.full_report(primary_equity, all_trades)

        return BacktestResult(
            equity_curve=primary_equity,
            trades=all_trades,
            metrics=metrics,
            symbol=getattr(env, "_episode_symbol", ""),
            start_step=getattr(env, "current_step", 0),
        )

    def run_multi_symbol_backtest(
        self,
        agent,
        env_factory,
        symbols: List[str],
    ) -> Dict[str, BacktestResult]:
        """Run separate backtests for each symbol and return results keyed by symbol."""
        results = {}
        for symbol in symbols:
            env = env_factory(symbol)
            result = self.run_backtest(agent, env)
            result.symbol = symbol
            results[symbol] = result
            logger.info(
                f"{symbol}: return={result.total_return*100:.2f}% "
                f"sharpe={result.sharpe:.3f} "
                f"max_dd={result.max_drawdown*100:.2f}%"
            )
        return results

    def print_report(self, result: BacktestResult):
        """Print a formatted performance report to the console."""
        self.console.print(
            f"\n[bold cyan]Backtest Report"
            + (f" — {result.symbol}" if result.symbol else "")
            + "[/bold cyan]"
        )

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=22)
        table.add_column("Value", style="white", justify="right", width=14)

        m = result.metrics
        rows = [
            ("Total Return", f"{m.get('total_return', 0)*100:.2f}%"),
            ("Annualized Return", f"{m.get('annualized_return', 0)*100:.2f}%"),
            ("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.3f}"),
            ("Sortino Ratio", f"{m.get('sortino_ratio', 0):.3f}"),
            ("Max Drawdown", f"{m.get('max_drawdown', 0)*100:.2f}%"),
            ("Calmar Ratio", f"{m.get('calmar_ratio', 0):.3f}"),
            ("Win Rate", f"{m.get('win_rate', 0)*100:.1f}%"),
            ("Profit Factor", f"{m.get('profit_factor', 0):.2f}"),
            ("Expectancy ($/trade)", f"${m.get('expectancy', 0):.2f}"),
            ("Avg Win", f"${m.get('avg_win', 0):.2f}"),
            ("Avg Loss", f"${m.get('avg_loss', 0):.2f}"),
            ("Total Trades", str(m.get("n_trades", 0))),
        ]

        for label, value in rows:
            # Color-code key metrics
            style = ""
            if "Return" in label or "Ratio" in label or "Win Rate" in label:
                try:
                    num = float(value.replace("%", "").replace("$", ""))
                    style = "green" if num > 0 else "red"
                except ValueError:
                    pass
            if "Drawdown" in label:
                try:
                    num = float(value.replace("%", ""))
                    style = "red" if num < -10 else ("yellow" if num < -5 else "green")
                except ValueError:
                    pass
            table.add_row(label, f"[{style}]{value}[/{style}]" if style else value)

        self.console.print(table)

    def compare_reports(self, results: Dict[str, BacktestResult]):
        """Print a side-by-side comparison of multiple backtest results."""
        self.console.print("\n[bold cyan]Multi-Symbol Comparison[/bold cyan]")

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        for symbol in results:
            table.add_column(symbol, justify="right", width=10)

        metrics_keys = [
            ("Total Return", "total_return", True),
            ("Sharpe", "sharpe_ratio", False),
            ("Max Drawdown", "max_drawdown", True),
            ("Win Rate", "win_rate", True),
            ("Trades", "n_trades", False),
        ]

        for label, key, is_pct in metrics_keys:
            row = [label]
            for symbol, result in results.items():
                val = result.metrics.get(key, 0)
                if is_pct:
                    row.append(f"{val*100:.1f}%")
                elif isinstance(val, float):
                    row.append(f"{val:.3f}")
                else:
                    row.append(str(val))
            table.add_row(*row)

        self.console.print(table)
