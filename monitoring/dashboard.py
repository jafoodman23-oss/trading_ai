"""
Live terminal dashboard using Rich.
Shows positions, recent trades, P&L, and model metrics in real time.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


class Dashboard:
    """
    Rich-based live terminal dashboard.

    Usage:
        dashboard = Dashboard(mode="paper", initial_capital=100_000)
        with dashboard:
            while running:
                dashboard.update(account, positions, trades, metrics, logs)
                time.sleep(1)
    """

    def __init__(self, mode: str = "paper", initial_capital: float = 100_000.0):
        self.mode = mode
        self.initial_capital = initial_capital
        self.console = Console()
        self._live: Optional[Live] = None

        # State
        self._account: dict = {}
        self._positions: list = []
        self._recent_trades: list = []
        self._metrics: dict = {}
        self._log_lines: List[str] = []
        self._equity_history: List[float] = [initial_capital]
        self._start_time = datetime.now()

    # ------------------------------------------------------------------ #
    #  Context manager                                                     #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=1,
            screen=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)

    def start(self):
        self.__enter__()

    def stop(self):
        if self._live:
            self._live.stop()

    # ------------------------------------------------------------------ #
    #  Public update API                                                   #
    # ------------------------------------------------------------------ #

    def update(
        self,
        account: dict,
        positions: list,
        recent_trades: list,
        metrics: dict,
        log_lines: Optional[List[str]] = None,
    ):
        """Refresh dashboard with latest data."""
        self._account = account
        self._positions = positions
        self._recent_trades = recent_trades[-10:]  # keep last 10
        self._metrics = metrics
        if log_lines:
            self._log_lines = log_lines[-15:]

        equity = account.get("equity", self.initial_capital)
        self._equity_history.append(equity)
        if len(self._equity_history) > 500:
            self._equity_history = self._equity_history[-500:]

        if self._live:
            self._live.update(self._render())

    def log(self, message: str):
        """Append a message to the dashboard log panel."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_lines.append(f"[{ts}] {message}")
        if len(self._log_lines) > 15:
            self._log_lines = self._log_lines[-15:]

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _render(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=12),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="account", size=10),
            Layout(name="positions"),
        )
        layout["right"].split_column(
            Layout(name="metrics", size=14),
            Layout(name="trades"),
        )

        layout["header"].update(self._render_header())
        layout["account"].update(self._render_account())
        layout["positions"].update(self._render_positions())
        layout["metrics"].update(self._render_metrics())
        layout["trades"].update(self._render_trades())
        layout["footer"].update(self._render_log())

        return Panel(layout, title="[bold cyan]Trading AI Dashboard[/bold cyan]", border_style="cyan")

    def _render_header(self) -> Panel:
        mode_color = "green" if self.mode == "paper" else "bold red"
        mode_label = f"[{mode_color}]{'PAPER TRADING' if self.mode == 'paper' else '⚠ LIVE TRADING'}[/{mode_color}]"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = str(datetime.now() - self._start_time).split(".")[0]
        text = Text()
        text.append(f"  Mode: ")
        text.append(f"{'PAPER' if self.mode == 'paper' else 'LIVE'}", style=mode_color)
        text.append(f"  |  Time: {now}  |  Uptime: {elapsed}  |  Self-Learning AI v1.0")
        return Panel(text, style="bold", box=box.SIMPLE)

    def _render_account(self) -> Panel:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Key", style="cyan", width=18)
        table.add_column("Value", style="white")

        equity = self._account.get("equity", self.initial_capital)
        cash = self._account.get("cash", self.initial_capital)
        pnl = equity - self.initial_capital
        pnl_pct = (pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0

        pnl_color = "green" if pnl >= 0 else "red"
        pnl_sign = "+" if pnl >= 0 else ""

        table.add_row("Portfolio Value", f"${equity:,.2f}")
        table.add_row("Cash", f"${cash:,.2f}")
        table.add_row("Starting Capital", f"${self.initial_capital:,.2f}")
        table.add_row(
            "Total P&L",
            f"[{pnl_color}]{pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)[/{pnl_color}]"
        )
        table.add_row(
            "Unrealized P&L",
            f"${self._account.get('unrealized_pnl', 0):,.2f}"
        )

        return Panel(table, title="[bold]Account[/bold]", border_style="blue")

    def _render_positions(self) -> Panel:
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        table.add_column("Symbol", width=8)
        table.add_column("Qty", justify="right", width=10)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("Current", justify="right", width=10)
        table.add_column("Unrlzd P&L", justify="right", width=12)
        table.add_column("Mkt Value", justify="right", width=12)

        for pos in self._positions:
            sym = pos.get("symbol", "?")
            qty = pos.get("qty", 0)
            entry = pos.get("avg_entry_price", 0)
            current = pos.get("current_price", 0)
            upnl = pos.get("unrealized_pnl", 0)
            mval = pos.get("market_value", 0)
            color = "green" if upnl >= 0 else "red"
            sign = "+" if upnl >= 0 else ""
            table.add_row(
                sym,
                f"{qty:.2f}",
                f"${entry:.2f}",
                f"${current:.2f}",
                f"[{color}]{sign}${upnl:.2f}[/{color}]",
                f"${mval:.2f}",
            )

        if not self._positions:
            table.add_row("[dim]No open positions[/dim]", "", "", "", "", "")

        return Panel(table, title="[bold]Open Positions[/bold]", border_style="blue")

    def _render_metrics(self) -> Panel:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=18)
        table.add_column("Value", style="white")

        m = self._metrics
        sharpe = m.get("sharpe_ratio", 0)
        sharpe_color = "green" if sharpe > 1 else ("yellow" if sharpe > 0 else "red")
        wr = m.get("win_rate", 0)
        wr_color = "green" if wr > 0.5 else ("yellow" if wr > 0.4 else "red")
        mdd = m.get("max_drawdown", 0)
        mdd_color = "red" if mdd < -0.1 else ("yellow" if mdd < -0.05 else "green")

        table.add_row("Sharpe Ratio", f"[{sharpe_color}]{sharpe:.3f}[/{sharpe_color}]")
        table.add_row("Sortino Ratio", f"{m.get('sortino_ratio', 0):.3f}")
        table.add_row("Win Rate", f"[{wr_color}]{wr*100:.1f}%[/{wr_color}]")
        table.add_row("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
        table.add_row("Max Drawdown", f"[{mdd_color}]{mdd*100:.2f}%[/{mdd_color}]")
        table.add_row("Total Trades", f"{m.get('n_trades', 0)}")
        table.add_row("Expectancy", f"${m.get('expectancy', 0):.2f}")
        table.add_row("Model Confidence", f"{m.get('model_confidence', 0)*100:.1f}%")
        table.add_row("Training Steps", f"{m.get('training_steps', 0):,}")
        table.add_row("Last Retrain", m.get("last_retrain", "Never"))

        return Panel(table, title="[bold]Performance Metrics[/bold]", border_style="blue")

    def _render_trades(self) -> Panel:
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        table.add_column("Time", width=10)
        table.add_column("Sym", width=6)
        table.add_column("Side", width=5)
        table.add_column("Qty", justify="right", width=8)
        table.add_column("Price", justify="right", width=9)
        table.add_column("P&L", justify="right", width=10)

        for trade in reversed(self._recent_trades):
            side = trade.get("side", trade.get("action", "?")).upper()
            side_color = "green" if side == "BUY" else "red"
            pnl = trade.get("pnl", 0)
            pnl_color = "green" if pnl > 0 else ("red" if pnl < 0 else "white")
            pnl_str = f"[{pnl_color}]{'+' if pnl > 0 else ''}${pnl:.2f}[/{pnl_color}]" if pnl else "-"
            ts = trade.get("timestamp", "")
            if hasattr(ts, "strftime"):
                ts = ts.strftime("%H:%M:%S")
            elif isinstance(ts, str) and len(ts) > 8:
                ts = ts[-8:]
            table.add_row(
                str(ts),
                trade.get("symbol", "?"),
                f"[{side_color}]{side}[/{side_color}]",
                f"{trade.get('qty', 0):.2f}",
                f"${trade.get('price', 0):.2f}",
                pnl_str,
            )

        if not self._recent_trades:
            table.add_row("[dim]No trades yet[/dim]", "", "", "", "", "")

        return Panel(table, title="[bold]Recent Trades[/bold]", border_style="blue")

    def _render_log(self) -> Panel:
        lines = "\n".join(self._log_lines) if self._log_lines else "[dim]No log messages[/dim]"
        return Panel(lines, title="[bold]System Log[/bold]", border_style="dim")

    # ------------------------------------------------------------------ #
    #  Static / non-live print                                            #
    # ------------------------------------------------------------------ #

    def print_summary(self, metrics: dict, trades: list, equity_curve: list):
        """Print a final summary after a backtest or training run."""
        self.console.print("\n[bold cyan]═══ Run Summary ═══[/bold cyan]")
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white", justify="right")

        for key, val in metrics.items():
            if isinstance(val, float):
                if "return" in key or "drawdown" in key or "rate" in key:
                    table.add_row(key.replace("_", " ").title(), f"{val*100:.2f}%")
                else:
                    table.add_row(key.replace("_", " ").title(), f"{val:.4f}")
            else:
                table.add_row(key.replace("_", " ").title(), str(val))

        self.console.print(table)
