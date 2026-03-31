"""
Trading AI — Main Entry Point

Usage:
    python main.py train   [--symbols SPY AAPL] [--days 365] [--timesteps 500000]
    python main.py run     [--mode paper]
    python main.py backtest [--start 2024-01-01] [--end 2025-01-01] [--symbols SPY]
    python main.py dashboard

Switch paper → live: edit config/config.yaml and set mode: "live"
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


# ------------------------------------------------------------------ #
#  Helper: graceful shutdown                                           #
# ------------------------------------------------------------------ #

_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    logger.info(f"Received signal {sig} — shutting down gracefully...")
    _shutdown = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ------------------------------------------------------------------ #
#  Command: train                                                      #
# ------------------------------------------------------------------ #

def cmd_train(args, settings):
    """Pre-train the RL agent on historical market data."""
    from data.storage import DataStorage
    from models.rl_agent import RLAgent
    from training.trainer import Trainer
    from training.evaluator import Evaluator
    from utils.logger import setup_logger

    setup_logger(settings.paths.logs_dir)
    logger.info("=== TRAINING MODE ===")
    logger.info(f"Symbols: {args.symbols or settings.symbols}")
    logger.info(f"Days: {args.days}, Timesteps: {args.timesteps:,}")

    storage = DataStorage(settings.paths.db_path)
    evaluator = Evaluator()

    # Agent will be initialized inside trainer once env is built
    # Pass a placeholder; Trainer replaces it after building env
    trainer = Trainer(
        settings=settings,
        agent=_build_placeholder_agent(settings),
        storage=storage,
        evaluator=evaluator,
    )

    # Try to resume from a saved checkpoint
    trainer.load_best_model()

    result = trainer.pretrain_historical(
        symbols=args.symbols or settings.symbols,
        days=args.days,
        total_timesteps=args.timesteps,
    )

    logger.info(
        f"Training complete. "
        f"Sharpe={result.sharpe:.3f}, "
        f"Return={result.total_return*100:.2f}%, "
        f"WinRate={result.win_rate*100:.1f}%"
    )


# ------------------------------------------------------------------ #
#  Command: run (paper or live trading)                                #
# ------------------------------------------------------------------ #

async def cmd_run_async(args, settings):
    """Start the live trading loop (paper or live)."""
    from data.storage import DataStorage
    from data.collector import DataCollector
    from data.preprocessor import FeatureEngineer
    from models.rl_agent import RLAgent
    from models.ensemble import EnsembleAgent
    from trading.paper_broker import PaperBroker
    from trading.alpaca_broker import AlpacaBroker
    from trading.risk_manager import RiskManager
    from training.trainer import Trainer
    from training.evaluator import Evaluator
    from monitoring.dashboard import Dashboard
    from monitoring.metrics import MetricsCalculator
    from utils.logger import setup_logger

    setup_logger(settings.paths.logs_dir)

    mode = args.mode if hasattr(args, "mode") else settings.mode
    settings.mode = mode
    logger.info(f"=== TRADING MODE: {mode.upper()} ===")

    storage = DataStorage(settings.paths.db_path)
    evaluator = Evaluator()
    feature_engineer = FeatureEngineer(settings)
    calc = MetricsCalculator()

    # Broker selection: paper broker needs no API keys
    if mode == "paper" and (
        settings.alpaca.paper_api_key.startswith("YOUR_") or
        not settings.alpaca.paper_api_key
    ):
        logger.info("No Alpaca API key configured — using internal PaperBroker")
        broker = PaperBroker(
            initial_capital=settings.trading.initial_capital,
            commission_rate=settings.trading.commission_rate,
            slippage_rate=settings.trading.slippage_rate,
        )
        use_alpaca = False
    else:
        logger.info(f"Connecting to Alpaca ({mode})")
        broker = AlpacaBroker(settings)
        use_alpaca = True

    risk_manager = RiskManager(settings, broker)

    # Build a minimal env for loading the agent
    placeholder_agent = _build_placeholder_agent(settings)
    trainer = Trainer(settings, placeholder_agent, storage, evaluator)

    # Load best saved model or train first if none exists
    loaded = trainer.load_best_model()
    if not loaded:
        logger.warning("No trained model found. Run `python main.py train` first.")
        logger.info("Starting with an untrained agent (random actions until training completes)")

    agent = trainer.agent
    dashboard = Dashboard(mode=mode, initial_capital=settings.trading.initial_capital)

    # Feature buffer for real-time inference: symbol -> list of bars
    bar_buffer: dict = {s: [] for s in settings.symbols}
    recent_trades: list = []
    all_trades: list = []

    def on_bar(symbol: str, bar: dict):
        """Called for every new bar received from the data stream."""
        nonlocal bar_buffer, recent_trades, all_trades

        # Update broker prices
        price = bar.get("close", bar.get("price", 0))
        broker.update_prices({symbol: price}) if hasattr(broker, "update_prices") else None

        # Add to buffer
        bar_buffer[symbol].append(bar)
        lookback = settings.model.lookback_window
        if len(bar_buffer[symbol]) > lookback * 3:
            bar_buffer[symbol] = bar_buffer[symbol][-(lookback * 3):]

        # Only act when we have enough history
        if len(bar_buffer[symbol]) < lookback + 5:
            return

        # Build features
        try:
            import pandas as pd
            df = pd.DataFrame(bar_buffer[symbol])
            features, prices_arr = feature_engineer.compute_features(df)
            if features is None or len(features) < lookback:
                return

            # Get the latest observation window
            obs_market = features[-lookback:]
            # Build a simple observation (the env appends portfolio features, but
            # for live inference we build them manually)
            account = broker.get_account()
            equity = account.equity if hasattr(account, "equity") else account.get("equity", settings.trading.initial_capital)
            position = broker.get_position(symbol)
            pos_qty = position.qty if position else 0.0
            pos_price = position.avg_entry_price if position else 0.0

            cash_pct = account.cash / equity if equity > 0 else 1.0
            pos_val = pos_qty * price
            pos_pct = pos_val / equity if equity > 0 else 0.0
            unrlzd = (price - pos_price) / pos_price if pos_price > 0 else 0.0
            bars_held = 0  # simplified

            portfolio_feat = [cash_pct, pos_pct, unrlzd, bars_held]
            import numpy as np
            pf = np.array(portfolio_feat, dtype=np.float32)
            pf_window = np.tile(pf, (lookback, 1))
            obs = np.concatenate([obs_market, pf_window], axis=1).astype(np.float32)
            obs = np.clip(obs, -10.0, 10.0)

            # Predict action
            action, confidence = agent.predict(obs)

            # Risk check and order placement
            if action == 1 and pos_qty == 0:  # BUY
                qty = risk_manager.calculate_position_size(symbol, price, None, account)
                approved, adj_qty, reason = risk_manager.check_trade(symbol, "buy", qty, price)
                if approved and adj_qty > 0 and not risk_manager.is_halted():
                    order = broker.place_order(symbol, "buy", adj_qty)
                    trade_record = {
                        "symbol": symbol, "side": "buy", "qty": adj_qty,
                        "price": price, "timestamp": datetime.now(),
                    }
                    recent_trades.append(trade_record)
                    all_trades.append(trade_record)
                    dashboard.log(f"BUY {adj_qty:.2f} {symbol} @ ${price:.2f} (confidence: {confidence:.0%})")
                else:
                    dashboard.log(f"Trade blocked: {reason}")

            elif action == 2 and pos_qty > 0:  # SELL
                approved, adj_qty, reason = risk_manager.check_trade(symbol, "sell", pos_qty, price)
                if approved:
                    order = broker.place_order(symbol, "sell", pos_qty)
                    pnl = (price - pos_price) * pos_qty if pos_price > 0 else 0
                    trade_record = {
                        "symbol": symbol, "side": "sell", "qty": pos_qty,
                        "price": price, "pnl": pnl, "timestamp": datetime.now(),
                    }
                    recent_trades.append(trade_record)
                    all_trades.append(trade_record)
                    dashboard.log(f"SELL {pos_qty:.2f} {symbol} @ ${price:.2f} P&L: ${pnl:.2f}")

            # Notify trainer of new bar for continuous learning
            trainer.on_new_bar(symbol, bar)

        except Exception as e:
            logger.error(f"Error processing bar for {symbol}: {e}")
            dashboard.log(f"ERROR: {e}")

    # Start data collection
    collector = None
    if use_alpaca:
        from data.collector import DataCollector
        collector = DataCollector(settings, callback=on_bar)

    # Start continuous training loop in background
    training_task = asyncio.create_task(trainer.continuous_train_loop())

    logger.info("Starting trading loop. Press Ctrl+C to stop.")

    with dashboard:
        if collector:
            collect_task = asyncio.create_task(collector.start())

        last_dashboard_update = time.time()

        while not _shutdown:
            await asyncio.sleep(1)

            # Update dashboard every second
            if time.time() - last_dashboard_update >= 1:
                try:
                    account = broker.get_account()
                    positions_raw = broker.get_all_positions() if hasattr(broker, "get_all_positions") else []
                    positions_dicts = []
                    for p in positions_raw:
                        positions_dicts.append({
                            "symbol": p.symbol,
                            "qty": p.qty,
                            "avg_entry_price": p.avg_entry_price,
                            "current_price": p.current_price,
                            "unrealized_pnl": p.unrealized_pnl,
                            "market_value": p.market_value,
                        })

                    equity_curve = broker.get_equity_curve() if hasattr(broker, "get_equity_curve") else [account.equity]
                    eq_arr = import_numpy_array(equity_curve)
                    metrics = calc.full_report(eq_arr, all_trades)
                    metrics["model_confidence"] = 0.5
                    metrics["training_steps"] = trainer.stats["total_training_steps"]
                    metrics["last_retrain"] = trainer.stats["last_retrain"]

                    account_dict = {
                        "equity": account.equity,
                        "cash": account.cash,
                        "unrealized_pnl": getattr(account, "unrealized_pnl", 0),
                    }

                    dashboard.update(
                        account=account_dict,
                        positions=positions_dicts,
                        recent_trades=recent_trades[-10:],
                        metrics=metrics,
                    )
                    last_dashboard_update = time.time()
                    risk_manager.update(account)
                except Exception as e:
                    logger.error(f"Dashboard update error: {e}")

        # Graceful shutdown
        logger.info("Shutting down...")
        training_task.cancel()
        if collector:
            await collector.stop()
            collect_task.cancel()

        # Close all positions on shutdown
        logger.info("Closing all open positions...")
        positions = broker.get_all_positions() if hasattr(broker, "get_all_positions") else []
        for pos in positions:
            try:
                broker.place_order(pos.symbol, "sell", pos.qty)
                logger.info(f"Closed position: {pos.symbol} qty={pos.qty}")
            except Exception as e:
                logger.error(f"Failed to close {pos.symbol}: {e}")

        logger.info("Trading session ended.")


def import_numpy_array(lst):
    import numpy as np
    return np.array(lst) if lst else np.array([100_000.0])


def cmd_run(args, settings):
    asyncio.run(cmd_run_async(args, settings))


# ------------------------------------------------------------------ #
#  Command: backtest                                                   #
# ------------------------------------------------------------------ #

def cmd_backtest(args, settings):
    """Run a historical backtest and print performance report."""
    from data.storage import DataStorage
    from data.historical import HistoricalDataFetcher
    from data.preprocessor import FeatureEngineer
    from environment.trading_env import TradingEnv
    from training.evaluator import Evaluator
    from utils.logger import setup_logger

    setup_logger(settings.paths.logs_dir)
    logger.info("=== BACKTEST MODE ===")

    symbols = args.symbols or settings.symbols
    start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else datetime.now() - timedelta(days=180)
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()

    logger.info(f"Backtest: {symbols} | {start.date()} → {end.date()}")

    fetcher = HistoricalDataFetcher(settings)
    engineer = FeatureEngineer(settings.data.features)
    storage = DataStorage(settings.paths.db_path)
    evaluator = Evaluator()

    # Load trained agent
    placeholder_agent = _build_placeholder_agent(settings)
    from training.trainer import Trainer
    trainer = Trainer(settings, placeholder_agent, storage, evaluator)
    loaded = trainer.load_best_model()
    if not loaded:
        logger.warning("No trained model found — running backtest with random agent")

    all_features = {}
    all_prices = {}

    for symbol in symbols:
        try:
            df = fetcher.fetch_bars(symbol, start, end, settings.data.timeframe)
            if df is None or len(df) < settings.model.lookback_window + 10:
                logger.warning(f"Insufficient data for {symbol}")
                continue
            features, prices = engineer.compute_features(df)
            if features is not None and len(features) > 0:
                all_features[symbol] = features
                all_prices[symbol] = prices
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")

    if not all_features:
        logger.error("No data available for backtest")
        return

    results = {}
    for symbol in all_features:
        env = TradingEnv(
            data={symbol: all_features[symbol]},
            prices={symbol: all_prices[symbol]},
            config=settings,
            mode="eval",
            initial_capital=settings.trading.initial_capital,
        )
        result = evaluator.run_backtest(trainer.agent, env, n_episodes=1)
        result.symbol = symbol
        results[symbol] = result
        evaluator.print_report(result)

    if len(results) > 1:
        evaluator.compare_reports(results)


# ------------------------------------------------------------------ #
#  Command: dashboard (read-only view of latest state)                 #
# ------------------------------------------------------------------ #

def cmd_dashboard(args, settings):
    """Show a static snapshot of the latest saved metrics."""
    from data.storage import DataStorage
    from monitoring.dashboard import Dashboard
    from monitoring.metrics import MetricsCalculator
    from rich.console import Console

    console = Console()
    storage = DataStorage(settings.paths.db_path)

    snapshot = storage.get_latest_snapshot()
    if snapshot:
        console.print(f"\n[bold cyan]Latest Model Snapshot[/bold cyan]")
        console.print(f"  Version  : {snapshot.get('version', 'N/A')}")
        console.print(f"  Sharpe   : {snapshot.get('sharpe', 0):.3f}")
        console.print(f"  Win Rate : {snapshot.get('win_rate', 0)*100:.1f}%")
        console.print(f"  Return   : {snapshot.get('total_return', 0)*100:.2f}%")
        console.print(f"  Saved    : {snapshot.get('timestamp', 'N/A')}")
    else:
        console.print("[yellow]No model snapshots found. Run `python main.py train` first.[/yellow]")

    trades = storage.get_recent_trades(limit=50)
    if trades:
        calc = MetricsCalculator()
        from monitoring.dashboard import Dashboard
        dash = Dashboard(mode=settings.mode, initial_capital=settings.trading.initial_capital)

        # Try to reconstruct equity from trades
        import numpy as np
        equity = settings.trading.initial_capital
        equity_curve = [equity]
        for t in trades:
            pnl = t.get("pnl", 0) or 0
            equity += pnl
            equity_curve.append(equity)

        metrics = calc.full_report(np.array(equity_curve), trades)
        dash.print_summary(metrics, trades, equity_curve)
    else:
        console.print("[dim]No trade history found.[/dim]")


# ------------------------------------------------------------------ #
#  Helper                                                              #
# ------------------------------------------------------------------ #

def _build_placeholder_agent(settings):
    """Build a dummy agent object that will be replaced during training/loading."""

    class _PlaceholderAgent:
        model = None
        _is_trained = False

        def predict(self, obs):
            import numpy as np
            return int(np.random.randint(0, 3)), 0.33

        def save(self, path):
            pass

        def load(self, path):
            pass

        def train(self, total_timesteps):
            pass

        def update_online(self, env, additional_timesteps=50000):
            pass

    return _PlaceholderAgent()


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-Learning Trading AI — paper and live trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --symbols SPY AAPL NVDA --days 365 --timesteps 500000
  python main.py run --mode paper
  python main.py backtest --start 2024-01-01 --end 2025-01-01 --symbols SPY QQQ
  python main.py dashboard

To switch from paper to live trading:
  Edit config/config.yaml  →  set mode: "live"  and fill in live_api_key / live_api_secret
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # train
    train_p = subparsers.add_parser("train", help="Pre-train the agent on historical data")
    train_p.add_argument("--symbols", nargs="+", help="Symbols to train on (default: from config)")
    train_p.add_argument("--days", type=int, default=365, help="Days of historical data")
    train_p.add_argument("--timesteps", type=int, default=500_000, help="PPO training timesteps")

    # run
    run_p = subparsers.add_parser("run", help="Start live paper/live trading")
    run_p.add_argument(
        "--mode", choices=["paper", "live"], default=None,
        help="Override config mode (paper or live)"
    )

    # backtest
    bt_p = subparsers.add_parser("backtest", help="Run a historical backtest")
    bt_p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    bt_p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    bt_p.add_argument("--symbols", nargs="+", help="Symbols to backtest")

    # dashboard
    subparsers.add_parser("dashboard", help="Show latest metrics snapshot")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Load config
    try:
        from config.settings import Settings
        settings = Settings.load()
    except FileNotFoundError as e:
        print(f"Error loading config: {e}")
        print("Make sure config/config.yaml exists.")
        sys.exit(1)

    # Override mode if specified
    if hasattr(args, "mode") and args.mode is not None:
        settings.mode = args.mode

    # Dispatch
    if args.command == "train":
        cmd_train(args, settings)
    elif args.command == "run":
        cmd_run(args, settings)
    elif args.command == "backtest":
        cmd_backtest(args, settings)
    elif args.command == "dashboard":
        cmd_dashboard(args, settings)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
