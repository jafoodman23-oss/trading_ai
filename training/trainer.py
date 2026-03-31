"""
Trainer: orchestrates historical pre-training and continuous online learning.

Phase 1 — Historical pre-training:
    Fetch N days of historical bars → compute features → train PPO on full dataset.

Phase 2 — Continuous learning (runs forever):
    Stream live bars → every N bars, evaluate rolling performance →
    if Sharpe drops below threshold, trigger a retrain → save checkpoint if improved.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from config.settings import Settings
from data.historical import HistoricalDataFetcher
from data.preprocessor import FeatureEngineer
from data.storage import DataStorage
from environment.trading_env import TradingEnv
from models.rl_agent import RLAgent
from training.evaluator import Evaluator, BacktestResult


class Trainer:
    """
    Manages the full training lifecycle of the RL trading agent.

    Parameters
    ----------
    settings : Settings
    agent    : RLAgent (or any BaseAgent)
    storage  : DataStorage for persisting bars and model snapshots
    evaluator: Evaluator for performance measurement
    """

    def __init__(
        self,
        settings: Settings,
        agent: RLAgent,
        storage: DataStorage,
        evaluator: Optional[Evaluator] = None,
    ):
        self.settings = settings
        self.agent = agent
        self.storage = storage
        self.evaluator = evaluator or Evaluator()
        self.fetcher = HistoricalDataFetcher(settings, storage)
        self.feature_engineer = FeatureEngineer(settings.data.features)

        self._best_sharpe: float = -float("inf")
        self._total_training_steps: int = 0
        self._last_retrain_time: datetime = datetime.now()
        self._bar_count: int = 0

        # Live data buffer: symbol -> list of raw bar dicts
        self._live_bars: Dict[str, List[dict]] = {s: [] for s in settings.symbols}

    # ------------------------------------------------------------------ #
    #  Phase 1: Historical pre-training                                    #
    # ------------------------------------------------------------------ #

    def pretrain_historical(
        self,
        symbols: Optional[List[str]] = None,
        days: Optional[int] = None,
        total_timesteps: Optional[int] = None,
    ) -> BacktestResult:
        """
        Fetch historical data, build features, train the agent on all available data.

        Returns the evaluation BacktestResult from after training.
        """
        symbols = symbols or self.settings.symbols
        days = days or self.settings.data.historical_days
        timesteps = total_timesteps or self.settings.model.training_timesteps

        logger.info(f"Starting historical pre-training: {symbols}, {days} days, {timesteps:,} timesteps")

        # 1. Fetch historical bars
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        all_features: Dict[str, np.ndarray] = {}
        all_prices: Dict[str, np.ndarray] = {}

        for symbol in symbols:
            logger.info(f"Fetching historical data for {symbol}...")
            try:
                df = self.fetcher.fetch_bars(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    timeframe=self.settings.data.timeframe,
                )
                if df is None or len(df) < self.settings.model.lookback_window + 50:
                    logger.warning(f"Not enough data for {symbol} ({len(df) if df is not None else 0} bars), skipping")
                    continue

                # 2. Compute features
                features, prices = self.feature_engineer.compute_features(df)
                if features is None or len(features) == 0:
                    logger.warning(f"Feature engineering failed for {symbol}")
                    continue

                all_features[symbol] = features
                all_prices[symbol] = prices
                logger.info(f"{symbol}: {len(features)} bars, {features.shape[1]} features")

                # Persist to DB
                df_to_save = df.copy().reset_index()
                if "timestamp" not in df_to_save.columns and df_to_save.columns[0] != "timestamp":
                    df_to_save = df_to_save.rename(columns={df_to_save.columns[0]: "timestamp"})
                self.storage.insert_bars(df_to_save, symbol)

            except Exception as e:
                logger.error(f"Failed to fetch/process {symbol}: {e}")
                continue

        if not all_features:
            raise RuntimeError("No valid data fetched for any symbol. Check API keys and symbols.")

        # 3. Build training environment
        train_env = TradingEnv(
            data=all_features,
            prices=all_prices,
            config=self.settings,
            mode="train",
            initial_capital=self.settings.trading.initial_capital,
        )

        # 4. Initialize agent with environment
        if not hasattr(self.agent, "model") or self.agent.model is None:
            logger.info("Initializing new RL model...")
            self.agent = RLAgent(train_env, self.settings)
        else:
            self.agent.model.set_env(
                __import__("stable_baselines3.common.vec_env", fromlist=["DummyVecEnv"]).DummyVecEnv([lambda: train_env])
            )

        # 5. Train
        logger.info(f"Training for {timesteps:,} timesteps...")
        self.agent.train(total_timesteps=timesteps)
        self._total_training_steps += timesteps

        # 6. Evaluate
        eval_env = TradingEnv(
            data=all_features,
            prices=all_prices,
            config=self.settings,
            mode="eval",
            initial_capital=self.settings.trading.initial_capital,
        )
        result = self.evaluator.run_backtest(self.agent, eval_env, n_episodes=3)
        self.evaluator.print_report(result)

        # 7. Save if best
        self._maybe_save_checkpoint(result)

        logger.info(
            f"Pre-training complete. Sharpe={result.sharpe:.3f}, "
            f"Return={result.total_return*100:.2f}%, "
            f"WinRate={result.win_rate*100:.1f}%"
        )
        return result

    # ------------------------------------------------------------------ #
    #  Phase 2: Continuous / online learning                               #
    # ------------------------------------------------------------------ #

    def on_new_bar(self, symbol: str, bar: dict):
        """
        Called by the data collector whenever a new live bar arrives.
        Accumulates bars and triggers retraining when threshold reached.
        """
        self._live_bars[symbol].append(bar)
        self._bar_count += 1

        retrain_every = self.settings.continuous_training.retrain_every_n_bars
        if self._bar_count % retrain_every == 0:
            self._continuous_retrain_check()

    def _continuous_retrain_check(self):
        """Evaluate rolling performance; retrain if needed."""
        if not self.settings.continuous_training.enabled:
            return

        logger.info(f"Continuous training check at bar #{self._bar_count}")

        # Build updated feature arrays from accumulated live bars
        updated_features = {}
        updated_prices = {}

        for symbol, bars in self._live_bars.items():
            if len(bars) < self.settings.model.lookback_window + 10:
                continue
            try:
                import pandas as pd
                df = pd.DataFrame(bars)
                df = df.sort_values("timestamp").reset_index(drop=True)
                features, prices = self.feature_engineer.compute_features(df)
                if features is not None and len(features) > 0:
                    updated_features[symbol] = features
                    updated_prices[symbol] = prices
            except Exception as e:
                logger.warning(f"Feature update failed for {symbol}: {e}")

        if not updated_features:
            return

        # Quick evaluation on recent data
        eval_env = TradingEnv(
            data=updated_features,
            prices=updated_prices,
            config=self.settings,
            mode="eval",
            initial_capital=self.settings.trading.initial_capital,
        )
        result = self.evaluator.run_backtest(self.agent, eval_env, n_episodes=2)
        rolling_sharpe = result.sharpe
        logger.info(f"Rolling Sharpe: {rolling_sharpe:.3f} (threshold: {self.settings.continuous_training.performance_threshold})")

        # If performance dropped, do an online update
        if rolling_sharpe < self.settings.continuous_training.performance_threshold:
            logger.info("Performance below threshold — triggering online update...")
            new_env = TradingEnv(
                data=updated_features,
                prices=updated_prices,
                config=self.settings,
                mode="train",
                initial_capital=self.settings.trading.initial_capital,
            )
            additional_steps = min(50_000, self.settings.model.training_timesteps // 10)
            self.agent.update_online(new_env, additional_timesteps=additional_steps)
            self._total_training_steps += additional_steps
            self._last_retrain_time = datetime.now()

            # Re-evaluate after update
            result = self.evaluator.run_backtest(self.agent, eval_env, n_episodes=2)
            logger.info(f"Post-update Sharpe: {result.sharpe:.3f}")

        self._maybe_save_checkpoint(result)

    async def continuous_train_loop(self):
        """
        Async loop for continuous training. Run this as a background task.
        Sleeps between checks so it doesn't hog the CPU.
        """
        logger.info("Continuous training loop started")
        while True:
            await asyncio.sleep(60)  # check every minute

            retrain_every = self.settings.continuous_training.retrain_every_n_bars
            if self._bar_count > 0 and self._bar_count % retrain_every == 0:
                self._continuous_retrain_check()

    # ------------------------------------------------------------------ #
    #  Checkpoint management                                               #
    # ------------------------------------------------------------------ #

    def _maybe_save_checkpoint(self, result: BacktestResult):
        """Save model if this is the best Sharpe so far."""
        if result.sharpe > self._best_sharpe:
            self._best_sharpe = result.sharpe
            path = self._checkpoint_path(result)
            self.agent.save(str(path))
            self.storage.save_model_snapshot(
                version=f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                path=str(path),
                sharpe=result.sharpe,
                win_rate=result.win_rate,
                total_return=result.total_return,
            )
            logger.info(f"New best model saved: Sharpe={result.sharpe:.3f} -> {path}")

    def _checkpoint_path(self, result: BacktestResult) -> Path:
        models_dir = Path(self.settings.paths.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sharpe_str = f"{result.sharpe:.2f}".replace("-", "neg")
        return models_dir / f"ppo_sharpe{sharpe_str}_{ts}"

    def load_best_model(self) -> bool:
        """Load the best saved model checkpoint from DB."""
        snapshot = self.storage.get_latest_snapshot()
        if snapshot is None:
            logger.warning("No saved model snapshots found")
            return False
        try:
            self.agent.load(snapshot["path"])
            self._best_sharpe = snapshot.get("sharpe", -float("inf"))
            logger.info(f"Loaded best model: {snapshot['path']} (Sharpe={self._best_sharpe:.3f})")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    @property
    def stats(self) -> dict:
        return {
            "total_training_steps": self._total_training_steps,
            "best_sharpe": self._best_sharpe,
            "bar_count": self._bar_count,
            "last_retrain": self._last_retrain_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
