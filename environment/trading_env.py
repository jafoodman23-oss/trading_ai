"""
Custom Gymnasium trading environment for reinforcement learning.

Observation space: (lookback_window, n_market_features + n_portfolio_features)
    portfolio_features: [cash_pct, position_pct, unrealized_pnl_pct, n_bars_held_norm]

Action space: Discrete(3)
    0 = hold
    1 = buy (go full long, size by available capital)
    2 = sell (close position / go flat)

Reward: log_return - transaction_cost_penalty - risk_penalty
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Multi-symbol trading environment.

    Parameters
    ----------
    data : dict mapping symbol -> np.ndarray of shape (n_bars, n_features)
    config : Settings object (uses config.trading and config.model)
    mode : "train" or "eval"
    """

    metadata = {"render_modes": ["human", "ansi"]}

    N_PORTFOLIO_FEATURES = 4  # cash_pct, position_pct, unrealized_pnl_pct, bars_held_norm

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        prices: Dict[str, np.ndarray],  # raw close prices aligned with data arrays
        config,
        mode: str = "train",
        initial_capital: float = 100_000.0,
    ):
        super().__init__()

        self.data = data          # symbol -> feature array
        self.prices = prices      # symbol -> close price array (aligned with features)
        self.config = config
        self.mode = mode
        self.initial_capital = initial_capital

        self.symbols = list(data.keys())
        self.n_symbols = len(self.symbols)
        self.lookback = config.model.lookback_window

        # Validate alignment
        for sym in self.symbols:
            assert len(self.data[sym]) == len(self.prices[sym]), \
                f"Data/price length mismatch for {sym}: {len(self.data[sym])} vs {len(self.prices[sym])}"
            assert len(self.data[sym]) >= self.lookback, \
                f"Not enough bars for {sym}: need {self.lookback}, got {len(self.data[sym])}"

        # Pick the symbol with the most data as primary
        self.primary_symbol = max(self.symbols, key=lambda s: len(self.data[s]))
        self.n_market_features = self.data[self.primary_symbol].shape[1]
        self.n_obs_features = self.n_market_features + self.N_PORTFOLIO_FEATURES

        # Commission and slippage
        self.commission_rate = getattr(config.trading, 'commission_rate', 0.0001)
        self.slippage_rate = getattr(config.trading, 'slippage_rate', 0.0001)

        # Action / observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.lookback, self.n_obs_features),
            dtype=np.float32,
        )

        # State variables (initialized in reset)
        self.current_step: int = 0
        self.cash: float = initial_capital
        self.position_qty: float = 0.0
        self.position_entry_price: float = 0.0
        self.bars_held: int = 0
        self.equity_curve: List[float] = []
        self.trades: List[dict] = []
        self._current_symbol_idx: int = 0

        # Track which symbol is active for this episode
        self._episode_symbol: str = self.primary_symbol
        self._episode_data: np.ndarray = self.data[self.primary_symbol]
        self._episode_prices: np.ndarray = self.prices[self.primary_symbol]
        self._episode_len: int = len(self._episode_prices)

    # ------------------------------------------------------------------ #
    #  Gymnasium interface                                                 #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # In training mode, randomly pick a symbol and start position
        if self.mode == "train" and self.n_symbols > 1:
            self._episode_symbol = self.np_random.choice(self.symbols)
        else:
            self._episode_symbol = self.primary_symbol

        self._episode_data = self.data[self._episode_symbol]
        self._episode_prices = self.prices[self._episode_symbol]
        self._episode_len = len(self._episode_prices)

        # In training mode, randomly start within the data to improve coverage
        if self.mode == "train":
            max_start = max(self._episode_len - self.lookback - 1, self.lookback)
            start = int(self.np_random.integers(self.lookback, max(self.lookback + 1, max_start)))
        else:
            start = self.lookback

        self.current_step = start
        self.cash = self.initial_capital
        self.position_qty = 0.0
        self.position_entry_price = 0.0
        self.bars_held = 0
        self.equity_curve = [self.initial_capital]
        self.trades = []

        obs = self._get_obs()
        info = {"symbol": self._episode_symbol, "start_step": start}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_portfolio_value = self._portfolio_value()
        current_price = self._current_price()

        reward = 0.0
        trade_cost = 0.0

        # ---- Execute action -----------------------------------------------
        if action == 1:  # BUY
            if self.position_qty == 0 and self.cash > 0:
                # Calculate buy price with slippage
                buy_price = current_price * (1 + self.slippage_rate)
                max_shares = self.cash / buy_price
                # Round down to whole shares
                shares = max(0.0, max_shares)
                if shares > 0:
                    cost = shares * buy_price
                    commission = cost * self.commission_rate
                    total_cost = cost + commission
                    if total_cost <= self.cash:
                        self.cash -= total_cost
                        self.position_qty = shares
                        self.position_entry_price = buy_price
                        self.bars_held = 0
                        trade_cost = commission
                        self.trades.append({
                            "step": self.current_step,
                            "action": "buy",
                            "price": buy_price,
                            "qty": shares,
                            "commission": commission,
                        })

        elif action == 2:  # SELL
            if self.position_qty > 0:
                sell_price = current_price * (1 - self.slippage_rate)
                proceeds = self.position_qty * sell_price
                commission = proceeds * self.commission_rate
                net_proceeds = proceeds - commission
                pnl = net_proceeds - (self.position_qty * self.position_entry_price)
                self.cash += net_proceeds
                trade_cost = commission
                self.trades.append({
                    "step": self.current_step,
                    "action": "sell",
                    "price": sell_price,
                    "qty": self.position_qty,
                    "commission": commission,
                    "pnl": pnl,
                })
                self.position_qty = 0.0
                self.position_entry_price = 0.0
                self.bars_held = 0

        # ---- Update position tracking ------------------------------------
        if self.position_qty > 0:
            self.bars_held += 1

        # ---- Advance step -----------------------------------------------
        self.current_step += 1

        # ---- Calculate reward -------------------------------------------
        new_portfolio_value = self._portfolio_value()
        self.equity_curve.append(new_portfolio_value)
        reward = self._calculate_reward(prev_portfolio_value, new_portfolio_value, action, trade_cost)

        # ---- Check terminal conditions ----------------------------------
        terminated = self.current_step >= self._episode_len - 1
        truncated = False

        # Force close position at end of episode
        if terminated and self.position_qty > 0:
            final_price = self._current_price() * (1 - self.slippage_rate)
            proceeds = self.position_qty * final_price
            commission = proceeds * self.commission_rate
            self.cash += proceeds - commission
            self.position_qty = 0.0

        obs = self._get_obs()
        info = {
            "portfolio_value": new_portfolio_value,
            "cash": self.cash,
            "position_qty": self.position_qty,
            "n_trades": len(self.trades),
            "bars_held": self.bars_held,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        price = self._current_price()
        pv = self._portfolio_value()
        unrealized = (
            self.position_qty * (price - self.position_entry_price)
            if self.position_qty > 0 else 0.0
        )
        total_return = (pv / self.initial_capital - 1) * 100

        msg = (
            f"Step {self.current_step:5d} | {self._episode_symbol} | "
            f"Price: ${price:8.2f} | "
            f"Cash: ${self.cash:10.2f} | "
            f"Pos: {self.position_qty:8.2f} | "
            f"Unrealized: ${unrealized:8.2f} | "
            f"Portfolio: ${pv:10.2f} | "
            f"Return: {total_return:+.2f}%"
        )

        if mode == "human":
            print(msg)
        return msg

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _current_price(self) -> float:
        idx = min(self.current_step, len(self._episode_prices) - 1)
        return float(self._episode_prices[idx])

    def _portfolio_value(self) -> float:
        price = self._current_price()
        position_value = self.position_qty * price
        return self.cash + position_value

    def _unrealized_pnl_pct(self) -> float:
        if self.position_qty <= 0 or self.position_entry_price <= 0:
            return 0.0
        current_price = self._current_price()
        return (current_price - self.position_entry_price) / self.position_entry_price

    def _get_obs(self) -> np.ndarray:
        """
        Build the observation window: (lookback, n_market_features + n_portfolio_features).
        Market features are taken from the feature array.
        Portfolio features are appended as the last 4 columns.
        """
        # Market features window
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = self.current_step

        market_window = self._episode_data[start_idx:end_idx]

        # Pad with zeros if we don't have enough history
        if len(market_window) < self.lookback:
            pad = np.zeros((self.lookback - len(market_window), self.n_market_features), dtype=np.float32)
            market_window = np.vstack([pad, market_window])

        # Portfolio features (same for all timesteps in the window)
        pv = self._portfolio_value()
        cash_pct = self.cash / pv if pv > 0 else 1.0
        position_value = self.position_qty * self._current_price()
        position_pct = position_value / pv if pv > 0 else 0.0
        unrealized_pnl_pct = self._unrealized_pnl_pct()
        bars_held_norm = min(self.bars_held / 100.0, 1.0)

        portfolio_features = np.array(
            [cash_pct, position_pct, unrealized_pnl_pct, bars_held_norm],
            dtype=np.float32,
        )

        # Broadcast portfolio features across the lookback window
        portfolio_window = np.tile(portfolio_features, (self.lookback, 1))

        obs = np.concatenate([market_window, portfolio_window], axis=1).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _calculate_reward(
        self,
        prev_value: float,
        curr_value: float,
        action: int,
        trade_cost: float,
    ) -> float:
        """
        Reward = log_return - transaction_cost_penalty - risk_penalty

        risk_penalty: penalizes high volatility of recent returns to encourage
                      stable, risk-adjusted growth.
        """
        if prev_value <= 0:
            return 0.0

        # Log return of portfolio
        log_return = np.log(curr_value / prev_value) if curr_value > 0 else -1.0

        # Transaction cost penalty (encourage efficient trading)
        cost_penalty = trade_cost / prev_value if trade_cost > 0 else 0.0

        # Volatility penalty (risk-adjusted reward)
        risk_penalty = 0.0
        if len(self.equity_curve) >= 20:
            recent = np.array(self.equity_curve[-20:])
            returns = np.diff(recent) / recent[:-1]
            vol = float(np.std(returns))
            risk_penalty = vol * 0.1  # penalty coefficient

        # Holding penalty: small negative reward for doing nothing
        holding_penalty = 0.001 if action == 0 and self.position_qty == 0 else 0.0

        reward = log_return - cost_penalty - risk_penalty - holding_penalty
        return float(np.clip(reward, -1.0, 1.0))

    # ------------------------------------------------------------------ #
    #  Episode summary                                                     #
    # ------------------------------------------------------------------ #

    def episode_summary(self) -> dict:
        """Return key metrics for the completed episode."""
        equity = np.array(self.equity_curve)
        total_return = (equity[-1] / equity[0] - 1) if len(equity) > 1 else 0.0

        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0.0])
        sharpe = (
            float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 390))
            if np.std(returns) > 0 else 0.0
        )

        sell_trades = [t for t in self.trades if t["action"] == "sell"]
        wins = sum(1 for t in sell_trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(sell_trades) if sell_trades else 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
            "n_trades": len(sell_trades),
            "final_equity": float(equity[-1]) if len(equity) > 0 else self.initial_capital,
        }
