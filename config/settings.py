"""
Configuration settings loader using Pydantic models.
Reads from config/config.yaml and provides typed access to all settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class AlpacaConfig(BaseModel):
    paper_api_key: str = "YOUR_PAPER_API_KEY"
    paper_api_secret: str = "YOUR_PAPER_API_SECRET"
    paper_base_url: str = "https://paper-api.alpaca.markets"
    live_api_key: str = "YOUR_LIVE_API_KEY"
    live_api_secret: str = "YOUR_LIVE_API_SECRET"
    live_base_url: str = "https://api.alpaca.markets"

    def get_api_key(self, mode: str) -> str:
        return self.paper_api_key if mode == "paper" else self.live_api_key

    def get_api_secret(self, mode: str) -> str:
        return self.paper_api_secret if mode == "paper" else self.live_api_secret

    def get_base_url(self, mode: str) -> str:
        return self.paper_base_url if mode == "paper" else self.live_base_url


class TradingConfig(BaseModel):
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    commission_bps: float = 1.0
    slippage_bps: float = 1.0

    @property
    def commission_rate(self) -> float:
        return self.commission_bps / 10_000.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0


class ModelConfig(BaseModel):
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    training_timesteps: int = 500_000
    lookback_window: int = 60
    hidden_size: int = 128
    lstm_layers: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10


class RiskConfig(BaseModel):
    max_drawdown_halt: float = 0.15
    daily_loss_limit: float = 0.05
    max_open_positions: int = 5
    kelly_fraction: float = 0.25
    atr_multiplier_stop: float = 2.0
    atr_multiplier_take: float = 4.0


class ContinuousTrainingConfig(BaseModel):
    enabled: bool = True
    retrain_every_n_bars: int = 500
    performance_threshold: float = 0.5
    min_trades_for_eval: int = 10
    eval_window_bars: int = 200


class DataConfig(BaseModel):
    timeframe: str = "1Min"
    historical_days: int = 365
    features: List[str] = Field(default_factory=lambda: [
        "rsi_7", "rsi_14", "rsi_21",
        "macd_line", "macd_signal", "macd_hist",
        "bb_upper_pct", "bb_lower_pct", "bb_bandwidth",
        "ema_9_dist", "ema_21_dist", "ema_50_dist", "ema_200_dist",
        "atr_14_norm",
        "stoch_k", "stoch_d",
        "cci_20", "williams_r", "adx_14",
        "obv_zscore", "volume_ratio",
        "return_1", "return_5", "return_10", "return_20",
        "hl_range_norm", "gap",
    ])

    @property
    def n_features(self) -> int:
        return len(self.features)


class PathsConfig(BaseModel):
    models_dir: str = "checkpoints"
    logs_dir: str = "logs"
    db_path: str = "trading_ai.db"

    def resolve(self, base: Path) -> "PathsConfig":
        """Return a new PathsConfig with all paths resolved relative to base."""
        return PathsConfig(
            models_dir=str(base / self.models_dir),
            logs_dir=str(base / self.logs_dir),
            db_path=str(base / self.db_path),
        )


class Settings(BaseModel):
    mode: str = "paper"
    symbols: List[str] = Field(default_factory=lambda: ["SPY", "AAPL", "TSLA", "QQQ", "NVDA"])
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    continuous_training: ContinuousTrainingConfig = Field(default_factory=ContinuousTrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def load(cls, path: str = "config/config.yaml") -> "Settings":
        """Load settings from YAML file. Resolves paths relative to config file location."""
        config_path = Path(path)
        if not config_path.exists():
            # Try relative to this file's directory
            config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        settings = cls.model_validate(raw)

        # Resolve paths relative to project root (parent of config dir)
        project_root = config_path.parent.parent
        settings.paths = settings.paths.resolve(project_root)

        # Ensure directories exist
        Path(settings.paths.models_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.paths.logs_dir).mkdir(parents=True, exist_ok=True)

        return settings

    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def get_alpaca_credentials(self) -> tuple[str, str, str]:
        """Returns (api_key, api_secret, base_url) for the current mode."""
        return (
            self.alpaca.get_api_key(self.mode),
            self.alpaca.get_api_secret(self.mode),
            self.alpaca.get_base_url(self.mode),
        )
