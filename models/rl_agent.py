"""
Reinforcement Learning agent using PPO (Proximal Policy Optimization)
from Stable-Baselines3 with a custom Conv1D + LSTM feature extractor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.error("stable-baselines3 not installed — RL agent disabled")

from .base_agent import BaseAgent


class TradingFeatureExtractor(BaseFeaturesExtractor if SB3_AVAILABLE else object):
    """
    Custom feature extractor: Conv1D -> LSTM -> FC
    Designed to capture temporal patterns in the (lookback, n_features) input.

    Architecture:
        Input:  (batch, lookback, n_features)
        Conv1D: extract local patterns (3 filters of varying size)
        LSTM:   capture sequential dependencies
        FC:     project to features_dim
    """

    def __init__(self, observation_space, features_dim: int = 256):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required")

        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[1]  # n_features
        seq_len = observation_space.shape[0]           # lookback

        # 1D Convolution over time axis (transpose to batch, channels, seq)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # LSTM to capture long-range temporal dependencies
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, lookback, n_features)
        # Conv1d expects (batch, channels, seq_len)
        x = observations.transpose(1, 2)          # -> (batch, n_features, lookback)
        x = self.conv_block(x)                    # -> (batch, 64, lookback)
        x = x.transpose(1, 2)                     # -> (batch, lookback, 64)
        lstm_out, (h_n, _) = self.lstm(x)         # h_n: (num_layers, batch, hidden)
        x = h_n[-1]                               # last layer hidden state: (batch, hidden)
        x = self.fc(x)                            # -> (batch, features_dim)
        return x


class RLAgent(BaseAgent):
    """
    PPO-based RL agent with a custom temporal feature extractor.

    The same agent works for training, paper trading, and live trading
    by simply swapping the underlying environment.
    """

    def __init__(self, env, config, device: str = "auto"):
        super().__init__(name="PPO_TradingAgent")
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required")

        self.config = config
        self.device = device

        policy_kwargs = dict(
            features_extractor_class=TradingFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=nn.ReLU,
        )

        # Wrap in DummyVecEnv if not already vectorized
        if not hasattr(env, 'num_envs'):
            vec_env = DummyVecEnv([lambda: env])
        else:
            vec_env = env

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=config.model.learning_rate,
            n_steps=config.model.n_steps,
            batch_size=config.model.batch_size,
            gamma=config.model.gamma,
            gae_lambda=config.model.gae_lambda,
            clip_range=config.model.clip_range,
            ent_coef=config.model.ent_coef,
            vf_coef=config.model.vf_coef,
            max_grad_norm=config.model.max_grad_norm,
            n_epochs=config.model.n_epochs,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
            tensorboard_log=None,
        )

        self._vec_env = vec_env
        logger.info(f"RLAgent initialized (device={device})")

    def train(self, total_timesteps: int, callback=None, progress_bar: bool = True) -> dict:
        """
        Train the agent for the specified number of timesteps.
        Returns a dict with the training info.
        """
        logger.info(f"Starting PPO training: {total_timesteps:,} timesteps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
            reset_num_timesteps=False,
        )
        self._is_trained = True
        logger.info("Training complete")
        return {"timesteps": total_timesteps}

    def predict(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Predict action for a single observation.

        Parameters
        ----------
        obs : np.ndarray of shape (lookback, n_features)

        Returns
        -------
        (action, confidence)  — confidence from action probability distribution
        """
        if not self._is_trained:
            logger.warning("Agent not trained yet — returning random action")
            return int(np.random.randint(0, 3)), 0.33

        # SB3 expects (1, lookback, n_features) for single obs
        obs_batch = obs[np.newaxis, ...]
        action, _states = self.model.predict(obs_batch, deterministic=True)
        action_int = int(action[0])

        # Get action probabilities for confidence
        try:
            obs_tensor = torch.FloatTensor(obs_batch).to(self.model.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
            confidence = float(probs[action_int])
        except Exception:
            confidence = 0.5

        return action_int, confidence

    def predict_batch(self, obs_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Efficient batch prediction."""
        if not self._is_trained:
            n = len(obs_batch)
            return np.zeros(n, dtype=np.int32), np.full(n, 0.33, dtype=np.float32)

        actions, _ = self.model.predict(obs_batch, deterministic=True)
        confidences = np.full(len(actions), 0.5, dtype=np.float32)
        return actions.astype(np.int32), confidences

    def save(self, path: str):
        """Save the PPO model to disk."""
        path = str(Path(path).with_suffix(""))  # SB3 adds .zip
        self.model.save(path)
        logger.info(f"RLAgent saved to {path}.zip")

    def load(self, path: str):
        """Load a PPO model from disk."""
        path_obj = Path(path)
        if not path_obj.suffix:
            path_obj = path_obj.with_suffix(".zip")

        if not path_obj.exists():
            # Try with .zip extension
            path_obj_zip = Path(str(path) + ".zip")
            if path_obj_zip.exists():
                path_obj = path_obj_zip
            else:
                raise FileNotFoundError(f"Model file not found: {path}")

        self.model = PPO.load(str(path_obj), env=self._vec_env, device=self.device)
        self._is_trained = True
        logger.info(f"RLAgent loaded from {path_obj}")

    def update_online(self, new_env, additional_timesteps: int = 50_000):
        """
        Continue training on a new/updated environment.
        Used for continuous self-improvement as new data arrives.
        """
        if not hasattr(new_env, 'num_envs'):
            new_vec_env = DummyVecEnv([lambda: new_env])
        else:
            new_vec_env = new_env

        self.model.set_env(new_vec_env)
        self._vec_env = new_vec_env

        logger.info(f"Continuing training for {additional_timesteps:,} more timesteps")
        self.model.learn(
            total_timesteps=additional_timesteps,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        logger.info("Online update complete")

    def get_policy_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
