"""
Abstract base class for all trading agents.
Defines the interface that RLAgent, PatternDetector, and EnsembleAgent must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class BaseAgent(ABC):
    """
    Common interface for trading agents.

    All agents receive observations of shape (lookback, n_features)
    and return a discrete action: 0=hold, 1=buy, 2=sell.
    """

    def __init__(self, name: str):
        self.name = name
        self._is_trained = False

    @abstractmethod
    def predict(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Given an observation, return (action, confidence).

        Parameters
        ----------
        obs : np.ndarray of shape (lookback, n_features)

        Returns
        -------
        action : int — 0=hold, 1=buy, 2=sell
        confidence : float in [0, 1] — model's confidence in the action
        """
        ...

    @abstractmethod
    def save(self, path: str):
        """Persist the model to disk."""
        ...

    @abstractmethod
    def load(self, path: str):
        """Load a previously saved model from disk."""
        ...

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def predict_batch(self, obs_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict actions for a batch of observations.
        Default implementation calls predict() in a loop.
        Subclasses may override for efficiency.

        Parameters
        ----------
        obs_batch : np.ndarray of shape (batch, lookback, n_features)

        Returns
        -------
        actions : np.ndarray of shape (batch,) with int actions
        confidences : np.ndarray of shape (batch,) with float confidences
        """
        actions = []
        confidences = []
        for obs in obs_batch:
            a, c = self.predict(obs)
            actions.append(a)
            confidences.append(c)
        return np.array(actions, dtype=np.int32), np.array(confidences, dtype=np.float32)

    def __repr__(self) -> str:
        trained_str = "trained" if self._is_trained else "untrained"
        return f"{self.__class__.__name__}(name={self.name!r}, {trained_str})"
