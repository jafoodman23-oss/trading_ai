"""
Ensemble agent that combines RL (PPO) and Pattern Detector signals.
Weights are dynamically adjusted based on each model's recent performance.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .base_agent import BaseAgent
from .rl_agent import RLAgent
from .pattern_detector import PatternDetector


class ModelPerformanceTracker:
    """
    Tracks rolling prediction accuracy and P&L attribution for a single model.
    Used to update ensemble weights.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._correct: Deque[bool] = deque(maxlen=window)
        self._pnl: Deque[float] = deque(maxlen=window)

    def record(self, was_correct: bool, pnl: float = 0.0):
        self._correct.append(was_correct)
        self._pnl.append(pnl)

    @property
    def accuracy(self) -> float:
        if not self._correct:
            return 0.5
        return sum(self._correct) / len(self._correct)

    @property
    def avg_pnl(self) -> float:
        if not self._pnl:
            return 0.0
        return float(np.mean(self._pnl))

    @property
    def score(self) -> float:
        """Combined performance score used for weight calculation."""
        return 0.6 * self.accuracy + 0.4 * max(0.0, self.avg_pnl + 0.5)

    def __len__(self) -> int:
        return len(self._correct)


class EnsembleAgent(BaseAgent):
    """
    Combines RL agent and Pattern Detector via adaptive weighted voting.

    Weight update rule:
        After each trade outcome is known, score each model by its prediction accuracy.
        Weights = softmax(scores) with a minimum floor to avoid degenerate solutions.

    Signal aggregation:
        Each model outputs action probabilities (3 classes).
        Weighted average of probabilities -> argmax -> final action.
    """

    MIN_WEIGHT = 0.1   # floor to prevent any model from being ignored
    WEIGHT_UPDATE_FREQ = 50  # update weights every N predictions

    def __init__(
        self,
        rl_agent: RLAgent,
        pattern_detector: PatternDetector,
        rl_weight: float = 0.6,
        pattern_weight: float = 0.4,
    ):
        super().__init__(name="EnsembleAgent")
        self.rl_agent = rl_agent
        self.pattern_detector = pattern_detector

        # Normalize weights
        total = rl_weight + pattern_weight
        self._rl_weight = rl_weight / total
        self._pattern_weight = pattern_weight / total

        self._rl_tracker = ModelPerformanceTracker(window=200)
        self._pattern_tracker = ModelPerformanceTracker(window=200)

        self._prediction_count = 0

        # History for feedback
        self._last_predictions: List[Dict] = []

        logger.info(
            f"EnsembleAgent initialized: RL={self._rl_weight:.2f}, "
            f"Pattern={self._pattern_weight:.2f}"
        )

    @property
    def rl_weight(self) -> float:
        return self._rl_weight

    @property
    def pattern_weight(self) -> float:
        return self._pattern_weight

    def predict(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Combine predictions from both models using weighted voting.

        Returns
        -------
        (action, confidence)
        """
        # Get action probabilities from each model
        rl_probs = self._get_rl_probs(obs)
        pattern_probs = self._get_pattern_probs(obs)

        # Weighted ensemble
        combined_probs = self._rl_weight * rl_probs + self._pattern_weight * pattern_probs

        # Final decision
        action = int(np.argmax(combined_probs))
        confidence = float(combined_probs[action])

        # Store for later feedback
        self._last_predictions.append({
            "action": action,
            "rl_action": int(np.argmax(rl_probs)),
            "pattern_action": int(np.argmax(pattern_probs)),
            "rl_probs": rl_probs.tolist(),
            "pattern_probs": pattern_probs.tolist(),
            "combined_probs": combined_probs.tolist(),
        })

        self._prediction_count += 1
        if self._prediction_count % self.WEIGHT_UPDATE_FREQ == 0:
            self._update_weights()

        return action, confidence

    def record_outcome(self, action_taken: int, pnl: float):
        """
        Feed back the outcome of a trade to update model weights.

        Parameters
        ----------
        action_taken : the actual action that was executed
        pnl : the profit/loss resulting from this action (0 for holds)
        """
        if not self._last_predictions:
            return

        last = self._last_predictions[-1]
        was_profitable = pnl > 0 if action_taken in (1, 2) else pnl >= 0

        # Give credit to each model based on whether it agreed with the profitable action
        rl_correct = (last["rl_action"] == action_taken) and was_profitable
        pattern_correct = (last["pattern_action"] == action_taken) and was_profitable

        self._rl_tracker.record(rl_correct, pnl)
        self._pattern_tracker.record(pattern_correct, pnl)

    def _get_rl_probs(self, obs: np.ndarray) -> np.ndarray:
        """Get RL agent action probability distribution."""
        if not self.rl_agent.is_trained:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

        try:
            import torch
            obs_batch = obs[np.newaxis, ...]
            obs_tensor = torch.FloatTensor(obs_batch).to(self.rl_agent.model.device)
            with torch.no_grad():
                dist = self.rl_agent.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
            return probs.astype(np.float32)
        except Exception as e:
            logger.debug(f"RL prob extraction failed: {e}, using predict")
            action, confidence = self.rl_agent.predict(obs)
            probs = np.full(3, (1 - confidence) / 2, dtype=np.float32)
            probs[action] = confidence
            return probs

    def _get_pattern_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        Get pattern detector probability distribution.
        Pattern detector outputs [P(down), P(flat), P(up)].
        Map to [P(hold), P(buy), P(sell)].
        """
        if not self.pattern_detector.is_trained:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

        try:
            # obs is (lookback, n_features) — take only market features (exclude portfolio)
            # The pattern detector was trained on market features only
            n_market = self.pattern_detector.net.input_norm.normalized_shape[0]
            market_obs = obs[:, :n_market]

            raw_probs = self.pattern_detector.predict_proba(market_obs)
            # raw_probs: [P(down), P(flat), P(up)]
            # Map to trading actions: down->sell, flat->hold, up->buy
            # Actions: 0=hold, 1=buy, 2=sell
            mapped_probs = np.array([
                raw_probs[1],  # hold <- flat
                raw_probs[2],  # buy  <- up
                raw_probs[0],  # sell <- down
            ], dtype=np.float32)
            return mapped_probs
        except Exception as e:
            logger.debug(f"Pattern prob extraction failed: {e}")
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    def _update_weights(self):
        """
        Update model weights based on rolling performance scores.
        Uses softmax with temperature, ensuring minimum weight floor.
        """
        rl_score = self._rl_tracker.score
        pattern_score = self._pattern_tracker.score

        # Only update if we have enough data
        if len(self._rl_tracker) < 20:
            return

        # Softmax with temperature
        temperature = 2.0
        scores = np.array([rl_score, pattern_score]) / temperature
        weights = np.exp(scores - scores.max())
        weights = weights / weights.sum()

        # Apply minimum weight floor
        weights = np.maximum(weights, self.MIN_WEIGHT)
        weights = weights / weights.sum()

        old_rl = self._rl_weight
        self._rl_weight = float(weights[0])
        self._pattern_weight = float(weights[1])

        if abs(self._rl_weight - old_rl) > 0.05:
            logger.debug(
                f"Ensemble weights updated: RL={self._rl_weight:.3f} "
                f"(acc={self._rl_tracker.accuracy:.3f}), "
                f"Pattern={self._pattern_weight:.3f} "
                f"(acc={self._pattern_tracker.accuracy:.3f})"
            )

    @property
    def is_trained(self) -> bool:
        """Ensemble is ready if at least the RL agent is trained."""
        return self.rl_agent.is_trained

    def save(self, path: str):
        """Save both sub-models."""
        from pathlib import Path
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)
        self.rl_agent.save(str(base / "rl_agent"))
        if self.pattern_detector.is_trained:
            self.pattern_detector.save(str(base / "pattern_detector.pt"))
        logger.info(f"EnsembleAgent saved to {base}")

    def load(self, path: str):
        """Load both sub-models."""
        from pathlib import Path
        base = Path(path)
        rl_path = base / "rl_agent.zip"
        if rl_path.exists():
            self.rl_agent.load(str(rl_path))
        pattern_path = base / "pattern_detector.pt"
        if pattern_path.exists():
            self.pattern_detector.load(str(pattern_path))
        logger.info(f"EnsembleAgent loaded from {base}")

    def get_weights(self) -> Dict[str, float]:
        return {"rl": self._rl_weight, "pattern": self._pattern_weight}

    def get_performance_stats(self) -> Dict:
        return {
            "rl_accuracy": self._rl_tracker.accuracy,
            "pattern_accuracy": self._pattern_tracker.accuracy,
            "rl_avg_pnl": self._rl_tracker.avg_pnl,
            "pattern_avg_pnl": self._pattern_tracker.avg_pnl,
            "rl_weight": self._rl_weight,
            "pattern_weight": self._pattern_weight,
            "n_predictions": self._prediction_count,
        }
