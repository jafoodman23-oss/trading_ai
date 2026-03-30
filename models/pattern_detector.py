"""
PyTorch Bidirectional LSTM pattern detector.
Classifies market states as: 0=down, 1=flat, 2=up.
Used as a complementary signal alongside the RL agent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from .base_agent import BaseAgent


class PatternDetectorNet(nn.Module):
    """
    Bidirectional LSTM for market pattern classification.

    Input:  (batch, seq_len, input_size)
    Output: (batch, output_size) — log probabilities over [down, flat, up]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention mechanism over LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        log_probs : (batch, output_size)
        """
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention-weighted sum over time steps
        attn_scores = self.attention(lstm_out)   # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        logits = self.classifier(context)
        return F.log_softmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (not log-probs)."""
        log_probs = self.forward(x)
        return torch.exp(log_probs)


class PatternDetector(BaseAgent):
    """
    Wrapper around PatternDetectorNet with training and inference utilities.

    Labels are derived from future returns:
        0 (down)  : next N-bar return < -threshold
        1 (flat)  : |next N-bar return| < threshold
        2 (up)    : next N-bar return > threshold
    """

    LABEL_DOWN = 0
    LABEL_FLAT = 1
    LABEL_UP = 2
    RETURN_THRESHOLD = 0.005  # 0.5% threshold for up/down labeling
    LOOKAHEAD = 5             # bars ahead for label computation

    def __init__(
        self,
        config,
        input_size: Optional[int] = None,
        device: str = "auto",
    ):
        super().__init__(name="PatternDetector")
        self.config = config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        input_size = input_size or config.data.n_features

        self.net = PatternDetectorNet(
            input_size=input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.lstm_layers,
            output_size=3,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=config.model.learning_rate,
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        logger.info(
            f"PatternDetector initialized: {sum(p.numel() for p in self.net.parameters()):,} params"
        )

    def _make_labels(self, prices: np.ndarray, lookahead: int = None) -> np.ndarray:
        """
        Compute integer class labels from price array based on future returns.

        Parameters
        ----------
        prices : np.ndarray of shape (n,)

        Returns
        -------
        labels : np.ndarray of shape (n - lookahead,) with values in {0, 1, 2}
        """
        lh = lookahead or self.LOOKAHEAD
        n = len(prices)
        labels = np.ones(n - lh, dtype=np.int64)  # default = flat

        for i in range(n - lh):
            future_return = (prices[i + lh] - prices[i]) / prices[i]
            if future_return > self.RETURN_THRESHOLD:
                labels[i] = self.LABEL_UP
            elif future_return < -self.RETURN_THRESHOLD:
                labels[i] = self.LABEL_DOWN

        return labels

    def fit(
        self,
        X: np.ndarray,
        prices: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> dict:
        """
        Train the pattern detector.

        Parameters
        ----------
        X : np.ndarray of shape (n, seq_len, n_features) — feature windows
        prices : np.ndarray of shape (n,) — close prices (for label generation)
        epochs : training epochs
        batch_size : mini-batch size
        val_split : fraction of data for validation

        Returns
        -------
        dict with training history
        """
        lh = self.LOOKAHEAD
        labels = self._make_labels(prices, lh)
        n = len(labels)

        # Trim X to match labels
        X_trimmed = X[:n]

        # Train/val split
        val_n = max(1, int(n * val_split))
        X_train, X_val = X_trimmed[:-val_n], X_trimmed[-val_n:]
        y_train, y_val = labels[:-val_n], labels[-val_n:]

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Class weights for imbalance handling
        class_counts = np.bincount(y_train, minlength=3)
        total = len(y_train)
        weights = torch.FloatTensor([total / max(c, 1) for c in class_counts]).to(self.device)
        criterion = nn.NLLLoss(weight=weights)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        self.net.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                log_probs = self.net(X_batch)
                loss = criterion(log_probs, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_log_probs = self.net(X_val_t)
                val_loss = criterion(val_log_probs, y_val_t).item()
                val_preds = val_log_probs.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()
            self.net.train()

            self.scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"PatternDetector Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.3f}"
                )

        self._is_trained = True
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for a batch of windows.

        Parameters
        ----------
        X : np.ndarray of shape (n, seq_len, n_features) or (seq_len, n_features)

        Returns
        -------
        probs : np.ndarray of shape (n, 3) or (3,) — [P(down), P(flat), P(up)]
        """
        single = X.ndim == 2
        if single:
            X = X[np.newaxis, ...]

        self.net.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            probs = self.net.predict_proba(X_t).cpu().numpy()

        return probs[0] if single else probs

    def predict(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Predict action from observation.
        Maps class labels to trading actions:
            down -> sell (2)
            flat -> hold (0)
            up   -> buy (1)
        """
        if not self._is_trained:
            return 0, 0.33

        probs = self.predict_proba(obs)
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        # Map: 0=down->sell(2), 1=flat->hold(0), 2=up->buy(1)
        action_map = {self.LABEL_DOWN: 2, self.LABEL_FLAT: 0, self.LABEL_UP: 1}
        action = action_map[class_idx]
        return action, confidence

    def save(self, path: str):
        """Save model weights and optimizer state."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "is_trained": self._is_trained,
        }, str(path_obj))
        logger.info(f"PatternDetector saved to {path_obj}")

    def load(self, path: str):
        """Load model weights from disk."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(str(path_obj), map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._is_trained = checkpoint.get("is_trained", True)
        self.net.eval()
        logger.info(f"PatternDetector loaded from {path_obj}")
