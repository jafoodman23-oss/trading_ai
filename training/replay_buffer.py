"""
Prioritized Experience Replay Buffer.
Uses numpy arrays for fast storage and sampling.
Priorities are maintained in a segment tree for O(log n) updates and sampling.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class SegmentTree:
    """
    Segment tree for efficient O(log n) priority updates and sum queries.
    Used to implement proportional prioritized replay.
    """

    def __init__(self, capacity: int, operation, neutral_element):
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        self.tree = np.full(2 * capacity, neutral_element, dtype=np.float64)

    def _propagate(self, idx: int):
        parent = idx // 2
        while parent >= 1:
            left = parent * 2
            right = parent * 2 + 1
            self.tree[parent] = self.operation(self.tree[left], self.tree[right])
            parent //= 2

    def update(self, idx: int, value: float):
        """Update a leaf node."""
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def query(self, left: int, right: int) -> float:
        """Query [left, right) range."""
        result = self.neutral_element
        left += self.capacity
        right += self.capacity
        while left < right:
            if left & 1:
                result = self.operation(result, self.tree[left])
                left += 1
            if right & 1:
                right -= 1
                result = self.operation(result, self.tree[right])
            left >>= 1
            right >>= 1
        return result

    def total(self) -> float:
        return self.tree[1] if self.capacity > 0 else self.neutral_element

    def retrieve(self, value: float) -> int:
        """Find the index for a given cumulative sum value."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = 2 * idx + 1
            if self.tree[left] > value:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - self.capacity


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay buffer.

    Stores (state, action, reward, next_state, done) tuples and samples
    them with probability proportional to their TD error priority.

    Parameters
    ----------
    capacity : maximum number of transitions to store
    alpha : priority exponent (0 = uniform, 1 = full prioritization)
    beta : importance sampling correction exponent (0 = no correction, 1 = full)
    beta_increment : increment beta toward 1.0 each sample
    eps : small constant added to priorities to ensure non-zero probability
    """

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple,
        action_dim: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps

        # Priority trees
        self._sum_tree = SegmentTree(capacity, lambda a, b: a + b, 0.0)
        self._min_tree = SegmentTree(capacity, min, float("inf"))

        # Storage arrays (pre-allocated)
        self._states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.int32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)

        self._write_pos = 0
        self._size = 0
        self._max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None,
    ):
        """
        Add a transition to the buffer.

        Parameters
        ----------
        state : observation of shape state_shape
        action : action taken
        reward : received reward
        next_state : next observation
        done : episode terminal flag
        td_error : initial TD error (uses max priority if None)
        """
        priority = (
            (abs(td_error) + self.eps) ** self.alpha
            if td_error is not None
            else self._max_priority
        )

        idx = self._write_pos
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._dones[idx] = done

        self._sum_tree.update(idx, priority)
        self._min_tree.update(idx, priority)
        self._max_priority = max(self._max_priority, priority)

        self._write_pos = (self._write_pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions with prioritized probability.

        Returns
        -------
        (states, actions, rewards, next_states, dones, weights, indices)
        weights : importance sampling weights for bias correction
        indices : buffer indices for priority updates
        """
        assert self._size >= batch_size, "Buffer has fewer samples than batch_size"

        indices = self._sample_indices(batch_size)

        states = self._states[indices]
        actions = self.actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        weights = self._compute_weights(indices)

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, weights, indices

    @property
    def actions(self) -> np.ndarray:
        return self._actions.squeeze(-1) if self.action_dim == 1 else self._actions

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        """Sample indices proportional to priorities."""
        total = self._sum_tree.total()
        segment = total / batch_size
        indices = np.zeros(batch_size, dtype=np.int64)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            val = np.random.uniform(low, high)
            indices[i] = self._sum_tree.retrieve(val)

        return np.clip(indices, 0, self._size - 1)

    def _compute_weights(self, indices: np.ndarray) -> np.ndarray:
        """Compute importance sampling weights for bias correction."""
        min_prob = self._min_tree.total() / self._sum_tree.total()
        max_weight = (min_prob * self._size) ** (-self.beta)

        weights = np.zeros(len(indices), dtype=np.float32)
        for i, idx in enumerate(indices):
            priority = self._sum_tree.query(idx, idx + 1)
            prob = priority / self._sum_tree.total()
            weight = (prob * self._size) ** (-self.beta)
            weights[i] = weight / max_weight

        return weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled transitions based on new TD errors.

        Parameters
        ----------
        indices : buffer indices from the last sample() call
        td_errors : new TD errors for those transitions
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self.eps) ** self.alpha
            self._sum_tree.update(int(idx), priority)
            self._min_tree.update(int(idx), priority)
            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True when buffer has at least one full batch available."""
        return self._size > 0

    @property
    def fill_ratio(self) -> float:
        """Fraction of capacity used."""
        return self._size / self.capacity

    def clear(self):
        """Reset the buffer."""
        self._size = 0
        self._write_pos = 0
        self._max_priority = 1.0
        self._states[:] = 0
        self._actions[:] = 0
        self._rewards[:] = 0
        self._next_states[:] = 0
        self._dones[:] = False
