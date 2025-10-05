"""Experience replay buffer for DQN training."""

import numpy as np
from collections import deque
import random
from typing import Dict


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int = 100000):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing batched transitions:
                - states: np.ndarray of shape (batch_size, state_dim)
                - actions: np.ndarray of shape (batch_size,)
                - rewards: np.ndarray of shape (batch_size,)
                - next_states: np.ndarray of shape (batch_size, state_dim)
                - dones: np.ndarray of shape (batch_size,)
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t['state'] for t in batch], dtype=np.float32)
        actions = np.array([t['action'] for t in batch], dtype=np.int64)
        rewards = np.array([t['reward'] for t in batch], dtype=np.float32)
        next_states = np.array([t['next_state'] for t in batch], dtype=np.float32)
        dones = np.array([t['done'] for t in batch], dtype=np.float32)

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer (optional enhancement).

    Uses prioritized sampling based on TD error for more efficient learning.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = None
    ):
        """Add a transition to the buffer with priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            priority: Priority value (defaults to max priority if None)
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        if priority is None:
            priority = max_priority

        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample a batch of transitions based on priorities.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (batch_dict, indices, weights):
                - batch_dict: Dictionary containing batched transitions
                - indices: Indices of sampled transitions
                - weights: Importance sampling weights
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

        # Calculate importance sampling weights
        beta = self._compute_beta()
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        # Get batch
        batch = [self.buffer[idx] for idx in indices]
        states = np.array([t['state'] for t in batch], dtype=np.float32)
        actions = np.array([t['action'] for t in batch], dtype=np.int64)
        rewards = np.array([t['reward'] for t in batch], dtype=np.float32)
        next_states = np.array([t['next_state'] for t in batch], dtype=np.float32)
        dones = np.array([t['done'] for t in batch], dtype=np.float32)

        batch_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

        self.frame += 1
        return batch_dict, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # small epsilon to avoid zero priority

    def _compute_beta(self) -> float:
        """Compute current beta value for importance sampling."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size

    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
