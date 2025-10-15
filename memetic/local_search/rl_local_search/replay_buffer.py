"""Experience replay buffer for DQN training."""

import numpy as np
from collections import deque
import random
from typing import Dict, Tuple


# ============================================================================
# STANDARD REPLAY BUFFER (Original - 1-step only)
# ============================================================================
# class ReplayBuffer:
#     """Experience replay buffer for storing and sampling transitions."""
#
#     def __init__(self, capacity: int = 100000):
#         """Initialize the replay buffer.
#
#         Args:
#             capacity: Maximum number of transitions to store
#         """
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity)
#
#     def add(
#         self,
#         state: np.ndarray,
#         action: int,
#         reward: float,
#         next_state: np.ndarray,
#         done: bool
#     ):
#         """Add a transition to the buffer.
#
#         Args:
#             state: Current state
#             action: Action taken
#             reward: Reward received
#             next_state: Next state
#             done: Whether episode is done
#         """
#         self.buffer.append({
#             'state': state,
#             'action': action,
#             'reward': reward,
#             'next_state': next_state,
#             'done': done
#         })
#
#     def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
#         """Sample a batch of transitions from the buffer.
#
#         Args:
#             batch_size: Number of transitions to sample
#
#         Returns:
#             Dictionary containing batched transitions:
#                 - states: np.ndarray of shape (batch_size, state_dim)
#                 - actions: np.ndarray of shape (batch_size,)
#                 - rewards: np.ndarray of shape (batch_size,)
#                 - next_states: np.ndarray of shape (batch_size, state_dim)
#                 - dones: np.ndarray of shape (batch_size,)
#         """
#         batch = random.sample(self.buffer, batch_size)
#
#         states = np.array([t['state'] for t in batch], dtype=np.float32)
#         actions = np.array([t['action'] for t in batch], dtype=np.int64)
#         rewards = np.array([t['reward'] for t in batch], dtype=np.float32)
#         next_states = np.array([t['next_state'] for t in batch], dtype=np.float32)
#         dones = np.array([t['done'] for t in batch], dtype=np.float32)
#
#         return {
#             'states': states,
#             'actions': actions,
#             'rewards': rewards,
#             'next_states': next_states,
#             'dones': dones
#         }
#
#     def __len__(self) -> int:
#         """Return the current size of the buffer."""
#         return len(self.buffer)
#
#     def clear(self):
#         """Clear all transitions from the buffer."""
#         self.buffer.clear()


# ============================================================================
# N-STEP REPLAY BUFFER (Current Implementation)
# ============================================================================
class ReplayBuffer:
    """Experience replay buffer with n-step returns support.

    Supports both 1-step and n-step temporal difference learning.
    n-step returns provide better credit assignment by using actual rewards
    from the next n steps instead of bootstrapping immediately.
    """

    def __init__(self, capacity: int = 100000, n_step: int = 1, gamma: float = 0.99):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            n_step: Number of steps for n-step returns (1 = standard TD)
            gamma: Discount factor for computing n-step returns
        """
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)

        # Temporary buffer for computing n-step returns
        self.n_step_buffer = deque(maxlen=n_step)

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
        # Add to n-step buffer
        self.n_step_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

        # If we have enough steps or episode is done, compute n-step return
        if len(self.n_step_buffer) == self.n_step or done:
            # Compute n-step return
            n_step_return = 0.0
            for i, transition in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * transition['reward']

            # Get the initial state and action
            first_transition = self.n_step_buffer[0]

            # Store n-step transition
            self.buffer.append({
                'state': first_transition['state'],
                'action': first_transition['action'],
                'reward': n_step_return,
                'next_state': next_state,
                'done': done,
                'n_steps': len(self.n_step_buffer)
            })

            # If episode is done, clear n-step buffer
            if done:
                self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing batched transitions with n-step returns
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t['state'] for t in batch], dtype=np.float32)
        actions = np.array([t['action'] for t in batch], dtype=np.int64)
        rewards = np.array([t['reward'] for t in batch], dtype=np.float32)
        next_states = np.array([t['next_state'] for t in batch], dtype=np.float32)
        dones = np.array([t['done'] for t in batch], dtype=np.float32)
        n_steps = np.array([t['n_steps'] for t in batch], dtype=np.int32)

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'n_steps': n_steps
        }

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer.clear()
        self.n_step_buffer.clear()


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer with n-step returns support.

    Uses prioritized sampling based on TD error for more efficient learning.
    Samples important transitions (high TD error) more frequently.
    """

    def __init__(
        self,
        capacity: int = 100000,
        n_step: int = 1,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            n_step: Number of steps for n-step returns (1 = standard TD)
            gamma: Discount factor for computing n-step returns
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

        # Temporary buffer for computing n-step returns
        self.n_step_buffer = deque(maxlen=n_step)

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
        # Add to n-step buffer
        self.n_step_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

        # If we have enough steps or episode is done, compute n-step return
        if len(self.n_step_buffer) == self.n_step or done:
            # Compute n-step return
            n_step_return = 0.0
            for i, transition in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * transition['reward']

            # Get the initial state and action
            first_transition = self.n_step_buffer[0]

            # Create n-step transition
            max_priority = self.priorities.max() if self.buffer else 1.0
            if priority is None:
                priority = max_priority

            transition = {
                'state': first_transition['state'],
                'action': first_transition['action'],
                'reward': n_step_return,
                'next_state': next_state,
                'done': done,
                'n_steps': len(self.n_step_buffer)
            }

            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.position] = transition

            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

            # If episode is done, clear n-step buffer
            if done:
                self.n_step_buffer.clear()

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
        n_steps = np.array([t['n_steps'] for t in batch], dtype=np.int32)

        batch_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'n_steps': n_steps
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
        self.n_step_buffer.clear()
