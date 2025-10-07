"""Deep Q-Network implementation for local search operator selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DQNNetwork(nn.Module):
    """Deep Q-Network for estimating Q-values of local search operators."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [128, 128, 64],
        dropout_rate: float = 0.05  
    ):
        """Initialize the DQN network.

        Args:
            state_dim: Dimension of state/observation space
            action_dim: Number of possible actions (operators)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build MLP layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer (Q-values for each action)
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Q-values for each action of shape (batch_size, action_dim)
        """
        return self.network(state)


class DQNAgent:
    """DQN agent for learning operator selection policy."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [128, 128, 64],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        target_update_interval: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions for Q-network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            target_update_interval: Steps between target network updates
            device: Device to use for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.device = torch.device(device)

        # Q-network and target network
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # State normalization (running mean and std)
        self.state_mean = None
        self.state_std = None
        self.normalization_samples = 0

        # Training statistics
        self.update_count = 0
        self.training_losses = []

    def normalize_state(self, state: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """Normalize state using running mean and std.

        Args:
            state: State to normalize
            update_stats: Whether to update running statistics

        Returns:
            Normalized state
        """
        if self.state_mean is None:
            self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
            self.state_std = np.ones(self.state_dim, dtype=np.float32)

        if update_stats and self.normalization_samples < 10000:
            # Update running statistics (online mean/std)
            self.normalization_samples += 1
            alpha = 1.0 / self.normalization_samples
            delta = state - self.state_mean
            self.state_mean += alpha * delta
            self.state_std = np.sqrt((1 - alpha) * (self.state_std ** 2) + alpha * (delta ** 2))

        # Normalize with small epsilon to avoid division by zero
        normalized = (state - self.state_mean) / (self.state_std + 1e-8)
        # Clip to reasonable range
        # normalized = np.clip(normalized, -10, 10)
        return normalized

    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Current state observation
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Selected action index
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            # Normalize state
            normalized_state = self.normalize_state(state, update_stats=True)

            # Greedy action
            self.q_network.eval()  # Set to eval mode to disable dropout
            with torch.no_grad():
                state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
            self.q_network.train()  # Set back to train mode
            return action

    def update(self, batch: dict) -> float:
        """Update the Q-network using a batch of transitions.

        Args:
            batch: Dictionary containing:
                - states: np.ndarray of shape (batch_size, state_dim)
                - actions: np.ndarray of shape (batch_size,)
                - rewards: np.ndarray of shape (batch_size,)
                - next_states: np.ndarray of shape (batch_size, state_dim)
                - dones: np.ndarray of shape (batch_size,)

        Returns:
            Loss value
        """
        # Normalize states
        normalized_states = np.array([self.normalize_state(s, update_stats=False) for s in batch['states']])
        normalized_next_states = np.array([self.normalize_state(s, update_stats=False) for s in batch['next_states']])

        # Convert to tensors
        states = torch.FloatTensor(normalized_states).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(normalized_next_states).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # --- OPTION 1: Double DQN (DDQN) ---
            # next_actions = self.q_network(next_states).argmax(dim=1)  # Select with Q-network
            # next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # Evaluate with target

            # --- OPTION 2: Standard DQN ---
            # Uncomment this line and comment out the two DDQN lines above:
            next_q_values = self.target_network(next_states).max(dim=1)[0]

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Track loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)

        return loss_value

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """Save the agent's networks and state.

        Args:
            path: Path to save the checkpoint
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'training_losses': self.training_losses,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'normalization_samples': self.normalization_samples
        }, path)

    def load(self, path: str):
        """Load the agent's networks and state.

        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        self.training_losses = checkpoint['training_losses']

        # Load normalization statistics if available (backward compatibility)
        self.state_mean = checkpoint.get('state_mean', None)
        self.state_std = checkpoint.get('state_std', None)
        self.normalization_samples = checkpoint.get('normalization_samples', 0)
