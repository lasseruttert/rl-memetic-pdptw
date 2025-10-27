"""Simplified Deep Q-Network for mutation operator selection."""

import torch
import torch.nn as nn


class DQNMutationNetwork(nn.Module):
    """Dueling Deep Q-Network for mutation operator selection.

    Uses a simple architecture with shared feature extraction followed by
    separate value and advantage streams (Dueling DQN).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [128, 128, 64],
        dropout_rate: float = 0.05
    ):
        """Initialize the Dueling DQN network.

        Args:
            state_dim: Dimension of state/observation space
            action_dim: Number of possible actions (operators)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction layers
        feature_layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims[:-1]:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*feature_layers)

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], 1)
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], action_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor):
        """Forward pass through the network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Q-values for each action of shape (batch_size, action_dim)
        """
        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine using dueling architecture
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
