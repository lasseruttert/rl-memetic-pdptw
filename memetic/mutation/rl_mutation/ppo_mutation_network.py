"""Simplified PPO Actor-Critic Network for mutation operator selection."""

import torch
import torch.nn as nn


class PPOMutationNetwork(nn.Module):
    """Actor-Critic Network for mutation operator selection.

    Uses a simple architecture with shared feature extraction followed by
    separate policy (actor) and value (critic) heads.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [128, 128, 64],
        dropout_rate: float = 0.05
    ):
        """Initialize the Actor-Critic network.

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

        # Policy head (actor): outputs action logits
        self.policy_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], action_dim)
        )

        # Value head (critic): outputs state value
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], 1)
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
            Tuple of (action_logits, state_values):
                - action_logits: Logits for action distribution (batch_size, action_dim)
                - state_values: State value estimates (batch_size, 1)
        """
        features = self.feature_extractor(state)
        action_logits = self.policy_head(features)
        state_values = self.value_head(features)

        return action_logits, state_values


def detect_ppo_from_checkpoint(checkpoint_path: str) -> bool:
    """Detect if a checkpoint is from a PPO model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        True if PPO model, False if DQN model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # PPO checkpoints have 'policy_network_state_dict'
    # DQN checkpoints have 'q_network_state_dict'
    if 'policy_network_state_dict' in checkpoint:
        return True
    elif 'q_network_state_dict' in checkpoint:
        return False
    else:
        raise ValueError(f"Unknown checkpoint format in {checkpoint_path}")
