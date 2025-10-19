"""Proximal Policy Optimization (PPO) network for local search operator selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO with shared backbone.

    Architecture:
    - Shared feature extractor processes state information
    - Policy head (actor): outputs action logits for discrete action distribution
    - Value head (critic): outputs scalar state value estimate

    Supports both standard concatenation and operator attention mechanisms.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [128, 128, 64],
        dropout_rate: float = 0.05,
        use_operator_attention: bool = False,
        solution_feature_dim: Optional[int] = None,
        operator_feature_dim_per_op: Optional[int] = None,
        num_operators: Optional[int] = None
    ):
        """Initialize the PPO Actor-Critic network.

        Args:
            state_dim: Dimension of state/observation space
            action_dim: Number of possible actions (operators)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_operator_attention: Whether to use operator attention mechanism
            solution_feature_dim: Dimension of solution features (required if use_operator_attention=True)
            operator_feature_dim_per_op: Features per operator (required if use_operator_attention=True)
            num_operators: Number of operators (required if use_operator_attention=True)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_operator_attention = use_operator_attention
        self.solution_feature_dim = solution_feature_dim
        self.operator_feature_dim_per_op = operator_feature_dim_per_op
        self.num_operators = num_operators

        # Validate attention parameters
        if use_operator_attention:
            if solution_feature_dim is None or operator_feature_dim_per_op is None or num_operators is None:
                raise ValueError("When use_operator_attention=True, must provide solution_feature_dim, "
                                 "operator_feature_dim_per_op, and num_operators")

        # Build architecture based on attention flag
        if not use_operator_attention:
            # STANDARD BRANCH: Simple concatenation architecture
            # Shared feature extraction layers
            feature_layers = []
            prev_dim = state_dim

            for hidden_dim in hidden_dims[:-1]:  # All but last layer
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

        else:
            # ATTENTION BRANCH: Operator Attention Architecture
            # Model dimension for attention
            d_model = hidden_dims[0]

            # Solution feature encoder
            self.solution_encoder = nn.Sequential(
                nn.Linear(solution_feature_dim, d_model),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

            # Operator feature encoder (shared across all operators)
            self.operator_encoder = nn.Sequential(
                nn.Linear(operator_feature_dim_per_op, d_model),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

            # Cross-attention: Solution (query) attends to Operators (key/value)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            )

            # Combined feature processing
            combined_dim = d_model * 2  # solution_embed + attended_operators
            feature_layers = []
            prev_dim = combined_dim

            for hidden_dim in hidden_dims[1:-1]:  # Skip first (used as d_model), all but last
                feature_layers.append(nn.Linear(prev_dim, hidden_dim))
                feature_layers.append(nn.ReLU())
                feature_layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim

            if feature_layers:
                self.combined_processor = nn.Sequential(*feature_layers)
            else:
                # If no intermediate layers, use identity
                self.combined_processor = nn.Identity()
                prev_dim = combined_dim

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

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Tuple of (action_logits, state_values):
                - action_logits: Logits for action distribution (batch_size, action_dim)
                - state_values: State value estimates (batch_size, 1)
        """
        if not self.use_operator_attention:
            # STANDARD FORWARD: Simple concatenation
            # Extract shared features
            features = self.feature_extractor(state)

            # Compute policy logits and state value
            action_logits = self.policy_head(features)  # (batch_size, action_dim)
            state_values = self.value_head(features)    # (batch_size, 1)

            return action_logits, state_values

        else:
            # ATTENTION FORWARD: Operator Attention
            batch_size = state.shape[0]

            # Split state into solution features and operator features
            solution_features = state[:, :self.solution_feature_dim]  # (batch_size, solution_dim)
            operator_features_flat = state[:, self.solution_feature_dim:]  # (batch_size, num_ops * op_dim)

            # Reshape operator features: (batch_size, num_ops, op_dim_per_op)
            operator_features = operator_features_flat.reshape(
                batch_size, self.num_operators, self.operator_feature_dim_per_op
            )

            # Encode solution features
            solution_embed = self.solution_encoder(solution_features)  # (batch_size, d_model)

            # Encode operator features (apply encoder to each operator)
            # Input: (batch_size, num_ops, op_dim_per_op)
            # Reshape for encoder: (batch_size * num_ops, op_dim_per_op)
            operator_features_reshaped = operator_features.reshape(-1, self.operator_feature_dim_per_op)
            operator_embed_reshaped = self.operator_encoder(operator_features_reshaped)  # (batch * num_ops, d_model)
            # Reshape back: (batch_size, num_ops, d_model)
            operator_embed = operator_embed_reshaped.reshape(batch_size, self.num_operators, -1)

            # Cross-attention: Solution (query) attends to Operators (key/value)
            # Query: (batch_size, 1, d_model) - unsqueeze for attention
            solution_query = solution_embed.unsqueeze(1)  # (batch_size, 1, d_model)
            # Key/Value: (batch_size, num_ops, d_model)
            attended_output, attention_weights = self.cross_attention(
                query=solution_query,
                key=operator_embed,
                value=operator_embed
            )  # attended_output: (batch_size, 1, d_model)

            # Squeeze attended output
            attended_output = attended_output.squeeze(1)  # (batch_size, d_model)

            # Combine solution embedding and attended operator features
            combined_features = torch.cat([solution_embed, attended_output], dim=1)  # (batch_size, d_model * 2)

            # Process combined features
            features = self.combined_processor(combined_features)

            # Compute policy logits and state value
            action_logits = self.policy_head(features)  # (batch_size, action_dim)
            state_values = self.value_head(features)    # (batch_size, 1)

            return action_logits, state_values


def detect_ppo_from_checkpoint(path: str) -> bool:
    """Detect whether a saved model is PPO-based.

    This function inspects the saved state_dict keys to determine if the model
    uses PPO architecture (actor-critic) or DQN architecture.

    Args:
        path: Path to the saved checkpoint (.pt file)

    Returns:
        True if model is PPO, False if DQN

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint format is invalid
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        # Load only the keys for efficiency
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Check for PPO-specific keys
        if 'policy_network_state_dict' in checkpoint:
            return True  # PPO model
        elif 'q_network_state_dict' in checkpoint:
            return False  # DQN model
        else:
            raise RuntimeError(
                f"Could not determine architecture from checkpoint {path}. "
                f"Found keys: {list(checkpoint.keys())}"
            )

    except Exception as e:
        if isinstance(e, (FileNotFoundError, RuntimeError)):
            raise
        raise RuntimeError(f"Error loading checkpoint {path}: {e}")
