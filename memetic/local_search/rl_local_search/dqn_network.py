"""Deep Q-Network implementation for local search operator selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class DQNNetwork(nn.Module):
    """Dueling Deep Q-Network for estimating Q-values of local search operators.

    Dueling DQN separates the Q-value into two components:
    - V(s): Value function - "How good is this state in general?"
    - A(s,a): Advantage function - "How much better is action a compared to others?"

    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
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
        """Initialize the Dueling DQN network.

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
            # Shared feature extraction layers
            feature_layers = []
            prev_dim = state_dim

            for hidden_dim in hidden_dims[:-1]:
                feature_layers.append(nn.Linear(prev_dim, hidden_dim))
                feature_layers.append(nn.ReLU())
                feature_layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim

            self.feature_extractor = nn.Sequential(*feature_layers)

            # Value stream: V(s) - scalar value representing state quality
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[-1], 1)
            )

            # Advantage stream: A(s,a) - relative advantage of each action
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[-1], action_dim)
            )

        else:
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Q-values for each action of shape (batch_size, action_dim)
        """
        if not self.use_operator_attention:
            # ORIGINAL FORWARD: Standard Dueling DQN
            # Extract shared features
            features = self.feature_extractor(state)

            # Compute value and advantage
            value = self.value_stream(features)  # (batch_size, 1)
            advantage = self.advantage_stream(features)  # (batch_size, action_dim)

            # Combine using dueling architecture formula
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            # Subtracting the mean ensures identifiability (unique V and A)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

            return q_values

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
            # Reshape for encoder: (batch_size * num_ops, op_dim_per_op)
            operator_features_reshaped = operator_features.reshape(-1, self.operator_feature_dim_per_op)
            operator_embed_reshaped = self.operator_encoder(operator_features_reshaped)  # (batch * num_ops, d_model)
            # Reshape back: (batch_size, num_ops, d_model)
            operator_embed = operator_embed_reshaped.reshape(batch_size, self.num_operators, -1)

            # Cross-attention: Solution (query) attends to Operators (key/value)
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

            # Compute value and advantage
            value = self.value_stream(features)  # (batch_size, 1)
            advantage = self.advantage_stream(features)  # (batch_size, action_dim)

            # Combine using dueling architecture formula
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

            return q_values


def detect_attention_from_checkpoint(path: str) -> bool:
    """Detect whether a saved model uses operator attention architecture.

    This function inspects the saved state_dict keys to determine if the model
    was trained with or without operator attention mechanism.

    Args:
        path: Path to the saved checkpoint (.pt file)

    Returns:
        True if model uses attention, False otherwise

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

        if 'q_network_state_dict' not in checkpoint:
            raise RuntimeError(f"Invalid checkpoint format: missing 'q_network_state_dict' in {path}")

        state_dict = checkpoint['q_network_state_dict']
        keys = set(state_dict.keys())

        # Check for attention-specific layers
        has_solution_encoder = any('solution_encoder' in k for k in keys)
        has_cross_attention = any('cross_attention' in k for k in keys)

        # Check for non-attention layers
        has_feature_extractor = any('feature_extractor' in k for k in keys)

        if has_solution_encoder and has_cross_attention:
            return True  # Attention model
        elif has_feature_extractor:
            return False  # Non-attention model
        else:
            raise RuntimeError(
                f"Could not determine architecture from checkpoint {path}. "
                f"Found keys: {list(keys)[:5]}..."
            )

    except Exception as e:
        if isinstance(e, (FileNotFoundError, RuntimeError)):
            raise
        raise RuntimeError(f"Error loading checkpoint {path}: {e}")


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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_operator_attention: bool = False,
        solution_feature_dim: Optional[int] = None,
        operator_feature_dim_per_op: Optional[int] = None,
        num_operators: Optional[int] = None
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
            use_operator_attention: Whether to use operator attention mechanism
            solution_feature_dim: Dimension of solution features (for attention)
            operator_feature_dim_per_op: Features per operator (for attention)
            num_operators: Number of operators (for attention)
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
        self.q_network = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_operator_attention=use_operator_attention,
            solution_feature_dim=solution_feature_dim,
            operator_feature_dim_per_op=operator_feature_dim_per_op,
            num_operators=num_operators
        ).to(self.device)

        self.target_network = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_operator_attention=use_operator_attention,
            solution_feature_dim=solution_feature_dim,
            operator_feature_dim_per_op=operator_feature_dim_per_op,
            num_operators=num_operators
        ).to(self.device)

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

        if update_stats and self.normalization_samples < 100000:
            # Update running statistics
            self.normalization_samples += 1
            alpha = 1.0 / self.normalization_samples
            delta = state - self.state_mean
            self.state_mean += alpha * delta
            self.state_std = np.sqrt((1 - alpha) * (self.state_std ** 2) + alpha * (delta ** 2))

        # Normalize with small epsilon to avoid division by zero
        normalized = (state - self.state_mean) / (self.state_std + 1e-8)
        # Clip to reasonable range
        normalized = np.clip(normalized, -10, 10)
        return normalized

    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None, update_stats: bool = True) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Current state observation
            epsilon: Exploration rate (uses self.epsilon if None)
            update_stats: Whether to update normalization statistics (False during testing)

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
            normalized_state = self.normalize_state(state, update_stats=update_stats)

            # Greedy action
            self.q_network.eval()  # Set to eval mode to disable dropout
            with torch.no_grad():
                state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
            self.q_network.train()  # Set back to train mode
            return action

    def get_q_values(self, state: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """Get Q-values for all actions for a given state.

        Args:
            state: Current state observation
            update_stats: Whether to update normalization statistics

        Returns:
            Array of Q-values for each action
        """
        normalized_state = self.normalize_state(state, update_stats=update_stats)

        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
        self.q_network.train()

        return q_values.cpu().numpy().flatten()

    # ORIGINAL UPDATE (1-step, no prioritized replay) 
    # def update(self, batch: dict) -> float:
    #     """Update the Q-network using a batch of transitions.
    #
    #     Args:
    #         batch: Dictionary containing:
    #             - states: np.ndarray of shape (batch_size, state_dim)
    #             - actions: np.ndarray of shape (batch_size,)
    #             - rewards: np.ndarray of shape (batch_size,)
    #             - next_states: np.ndarray of shape (batch_size, state_dim)
    #             - dones: np.ndarray of shape (batch_size,)
    #
    #     Returns:
    #         Loss value
    #     """
    #     # Normalize states
    #     normalized_states = np.array([self.normalize_state(s, update_stats=False) for s in batch['states']])
    #     normalized_next_states = np.array([self.normalize_state(s, update_stats=False) for s in batch['next_states']])
    #
    #     # Convert to tensors
    #     states = torch.FloatTensor(normalized_states).to(self.device)
    #     actions = torch.LongTensor(batch['actions']).to(self.device)
    #     rewards = torch.FloatTensor(batch['rewards']).to(self.device)
    #     next_states = torch.FloatTensor(normalized_next_states).to(self.device)
    #     dones = torch.FloatTensor(batch['dones']).to(self.device)
    #
    #     # Current Q-values
    #     current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    #
    #     with torch.no_grad():
    #         # --- OPTION 1: Double DQN (DDQN) ---
    #         next_actions = self.q_network(next_states).argmax(dim=1)  # Select with Q-network
    #         next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # Evaluate with target
    #
    #         # --- OPTION 2: Standard DQN ---
    #         # Uncomment this line and comment out the two DDQN lines above:
    #         # next_q_values = self.target_network(next_states).max(dim=1)[0]
    #
    #         target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
    #
    #     # Compute loss (Huber loss for stability)
    #     loss = F.smooth_l1_loss(current_q_values, target_q_values)
    #
    #     # Optimize
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     # Gradient clipping for stability
    #     torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
    #     self.optimizer.step()
    #
    #     # Update target network periodically
    #     self.update_count += 1
    #     if self.update_count % self.target_update_interval == 0:
    #         self.target_network.load_state_dict(self.q_network.state_dict())
    #
    #     # Track loss
    #     loss_value = loss.item()
    #     self.training_losses.append(loss_value)
    #
    #     return loss_value

    # CURRENT UPDATE (n-step + optional prioritized replay support)
    def update(self, batch: dict, weights: Optional[np.ndarray] = None) -> tuple[float, Optional[np.ndarray]]:
        """Update the Q-network using a batch of transitions.

        Supports n-step returns and prioritized experience replay.

        Args:
            batch: Dictionary containing:
                - states: np.ndarray of shape (batch_size, state_dim)
                - actions: np.ndarray of shape (batch_size,)
                - rewards: np.ndarray of shape (batch_size,) - n-step returns
                - next_states: np.ndarray of shape (batch_size, state_dim)
                - dones: np.ndarray of shape (batch_size,)
                - n_steps: np.ndarray of shape (batch_size,) - number of steps for each transition
            weights: Optional importance sampling weights for prioritized replay

        Returns:
            Tuple of (loss_value, td_errors):
                - loss_value: Scalar loss value
                - td_errors: TD errors for priority updates (None if not using prioritized replay)
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
        n_steps = torch.LongTensor(batch['n_steps']).to(self.device)

        # Importance sampling weights
        if weights is not None:
            weights_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            weights_tensor = torch.ones(len(batch['states'])).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: Select actions with Q-network, evaluate with target network
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            gamma_n = self.gamma ** n_steps.float()
            target_q_values = rewards + (1 - dones) * gamma_n * next_q_values

        # TD errors for prioritized replay
        td_errors = (current_q_values - target_q_values).abs().detach().cpu().numpy()

        # Weighted loss 
        element_wise_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        loss = (weights_tensor * element_wise_loss).mean()

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

        return loss_value, td_errors

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
