"""PPO agent for learning mutation operator selection policy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from memetic.mutation.rl_mutation.ppo_mutation_network import PPOMutationNetwork


class PPOMutationAgent:
    """PPO agent for learning mutation operator selection policy.

    Uses Proximal Policy Optimization with clipped objective for stable training.
    Collects trajectories on-policy and updates with multiple epochs per trajectory.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [128, 128, 64],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_epochs: int = 2,
        batch_size: int = 2048,
        num_minibatches: int = 2,
        normalize_advantages: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the PPO agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions for networks
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: Lambda for Generalized Advantage Estimation
            clip_epsilon: Clipping parameter for PPO objective
            entropy_coef: Coefficient for entropy bonus
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of epochs to update per trajectory
            batch_size: Minimum trajectory size before update
            num_minibatches: Number of minibatches per epoch
            normalize_advantages: Whether to normalize advantages
            device: Device to use for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_minibatches = num_minibatches
        self.normalize_advantages = normalize_advantages
        self.device = torch.device(device)

        # Policy network (actor-critic)
        self.policy_network = PPOMutationNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        # Optimizer for both policy and value networks (shared)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # State normalization (running mean and std)
        self.state_mean = None
        self.state_std = None
        self.normalization_samples = 0

        # Trajectory storage (cleared after each update)
        self.trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }

        # Training statistics
        self.update_count = 0
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'clip_fractions': [],
            'approx_kls': []
        }

    def normalize_state(self, state: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """Normalize state using running mean and std.

        Uses Welford's online algorithm for numerically stable variance calculation.

        Args:
            state: State to normalize
            update_stats: Whether to update running statistics

        Returns:
            Normalized state
        """
        if self.state_mean is None:
            self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
            self.state_std = np.ones(self.state_dim, dtype=np.float32)
            self.state_m2 = np.zeros(self.state_dim, dtype=np.float32)  # For Welford's algorithm

        if update_stats and self.normalization_samples < 100000:
            # Welford's online algorithm for numerically stable variance
            self.normalization_samples += 1
            delta = state - self.state_mean
            self.state_mean += delta / self.normalization_samples
            delta2 = state - self.state_mean
            self.state_m2 += delta * delta2

            # Calculate std from accumulated M2
            if self.normalization_samples > 1:
                variance = self.state_m2 / (self.normalization_samples - 1)
                self.state_std = np.sqrt(variance + 1e-8)  # Add epsilon for numerical stability

        # Normalize with small epsilon to avoid division by zero
        normalized = (state - self.state_mean) / (self.state_std + 1e-8)
        # Clip to reasonable range
        normalized = np.clip(normalized, -10, 10)
        return normalized

    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None, update_stats: bool = True,
                   deterministic: bool = False) -> Tuple[int, float, float]:
        """Select an action using the policy network.

        Args:
            state: Current state observation
            epsilon: Not used in PPO (kept for compatibility with DQN interface)
            update_stats: Whether to update normalization statistics
            deterministic: If True, select argmax action (for evaluation)

        Returns:
            Tuple of (action, log_prob, value):
                - action: Selected action index
                - log_prob: Log probability of the action
                - value: State value estimate
        """
        # Normalize state
        normalized_state = self.normalize_state(state, update_stats=update_stats)

        self.policy_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
            action_logits, state_value = self.policy_network(state_tensor)

            # Create action distribution
            action_probs = F.softmax(action_logits, dim=-1)

            if deterministic:
                # Deterministic selection (for evaluation)
                action = action_probs.argmax(dim=-1).item()
            else:
                # Sample from distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()

            # Compute log probability
            log_prob = F.log_softmax(action_logits, dim=-1)[0, action].item()
            value = state_value.item()

        self.policy_network.train()
        return action, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """Store a transition in the trajectory buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of the action
            value: State value estimate
            done: Whether episode is done
        """
        self.trajectory['states'].append(state)
        self.trajectory['actions'].append(action)
        self.trajectory['rewards'].append(reward)
        self.trajectory['log_probs'].append(log_prob)
        self.trajectory['values'].append(value)
        self.trajectory['dones'].append(done)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            next_value: Value estimate for the next state after trajectory

        Returns:
            Tuple of (advantages, returns):
                - advantages: Advantage estimates
                - returns: Discounted returns (targets for value function)
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]

            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage

        returns = advantages + values

        return advantages, returns

    def update(self, next_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Update the policy and value networks using collected trajectory.

        Args:
            next_state: Next state after trajectory (for computing final value estimate)

        Returns:
            Dictionary with training statistics
        """
        if len(self.trajectory['states']) < self.batch_size:
            # Not enough data yet
            return {}

        # Convert trajectory to numpy arrays
        states = np.array(self.trajectory['states'], dtype=np.float32)
        actions = np.array(self.trajectory['actions'], dtype=np.int64)
        rewards = np.array(self.trajectory['rewards'], dtype=np.float32)
        old_log_probs = np.array(self.trajectory['log_probs'], dtype=np.float32)
        values = np.array(self.trajectory['values'], dtype=np.float32)
        dones = np.array(self.trajectory['dones'], dtype=np.float32)

        # Compute next value for GAE
        if next_state is not None:
            normalized_next_state = self.normalize_state(next_state, update_stats=False)
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(normalized_next_state).unsqueeze(0).to(self.device)
                _, next_value_tensor = self.policy_network(next_state_tensor)
                next_value = next_value_tensor.item()
        else:
            next_value = 0.0

        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize states
        normalized_states = np.array([self.normalize_state(s, update_stats=False) for s in states])

        # Convert to tensors
        states_tensor = torch.FloatTensor(normalized_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Update for multiple epochs
        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clip_fraction': [],
            'approx_kl': []
        }

        batch_size = len(states)
        minibatch_size = batch_size // self.num_minibatches

        for epoch in range(self.num_epochs):
            # Shuffle indices
            indices = np.random.permutation(batch_size)

            # Split into minibatches
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                if end > batch_size:
                    break

                mb_indices = indices[start:end]

                # Get minibatch
                mb_states = states_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]

                # Forward pass
                action_logits, state_values = self.policy_network(mb_states)
                state_values = state_values.squeeze(-1)

                # Compute action probabilities and log probs
                action_probs = F.softmax(action_logits, dim=-1)
                log_probs = F.log_softmax(action_logits, dim=-1)
                mb_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)

                # PPO clipped objective
                ratio = torch.exp(mb_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()

                # Value loss (clipped)
                old_values = torch.FloatTensor(values[mb_indices]).to(self.device)
                value_pred_clipped = old_values + torch.clamp(
                    state_values - old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_unclipped = F.mse_loss(state_values, mb_returns)
                value_loss_clipped = F.mse_loss(value_pred_clipped, mb_returns)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                # Entropy bonus (for exploration)
                entropy = -(action_probs * log_probs).sum(dim=-1).mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track statistics
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl = (mb_old_log_probs - mb_log_probs).mean().item()

                epoch_stats['policy_loss'].append(policy_loss.item())
                epoch_stats['value_loss'].append(value_loss.item())
                epoch_stats['entropy'].append(entropy.item())
                epoch_stats['clip_fraction'].append(clip_fraction)
                epoch_stats['approx_kl'].append(approx_kl)

        # Aggregate statistics
        stats = {
            'policy_loss': np.mean(epoch_stats['policy_loss']),
            'value_loss': np.mean(epoch_stats['value_loss']),
            'entropy': np.mean(epoch_stats['entropy']),
            'clip_fraction': np.mean(epoch_stats['clip_fraction']),
            'approx_kl': np.mean(epoch_stats['approx_kl']),
            'explained_variance': self._explained_variance(values, returns)
        }

        # Store in history
        self.training_stats['policy_losses'].append(stats['policy_loss'])
        self.training_stats['value_losses'].append(stats['value_loss'])
        self.training_stats['entropies'].append(stats['entropy'])
        self.training_stats['clip_fractions'].append(stats['clip_fraction'])
        self.training_stats['approx_kls'].append(stats['approx_kl'])

        # Clear trajectory buffer
        self.clear_trajectory()

        self.update_count += 1

        return stats

    def _explained_variance(self, values: np.ndarray, returns: np.ndarray) -> float:
        """Compute explained variance of value function.

        Args:
            values: Value predictions
            returns: Actual returns

        Returns:
            Explained variance
        """
        var_returns = np.var(returns)
        if var_returns == 0:
            return 0.0
        return 1.0 - np.var(returns - values) / var_returns

    def clear_trajectory(self):
        """Clear the trajectory buffer."""
        self.trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }

    def get_q_values(self, state: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """Get action preferences (policy logits) for a given state.

        This method is for compatibility with DQN interface. For PPO, we return
        the policy logits which can be used for action ranking.

        Args:
            state: Current state observation
            update_stats: Whether to update normalization statistics

        Returns:
            Array of policy logits for each action
        """
        normalized_state = self.normalize_state(state, update_stats=update_stats)

        self.policy_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
            action_logits, _ = self.policy_network(state_tensor)
            self.policy_network.train()
            return action_logits.cpu().numpy().flatten()

    def save(self, path: str):
        """Save the agent's network and state.

        Args:
            path: Path to save the checkpoint
        """
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'training_stats': self.training_stats,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'state_m2': getattr(self, 'state_m2', None),
            'normalization_samples': self.normalization_samples
        }, path)

    def load(self, path: str):
        """Load the agent's network and state.

        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        self.training_stats = checkpoint['training_stats']

        # Load normalization statistics if available (backward compatibility)
        self.state_mean = checkpoint.get('state_mean', None)
        self.state_std = checkpoint.get('state_std', None)
        self.state_m2 = checkpoint.get('state_m2', None)
        self.normalization_samples = checkpoint.get('normalization_samples', 0)

        # Initialize state_m2 if not in checkpoint (backward compatibility)
        if self.state_m2 is None and self.state_mean is not None:
            self.state_m2 = np.zeros(self.state_dim, dtype=np.float32)
