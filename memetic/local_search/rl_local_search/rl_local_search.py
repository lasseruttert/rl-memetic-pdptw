"""Reinforcement Learning-based local search for PDPTW."""

import numpy as np
import random
import time
from typing import Callable, Optional, Dict, List, Tuple

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.local_search.base_local_search import BaseLocalSearch
from memetic.solution_operators.base_operator import BaseOperator
from memetic.fitness.fitness import fitness

from memetic.local_search.rl_local_search.local_search_env import LocalSearchEnv
from memetic.local_search.rl_local_search.dqn_network import DQNAgent
from memetic.local_search.rl_local_search.replay_buffer import ReplayBuffer

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class RLLocalSearch(BaseLocalSearch):
    """RL-based local search that learns to select operators adaptively.

    Uses Deep Q-Learning (DQN) to learn which operators to apply based on
    problem and solution features, enabling context-sensitive operator selection.
    """

    def __init__(
        self,
        operators: List[BaseOperator],
        hidden_dims: List[int] = [128, 128, 64],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        target_update_interval: int = 100,
        alpha: float = 1.0,
        beta: float = 1.0,
        acceptance_strategy: str = "greedy",
        reward_strategy: str = "initial_improvement",
        type: str = "OneShot",  # "OneShot", "Roulette", or "Ranking"
        max_iterations: int = 100,
        max_no_improvement: Optional[int] = None,
        replay_buffer_capacity: int = 100000,
        batch_size: int = 64,
        n_step: int = 3,
        use_prioritized_replay: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        use_operator_attention: bool = False,
        device: str = "cuda",
        verbose: bool = False
    ):
        """Initialize RL-based local search.

        Args:
            operators: List of local search operators to choose from
            hidden_dims: Hidden layer dimensions for Q-network
            learning_rate: Learning rate for DQN optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
            target_update_interval: Steps between target network updates
            alpha: Weight for fitness improvement in reward (normalized by distance_baseline)
            beta: Weight for feasibility improvement in reward (normalized to [0,1])
            acceptance_strategy: "greedy" or "always"
            max_iterations: Maximum iterations for both training episodes and inference
            max_no_improvement: Early stopping after N steps without improvement (None to disable)
            replay_buffer_capacity: Size of replay buffer
            batch_size: Batch size for training
            n_step: Number of steps for n-step returns (1 = standard TD, 3-5 recommended)
            use_prioritized_replay: Whether to use prioritized experience replay
            per_alpha: Prioritization strength (0 = uniform, 1 = full prioritization)
            per_beta_start: Initial importance sampling weight
            use_operator_attention: Whether to use operator attention mechanism for operator selection
            device: Device for training ("cuda" or "cpu")
            verbose: Whether to print training progress
        """
        super().__init__()

        self.operators = operators
        self.acceptance_strategy = acceptance_strategy
        self.reward_strategy = reward_strategy
        self.type = type
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.n_step = n_step
        self.use_prioritized_replay = use_prioritized_replay
        self.use_operator_attention = use_operator_attention
        self.verbose = verbose

        # Validate configuration
        if self.type in ["Ranking", "Roulette"] and self.acceptance_strategy != "greedy":
            raise ValueError(f"{self.type} type requires greedy acceptance strategy")

        # Environment
        self.env = LocalSearchEnv(
            operators=operators,
            alpha=alpha,
            acceptance_strategy=acceptance_strategy,
            reward_strategy=reward_strategy,
            max_steps=max_iterations,
            max_no_improvement=max_no_improvement
        )

        # DQN Agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = len(operators)

        # Extract feature dimensions from environment for attention mechanism
        solution_feature_dim = self.env.solution_feature_dim
        operator_feature_dim_per_op = self.env.operator_feature_dim_per_op

        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update_interval=target_update_interval,
            device=device,
            use_operator_attention=use_operator_attention,
            solution_feature_dim=solution_feature_dim,
            operator_feature_dim_per_op=operator_feature_dim_per_op,
            num_operators=action_dim
        )

        # Replay buffer 
        if use_prioritized_replay:
            from memetic.local_search.rl_local_search.replay_buffer import PrioritizedReplayBuffer
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=replay_buffer_capacity,
                n_step=n_step,
                gamma=gamma,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=100000 
            )
            if self.verbose:
                print(f"Using Prioritized Replay Buffer (alpha={per_alpha}, beta_start={per_beta_start})")
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=replay_buffer_capacity,
                n_step=n_step,
                gamma=gamma
            )
            if self.verbose:
                print(f"Using Standard Replay Buffer")

        if self.verbose:
            print(f"n-step returns: n={n_step}")
            if use_operator_attention:
                print(f"Operator Attention: ENABLED")
                print(f"  - Solution features: {solution_feature_dim}")
                print(f"  - Operator features per operator: {operator_feature_dim_per_op}")
                print(f"  - Number of operators: {action_dim}")
            else:
                print(f"Operator Attention: DISABLED (standard concatenation)")

        # Training mode flag
        self.training_mode = False

        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_best_fitness': [],
            'losses': [],
            'epsilon_values': []
        }

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)

        # Set PyTorch seeds
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            if self.verbose:
                print("Warning: PyTorch not available for seeding")

    def train(
        self,
        problem_generator: Callable[[], PDPTWProblem],
        initial_solution_generator: Callable[[PDPTWProblem], PDPTWSolution],
        num_episodes: int = 2000,
        new_instance_interval: int = 50,
        new_solution_interval: int = 5,
        update_interval: int = 1,
        warmup_episodes: int = 10,
        save_interval: int = 100,
        save_path: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        seed: Optional[int] = None,
        log_interval: int = 10
    ) -> Dict:
        """Train the RL policy for operator selection.

        Args:
            problem_generator: Function that generates problem instances
            initial_solution_generator: Function that generates initial solutions
            num_episodes: Number of training episodes
            new_instance_interval: Generate new instance every N episodes
            new_solution_interval: Generate new solution every N episodes
            update_interval: Update policy every N steps
            warmup_episodes: Number of episodes before training starts
            save_interval: Save model every N episodes
            save_path: Path to save model checkpoints
            tensorboard_dir: Directory for TensorBoard logs (None to disable)
            seed: Random seed for reproducibility (None for random)
            log_interval: Log detailed metrics every N episodes (default: 10)

        Returns:
            Dictionary containing training history
        """
        # Set random seed if provided
        if seed is not None:
            self._set_seed(seed)
            if self.verbose:
                print(f"Random seed set to {seed}")

        self.training_mode = True
        start_time = time.time()

        # Update beta_frames for prioritized replay if using it
        if self.use_prioritized_replay:
            # Anneal beta from beta_start to 1.0 over the entire training
            self.replay_buffer.beta_frames = num_episodes * self.max_iterations
            if self.verbose:
                print(f"PER beta will anneal from {self.replay_buffer.beta_start} to 1.0 over {self.replay_buffer.beta_frames} frames")

        # Initialize TensorBoard writer
        writer = None
        if tensorboard_dir and TENSORBOARD_AVAILABLE:
            writer = SummaryWriter(log_dir=tensorboard_dir)
            if self.verbose:
                print(f"TensorBoard logging enabled: {tensorboard_dir}")

            # Log hyperparameters
            hparams = {
                'learning_rate': self.agent.optimizer.param_groups[0]['lr'],
                'gamma': self.agent.gamma,
                'epsilon_start': self.agent.epsilon,
                'epsilon_end': self.agent.epsilon_end,
                'epsilon_decay': self.agent.epsilon_decay,
                'batch_size': self.batch_size,
                'num_episodes': num_episodes,
                'max_iterations': self.max_iterations,
                'acceptance_strategy': self.acceptance_strategy,
                'reward_strategy': self.reward_strategy,
                'alpha': self.env.alpha,
                'new_instance_interval': new_instance_interval,
                'new_solution_interval': new_solution_interval,
                'seed': seed if seed is not None else 'random',
            }
            writer.add_text('Hyperparameters', str(hparams), 0)

        elif tensorboard_dir and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard requested but not available. Install with: pip install tensorboard")

        # Initialize instance and solution
        instance = problem_generator()
        current_solution = initial_solution_generator(instance)

        if self.verbose:
            print(f"Starting RL Local Search training for {num_episodes} episodes...")
            print(f"State dim: {self.agent.state_dim}, Action dim: {self.agent.action_dim}")
            print(f"Operators: {[op.name if hasattr(op, 'name') else type(op).__name__ for op in self.operators]}")

        for episode in range(num_episodes):
            episode_start_time = time.time()

            # Generate new instance periodically
            if episode % new_instance_interval == 0:
                instance = problem_generator()
                current_solution = initial_solution_generator(instance)  
                if self.verbose:
                    print(f"Episode {episode}: New instance generated ({instance.num_requests} requests)")

            # Generate new solution periodically 
            elif episode % new_solution_interval == 0:
                current_solution = initial_solution_generator(instance)

            # Reset environment with current instance and solution
            state, info = self.env.reset(instance, current_solution)

            episode_reward = 0.0
            episode_length = 0
            step_losses = []
            episode_actions = [] 

            # Metric tracking
            num_accepted = 0
            num_rejected = 0
            accepted_improvements = []
            rejected_degradations = []

            # Per-operator tracking (Metric 3)
            operator_uses = [0] * len(self.operators)
            operator_successes = [0] * len(self.operators)
            operator_accepted = [0] * len(self.operators)  
            operator_improvements = [[] for _ in range(len(self.operators))] 

            # Episode loop
            for step in range(self.max_iterations):
                # Select action (epsilon-greedy)
                action = self.agent.get_action(state)
                episode_actions.append(action)

                # Take step in environment
                next_state, reward, terminated, truncated, step_info = self.env.step(action)

                # Store transition
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, done)

                # Update statistics
                episode_reward += reward
                episode_length += 1

                # Track metrics
                # Metric 2: Acceptance tracking
                if step_info['accepted']:
                    num_accepted += 1
                    accepted_improvements.append(step_info['fitness_improvement'])
                else:
                    num_rejected += 1
                    rejected_degradations.append(step_info['fitness_improvement'])


                # Metric 3: Operator effectiveness
                operator_uses[action] += 1
                fitness_change = step_info['fitness_improvement']
                operator_improvements[action].append(fitness_change)
                if fitness_change > 0:  
                    operator_successes[action] += 1
                if step_info['accepted']:
                    operator_accepted[action] += 1

                # Update policy (after warmup and if enough samples)
                if episode >= warmup_episodes and len(self.replay_buffer) >= self.batch_size:
                    if step % update_interval == 0:
                        # Sample batch (with priorities if using PER)
                        if self.use_prioritized_replay:
                            batch, indices, weights = self.replay_buffer.sample(self.batch_size)
                            loss, td_errors = self.agent.update(batch, weights)
                            # Update priorities based on TD errors
                            self.replay_buffer.update_priorities(indices, td_errors)
                        else:
                            batch = self.replay_buffer.sample(self.batch_size)
                            loss, _ = self.agent.update(batch)
                        step_losses.append(loss)

                # Move to next state
                state = next_state

                if terminated or truncated:
                    break

            # Decay epsilon
            self.agent.decay_epsilon()

            # Get best solution from episode
            best_solution, best_fitness = self.env.get_best_solution()

            # Record episode statistics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['episode_best_fitness'].append(best_fitness)
            self.training_history['epsilon_values'].append(self.agent.epsilon)

            avg_loss = np.mean(step_losses) if step_losses else 0.0
            if step_losses:
                self.training_history['losses'].append(avg_loss)

            # TensorBoard logging
            if writer:
                # Always log core metrics
                writer.add_scalar('Episode/Reward', episode_reward, episode)
                writer.add_scalar('Episode/BestFitness', best_fitness, episode)
                writer.add_scalar('Episode/Length', episode_length, episode)
                writer.add_scalar('Training/Epsilon', self.agent.epsilon, episode)
                if step_losses:
                    writer.add_scalar('Training/Loss', avg_loss, episode)

                # Log detailed metrics every log_interval episodes
                if episode % log_interval == 0 or episode == num_episodes - 1:
                    # Log state features
                    for i, feature_val in enumerate(state):
                        writer.add_scalar(f'Features/feature_{i}', feature_val, episode)

                    # Log operator selection statistics
                    action_counts = np.bincount(episode_actions, minlength=len(self.operators))
                    for i, count in enumerate(action_counts):
                        writer.add_scalar(f'Operators/operator_{i}_count', count, episode)
                        writer.add_scalar(f'Operators/operator_{i}_ratio', count / len(episode_actions) if episode_actions else 0, episode)

                    # Log Q-values for current state
                    q_values = self.agent.get_q_values(state, update_stats=False)
                    for i, q_val in enumerate(q_values):
                        writer.add_scalar(f'QValues/operator_{i}_qvalue', q_val, episode)
                    writer.add_scalar('QValues/max_qvalue', np.max(q_values), episode)
                    writer.add_scalar('QValues/mean_qvalue', np.mean(q_values), episode)
                    writer.add_scalar('QValues/std_qvalue', np.std(q_values), episode)


                    # Metric 2: Acceptance rate tracking
                    total_moves = num_accepted + num_rejected
                    acceptance_rate = num_accepted / total_moves if total_moves > 0 else 0
                    avg_accepted_improvement = np.mean(accepted_improvements) if accepted_improvements else 0
                    avg_rejected_degradation = np.mean(rejected_degradations) if rejected_degradations else 0

                    writer.add_scalar('Acceptance/acceptance_rate', acceptance_rate, episode)
                    writer.add_scalar('Acceptance/num_accepted', num_accepted, episode)
                    writer.add_scalar('Acceptance/num_rejected', num_rejected, episode)
                    writer.add_scalar('Acceptance/avg_accepted_improvement', avg_accepted_improvement, episode)
                    writer.add_scalar('Acceptance/avg_rejected_degradation', avg_rejected_degradation, episode)

                    # Metric 3: Operator effectiveness
                    for i in range(len(self.operators)):
                        if operator_uses[i] > 0:
                            success_rate = operator_successes[i] / operator_uses[i]
                            acceptance_rate_op = operator_accepted[i] / operator_uses[i]
                            avg_improvement = np.mean(operator_improvements[i]) if operator_improvements[i] else 0
                        else:
                            success_rate = 0
                            acceptance_rate_op = 0
                            avg_improvement = 0

                        writer.add_scalar(f'Effectiveness/operator_{i}_success_rate', success_rate, episode)
                        writer.add_scalar(f'Effectiveness/operator_{i}_acceptance_rate', acceptance_rate_op, episode)
                        writer.add_scalar(f'Effectiveness/operator_{i}_avg_improvement', avg_improvement, episode)

                    # Metric 5: Raw solution metrics
                    writer.add_scalar('Solution/raw_total_distance', best_solution.total_distance, episode)
                    writer.add_scalar('Solution/raw_num_vehicles', best_solution.num_vehicles_used, episode)
                    writer.add_scalar('Solution/is_feasible', int(best_solution.is_feasible), episode)

                    # Check for invalid states
                    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                        print(f"WARNING: Invalid state at episode {episode}")
                        print(f"  NaN: {np.any(np.isnan(state))}, Inf: {np.any(np.isinf(state))}")
                        print(f"  State: {state}")

                # Rolling averages
                if len(self.training_history['episode_rewards']) >= 10:
                    avg_reward_10 = np.mean(self.training_history['episode_rewards'][-10:])
                    avg_fitness_10 = np.mean(self.training_history['episode_best_fitness'][-10:])
                    writer.add_scalar('Episode/Reward_Avg10', avg_reward_10, episode)
                    writer.add_scalar('Episode/Fitness_Avg10', avg_fitness_10, episode)

                if len(self.training_history['episode_rewards']) >= 100:
                    avg_reward_100 = np.mean(self.training_history['episode_rewards'][-100:])
                    avg_fitness_100 = np.mean(self.training_history['episode_best_fitness'][-100:])
                    writer.add_scalar('Episode/Reward_Avg100', avg_reward_100, episode)
                    writer.add_scalar('Episode/Fitness_Avg100', avg_fitness_100, episode)

            # Console logging
            if self.verbose and (episode % 10 == 0 or episode == num_episodes - 1):
                elapsed = time.time() - start_time
                episode_time = time.time() - episode_start_time
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                avg_fitness = np.mean(self.training_history['episode_best_fitness'][-10:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} (avg: {avg_reward:.2f}) | "
                      f"Fitness: {best_fitness:.2f} (avg: {avg_fitness:.2f}) | "
                      f"Steps: {episode_length} | "
                      f"eps: {self.agent.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Time: {episode_time:.2f}s | "
                      f"Total: {elapsed:.2f}s")

            # Save checkpoint
            if save_path and episode % save_interval == 0 and episode > 0:
                checkpoint_path = f"{save_path}_episode_{episode}.pt"
                self.agent.save(checkpoint_path)
                if self.verbose:
                    print(f"Saved checkpoint to {checkpoint_path}")

        # Final save
        if save_path:
            final_path = f"{save_path}_final.pt"
            self.agent.save(final_path)
            if self.verbose:
                print(f"Training completed! Saved final model to {final_path}")

        self.training_mode = False
        total_time = time.time() - start_time

        if self.verbose:
            print(f"\nTraining Summary:")
            print(f"Total episodes: {num_episodes}")
            print(f"Total time: {total_time:.2f}s ({total_time/num_episodes:.2f}s per episode)")
            print(f"Final epsilon: {self.agent.epsilon:.3f}")
            print(f"Replay buffer size: {len(self.replay_buffer)}")

        # Close TensorBoard writer
        if writer:
            writer.close()
            if self.verbose:
                print(f"TensorBoard logs saved to {tensorboard_dir}")

        return self.training_history

    def search(
        self,
        problem: PDPTWProblem,
        solution: PDPTWSolution,
        max_iterations: Optional[int] = None,
        epsilon: float = 0.0
    ) -> Tuple[PDPTWSolution, float]:
        """Perform local search using trained RL policy (inference mode).

        This method is used during the memetic algorithm to improve solutions.

        Args:
            problem: PDPTW problem instance
            solution: Initial solution to improve
            max_iterations: Maximum number of iterations (None to use init value)
            epsilon: Exploration rate (0.0 = fully greedy)

        Returns:
            Tuple of (best_solution, best_fitness)
        """
        # Use provided max_iterations or default from init
        if max_iterations is None:
            max_iterations = self.max_iterations

        # Temporarily disable step limit for inference (Ranking/Roulette may use many steps per iteration)
        original_max_steps = self.env.max_steps
        original_max_no_improvement = self.env.max_no_improvement
        self.env.max_steps = float('inf')
        self.env.max_no_improvement = float('inf')

        # Reset environment
        state, info = self.env.reset(problem, solution)

        best_solution = solution.clone()
        best_fitness = fitness(problem, best_solution)

        # Inference loop
        done = False
        for iteration in range(max_iterations):
            if done:
                break

            # Select action using learned policy
            if self.type == "OneShot":
                action = self.agent.get_action(state, epsilon=epsilon, update_stats=False)

                # Apply operator
                next_state, reward, terminated, truncated, step_info = self.env.step(action)

                # Update best solution
                if step_info['fitness'] < best_fitness:
                    best_solution = self.env.current_solution.clone()
                    best_fitness = step_info['fitness']

                # Move to next state
                state = next_state
                done = terminated or truncated

            elif self.type == "Roulette":
                q_values = self.agent.get_q_values(state, update_stats=False)

                # Sample sequence based on Q-values (weighted, without replacement)
                exp_q = np.exp(q_values - np.max(q_values))
                probs = exp_q / np.sum(exp_q)
                action_sequence = np.random.choice(len(self.operators), size=len(self.operators),
                                                  replace=False, p=probs)

                # Try operators in sequence until one improves (greedy acceptance handles rejection)
                for action in action_sequence:
                    # Apply operator (environment handles state management)
                    next_state, reward, terminated, truncated, step_info = self.env.step(action)
                    state = next_state

                    # Check if this operator improved fitness
                    if step_info['fitness'] < best_fitness:
                        best_solution = self.env.current_solution.clone()
                        best_fitness = step_info['fitness']
                        break  # Stop at first improvement

                    if terminated or truncated:
                        done = True
                        break

            elif self.type == "Ranking":
                q_values = self.agent.get_q_values(state, update_stats=False)

                # Rank operators by Q-value (best first, descending order)
                action_sequence = np.argsort(-q_values)

                # Try operators in ranked order until one improves (greedy acceptance handles rejection)
                for action in action_sequence:
                    # Apply operator (environment handles state management)
                    next_state, reward, terminated, truncated, step_info = self.env.step(action)
                    state = next_state

                    # Check if this operator improved fitness
                    if step_info['fitness'] < best_fitness:
                        best_solution = self.env.current_solution.clone()
                        best_fitness = step_info['fitness']
                        break  # Stop at first improvement

                    if terminated or truncated:
                        done = True
                        break

        # Get final best solution
        env_best_solution, env_best_fitness = self.env.get_best_solution()
        if env_best_fitness < best_fitness:
            best_solution = env_best_solution
            best_fitness = env_best_fitness

        # Restore original max_steps
        self.env.max_steps = original_max_steps
        self.env.max_no_improvement = original_max_no_improvement

        return best_solution, best_fitness

    def save(self, path: str):
        """Save the RL agent and training history.

        Args:
            path: Path to save the model
        """
        self.agent.save(path)
        # Optionally save training history as well
        import pickle
        history_path = path.replace('.pt', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)

    def load(self, path: str):
        """Load a trained RL agent.

        Args:
            path: Path to the saved model
        """
        self.agent.load(path)
        # Optionally load training history
        import pickle
        history_path = path.replace('.pt', '_history.pkl')
        try:
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)
        except FileNotFoundError:
            if self.verbose:
                print(f"Warning: Training history not found at {history_path}")
