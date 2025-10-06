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
        alpha: float = 1.0,  # Fitness weight (rewards are normalized by distance_baseline)
        beta: float = 1.0,   # Feasibility weight (normalized to [0, 1] range)
        acceptance_strategy: str = "greedy",
        max_steps_per_episode: int = 100,
        replay_buffer_capacity: int = 100000,
        batch_size: int = 64,
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
            max_steps_per_episode: Maximum steps per episode during training
            replay_buffer_capacity: Size of replay buffer
            batch_size: Batch size for training
            device: Device for training ("cuda" or "cpu")
            verbose: Whether to print training progress
        """
        super().__init__()

        self.operators = operators
        self.acceptance_strategy = acceptance_strategy
        self.max_steps_per_episode = max_steps_per_episode
        self.batch_size = batch_size
        self.verbose = verbose

        # Environment
        self.env = LocalSearchEnv(
            operators=operators,
            alpha=alpha,
            acceptance_strategy=acceptance_strategy,
            max_steps=max_steps_per_episode
        )

        # DQN Agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = len(operators)

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
            device=device
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

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
        save_path: Optional[str] = None
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

        Returns:
            Dictionary containing training history
        """
        self.training_mode = True
        start_time = time.time()

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
                current_solution = initial_solution_generator(instance)  # Always generate new solution with new instance!
                if self.verbose:
                    print(f"Episode {episode}: New instance generated ({instance.num_requests} requests)")

            # Generate new solution periodically (only if no new instance)
            elif episode % new_solution_interval == 0:
                current_solution = initial_solution_generator(instance)

            # Reset environment with current instance and solution
            state, info = self.env.reset(instance, current_solution)

            episode_reward = 0.0
            episode_length = 0
            step_losses = []

            # Episode loop
            for step in range(self.max_steps_per_episode):
                # Select action (epsilon-greedy)
                action = self.agent.get_action(state)

                # Take step in environment
                next_state, reward, terminated, truncated, step_info = self.env.step(action)

                # Store transition
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, done)

                # Update statistics
                episode_reward += reward
                episode_length += 1

                # Update policy (after warmup and if enough samples)
                if episode >= warmup_episodes and len(self.replay_buffer) >= self.batch_size:
                    if step % update_interval == 0:
                        batch = self.replay_buffer.sample(self.batch_size)
                        loss = self.agent.update(batch)
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

            # Logging # TODO add TensorBoard logging
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

        return self.training_history

    def search(
        self,
        problem: PDPTWProblem,
        solution: PDPTWSolution,
        max_iterations: int = 150,
        epsilon: float = 0.0
    ) -> Tuple[PDPTWSolution, float]:
        """Perform local search using trained RL policy (inference mode).

        This method is used during the memetic algorithm to improve solutions.

        Args:
            problem: PDPTW problem instance
            solution: Initial solution to improve
            max_iterations: Maximum number of iterations
            epsilon: Exploration rate (0.0 = fully greedy)

        Returns:
            Tuple of (best_solution, best_fitness)
        """
        # Reset environment
        state, info = self.env.reset(problem, solution)

        best_solution = solution.clone()
        best_fitness = fitness(problem, best_solution)

        # Inference loop
        for iteration in range(max_iterations):
            # Select action using learned policy (greedy or with small epsilon)
            action = self.agent.get_action(state, epsilon=epsilon)

            # Apply operator
            next_state, reward, terminated, truncated, step_info = self.env.step(action)

            # Update best solution
            if step_info['fitness'] < best_fitness:
                best_solution = self.env.current_solution.clone()
                best_fitness = step_info['fitness']

            # Move to next state
            state = next_state

            if terminated or truncated:
                break

        # Get final best solution
        env_best_solution, env_best_fitness = self.env.get_best_solution()
        if env_best_fitness < best_fitness:
            best_solution = env_best_solution
            best_fitness = env_best_fitness

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
