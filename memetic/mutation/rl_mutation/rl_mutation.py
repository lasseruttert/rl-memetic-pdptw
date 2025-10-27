"""Reinforcement Learning-based mutation for PDPTW with population awareness."""

import numpy as np
import random
import time
from typing import Callable, Optional, Dict, List, Tuple

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
from memetic.fitness.fitness import fitness
from memetic.mutation.base_mutation import BaseMutation

from memetic.mutation.rl_mutation.mutation_env import MutationEnv
from memetic.mutation.rl_mutation.dqn_mutation_agent import DQNMutationAgent
from memetic.mutation.rl_mutation.ppo_mutation_agent import PPOMutationAgent
from memetic.mutation.rl_mutation.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from memetic.local_search.naive_local_search import NaiveLocalSearch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class RLMutation(BaseMutation):
    """RL-based mutation that learns to select operators adaptively with population awareness.

    Supports two RL algorithms:
    - DQN (Deep Q-Learning): Off-policy value-based learning
    - PPO (Proximal Policy Optimization): On-policy policy gradient learning

    Both learn which operators to apply based on problem, solution, and population features,
    enabling context-sensitive and population-aware operator selection.
    """

    def __init__(
        self,
        operators: List[BaseOperator],
        rl_algorithm: str = "dqn",
        hidden_dims: List[int] = [128, 128, 64],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        # DQN-specific parameters
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        target_update_interval: int = 100,
        replay_buffer_capacity: int = 100000,
        batch_size: int = 64,
        n_step: int = 3,
        use_prioritized_replay: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        # PPO-specific parameters
        ppo_clip_epsilon: float = 0.2,
        ppo_entropy_coef: float = 0.01,
        ppo_value_coef: float = 0.5,
        ppo_gae_lambda: float = 0.95,
        ppo_max_grad_norm: float = 0.5,
        ppo_num_epochs: int = 2,
        ppo_num_minibatches: int = 2,
        ppo_normalize_advantages: bool = True,
        # Common parameters
        alpha: float = 1.0,
        acceptance_strategy: str = "greedy",
        reward_strategy: str = "binary",
        max_steps: int = 100,
        max_no_improvement: Optional[int] = None,
        device: str = "cuda",
        verbose: bool = False
    ):
        """Initialize RL-based mutation.

        Args:
            operators: List of mutation operators to choose from
            rl_algorithm: RL algorithm to use ("dqn" or "ppo")
            hidden_dims: Hidden layer dimensions for networks
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards

            DQN-specific parameters:
                epsilon_start: Initial exploration rate
                epsilon_end: Final exploration rate
                epsilon_decay: Epsilon decay rate per episode
                target_update_interval: Steps between target network updates
                replay_buffer_capacity: Size of replay buffer
                batch_size: Batch size for training
                n_step: Number of steps for n-step returns
                use_prioritized_replay: Whether to use prioritized experience replay
                per_alpha: Prioritization strength
                per_beta_start: Initial importance sampling weight

            PPO-specific parameters:
                ppo_clip_epsilon: Clipping parameter for PPO objective
                ppo_entropy_coef: Coefficient for entropy bonus
                ppo_value_coef: Coefficient for value loss
                ppo_gae_lambda: Lambda for GAE
                ppo_max_grad_norm: Maximum gradient norm for clipping
                ppo_num_epochs: Number of update epochs per trajectory
                ppo_num_minibatches: Number of minibatches per epoch
                ppo_normalize_advantages: Whether to normalize advantages

            Common parameters:
                alpha: Weight for fitness improvement in reward
                acceptance_strategy: Acceptance strategy for solutions
                reward_strategy: Strategy for calculating rewards
                max_steps: Maximum steps per episode
                max_no_improvement: Early stopping after N steps without improvement
                device: Device for training ("cuda" or "cpu")
                verbose: Whether to print training progress
        """
        super().__init__()

        # Validate rl_algorithm
        if rl_algorithm not in ["dqn", "ppo"]:
            raise ValueError(f"Invalid rl_algorithm: {rl_algorithm}. Must be 'dqn' or 'ppo'")

        self.rl_algorithm = rl_algorithm
        self.operators = operators
        self.acceptance_strategy = acceptance_strategy
        self.reward_strategy = reward_strategy
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.verbose = verbose

        # Algorithm-specific parameters
        if rl_algorithm == "dqn":
            self.n_step = n_step
            self.use_prioritized_replay = use_prioritized_replay

        # Environment
        self.env = MutationEnv(
            operators=operators,
            alpha=alpha,
            acceptance_strategy=acceptance_strategy,
            reward_strategy=reward_strategy,
            max_steps=max_steps,
            max_no_improvement=max_no_improvement
        )

        # Agent initialization (DQN or PPO)
        state_dim = self.env.observation_space.shape[0]
        action_dim = len(operators)

        if rl_algorithm == "dqn":
            # DQN Agent
            self.agent = DQNMutationAgent(
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

            # Replay buffer for DQN
            if use_prioritized_replay:
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
                print(f"Algorithm: DQN")
                print(f"n-step returns: n={n_step}")

        elif rl_algorithm == "ppo":
            # PPO Agent
            self.agent = PPOMutationAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=ppo_gae_lambda,
                clip_epsilon=ppo_clip_epsilon,
                entropy_coef=ppo_entropy_coef,
                value_coef=ppo_value_coef,
                max_grad_norm=ppo_max_grad_norm,
                num_epochs=ppo_num_epochs,
                batch_size=batch_size,
                num_minibatches=ppo_num_minibatches,
                normalize_advantages=ppo_normalize_advantages,
                device=device
            )

            # No replay buffer for PPO (on-policy)
            self.replay_buffer = None

            if self.verbose:
                print(f"Algorithm: PPO")
                print(f"Clip epsilon: {ppo_clip_epsilon}, Entropy coef: {ppo_entropy_coef}")
                print(f"GAE lambda: {ppo_gae_lambda}, Num epochs: {ppo_num_epochs}")

        # Training mode flag
        self.training_mode = False

        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_best_fitness': [],
            'episode_best_diversity': [],
            'losses': [],
            'epsilon_values': [] if rl_algorithm == "dqn" else []
        }

        self.ls = NaiveLocalSearch(max_iterations=10, max_no_improvement=10)

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
        population_generator: Callable[[PDPTWProblem], List[PDPTWSolution]],
        num_episodes: int = 2000,
        new_instance_interval: int = 50,
        new_population_interval: int = 10,
        new_solution_interval: int = 1,
        update_interval: int = 1,
        warmup_episodes: int = 10,
        save_interval: int = 100,
        save_path: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        seed: Optional[int] = None,
        log_interval: int = 10,
        validation_set: Optional[List[Tuple[PDPTWProblem, List[PDPTWSolution]]]] = None,
        validation_interval: int = 100
    ) -> Dict:
        """Train the RL policy for mutation operator selection.

        Training loop structure:
        - Generate new instance every new_instance_interval episodes
        - Generate new population every new_population_interval episodes
        - Select new solution from population every new_solution_interval episodes

        Args:
            problem_generator: Function that generates problem instances
            population_generator: Function that generates populations given a problem
            num_episodes: Number of training episodes
            new_instance_interval: Generate new instance every N episodes
            new_population_interval: Generate new population every N episodes
            new_solution_interval: Select new solution from population every N episodes
            update_interval: Update policy every N steps
            warmup_episodes: Number of episodes before training starts
            save_interval: Save model every N episodes
            save_path: Path to save model checkpoints
            tensorboard_dir: Directory for TensorBoard logs
            seed: Random seed for reproducibility
            log_interval: Log detailed metrics every N episodes
            validation_set: Optional list of (problem, population) tuples for validation
            validation_interval: Evaluate on validation set every N episodes

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

        # Update beta_frames for prioritized replay if using DQN
        if self.rl_algorithm == "dqn" and self.use_prioritized_replay:
            self.replay_buffer.beta_frames = num_episodes * self.max_steps
            if self.verbose:
                print(f"PER beta will anneal from {self.replay_buffer.beta_start} to 1.0 over {self.replay_buffer.beta_frames} frames")

        # Initialize TensorBoard writer
        writer = None
        if tensorboard_dir and TENSORBOARD_AVAILABLE:
            writer = SummaryWriter(log_dir=tensorboard_dir)
            if self.verbose:
                print(f"TensorBoard logging enabled: {tensorboard_dir}")

            # Log hyperparameters
            hparams = self._create_hparams_dict(num_episodes, new_instance_interval,
                                               new_population_interval, new_solution_interval, seed)
            writer.add_text('Hyperparameters', str(hparams), 0)

        elif tensorboard_dir and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard requested but not available. Install with: pip install tensorboard")

        # Initialize instance, population, and solution
        instance = problem_generator()
        population = population_generator(instance)
        population_fitnesses = [None] * len(population)
        for i in range(len(population)):
            # Apply local search to each solution in the population
            improved_solution, fit = self.ls.search(instance, population[i])
            population[i] = improved_solution
            population_fitnesses[i] = fit
        random_index = random.randint(0, len(population) - 1)
        current_solution = population[random_index]

        # Pre-calculate fitnesses and vehicle counts for efficiency
        population_num_vehicles = [sol.num_vehicles_used for sol in population]

        if self.verbose:
            print(f"Starting RL Mutation training for {num_episodes} episodes...")
            print(f"State dim: {self.agent.state_dim}, Action dim: {self.agent.action_dim}")
            print(f"Operators: {[op.name if hasattr(op, 'name') else type(op).__name__ for op in self.operators]}")

        # PPO-specific: Track accumulated steps across episodes
        if self.rl_algorithm == "ppo":
            ppo_accumulated_steps = 0
            if self.verbose:
                print(f"PPO will update every {self.batch_size} steps (accumulating across episodes)")

        # Main training loop
        for episode in range(num_episodes):
            episode_start_time = time.time()

            # Generate new instance periodically
            if episode % new_instance_interval == 0:
                instance = problem_generator()
                population = population_generator(instance)
                population_fitnesses = [None] * len(population)
                for i in range(len(population)):
                    # Apply local search to each solution in the population
                    improved_solution, fitness = self.ls.search(instance, population[i])
                    population[i] = improved_solution
                    population_fitnesses[i] = fitness
                random_index = random.randint(0, len(population) - 1)
                current_solution = population[random_index]
                population_num_vehicles = [sol.num_vehicles_used for sol in population]
                if self.verbose:
                    print(f"Episode {episode}: New instance generated ({instance.num_requests} requests)")

            # Generate new population periodically
            elif episode % new_population_interval == 0:
                population = population_generator(instance)
                population_fitnesses = [None] * len(population)
                for i in range(len(population)):
                    # Apply local search to each solution in the population
                    improved_solution, fitness = self.ls.search(instance, population[i])
                    population[i] = improved_solution
                    population_fitnesses[i] = fitness
                random_index = random.randint(0, len(population) - 1)
                current_solution = population[random_index]
                population_num_vehicles = [sol.num_vehicles_used for sol in population]
                if self.verbose and episode % (new_population_interval * 5) == 0:
                    print(f"Episode {episode}: New population generated")

            # Select new solution from population periodically
            elif episode % new_solution_interval == 0:
                random_index = random.randint(0, len(population) - 1)
                current_solution = population[random_index]

            # Reset environment with current instance, population, and solution
            state, info = self.env.reset(
                instance,
                population,
                population_fitnesses,
                population_num_vehicles,
                current_solution
            )

            episode_reward = 0.0
            episode_length = 0
            step_losses = []
            episode_actions = []

            # Metric tracking
            num_accepted = 0
            num_rejected = 0
            accepted_improvements = []
            rejected_degradations = []

            # Per-operator tracking
            operator_uses = [0] * len(self.operators)
            operator_successes = [0] * len(self.operators)
            operator_accepted = [0] * len(self.operators)
            operator_improvements = [[] for _ in range(len(self.operators))]

            # Population-specific tracking
            diversity_changes = []
            population_fitness_changes = []

            # Episode loop - conditional logic for DQN vs PPO
            if self.rl_algorithm == "dqn":
                # DQN: Step-by-step updates with replay buffer
                for step in range(self.max_steps):
                    # Select action (epsilon-greedy)
                    action = self.agent.get_action(state)
                    episode_actions.append(action)

                    # Take step in environment
                    next_state, reward, terminated, truncated, step_info = self.env.step(action)

                    # Store transition in replay buffer
                    done = terminated or truncated
                    self.replay_buffer.add(state, action, reward, next_state, done)

                    # Update statistics
                    episode_reward += reward
                    episode_length += 1

                    # Track metrics
                    if step_info['accepted']:
                        num_accepted += 1
                        fitness_improvement = step_info.get('new_fitness', 0) - step_info['fitness']
                        accepted_improvements.append(fitness_improvement)
                    else:
                        num_rejected += 1
                        fitness_improvement = step_info.get('new_fitness', 0) - step_info['fitness']
                        rejected_degradations.append(fitness_improvement)

                    # Operator effectiveness tracking
                    operator_uses[action] += 1
                    if step_info.get('improvement', False):
                        operator_successes[action] += 1
                    if step_info['accepted']:
                        operator_accepted[action] += 1
                    fitness_change = step_info.get('new_fitness', 0) - step_info['fitness']
                    operator_improvements[action].append(fitness_change)

                    # Population metrics tracking
                    if 'population_mean_fitness' in step_info:
                        pop_fitness_change = info.get('population_mean_fitness', 0) - step_info.get('population_mean_fitness', 0)
                        population_fitness_changes.append(pop_fitness_change)

                    # Update policy (after warmup and if enough samples)
                    if episode >= warmup_episodes and len(self.replay_buffer) >= self.batch_size:
                        if step % update_interval == 0:
                            # Sample batch (with priorities if using PER)
                            if self.use_prioritized_replay:
                                batch, indices, weights = self.replay_buffer.sample(self.batch_size)
                                loss, td_errors = self.agent.update(batch, weights)
                                self.replay_buffer.update_priorities(indices, td_errors)
                            else:
                                batch = self.replay_buffer.sample(self.batch_size)
                                loss, _ = self.agent.update(batch)
                            step_losses.append(loss)

                    # Move to next state
                    state = next_state

                    if terminated or truncated:
                        break

                # Decay epsilon for DQN
                self.agent.decay_epsilon()

            elif self.rl_algorithm == "ppo":
                # PPO: Collect full trajectory, then update
                for step in range(self.max_steps):
                    # Select action using policy
                    action, log_prob, value = self.agent.get_action(state)
                    episode_actions.append(action)

                    # Take step in environment
                    next_state, reward, terminated, truncated, step_info = self.env.step(action)

                    # Store transition in trajectory
                    done = terminated or truncated
                    self.agent.store_transition(state, action, reward, log_prob, value, done)

                    # Update statistics
                    episode_reward += reward
                    episode_length += 1

                    # Track metrics (same as DQN)
                    if step_info['accepted']:
                        num_accepted += 1
                        fitness_improvement = step_info.get('new_fitness', 0) - step_info['fitness']
                        accepted_improvements.append(fitness_improvement)
                    else:
                        num_rejected += 1
                        fitness_improvement = step_info.get('new_fitness', 0) - step_info['fitness']
                        rejected_degradations.append(fitness_improvement)

                    # Operator effectiveness tracking
                    operator_uses[action] += 1
                    if step_info.get('improvement', False):
                        operator_successes[action] += 1
                    if step_info['accepted']:
                        operator_accepted[action] += 1
                    fitness_change = step_info.get('new_fitness', 0) - step_info['fitness']
                    operator_improvements[action].append(fitness_change)

                    # Population metrics tracking
                    if 'population_mean_fitness' in step_info:
                        pop_fitness_change = info.get('population_mean_fitness', 0) - step_info.get('population_mean_fitness', 0)
                        population_fitness_changes.append(pop_fitness_change)

                    # Move to next state
                    state = next_state

                    if terminated or truncated:
                        break

                # Accumulate steps for PPO
                ppo_accumulated_steps += episode_length

                # Update PPO policy when enough steps accumulated (if past warmup)
                if episode >= warmup_episodes and ppo_accumulated_steps >= self.batch_size:
                    ppo_stats = self.agent.update(next_state=state)
                    if ppo_stats:
                        step_losses.append(ppo_stats['policy_loss'])
                        # Store PPO-specific stats
                        if not hasattr(self, 'ppo_episode_stats'):
                            self.ppo_episode_stats = {
                                'policy_loss': [],
                                'value_loss': [],
                                'entropy': [],
                                'clip_fraction': [],
                                'approx_kl': [],
                                'explained_variance': []
                            }
                        for key in self.ppo_episode_stats:
                            self.ppo_episode_stats[key].append(ppo_stats[key])

                        if self.verbose and episode % 10 == 0:
                            print(f"  PPO update: {ppo_accumulated_steps} steps accumulated")
                        ppo_accumulated_steps = 0

            # Get best solution from episode
            best_solution, best_measures = self.env.get_best_solution()
            best_fitness = best_measures['fitness']
            best_diversity = best_measures['diversity_score']

            # Record episode statistics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['episode_best_fitness'].append(best_fitness)
            self.training_history['episode_best_diversity'].append(best_diversity)
            if self.rl_algorithm == "dqn":
                self.training_history['epsilon_values'].append(self.agent.epsilon)

            avg_loss = np.mean(step_losses) if step_losses else 0.0
            if step_losses:
                self.training_history['losses'].append(avg_loss)

            # TensorBoard logging
            if writer:
                self._log_tensorboard_metrics(writer, episode, episode_reward, best_fitness, best_measures,
                                             episode_length, avg_loss, step_losses, state, episode_actions,
                                             num_accepted, num_rejected, accepted_improvements,
                                             rejected_degradations, operator_uses, operator_successes,
                                             operator_accepted, operator_improvements, info, log_interval)

            # Console logging
            if self.verbose and (episode % 10 == 0 or episode == num_episodes - 1):
                self._print_console_log(episode, num_episodes, episode_reward, best_fitness,
                                       episode_length, avg_loss, start_time, episode_start_time)

            # Save checkpoint
            if save_path and episode % save_interval == 0 and episode > 0:
                checkpoint_path = f"{save_path}_episode_{episode}.pt"
                self.agent.save(checkpoint_path)
                if self.verbose:
                    print(f"Saved checkpoint to {checkpoint_path}")

        # Final PPO update if there are remaining accumulated steps
        if self.rl_algorithm == "ppo" and ppo_accumulated_steps > 0:
            if self.verbose:
                print(f"\nFinal PPO update with {ppo_accumulated_steps} remaining steps...")
            ppo_stats = self.agent.update(next_state=state)
            if ppo_stats and self.verbose:
                print(f"  Policy loss: {ppo_stats['policy_loss']:.4f}")

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
            if self.rl_algorithm == "dqn":
                print(f"Final epsilon: {self.agent.epsilon:.3f}")
                print(f"Replay buffer size: {len(self.replay_buffer)}")
            elif self.rl_algorithm == "ppo":
                print(f"Total updates: {self.agent.update_count}")

        # Close TensorBoard writer
        if writer:
            writer.close()
            if self.verbose:
                print(f"TensorBoard logs saved to {tensorboard_dir}")

        return self.training_history

    def _create_hparams_dict(self, num_episodes, new_instance_interval,
                           new_population_interval, new_solution_interval, seed):
        """Create hyperparameters dictionary for tensorboard logging."""
        hparams = {
            'rl_algorithm': self.rl_algorithm,
            'learning_rate': self.agent.optimizer.param_groups[0]['lr'],
            'gamma': self.agent.gamma,
            'batch_size': self.batch_size,
            'num_episodes': num_episodes,
            'max_steps': self.max_steps,
            'acceptance_strategy': self.acceptance_strategy,
            'reward_strategy': self.reward_strategy,
            'alpha': self.env.alpha,
            'new_instance_interval': new_instance_interval,
            'new_population_interval': new_population_interval,
            'new_solution_interval': new_solution_interval,
            'seed': seed if seed is not None else 'random',
        }

        # Add algorithm-specific hyperparameters
        if self.rl_algorithm == "dqn":
            hparams.update({
                'epsilon_start': self.agent.epsilon,
                'epsilon_end': self.agent.epsilon_end,
                'epsilon_decay': self.agent.epsilon_decay,
                'target_update_interval': self.agent.target_update_interval,
                'n_step': self.n_step,
                'use_prioritized_replay': self.use_prioritized_replay,
            })
        elif self.rl_algorithm == "ppo":
            hparams.update({
                'clip_epsilon': self.agent.clip_epsilon,
                'entropy_coef': self.agent.entropy_coef,
                'value_coef': self.agent.value_coef,
                'gae_lambda': self.agent.gae_lambda,
                'num_epochs': self.agent.num_epochs,
                'num_minibatches': self.agent.num_minibatches,
            })

        return hparams

    def _log_tensorboard_metrics(self, writer, episode, episode_reward, best_fitness, best_measures,
                                 episode_length, avg_loss, step_losses, state, episode_actions,
                                 num_accepted, num_rejected, accepted_improvements, rejected_degradations,
                                 operator_uses, operator_successes, operator_accepted, operator_improvements,
                                 info, log_interval):
        """Log metrics to TensorBoard."""
        # Always log core metrics
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/BestFitness', best_fitness, episode)
        writer.add_scalar('Episode/Diversity', best_measures['diversity_score'], episode)
        writer.add_scalar('Episode/Length', episode_length, episode)

        # Algorithm-specific metrics
        if self.rl_algorithm == "dqn":
            writer.add_scalar('Training/Epsilon', self.agent.epsilon, episode)
            if step_losses:
                writer.add_scalar('Training/Loss', avg_loss, episode)
        elif self.rl_algorithm == "ppo":
            if hasattr(self, 'ppo_episode_stats') and self.ppo_episode_stats['policy_loss']:
                writer.add_scalar('Training/PolicyLoss', self.ppo_episode_stats['policy_loss'][-1], episode)
                writer.add_scalar('Training/ValueLoss', self.ppo_episode_stats['value_loss'][-1], episode)
                writer.add_scalar('Training/Entropy', self.ppo_episode_stats['entropy'][-1], episode)
                writer.add_scalar('Training/ClipFraction', self.ppo_episode_stats['clip_fraction'][-1], episode)
                writer.add_scalar('Training/ApproxKL', self.ppo_episode_stats['approx_kl'][-1], episode)
                writer.add_scalar('Training/ExplainedVariance', self.ppo_episode_stats['explained_variance'][-1], episode)

        # Population metrics (NEW for mutation)
        if 'population_mean_fitness' in info:
            writer.add_scalar('Population/MeanFitness', info['population_mean_fitness'], episode)
            writer.add_scalar('Population/BestFitness', info['population_best_fitness'], episode)

        if 'diversity_score' in best_measures:
            writer.add_scalar('Diversity/BestSolutionDiversityScore', best_measures['diversity_score'], episode)
            writer.add_scalar('Diversity/AvgDistanceToPopulation', best_measures['avg_distance_to_pop'], episode)
            writer.add_scalar('Diversity/MinDistanceToPopulation', best_measures['min_distance_to_pop'], episode)

        # Log detailed metrics every log_interval episodes
        if episode % log_interval == 0 or episode == 0:
            # Operator selection statistics
            action_counts = np.bincount(episode_actions, minlength=len(self.operators))
            for i, count in enumerate(action_counts):
                writer.add_scalar(f'Operators/operator_{i}_count', count, episode)
                writer.add_scalar(f'Operators/operator_{i}_ratio',
                                count / len(episode_actions) if episode_actions else 0, episode)

            # Q-values/action preferences for current state
            if self.rl_algorithm == "dqn":
                q_values = self.agent.get_q_values(state, update_stats=False)
                for i, q_val in enumerate(q_values):
                    writer.add_scalar(f'QValues/operator_{i}_qvalue', q_val, episode)
                writer.add_scalar('QValues/max_qvalue', np.max(q_values), episode)
                writer.add_scalar('QValues/mean_qvalue', np.mean(q_values), episode)
                writer.add_scalar('QValues/std_qvalue', np.std(q_values), episode)
            elif self.rl_algorithm == "ppo":
                action_prefs = self.agent.get_q_values(state, update_stats=False)
                for i, pref in enumerate(action_prefs):
                    writer.add_scalar(f'ActionPrefs/operator_{i}_pref', pref, episode)

            # Acceptance rate tracking
            total_moves = num_accepted + num_rejected
            acceptance_rate = num_accepted / total_moves if total_moves > 0 else 0
            avg_accepted_improvement = np.mean(accepted_improvements) if accepted_improvements else 0
            avg_rejected_degradation = np.mean(rejected_degradations) if rejected_degradations else 0

            writer.add_scalar('Acceptance/acceptance_rate', acceptance_rate, episode)
            writer.add_scalar('Acceptance/num_accepted', num_accepted, episode)
            writer.add_scalar('Acceptance/num_rejected', num_rejected, episode)
            writer.add_scalar('Acceptance/avg_accepted_improvement', avg_accepted_improvement, episode)
            writer.add_scalar('Acceptance/avg_rejected_degradation', avg_rejected_degradation, episode)

            # Operator effectiveness
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

            # Raw solution metrics
            best_solution, _ = self.env.get_best_solution()
            writer.add_scalar('Solution/raw_total_distance', best_solution.total_distance, episode)
            writer.add_scalar('Solution/raw_num_vehicles', best_solution.num_vehicles_used, episode)
            writer.add_scalar('Solution/is_feasible', int(best_solution.is_feasible), episode)

        # Rolling averages
        if len(self.training_history['episode_rewards']) >= 10:
            avg_reward_10 = np.mean(self.training_history['episode_rewards'][-10:])
            avg_fitness_10 = np.mean(self.training_history['episode_best_fitness'][-10:])
            avg_diversity_10 = np.mean(self.training_history['episode_best_diversity'][-10:])
            writer.add_scalar('Episode/Reward_Avg10', avg_reward_10, episode)
            writer.add_scalar('Episode/Fitness_Avg10', avg_fitness_10, episode)
            writer.add_scalar('Episode/Diversity_Avg10', avg_diversity_10, episode)

        if len(self.training_history['episode_rewards']) >= 100:
            avg_reward_100 = np.mean(self.training_history['episode_rewards'][-100:])
            avg_fitness_100 = np.mean(self.training_history['episode_best_fitness'][-100:])
            avg_diversity_100 = np.mean(self.training_history['episode_best_diversity'][-100:])
            writer.add_scalar('Episode/Reward_Avg100', avg_reward_100, episode)
            writer.add_scalar('Episode/Fitness_Avg100', avg_fitness_100, episode)
            writer.add_scalar('Episode/Diversity_Avg100', avg_diversity_100, episode)

    def _print_console_log(self, episode, num_episodes, episode_reward, best_fitness,
                          episode_length, avg_loss, start_time, episode_start_time):
        """Print console log for training progress."""
        elapsed = time.time() - start_time
        episode_time = time.time() - episode_start_time
        avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
        avg_fitness = np.mean(self.training_history['episode_best_fitness'][-10:])

        log_msg = (f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} (avg: {avg_reward:.2f}) | "
                  f"Fitness: {best_fitness:.2f} (avg: {avg_fitness:.2f}) | "
                  f"Steps: {episode_length} | ")

        if self.rl_algorithm == "dqn":
            log_msg += f"eps: {self.agent.epsilon:.3f} | "
        elif self.rl_algorithm == "ppo" and hasattr(self, 'ppo_episode_stats') and self.ppo_episode_stats['entropy']:
            log_msg += f"entropy: {self.ppo_episode_stats['entropy'][-1]:.3f} | "

        log_msg += (f"Loss: {avg_loss:.4f} | "
                   f"Time: {episode_time:.2f}s | "
                   f"Total: {elapsed:.2f}s")

        print(log_msg)

    def save(self, path: str):
        """Save the RL agent and training history."""
        self.agent.save(path)
        import pickle
        history_path = path.replace('.pt', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)

    def load(self, path: str):
        """Load a trained RL agent."""
        self.agent.load(path)
        import pickle
        history_path = path.replace('.pt', '_history.pkl')
        try:
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)
        except FileNotFoundError:
            if self.verbose:
                print(f"Warning: Training history not found at {history_path}")

    def mutate(self, problem: PDPTWProblem, solution: PDPTWSolution, population: List[PDPTWSolution]) -> PDPTWSolution:
        """Mutate the given solution using trained RL agent for operator selection.

        This method uses the trained RL policy to adaptively select mutation operators
        based on problem, solution, and population features. Unlike naive mutation that
        randomly selects operators, this method uses learned operator selection.

        Args:
            problem: The PDPTW problem instance
            solution: The solution to mutate
            population: The current population (required for population-aware features)

        Returns:
            The best solution found during mutation (not necessarily the final solution)

        Raises:
            ValueError: If population is None (RL mutation requires population awareness)
        """
        # Validate population
        if population is None:
            raise ValueError("RLMutation requires a population for state features. Population cannot be None.")

        # Calculate fitness values for population and input solution
        pop_fitnesses = [fitness(problem, sol) for sol in population]
        pop_num_vehicles = [sol.num_vehicles_used for sol in population]

        # Reset environment with problem, population, and solution
        state, info = self.env.reset(problem, population, pop_fitnesses, pop_num_vehicles, solution)

        # Run mutation loop for max_steps iterations
        for step in range(self.max_steps):
            # Select action using trained agent (deterministic mode for inference)
            if self.rl_algorithm == "dqn":
                # DQN: Use epsilon=0 for greedy (deterministic) action selection
                action = self.agent.get_action(state, epsilon=0.0, update_stats=False)
            else:  # PPO
                # PPO: Use deterministic=True for argmax action selection
                action, _, _ = self.agent.get_action(state, deterministic=True, update_stats=False)

            # Apply selected operator through environment
            next_state, reward, terminated, truncated, step_info = self.env.step(action)

            # Update state
            state = next_state

            # Check if episode terminated
            if terminated or truncated:
                break

        # Return best solution found during mutation
        best_solution, best_measures = self.env.get_best_solution()
        return best_solution
