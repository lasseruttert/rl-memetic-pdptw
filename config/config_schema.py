"""Configuration schema for RL local search training.

This module defines dataclasses for all configuration sections, providing
type-safe, validated configuration management for the training pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class GeneralConfig:
    """General configuration settings."""
    seed: Optional[int] = 100
    problem_size: int = 100
    categories: List[str] = field(default_factory=lambda: ['lc1', 'lc2', 'lr1', 'lr2'])
    run_name_suffix: Optional[str] = None


@dataclass
class AcceptanceParams:
    """Sub-parameters for acceptance strategies."""
    epsilon_value: float = 0.1              # For epsilon_greedy
    initial_temp: float = 1000.0            # For simulated_annealing
    temp_decay: float = 0.995               # For simulated_annealing
    late_acceptance_length: int = 20        # For late_acceptance
    rising_epsilon_start: float = 0.05      # For rising_epsilon_greedy
    rising_epsilon_end: float = 0.5         # For rising_epsilon_greedy


@dataclass
class AlgorithmConfig:
    """RL algorithm configuration."""
    type: str = "dqn"                       # "dqn" or "ppo"
    acceptance_strategy: str = "greedy"     # Acceptance strategy for local search
    reward_strategy: str = "binary"         # Reward calculation strategy
    search_type: str = "OneShot"            # "OneShot", "Roulette", or "Ranking"
    acceptance_params: AcceptanceParams = field(default_factory=AcceptanceParams)


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])
    learning_rate: float = 1e-4
    gamma: float = 0.90                     # Discount factor
    dropout_rate: float = 0.05
    use_operator_attention: bool = False
    disable_operator_features: bool = False
    feature_weights: Optional[List[float]] = None


@dataclass
class DQNConfig:
    """DQN-specific hyperparameters."""
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9975
    target_update_interval: int = 100
    replay_buffer_capacity: int = 100000
    batch_size: int = 64
    n_step: int = 3                         # N-step returns (1-5 recommended)
    use_prioritized_replay: bool = True
    per_alpha: float = 0.6                  # Prioritization strength (0=uniform, 1=full)
    per_beta_start: float = 0.4             # Initial importance sampling weight


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""
    batch_size: int = 2048                  # Minimum steps before update
    clip_epsilon: float = 0.2               # Policy clipping range
    entropy_coef: float = 0.01              # Entropy regularization coefficient
    value_coef: float = 0.5                 # Value loss weight
    gae_lambda: float = 0.95                # GAE lambda (0.95-0.99 typical)
    max_grad_norm: float = 0.5              # Gradient clipping threshold
    num_epochs: int = 2                     # Update epochs per trajectory
    num_minibatches: int = 2                # Minibatches per epoch
    normalize_advantages: bool = True       # Normalize advantage estimates


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    num_episodes: int = 1000
    max_iterations: int = 200               # Max iterations per episode
    max_no_improvement: int = 50            # Early stopping threshold
    new_instance_interval: int = 5          # Generate new instance every N episodes
    new_solution_interval: int = 1          # Generate new solution every N episodes
    update_interval: int = 1                # Update policy every N steps
    warmup_episodes: int = 10               # Episodes before training starts
    log_interval: int = 10                  # Log detailed metrics every N episodes
    save_interval: int = 1000               # Save checkpoint every N episodes
    save_path: str = "models/rl_local_search"  # Base path for model checkpoints
    tensorboard_dir: Optional[str] = None   # Directory for TensorBoard logs (null = auto)


@dataclass
class EnvironmentConfig:
    """Local search environment configuration."""
    alpha: float = 10.0                     # Weight for fitness improvement in reward
    beta: float = 0.0                       # Weight for feasibility


@dataclass
class ValidationConfig:
    """Validation configuration."""
    skip_validation: bool = False           # Skip validation during training
    mode: str = "fixed_benchmark"           # "fixed_benchmark" or "random_sampled"
    num_instances: int = 10                 # Number of validation instances
    interval: int = 50                      # Evaluate every N episodes
    seeds: List[int] = field(default_factory=lambda: [42, 111, 222, 333, 444])
    runs_per_seed: int = 1                  # Runs per seed for each instance


@dataclass
class TestingConfig:
    """Testing/evaluation configuration."""
    skip_testing: bool = False              # Skip testing phase after training
    test_only: bool = False                 # Skip training, only run testing
    train_ratio: float = 0.8                # Ratio of instances for training
    num_test_problems: int = 50             # Unique test cases per seed
    runs_per_problem: int = 3               # Runs per test case with different RNG
    deterministic_test_rng: bool = False    # Use deterministic RNG during testing
    test_seeds: List[int] = field(default_factory=lambda: [42, 422, 100, 200, 300])
    model_paths: List[str] = field(default_factory=list)  # Model paths for evaluation


@dataclass
class OperatorSpec:
    """Specification for a single operator."""
    type: str                               # Operator class name
    params: Dict[str, Any] = field(default_factory=dict)  # Constructor parameters


@dataclass
class OperatorsConfig:
    """Operator configuration."""
    mode: str = "preset"                    # "preset" or "custom"
    custom_list: List[OperatorSpec] = field(default_factory=list)


@dataclass
class TrainingConfiguration:
    """Complete training configuration.

    This is the main configuration class that contains all configuration sections.
    It provides validation to ensure consistency across all settings.
    """
    general: GeneralConfig = field(default_factory=GeneralConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    operators: OperatorsConfig = field(default_factory=OperatorsConfig)

    def validate(self):
        """Validate configuration consistency and constraints.

        Raises:
            ValueError: If configuration is invalid with detailed error message
        """
        errors = []

        # Algorithm type validation
        if self.algorithm.type not in ["dqn", "ppo"]:
            errors.append(
                f"Invalid algorithm type: '{self.algorithm.type}'. "
                f"Valid options: dqn, ppo"
            )

        # Problem size validation
        valid_sizes = [100, 200, 400, 600, 1000]
        if self.general.problem_size not in valid_sizes:
            errors.append(
                f"Invalid problem size: {self.general.problem_size}. "
                f"Valid options: {', '.join(map(str, valid_sizes))}"
            )

        # Acceptance strategy validation
        valid_strategies = [
            "greedy", "always", "epsilon_greedy", "simulated_annealing",
            "late_acceptance", "rising_epsilon_greedy"
        ]
        if self.algorithm.acceptance_strategy not in valid_strategies:
            errors.append(
                f"Invalid acceptance strategy: '{self.algorithm.acceptance_strategy}'. "
                f"Valid options: {', '.join(valid_strategies)}"
            )

        # Reward strategy validation
        valid_rewards = [
            "binary", "initial_improvement", "old_improvement", "hybrid_improvement",
            "distance_baseline", "log_improvement", "tanh", "component"
        ]
        if self.algorithm.reward_strategy not in valid_rewards:
            errors.append(
                f"Invalid reward strategy: '{self.algorithm.reward_strategy}'. "
                f"Valid options: {', '.join(valid_rewards)}"
            )

        # Search type validation
        valid_search_types = ["OneShot", "Roulette", "Ranking"]
        if self.algorithm.search_type not in valid_search_types:
            errors.append(
                f"Invalid search type: '{self.algorithm.search_type}'. "
                f"Valid options: {', '.join(valid_search_types)}"
            )

        # Search type requires greedy acceptance for Ranking and Roulette
        if self.algorithm.search_type in ["Ranking", "Roulette"]:
            if self.algorithm.acceptance_strategy != "greedy":
                errors.append(
                    f"{self.algorithm.search_type} search type requires greedy acceptance strategy. "
                    f"Current: acceptance_strategy='{self.algorithm.acceptance_strategy}'. "
                    f"Change to: acceptance_strategy='greedy'"
                )

        # Test-only validation
        if self.testing.test_only and self.testing.skip_testing:
            errors.append(
                "Cannot use test_only and skip_testing together. "
                "test_only means 'skip training, run testing only'. "
                "skip_testing means 'skip testing phase'. "
                "These options are mutually exclusive."
            )

        # Operator mode validation
        if self.operators.mode not in ["preset", "custom"]:
            errors.append(
                f"Invalid operator mode: '{self.operators.mode}'. "
                f"Valid options: preset, custom"
            )

        # Custom operator mode requires operators
        if self.operators.mode == "custom" and not self.operators.custom_list:
            errors.append(
                "Custom operator mode requires at least one operator in custom_list. "
                "Either add operators or use mode='preset'"
            )

        # Validation mode validation
        if self.validation.mode not in ["fixed_benchmark", "random_sampled"]:
            errors.append(
                f"Invalid validation mode: '{self.validation.mode}'. "
                f"Valid options: fixed_benchmark, random_sampled"
            )

        # Category validation
        valid_categories = ['lc1', 'lc2', 'lr1', 'lr2', 'lrc1', 'lrc2']
        invalid_categories = [c for c in self.general.categories if c not in valid_categories]
        if invalid_categories:
            errors.append(
                f"Invalid categories: {', '.join(invalid_categories)}. "
                f"Valid options: {', '.join(valid_categories)}"
            )

        # Raise detailed error if validation failed
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValueError(error_message)
