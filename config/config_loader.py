"""Configuration loader for RL Local Search training."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Training configuration container."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from dictionary.

        Args:
            config_dict: Configuration dictionary from YAML
        """
        self.raw = config_dict
        self._parse_sections()

    def _parse_sections(self):
        """Parse configuration sections into attributes."""
        # Extract major sections
        self.problem = self.raw.get('problem', {})
        self.algorithm = self.raw.get('algorithm', {})
        self.network = self.raw.get('network', {})
        self.dqn = self.raw.get('dqn', {})
        self.ppo = self.raw.get('ppo', {})
        self.training = self.raw.get('training', {})
        self.validation = self.raw.get('validation', {})
        self.testing = self.raw.get('testing', {})
        self.operators = self.raw.get('operators', {})
        self.paths = self.raw.get('paths', {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by dot-notation key (e.g., 'problem.size').

        Args:
            key: Dot-notation key path
            default: Default value if key not found

        Returns:
            Value at key path or default
        """
        keys = key.split('.')
        value = self.raw
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def override(self, overrides: Dict[str, Any]):
        """Override config values with CLI arguments.

        Args:
            overrides: Dictionary of overrides (supports dot-notation keys)
        """
        for key, value in overrides.items():
            if value is None:
                continue
            keys = key.split('.')
            current = self.raw
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value
        self._parse_sections()


def load_config(config_path: str, cli_overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file
        cli_overrides: Dictionary of CLI argument overrides

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    # Check file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Apply defaults
    config_dict = apply_defaults(config_dict)

    # Validate schema
    from config.config_schema import validate_config
    validate_config(config_dict)

    # Create config object
    config = Config(config_dict)

    # Apply CLI overrides
    if cli_overrides:
        config.override(cli_overrides)

    return config


def apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values to missing config keys.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with defaults applied
    """

    defaults = {
        'problem': {
            'size': 100,
            'categories': ['lc1', 'lc2', 'lr1', 'lr2'],
            'train_ratio': 0.8,
        },
        'algorithm': {
            'acceptance_strategy': 'greedy',
            'reward_strategy': 'binary',
        },
        'network': {
            'hidden_dims': [128, 128, 64],
            'learning_rate': 1e-4,
            'gamma': 0.90,
        },
        'dqn': {
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.9975,
            'target_update_interval': 100,
            'batch_size': 64,
            'n_step': 3,
            'use_prioritized_replay': True,
            'per_alpha': 0.6,
            'per_beta_start': 0.4,
            'replay_buffer_capacity': 100000,
        },
        'ppo': {
            'batch_size': 2048,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'num_epochs': 2,
            'num_minibatches': 2,
        },
        'training': {
            'num_episodes': 1000,
            'max_iterations': 200,
            'max_no_improvement': 50,
            'warmup_episodes': 10,
            'save_interval': 1000,
            'alpha': 10.0,
            'beta': 0.0,
            'new_instance_interval': 5,
            'new_solution_interval': 1,
            'update_interval': 1,
        },
        'validation': {
            'skip_validation': False,
            'mode': 'fixed_benchmark',
            'num_instances': 10,
            'interval': 50,
            'seeds': [42, 111, 222, 333, 444],
            'runs_per_seed': 1,
        },
        'testing': {
            'num_test_problems': 50,
            'runs_per_problem': 3,
            'deterministic_test_rng': False,
            'test_seeds': [42, 422, 100, 200, 300],
        },
        'operators': {
            'preset': 'standard',
        },
        'paths': {
            'suffix': '',  # Default empty suffix
            'save_path': 'models/rl_local_search_{algorithm}_{problem_size}_{acceptance}_{reward}{attention}_{seed}_o',
            'tensorboard_dir': 'runs/{run_name}',
            'log_dir': 'logs',
        },
    }

    # Deep merge defaults
    return deep_merge(defaults, config)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary with defaults
        override: Dictionary with overrides

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
