"""Configuration module for RL local search training.

This module provides comprehensive YAML-based configuration management
for the RL-based local search training pipeline, including:
- Type-safe configuration schemas (dataclasses)
- YAML loading and saving
- Dynamic operator instantiation
- Configuration validation

Example usage:
    >>> from config import load_config
    >>> config = load_config('my_config.yaml')
    >>> config.validate()  # Raises ValueError if invalid
"""

from .config_schema import (
    TrainingConfiguration,
    GeneralConfig,
    AlgorithmConfig,
    AcceptanceParams,
    NetworkConfig,
    DQNConfig,
    PPOConfig,
    TrainingConfig,
    EnvironmentConfig,
    ValidationConfig,
    TestingConfig,
    OperatorsConfig,
    OperatorSpec,
)

from .config_loader import (
    load_config,
    save_config,
    load_yaml,
    dict_to_dataclass,
)

from .operator_factory import (
    create_operator,
    create_operators_from_config,
    validate_operator_params,
    get_operator_info,
    OPERATOR_REGISTRY,
)

__all__ = [
    # Main configuration class
    'TrainingConfiguration',
    # Configuration sections
    'GeneralConfig',
    'AlgorithmConfig',
    'AcceptanceParams',
    'NetworkConfig',
    'DQNConfig',
    'PPOConfig',
    'TrainingConfig',
    'EnvironmentConfig',
    'ValidationConfig',
    'TestingConfig',
    'OperatorsConfig',
    'OperatorSpec',
    # Loading and saving
    'load_config',
    'save_config',
    'load_yaml',
    'dict_to_dataclass',
    # Operator factory
    'create_operator',
    'create_operators_from_config',
    'validate_operator_params',
    'get_operator_info',
    'OPERATOR_REGISTRY',
]

__version__ = '1.0.0'
