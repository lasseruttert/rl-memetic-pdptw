"""Configuration loading and saving utilities.

This module provides functions to load YAML configuration files and convert
them to typed dataclass instances, with comprehensive error handling.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, get_origin, get_args
from dataclasses import asdict

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


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file and return dictionary.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If file is empty
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")

    if config_dict is None:
        raise ValueError(f"Empty configuration file: {path}")

    return config_dict


def dict_to_dataclass(data_dict: Dict[str, Any], dataclass_type):
    """Recursively convert dictionary to dataclass instance.

    Handles nested dataclasses and lists of dataclasses.

    Args:
        data_dict: Dictionary to convert (None returns default instance)
        dataclass_type: Target dataclass type

    Returns:
        Instantiated dataclass

    Raises:
        Warning for unknown keys (printed to stdout)
    """
    if data_dict is None:
        return dataclass_type()

    # Get field types from dataclass
    if not hasattr(dataclass_type, '__dataclass_fields__'):
        # Not a dataclass, return as-is
        return data_dict

    field_types = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data_dict.items():
        if key not in field_types:
            print(f"Warning: Unknown config key '{key}' in {dataclass_type.__name__}, ignoring")
            continue

        field_type = field_types[key]

        # Handle nested dataclasses
        if hasattr(field_type, '__dataclass_fields__'):
            kwargs[key] = dict_to_dataclass(value, field_type)
        # Handle lists of dataclasses (e.g., List[OperatorSpec])
        elif get_origin(field_type) is list:
            args = get_args(field_type)
            if args and hasattr(args[0], '__dataclass_fields__'):
                # List of dataclasses
                item_type = args[0]
                kwargs[key] = [dict_to_dataclass(item, item_type) for item in value]
            else:
                # Regular list
                kwargs[key] = value
        else:
            kwargs[key] = value

    return dataclass_type(**kwargs)


def load_config(path: str) -> TrainingConfiguration:
    """Load and validate complete training configuration from YAML.

    Args:
        path: Path to YAML configuration file

    Returns:
        TrainingConfiguration instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If configuration is invalid
        yaml.YAMLError: If YAML syntax is invalid
    """
    config_dict = load_yaml(path)

    # Start with default configuration
    config = TrainingConfiguration()

    # Convert each section to appropriate dataclass
    if 'general' in config_dict:
        config.general = dict_to_dataclass(config_dict['general'], GeneralConfig)

    if 'algorithm' in config_dict:
        algo_dict = config_dict['algorithm']
        # Handle nested acceptance_params
        if 'acceptance_params' in algo_dict:
            acceptance_params = dict_to_dataclass(
                algo_dict['acceptance_params'],
                AcceptanceParams
            )
            # Create new dict with converted acceptance_params
            algo_dict_copy = algo_dict.copy()
            algo_dict_copy['acceptance_params'] = acceptance_params
            # Remove the dict version and use dataclass
            config.algorithm = AlgorithmConfig(**{
                k: v for k, v in algo_dict_copy.items()
                if k in [f.name for f in AlgorithmConfig.__dataclass_fields__.values()]
            })
        else:
            config.algorithm = dict_to_dataclass(algo_dict, AlgorithmConfig)

    if 'network' in config_dict:
        config.network = dict_to_dataclass(config_dict['network'], NetworkConfig)

    if 'dqn' in config_dict:
        config.dqn = dict_to_dataclass(config_dict['dqn'], DQNConfig)

    if 'ppo' in config_dict:
        config.ppo = dict_to_dataclass(config_dict['ppo'], PPOConfig)

    if 'training' in config_dict:
        config.training = dict_to_dataclass(config_dict['training'], TrainingConfig)

    if 'environment' in config_dict:
        config.environment = dict_to_dataclass(config_dict['environment'], EnvironmentConfig)

    if 'validation' in config_dict:
        config.validation = dict_to_dataclass(config_dict['validation'], ValidationConfig)

    if 'testing' in config_dict:
        config.testing = dict_to_dataclass(config_dict['testing'], TestingConfig)

    if 'operators' in config_dict:
        op_dict = config_dict['operators']
        # Handle custom_list of OperatorSpec
        if 'custom_list' in op_dict and op_dict['custom_list']:
            custom_list = [
                dict_to_dataclass(spec, OperatorSpec)
                for spec in op_dict['custom_list']
            ]
            config.operators = OperatorsConfig(
                mode=op_dict.get('mode', 'preset'),
                custom_list=custom_list
            )
        else:
            config.operators = dict_to_dataclass(op_dict, OperatorsConfig)

    # Validate configuration
    config.validate()

    return config


def save_config(config: TrainingConfiguration, path: str):
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        path: Output path for YAML file
    """
    # Convert dataclass to dictionary
    config_dict = asdict(config)

    # Write to YAML with nice formatting
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=100
        )

    print(f"Configuration saved to: {path}")
