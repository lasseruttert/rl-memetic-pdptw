"""Configuration validation schemas."""

from typing import Dict, Any, List


def validate_config(config: Dict[str, Any]):
    """Validate configuration structure and values.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate problem section
    problem = config.get('problem', {})
    problem_size = problem.get('size')
    if problem_size not in [100, 200, 400, 600, 1000]:
        raise ValueError(f"Invalid problem size: {problem_size}. Must be one of [100, 200, 400, 600, 1000]")

    # Validate algorithm section
    algorithm = config.get('algorithm', {})
    valid_acceptance = [
        'greedy', 'always', 'epsilon_greedy', 'simulated_annealing',
        'late_acceptance', 'rising_epsilon_greedy'
    ]
    acceptance = algorithm.get('acceptance_strategy')
    if acceptance not in valid_acceptance:
        raise ValueError(f"Invalid acceptance strategy: {acceptance}. Must be one of {valid_acceptance}")

    valid_reward = [
        'binary', 'initial_improvement', 'old_improvement', 'hybrid_improvement',
        'distance_baseline', 'log_improvement', 'tanh', 'component'
    ]
    reward = algorithm.get('reward_strategy')
    if reward not in valid_reward:
        raise ValueError(f"Invalid reward strategy: {reward}. Must be one of {valid_reward}")

    # Validate operators section
    operators = config.get('operators', {})
    preset = operators.get('preset')
    if preset == 'custom':
        if 'custom' not in operators or not operators['custom']:
            raise ValueError("Custom operator preset requires 'custom' list with at least one operator")

        # Validate custom operator structure
        for i, op_spec in enumerate(operators['custom']):
            if 'type' not in op_spec:
                raise ValueError(f"Operator {i} missing 'type' field")
            if 'params' not in op_spec:
                raise ValueError(f"Operator {i} (type={op_spec['type']}) missing 'params' field")

    # Validate validation section
    validation = config.get('validation', {})
    valid_modes = ['fixed_benchmark', 'random_sampled']
    val_mode = validation.get('mode')
    if val_mode not in valid_modes:
        raise ValueError(f"Invalid validation mode: {val_mode}. Must be one of {valid_modes}")

    # Validate network section
    network = config.get('network', {})
    hidden_dims = network.get('hidden_dims')
    if not isinstance(hidden_dims, list) or len(hidden_dims) == 0:
        raise ValueError(f"Invalid hidden_dims: {hidden_dims}. Must be a non-empty list of integers")

    # Validate training section
    training = config.get('training', {})
    if training.get('num_episodes', 1) < 1:
        raise ValueError("num_episodes must be at least 1")

    print("[Config] Validation passed")
