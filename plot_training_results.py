"""Script to parse training logs and create comparison plots by reward and acceptance strategies."""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def parse_log_filename(filename):
    """Parse configuration from log filename.

    Expected format: training_rl_local_search_{algo}_{size}_{acceptance}_{reward}[_attention]_seed{seed}_{timestamp}.log

    Returns:
        dict with keys: rl_algorithm, problem_size, acceptance_strategy, reward_strategy,
                       use_attention, seed, timestamp
        or None if parsing fails
    """
    # Pattern for rl_local_search logs
    pattern = r'training_rl_local_search_(?P<algo>\w+)_(?P<size>\d+)_(?P<acceptance>\w+)_(?P<reward>\w+)(?:_attention)?_seed(?P<seed>\d+)_(?P<timestamp>\d+)\.log'

    match = re.match(pattern, filename)
    if not match:
        return None

    config = {
        'rl_algorithm': match.group('algo'),
        'problem_size': int(match.group('size')),
        'acceptance_strategy': match.group('acceptance'),
        'reward_strategy': match.group('reward'),
        'use_attention': '_attention' in filename,
        'seed': int(match.group('seed')),
        'timestamp': match.group('timestamp')
    }

    return config


def parse_episode_line(line):
    """Parse an episode log line to extract metrics.

    Example line:
    Episode 10/1000 | Reward: -64.00 (avg: -99.00) | Fitness: 1296.56 (avg: 4704.51) | Steps: 200 | eps: 0.973 | Loss: 0.5724 | Time: 13.38s | Total: 24.83s

    Returns:
        dict with episode metrics or None if parsing fails
    """
    episode_pattern = r'Episode (\d+)/\d+ \| Reward: ([+-]?\d+\.?\d*) \(avg: ([+-]?\d+\.?\d*)\) \| Fitness: ([+-]?\d+\.?\d*) \(avg: ([+-]?\d+\.?\d*)\) \| Steps: (\d+) \| eps: ([+-]?\d+\.?\d*) \| Loss: ([+-]?\d+\.?\d*)'

    match = re.search(episode_pattern, line)
    if not match:
        return None

    return {
        'episode': int(match.group(1)),
        'reward': float(match.group(2)),
        'reward_avg': float(match.group(3)),
        'fitness': float(match.group(4)),
        'fitness_avg': float(match.group(5)),
        'steps': int(match.group(6)),
        'epsilon': float(match.group(7)),
        'loss': float(match.group(8))
    }


def parse_log_file(log_path):
    """Parse a complete log file.

    Returns:
        dict with 'config' and 'episodes' keys
    """
    filename = os.path.basename(log_path)
    config = parse_log_filename(filename)

    if config is None:
        return None

    episodes = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            episode_data = parse_episode_line(line)
            if episode_data:
                episodes.append(episode_data)

    if not episodes:
        return None

    return {
        'config': config,
        'episodes': episodes
    }


def organize_data(parsed_logs):
    """Organize parsed logs by reward strategy and acceptance strategy.

    Returns:
        dict: {(rl_algo, reward_strategy): {acceptance_strategy: [list of episode data arrays]}}
    """
    organized = defaultdict(lambda: defaultdict(list))

    for log_data in parsed_logs:
        if log_data is None:
            continue

        config = log_data['config']
        episodes = log_data['episodes']

        key = (config['rl_algorithm'], config['reward_strategy'])
        acceptance = config['acceptance_strategy']

        # Convert episodes to structured arrays for easier plotting
        episode_nums = np.array([e['episode'] for e in episodes])
        rewards = np.array([e['reward'] for e in episodes])
        rewards_avg = np.array([e['reward_avg'] for e in episodes])
        fitness = np.array([e['fitness'] for e in episodes])
        fitness_avg = np.array([e['fitness_avg'] for e in episodes])
        epsilon = np.array([e['epsilon'] for e in episodes])
        loss = np.array([e['loss'] for e in episodes])

        organized[key][acceptance].append({
            'episode': episode_nums,
            'reward': rewards,
            'reward_avg': rewards_avg,
            'fitness': fitness,
            'fitness_avg': fitness_avg,
            'epsilon': epsilon,
            'loss': loss,
            'seed': config['seed']
        })

    return organized


def compute_statistics(runs):
    """Compute mean and std across multiple runs (seeds).

    Args:
        runs: List of dicts with episode data

    Returns:
        dict with 'episode', 'mean', 'std' for each metric
    """
    # Find common episode range (use minimum length)
    min_episodes = min(len(run['episode']) for run in runs)

    stats = {}
    metrics = ['reward', 'reward_avg', 'fitness', 'fitness_avg', 'epsilon', 'loss']

    for metric in metrics:
        # Collect data for this metric from all runs
        data = np.array([run[metric][:min_episodes] for run in runs])

        stats[metric] = {
            'episode': runs[0]['episode'][:min_episodes],
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0)
        }

    return stats


def plot_metric(data_by_acceptance, metric_key, rl_algo, reward_strategy, output_dir):
    """Create a single plot comparing acceptance strategies for a given metric.

    Args:
        data_by_acceptance: dict {acceptance_strategy: [list of runs]}
        metric_key: which metric to plot (e.g., 'reward_avg', 'fitness_avg')
        rl_algo: RL algorithm name (e.g., 'dqn', 'ppo')
        reward_strategy: Reward strategy name (e.g., 'binary', 'hybrid_improvement')
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define nice colors for different acceptance strategies
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_map = {}

    # Sort acceptance strategies for consistent ordering
    acceptance_strategies = sorted(data_by_acceptance.keys())

    for idx, acceptance in enumerate(acceptance_strategies):
        runs = data_by_acceptance[acceptance]
        color = colors[idx % len(colors)]
        color_map[acceptance] = color

        if len(runs) == 1:
            # Single run: plot directly
            run = runs[0]
            ax.plot(run['episode'], run[metric_key],
                   label=acceptance, color=color, linewidth=2, alpha=0.8)
        else:
            # Multiple runs: plot mean with std shading
            stats = compute_statistics(runs)
            episodes = stats[metric_key]['episode']
            mean = stats[metric_key]['mean']
            std = stats[metric_key]['std']

            ax.plot(episodes, mean, label=f'{acceptance} (n={len(runs)})',
                   color=color, linewidth=2, alpha=0.8)
            ax.fill_between(episodes, mean - std, mean + std,
                           color=color, alpha=0.2)

    # Format plot
    metric_display_names = {
        'reward': 'Episode Reward',
        'reward_avg': 'Average Episode Reward',
        'fitness': 'Episode Best Fitness',
        'fitness_avg': 'Average Best Fitness',
        'epsilon': 'Epsilon (Exploration Rate)',
        'loss': 'Training Loss'
    }

    metric_name = metric_display_names.get(metric_key, metric_key)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{rl_algo.upper()} - Reward Strategy: {reward_strategy} - {metric_name} by Acceptance Strategy',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save with specific naming
    filename = f'{rl_algo}_{reward_strategy}_{metric_key}_by_acceptance_strategy.png'
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  Saved: {filename}')


def create_all_plots(organized_data, output_dir='plots'):
    """Create all comparison plots.

    Args:
        organized_data: dict from organize_data()
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Metrics to plot
    metrics = ['reward_avg', 'fitness_avg', 'loss', 'epsilon']

    print(f'\nCreating plots in {output_dir}/...\n')

    for (rl_algo, reward_strategy), data_by_acceptance in organized_data.items():
        print(f'Processing: {rl_algo.upper()} - Reward: {reward_strategy}')
        print(f'  Acceptance strategies found: {list(data_by_acceptance.keys())}')

        for metric in metrics:
            # Skip epsilon for PPO (it doesn't use epsilon-greedy exploration)
            if metric == 'epsilon' and rl_algo == 'ppo':
                continue

            plot_metric(data_by_acceptance, metric, rl_algo, reward_strategy, output_dir)

        print()


def parse_testing_overall_summary(log_path):
    """Parse OVERALL SUMMARY ACROSS ALL INSTANCES section from log file.

    Returns:
        dict with config and method results, or None if section not found
    """
    filename = os.path.basename(log_path)
    config = parse_log_filename(filename)

    if config is None:
        return None

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the overall summary section - looking for the exact format from train_rl_local_search.py
    # The section starts with "OVERALL SUMMARY..." after a line of 80 '=' characters
    overall_pattern = r'={80}\nOVERALL SUMMARY ACROSS ALL INSTANCES\n={80}\nTotal evaluations: (\d+)\nUnique instances tested: (\d+)\n\nAverage initial fitness: ([\d.]+)\n\nRL Models:\n(.*?)\n\nBaseline Methods:\n(.*?)\n\n={80}'

    match = re.search(overall_pattern, content, re.DOTALL)
    if not match:
        return None

    total_evals = int(match.group(1))
    unique_instances = int(match.group(2))
    avg_initial = float(match.group(3))
    rl_models_text = match.group(4)
    baselines_text = match.group(5)

    # Parse RL model results
    # Format: "  {name}: {avg} ± {std} (Δ={improvement}, time: {time}s)"
    # Note: 2 spaces at start, method name can contain spaces/special chars, delta has explicit +/- sign
    method_pattern = r'^\s*(.+?):\s+([\d.]+)\s+±\s+([\d.]+)\s+\(Δ=([+-][\d.]+),\s+time:\s+([\d.]+)s\)'

    rl_models = {}
    for line in rl_models_text.strip().split('\n'):
        if not line.strip():
            continue
        match = re.match(method_pattern, line)
        if match:
            name = match.group(1).strip()
            rl_models[name] = {
                'avg_fitness': float(match.group(2)),
                'std_fitness': float(match.group(3)),
                'avg_improvement': float(match.group(4)),
                'avg_time': float(match.group(5))
            }

    baselines = {}
    for line in baselines_text.strip().split('\n'):
        if not line.strip():
            continue
        match = re.match(method_pattern, line)
        if match:
            name = match.group(1).strip()
            baselines[name] = {
                'avg_fitness': float(match.group(2)),
                'std_fitness': float(match.group(3)),
                'avg_improvement': float(match.group(4)),
                'avg_time': float(match.group(5))
            }

    return {
        'config': config,
        'total_evals': total_evals,
        'unique_instances': unique_instances,
        'avg_initial': avg_initial,
        'rl_models': rl_models,
        'baselines': baselines
    }


def organize_testing_data(parsed_testing_logs):
    """Organize parsed testing logs by reward and acceptance strategy.

    Returns:
        dict: {(rl_algo, reward_strategy, acceptance_strategy): testing_results}
    """
    organized = {}

    for log_data in parsed_testing_logs:
        if log_data is None:
            continue

        config = log_data['config']
        key = (config['rl_algorithm'], config['reward_strategy'], config['acceptance_strategy'])
        organized[key] = log_data

    return organized


def plot_testing_method_comparison(testing_data_by_config, output_dir):
    """Create bar plots comparing all methods for each configuration.

    Args:
        testing_data_by_config: dict {(rl_algo, reward, acceptance): testing_results}
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nCreating testing comparison plots in {output_dir}/...\n')

    for (rl_algo, reward_strategy, acceptance_strategy), results in testing_data_by_config.items():
        print(f'Processing: {rl_algo.upper()} - Reward: {reward_strategy} - Acceptance: {acceptance_strategy}')

        # Collect all methods and their results
        all_methods = {}
        all_methods.update(results['rl_models'])
        all_methods.update(results['baselines'])

        if not all_methods:
            continue

        method_names = list(all_methods.keys())
        avg_fitness = [all_methods[m]['avg_fitness'] for m in method_names]
        std_fitness = [all_methods[m]['std_fitness'] for m in method_names]
        avg_improvement = [all_methods[m]['avg_improvement'] for m in method_names]
        avg_time = [all_methods[m]['avg_time'] for m in method_names]

        # Plot 1: Average fitness with error bars
        fig, ax = plt.subplots(figsize=(14, 6))
        x_pos = np.arange(len(method_names))

        # Color RL models differently from baselines
        colors = []
        for name in method_names:
            if name in results['rl_models']:
                colors.append('steelblue')
            else:
                colors.append('coral')

        bars = ax.bar(x_pos, avg_fitness, yerr=std_fitness, capsize=5,
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Fitness (lower is better)', fontsize=12, fontweight='bold')
        ax.set_title(f'{rl_algo.upper()} - Reward: {reward_strategy} - Acceptance: {acceptance_strategy}\nTesting Performance: Method Comparison (Avg Fitness ± Std)',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='steelblue', label='RL Models'),
                          Patch(facecolor='coral', label='Baseline Methods')]
        ax.legend(handles=legend_elements, loc='upper right')

        filename = f'{rl_algo}_{reward_strategy}_{acceptance_strategy}_testing_method_comparison_fitness.png'
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {filename}')

        # Plot 2: Average improvement
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(x_pos, avg_improvement, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Improvement (Δ)', fontsize=12, fontweight='bold')
        ax.set_title(f'{rl_algo.upper()} - Reward: {reward_strategy} - Acceptance: {acceptance_strategy}\nTesting Performance: Method Comparison (Avg Improvement)',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(handles=legend_elements, loc='upper right')

        filename = f'{rl_algo}_{reward_strategy}_{acceptance_strategy}_testing_method_comparison_improvement.png'
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {filename}')

        # Plot 3: Average time
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(x_pos, avg_time, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{rl_algo.upper()} - Reward: {reward_strategy} - Acceptance: {acceptance_strategy}\nTesting Performance: Method Comparison (Avg Time)',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(handles=legend_elements, loc='upper right')

        filename = f'{rl_algo}_{reward_strategy}_{acceptance_strategy}_testing_method_comparison_time.png'
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {filename}')

        print()


def plot_testing_acceptance_strategy_comparison(testing_data_by_config, output_dir):
    """Create plots comparing acceptance strategies for the same reward strategy.

    Args:
        testing_data_by_config: dict {(rl_algo, reward, acceptance): testing_results}
        output_dir: Directory to save plots
    """
    # Group by (rl_algo, reward_strategy)
    grouped = defaultdict(dict)
    for (rl_algo, reward_strategy, acceptance_strategy), results in testing_data_by_config.items():
        grouped[(rl_algo, reward_strategy)][acceptance_strategy] = results

    print(f'\nCreating acceptance strategy comparison plots...\n')

    for (rl_algo, reward_strategy), acceptance_data in grouped.items():
        if len(acceptance_data) < 2:
            continue  # Need at least 2 acceptance strategies to compare

        print(f'Processing: {rl_algo.upper()} - Reward: {reward_strategy}')
        print(f'  Acceptance strategies: {list(acceptance_data.keys())}')

        # Collect unique method names across all acceptance strategies
        all_method_names = set()
        for results in acceptance_data.values():
            all_method_names.update(results['rl_models'].keys())
            all_method_names.update(results['baselines'].keys())

        method_names = sorted(all_method_names)
        acceptance_strategies = sorted(acceptance_data.keys())

        # Create grouped bar plot for fitness comparison
        fig, ax = plt.subplots(figsize=(16, 7))

        x = np.arange(len(method_names))
        width = 0.8 / len(acceptance_strategies)

        colors = plt.cm.Set2(np.linspace(0, 1, len(acceptance_strategies)))

        for idx, acceptance in enumerate(acceptance_strategies):
            results = acceptance_data[acceptance]
            all_methods = {**results['rl_models'], **results['baselines']}

            fitness_values = []
            std_values = []
            for method in method_names:
                if method in all_methods:
                    fitness_values.append(all_methods[method]['avg_fitness'])
                    std_values.append(all_methods[method]['std_fitness'])
                else:
                    fitness_values.append(0)
                    std_values.append(0)

            offset = width * (idx - len(acceptance_strategies) / 2 + 0.5)
            ax.bar(x + offset, fitness_values, width, label=acceptance,
                  color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Fitness (lower is better)', fontsize=12, fontweight='bold')
        ax.set_title(f'{rl_algo.upper()} - Reward: {reward_strategy}\nTesting Performance: Acceptance Strategy Comparison',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
        ax.legend(title='Acceptance Strategy', loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        filename = f'{rl_algo}_{reward_strategy}_testing_acceptance_strategy_comparison_fitness.png'
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {filename}')

        print()


def main():
    """Main function to parse logs and create plots."""
    log_dir = 'logs'
    training_output_dir = 'results/log_plots/training'
    testing_output_dir = 'results/log_plots/testing'

    print(f'Scanning {log_dir}/ for training logs...')

    # Find all log files
    log_files = []
    for filename in os.listdir(log_dir):
        if filename.startswith('training_rl_local_search_') and filename.endswith('.log'):
            log_files.append(os.path.join(log_dir, filename))

    print(f'Found {len(log_files)} log files\n')

    if not log_files:
        print('No training logs found!')
        return

    # Parse all logs for training data
    print('='*80)
    print('PARSING TRAINING DATA')
    print('='*80)
    parsed_logs = []
    for log_path in log_files:
        print(f'  Parsing: {os.path.basename(log_path)}')
        log_data = parse_log_file(log_path)
        if log_data:
            parsed_logs.append(log_data)
            config = log_data['config']
            print(f'    -> {config["rl_algorithm"].upper()} | Acceptance: {config["acceptance_strategy"]} | '
                  f'Reward: {config["reward_strategy"]} | Seed: {config["seed"]} | Episodes: {len(log_data["episodes"])}')
        else:
            print(f'    -> Failed to parse training data')

    print(f'\nSuccessfully parsed training data from {len(parsed_logs)} log files')

    # Parse all logs for testing data
    print('\n' + '='*80)
    print('PARSING TESTING DATA')
    print('='*80)
    parsed_testing_logs = []
    for log_path in log_files:
        print(f'  Parsing: {os.path.basename(log_path)}')
        testing_data = parse_testing_overall_summary(log_path)
        if testing_data:
            parsed_testing_logs.append(testing_data)
            config = testing_data['config']
            print(f'    -> {config["rl_algorithm"].upper()} | Acceptance: {config["acceptance_strategy"]} | '
                  f'Reward: {config["reward_strategy"]} | Methods: {len(testing_data["rl_models"]) + len(testing_data["baselines"])}')
        else:
            print(f'    -> No testing data found (training may not have completed)')

    print(f'\nSuccessfully parsed testing data from {len(parsed_testing_logs)} log files')

    # Create training plots
    if parsed_logs:
        print('\n' + '='*80)
        print('CREATING TRAINING PLOTS')
        print('='*80)

        # Organize data
        print('\nOrganizing training data by reward and acceptance strategies...')
        organized_data = organize_data(parsed_logs)

        print(f'Found {len(organized_data)} (RL algorithm, reward strategy) combinations:')
        for (rl_algo, reward_strategy), data_by_acceptance in organized_data.items():
            num_acceptance = len(data_by_acceptance)
            total_runs = sum(len(runs) for runs in data_by_acceptance.values())
            print(f'  {rl_algo.upper()} + {reward_strategy}: {num_acceptance} acceptance strategies, {total_runs} total runs')

        # Create plots
        create_all_plots(organized_data, training_output_dir)

    # Create testing plots
    if parsed_testing_logs:
        print('\n' + '='*80)
        print('CREATING TESTING PLOTS')
        print('='*80)

        # Organize testing data
        print('\nOrganizing testing data by configuration...')
        testing_data_by_config = organize_testing_data(parsed_testing_logs)

        print(f'Found {len(testing_data_by_config)} complete testing configurations')

        # Create testing plots
        plot_testing_method_comparison(testing_data_by_config, testing_output_dir)
        plot_testing_acceptance_strategy_comparison(testing_data_by_config, testing_output_dir)

    print(f'\n{"="*80}')
    print('ALL PLOTS CREATED SUCCESSFULLY')
    print(f'{"="*80}')
    if parsed_logs:
        print(f'Training plots saved to: {training_output_dir}/')
    if parsed_testing_logs:
        print(f'Testing plots saved to: {testing_output_dir}/')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
