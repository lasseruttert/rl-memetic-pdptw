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

    Acceptance strategies: always, greedy, epsilon_greedy, rising_epsilon_greedy
    Reward strategies: binary, component, tanh, distance_baseline, hybrid_improvement,
                      initial_improvement, old_improvement, log_improvement
    _attention is an optional flag after reward strategy

    Returns:
        dict with keys: rl_algorithm, problem_size, acceptance_strategy, reward_strategy,
                       use_attention, seed, timestamp
        or None if parsing fails
    """
    # Pattern: Match known acceptance strategies, then capture reward (may have underscores),
    # then optional _attention flag, then _seed
    # Use negative lookahead to prevent reward from capturing _attention
    pattern = r'training_rl_local_search_(?P<algo>\w+)_(?P<size>\d+)_(?P<acceptance>rising_epsilon_greedy|epsilon_greedy|always|greedy)_(?P<reward>[^_]+(?:_(?!attention)[^_]+)*)(?P<attention>_attention)?_seed(?P<seed>\d+)_(?P<timestamp>\d+)\.log'

    match = re.match(pattern, filename)
    if not match:
        return None

    config = {
        'rl_algorithm': match.group('algo'),
        'problem_size': int(match.group('size')),
        'acceptance_strategy': match.group('acceptance'),
        'reward_strategy': match.group('reward'),
        'use_attention': match.group('attention') is not None,
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
    """Organize parsed testing logs by acceptance strategy only.

    Returns:
        dict: {acceptance_strategy: {(rl_algo, reward_strategy): testing_results}}
    """
    organized = defaultdict(dict)

    for log_data in parsed_testing_logs:
        if log_data is None:
            continue

        config = log_data['config']
        acceptance = config['acceptance_strategy']
        inner_key = (config['rl_algorithm'], config['reward_strategy'])
        organized[acceptance][inner_key] = log_data

    return organized


def plot_testing_comparison(testing_data_by_acceptance, output_dir):
    """Create plots comparing reward strategies for each acceptance strategy.

    Args:
        testing_data_by_acceptance: dict {acceptance_strategy: {(rl_algo, reward_strategy): testing_results}}
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nCreating testing comparison plots in {output_dir}/...\n')

    for acceptance_strategy, algo_reward_data in testing_data_by_acceptance.items():
        print(f'Processing: Acceptance Strategy: {acceptance_strategy}')
        print(f'  Combinations found: {len(algo_reward_data)}')
        print(f'  Combinations: {sorted(algo_reward_data.keys())}')

        # Group by reward strategy to compare different rewards
        # For each reward strategy, collect data across all algorithms
        reward_grouped = defaultdict(list)
        for (rl_algo, reward_strategy), results in algo_reward_data.items():
            reward_grouped[reward_strategy].append((rl_algo, results))

        print(f'  Unique reward strategies: {sorted(reward_grouped.keys())} ({len(reward_grouped)} total)')

        if len(reward_grouped) < 2:
            print(f'  Skipping (need at least 2 reward strategies, only have {len(reward_grouped)})')
            continue

        reward_strategies = sorted(reward_grouped.keys())

        # Plot each metric separately
        for metric_key, metric_name, ylabel in [
            ('avg_fitness', 'fitness', 'Average Fitness (lower is better)'),
            ('avg_improvement', 'improvement', 'Average Improvement (Δ)'),
            ('avg_time', 'time', 'Average Time (seconds)')
        ]:
            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(reward_strategies))
            width = 0.6

            colors = plt.cm.tab10(np.linspace(0, 1, len(reward_strategies)))

            values = []
            std_values = []
            labels = []

            for reward in reward_strategies:
                # Average across all algorithms for this reward strategy
                algo_results = reward_grouped[reward]

                # Collect RL model results for this reward strategy
                all_values = []
                for rl_algo, results in algo_results:
                    # Get the first RL model (e.g., "RL (OneShot)")
                    if results['rl_models']:
                        first_model = list(results['rl_models'].keys())[0]
                        all_values.append(results['rl_models'][first_model][metric_key])

                if all_values:
                    values.append(np.mean(all_values))
                    std_values.append(np.std(all_values) if len(all_values) > 1 else 0)
                else:
                    values.append(0)
                    std_values.append(0)

                labels.append(reward)

            if metric_key == 'avg_fitness':
                ax.bar(x, values, width, yerr=std_values, capsize=5,
                      color=colors[:len(reward_strategies)], alpha=0.8,
                      edgecolor='black', linewidth=1.5)
            else:
                ax.bar(x, values, width,
                      color=colors[:len(reward_strategies)], alpha=0.8,
                      edgecolor='black', linewidth=1.5)

            ax.set_xlabel('Reward Strategy', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(f'Acceptance: {acceptance_strategy}\nTesting Performance: Reward Strategy Comparison ({metric_name.capitalize()})',
                        fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

            if metric_key == 'avg_improvement':
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

            filename = f'{acceptance_strategy}_testing_{metric_name}_by_reward_strategy.png'
            filepath = os.path.join(output_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {filename}')

        print()


def main():
    """Main function to parse logs and create plots."""
    log_dir = '_vm_logs'
    testing_output_dir = 'results/log_plots/testing_by_acceptance'

    print(f'Scanning {log_dir}/ for log files...')

    # Find all log files
    log_files = []
    for filename in os.listdir(log_dir):
        if filename.startswith('training_rl_local_search_') and filename.endswith('.log'):
            log_files.append(os.path.join(log_dir, filename))

    print(f'Found {len(log_files)} log files\n')

    if not log_files:
        print('No log files found!')
        return

    # Parse all logs for testing data
    print('='*80)
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
                  f'Reward: {config["reward_strategy"]} | Attention: {config["use_attention"]} | '
                  f'Methods: {len(testing_data["rl_models"]) + len(testing_data["baselines"])}')
        else:
            print(f'    -> No testing data found (training may not have completed)')

    print(f'\nSuccessfully parsed testing data from {len(parsed_testing_logs)} log files')

    # Create testing plots
    if parsed_testing_logs:
        print('\n' + '='*80)
        print('CREATING TESTING PLOTS')
        print('='*80)

        # Organize testing data by acceptance strategy only
        print('\nOrganizing testing data by acceptance strategy...')
        testing_data_by_acceptance = organize_testing_data(parsed_testing_logs)

        print(f'Found {len(testing_data_by_acceptance)} acceptance strategies')
        for acceptance_strategy, algo_reward_data in testing_data_by_acceptance.items():
            print(f'  {acceptance_strategy}: {len(algo_reward_data)} (algo, reward) combinations')

        # Create testing plots
        plot_testing_comparison(testing_data_by_acceptance, testing_output_dir)

        print(f'\n{"="*80}')
        print('ALL PLOTS CREATED SUCCESSFULLY')
        print(f'{"="*80}')
        print(f'Testing plots saved to: {testing_output_dir}/')
        print(f'{"="*80}\n')
    else:
        print('\nNo testing data to plot!')


if __name__ == '__main__':
    main()
