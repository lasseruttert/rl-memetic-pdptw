"""Script to create RL algorithm comparison plots from training logs.

This script parses training logs from RL local search experiments and creates
comparison plots showing performance across different operator sets, algorithms
(DQN vs PPO), and selection strategies (OneShot, Roulette, Ranking).

Output: 2 plots per operator set (fitness and time comparison)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict


def darken_color(color, factor=0.6):
    """Make a color darker by the given factor (0=black, 1=original)."""
    rgb = mcolors.to_rgb(color)
    return tuple(c * factor for c in rgb)

# Unified plot style (LaTeX-ready, thesis-optimized for maximum readability)
PLOT_STYLE = {
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.titlesize': 26,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 28,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.labelpad': 12,
    'xtick.major.pad': 10,
    'ytick.major.pad': 10,
    'lines.linewidth': 3.0,
    'axes.linewidth': 1.5,
}
plt.rcParams.update(PLOT_STYLE)

# Unified color palette
METHOD_COLORS = [
    '#2E86AB',  # Steel Blue
    '#A23B72',  # Plum Purple
    '#1D7874',  # Teal
    '#E8A838',  # Muted Gold
    '#6B4C9A',  # Violet
    '#D64550',  # Soft Red
    '#44AF69',  # Sage Green
    '#8B5E3C',  # Brown
]

# DQN color family (blue shades)
DQN_COLORS = ['#1A5276', '#2E86AB', '#5DADE2']  # Dark blue, Steel blue, Light blue

# PPO color family (purple shades)
PPO_COLORS = ['#6C3483', '#A23B72', '#D98880']  # Dark purple, Plum, Light rose

BASELINE_COLORS = ['#5A5A5A', '#7A7A7A', '#9A9A9A', '#BABABA']


def parse_log_filename(filename):
    """Parse configuration from log filename.

    Expected format: training_rl_local_search_{algo}_{size}_{acceptance}_{reward}_seed{seed}_set{set}_{timestamp}.log

    Returns:
        dict with keys: rl_algorithm, problem_size, acceptance_strategy, reward_strategy,
                       use_attention, seed, set_number, timestamp
        or None if parsing fails
    """
    # Pattern to match filename with set number before timestamp
    pattern = r'training_rl_local_search_(?P<algo>\w+)_(?P<size>\d+)_(?P<acceptance>[\w_]+)_(?P<reward>[\w_]+)(?P<attention>_attention)?_seed(?P<seed>\d+)_set(?P<set>\d+)_(?P<timestamp>\d+)\.log'

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
        'set_number': int(match.group('set')),
        'timestamp': match.group('timestamp')
    }

    return config


def parse_operator_list(log_path):
    """Extract operator list from log file header.

    Args:
        log_path: Path to log file

    Returns:
        dict with 'operators' (list of operator names) and 'num_operators' (int)
        or None if not found
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            # Read first 40 lines where operator list appears
            header_lines = []
            for i in range(40):
                try:
                    header_lines.append(next(f))
                except StopIteration:
                    break
    except Exception:
        return None

    # Look for line like: "Operators: ['Op1', 'Op2', ...]"
    for line in header_lines:
        if line.strip().startswith('Operators:'):
            # Extract the list content
            match = re.search(r'Operators: \[([^\]]+)\]', line)
            if match:
                # Parse operator names (quoted strings separated by commas)
                ops_str = match.group(1)
                # Split by comma and strip quotes/whitespace
                operators = [op.strip().strip("'\"") for op in ops_str.split(',')]
                return {
                    'operators': operators,
                    'num_operators': len(operators)
                }

    return None


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

    # Find the overall summary section
    overall_pattern = r'={80}\nOVERALL SUMMARY ACROSS ALL INSTANCES\n={80}\nTotal evaluations: (\d+)\nUnique instances tested: (\d+)\n\nAverage initial fitness: ([\d.]+)\n\nRL Models:\n(.*?)\n\nBaseline Methods:\n(.*?)\n\n={80}'

    match = re.search(overall_pattern, content, re.DOTALL)
    if not match:
        return None

    total_evals = int(match.group(1))
    unique_instances = int(match.group(2))
    avg_initial = float(match.group(3))
    rl_models_text = match.group(4)
    baselines_text = match.group(5)

    # Parse method results
    # Format: "  {name}: {avg} ± {std} (Δ={improvement}, time: {time}s)"
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


def organize_by_set_and_algorithm(log_dir):
    """Organize parsed testing logs by operator set and algorithm.

    Args:
        log_dir: Directory containing log files

    Returns:
        dict: {set_number: {'operators': [...], 'num_operators': N,
                           'DQN': {...}, 'PPO': {...}, 'baselines': {...}}}
    """
    data_by_set = defaultdict(lambda: {
        'operators': None,
        'num_operators': 0,
        'DQN': {},
        'PPO': {},
        'baselines': {}
    })

    # Find all log files
    log_files = []
    for filename in os.listdir(log_dir):
        if filename.startswith('training_rl_local_search_') and filename.endswith('.log'):
            log_files.append(os.path.join(log_dir, filename))

    print(f'Found {len(log_files)} log files\n')

    # Parse each log file
    for log_path in log_files:
        filename = os.path.basename(log_path)
        print(f'  Parsing: {filename}')

        # Parse testing data
        testing_data = parse_testing_overall_summary(log_path)
        if not testing_data:
            print(f'    -> No testing data found')
            continue

        # Parse operator list
        operator_data = parse_operator_list(log_path)
        if not operator_data:
            print(f'    -> No operator list found')
            continue

        config = testing_data['config']
        set_num = config['set_number']
        algo = config['rl_algorithm'].upper()  # DQN or PPO

        print(f'    -> {algo}, Set {set_num}, {operator_data["num_operators"]} operators')

        # Store operator info (only once per set)
        if data_by_set[set_num]['operators'] is None:
            data_by_set[set_num]['operators'] = operator_data['operators']
            data_by_set[set_num]['num_operators'] = operator_data['num_operators']

        # Extract RL model results - parse method name from parentheses
        for model_name, results in testing_data['rl_models'].items():
            # model_name format: "rl_local_search_dqn_..._set1_final.pt (OneShot)"
            match = re.search(r'\((\w+)\)', model_name)
            if match:
                method = match.group(1)  # OneShot, Roulette, or Ranking
                data_by_set[set_num][algo][method] = results

        # Store baselines (same for DQN/PPO, only store once)
        if not data_by_set[set_num]['baselines']:
            data_by_set[set_num]['baselines'] = testing_data['baselines']

    return dict(data_by_set)


def save_figure_multi_format(fig, filepath, formats=['png', 'pdf']):
    """Save figure in multiple formats for thesis use.

    Args:
        fig: Matplotlib figure object
        filepath: Base filepath (without extension)
        formats: List of formats to save ['png', 'pdf']
    """
    for fmt in formats:
        output_path = f"{filepath}.{fmt}"
        if fmt == 'pdf':
            fig.savefig(output_path, format='pdf', bbox_inches='tight',
                       dpi=300, backend='pdf')
        else:  # png
            fig.savefig(output_path, format='png', bbox_inches='tight',
                       dpi=300)
        print(f'    Saved: {output_path}')


def compute_axis_limits_with_outlier_truncation(values, std_values=None, start_at_zero=False, outlier_threshold=2.0):
    """Compute y-axis limits, truncating outliers that are > threshold * next highest.

    Args:
        values: List of values to plot
        std_values: List of std values for error bars (optional)
        start_at_zero: Whether y-axis should start at 0
        outlier_threshold: Truncate bars > this multiple of next highest value

    Returns:
        tuple: (y_min, y_max, truncated_indices) where truncated_indices are bars to clip
    """
    if std_values is None:
        std_values = [0] * len(values)

    # Calculate upper bounds (value + std) for proper axis scaling
    upper_bounds = [v + s for v, s in zip(values, std_values)]
    lower_bounds = [v - s for v, s in zip(values, std_values)]

    valid_upper = [u for u in upper_bounds if u > 0]
    if len(valid_upper) < 2:
        y_min = 0 if start_at_zero else min(lower_bounds) * 0.9
        y_max = max(upper_bounds) * 1.1
        return y_min, y_max, []

    sorted_upper = sorted(valid_upper, reverse=True)
    max_upper = sorted_upper[0]
    second_max_upper = sorted_upper[1]

    truncated_indices = []

    # Check if max is an outlier (more than threshold * second highest)
    if max_upper > outlier_threshold * second_max_upper:
        # Find all outlier values
        truncation_limit = second_max_upper * 1.5  # Show up to 1.5x the second highest
        for i, u in enumerate(upper_bounds):
            if u > truncation_limit:
                truncated_indices.append(i)

        if start_at_zero:
            y_min = 0
            y_max = truncation_limit  # No padding - bars go right to the edge
        else:
            non_outlier_lower = [lower_bounds[i] for i, u in enumerate(upper_bounds) if u <= truncation_limit]
            if non_outlier_lower:
                y_min = min(non_outlier_lower) * 0.95
                y_max = truncation_limit  # No padding - bars go right to the edge
            else:
                y_min = 0
                y_max = truncation_limit  # No padding - bars go right to the edge
    else:
        # No outliers - normal scaling with error bars accounted for
        if start_at_zero:
            y_min = 0
            y_max = max(upper_bounds) * 1.1
        else:
            y_min = min(lower_bounds) * 0.95
            y_max = max(upper_bounds) * 1.1

    return y_min, y_max, truncated_indices


def plot_metric(methods, colors, metric_key, std_key, ylabel, title, filename,
                output_dir, show_error_bars=True, start_at_zero=False):
    """Create a single bar chart for one metric.

    Args:
        methods: List of tuples [(method_name, results_dict), ...]
        colors: List of colors for bars
        metric_key: Key for metric value (e.g., 'avg_fitness')
        std_key: Key for std value (e.g., 'std_fitness') or None
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
        output_dir: Output directory
        show_error_bars: Whether to show error bars
        start_at_zero: Whether to start y-axis at 0
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    method_names = []
    values = []
    std_values = []

    for method_name, results in methods:
        method_names.append(method_name)
        if results:
            values.append(results[metric_key])
            if show_error_bars and std_key and std_key in results:
                std_values.append(results[std_key])
            else:
                std_values.append(0)
        else:
            values.append(0)
            std_values.append(0)

    x = np.arange(len(method_names))
    width = 0.7

    # Compute axis limits with outlier truncation (accounting for error bars)
    y_min, y_max, truncated_indices = compute_axis_limits_with_outlier_truncation(
        values, std_values=std_values if show_error_bars else None, start_at_zero=start_at_zero
    )

    # Draw bars
    bars = ax.bar(x, values, width,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    # Draw error bars separately with matching darkened colors
    if show_error_bars:
        for i, (val, std, color) in enumerate(zip(values, std_values, colors)):
            if std > 0:
                dark_color = darken_color(color, factor=0.66)
                ax.errorbar(x[i], val, yerr=std, fmt='none',
                           capsize=12, capthick=2.5, elinewidth=2.5,
                           ecolor=dark_color)

    # Styling
    ax.set_xlabel('Method')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add separator line between RL methods and baselines (6 RL methods: 3 DQN + 3 PPO)
    ax.axvline(x=5.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Set y-axis limits
    ax.set_ylim(bottom=y_min, top=y_max)

    # Add legend to distinguish algorithm types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DQN_COLORS[1], alpha=0.85, label='DQN'),
        Patch(facecolor=PPO_COLORS[1], alpha=0.85, label='PPO'),
        Patch(facecolor=BASELINE_COLORS[0], alpha=0.85, label='Baselines')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, edgecolor='gray')

    filepath = os.path.join(output_dir, filename.replace('.png', ''))
    plt.tight_layout()
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_algorithm_comparison_by_set(data_by_set, output_dir):
    """Create fitness and time comparison plots for each operator set.

    Args:
        data_by_set: Dict organized by set_number
        output_dir: Where to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nCreating plots in {output_dir}/...\n')

    for set_num in sorted(data_by_set.keys()):
        set_data = data_by_set[set_num]
        num_ops = set_data['num_operators']

        print(f'Set {set_num} ({num_ops} operators):')

        # Prepare data for all 10 methods
        methods = [
            ('DQN-OneShot', set_data['DQN'].get('OneShot')),
            ('DQN-Roulette', set_data['DQN'].get('Roulette')),
            ('DQN-Ranking', set_data['DQN'].get('Ranking')),
            ('PPO-OneShot', set_data['PPO'].get('OneShot')),
            ('PPO-Roulette', set_data['PPO'].get('Roulette')),
            ('PPO-Ranking', set_data['PPO'].get('Ranking')),
            ('Adaptive', set_data['baselines'].get('Adaptive')),
            ('Naive', set_data['baselines'].get('Naive')),
            ('Naive (best)', set_data['baselines'].get('Naive (best)')),
            ('Random', set_data['baselines'].get('Random'))
        ]

        # Define colors: DQN (blue shades), PPO (purple shades), Baselines (gray shades)
        colors = DQN_COLORS + PPO_COLORS + BASELINE_COLORS

        # Plot 1: Fitness Comparison
        plot_metric(
            methods, colors,
            metric_key='avg_fitness',
            std_key='std_fitness',
            ylabel='Fitness',
            title=f'RL Algorithm Comparison: Fitness (Set {set_num})',
            filename=f'set{set_num}_algorithm_comparison_fitness.png',
            output_dir=output_dir,
            show_error_bars=True
        )

        # Plot 2: Time Comparison
        plot_metric(
            methods, colors,
            metric_key='avg_time',
            std_key=None,
            ylabel='Time (s)',
            title=f'RL Algorithm Comparison: Time (Set {set_num})',
            filename=f'set{set_num}_algorithm_comparison_time.png',
            output_dir=output_dir,
            show_error_bars=False,
            start_at_zero=True
        )

        print()


def plot_algorithm_comparison_combined_by_metric(data_by_set, output_dir, metric_key,
                                                  ylabel, title_prefix, start_at_zero=False):
    """Create combined figure showing all sets for one metric.

    Args:
        data_by_set: Dict organized by set_number
        output_dir: Output directory
        metric_key: 'avg_fitness' or 'avg_time'
        ylabel: Y-axis label
        title_prefix: Title prefix ('Fitness' or 'Time')
        start_at_zero: Whether to start y-axis at 0
    """
    sets = sorted(data_by_set.keys())
    num_sets = len(sets)

    # Create figure with subplots (2x2 grid for 4 sets) - thesis-ready size
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    colors = DQN_COLORS + PPO_COLORS + BASELINE_COLORS

    for idx, set_num in enumerate(sets):
        ax = axes[idx]
        set_data = data_by_set[set_num]
        num_ops = set_data['num_operators']

        # Prepare methods data
        methods = [
            ('DQN-OneShot', set_data['DQN'].get('OneShot')),
            ('DQN-Roulette', set_data['DQN'].get('Roulette')),
            ('DQN-Ranking', set_data['DQN'].get('Ranking')),
            ('PPO-OneShot', set_data['PPO'].get('OneShot')),
            ('PPO-Roulette', set_data['PPO'].get('Roulette')),
            ('PPO-Ranking', set_data['PPO'].get('Ranking')),
            ('Adaptive', set_data['baselines'].get('Adaptive')),
            ('Naive', set_data['baselines'].get('Naive')),
            ('Naive (best)', set_data['baselines'].get('Naive (best)')),
            ('Random', set_data['baselines'].get('Random'))
        ]

        method_names = []
        values = []
        std_values = []

        for method_name, results in methods:
            method_names.append(method_name)
            if results:
                values.append(results[metric_key])
                # Add std for fitness metric
                if metric_key == 'avg_fitness' and 'std_fitness' in results:
                    std_values.append(results['std_fitness'])
                else:
                    std_values.append(0)
            else:
                values.append(0)
                std_values.append(0)

        x = np.arange(len(method_names))
        width = 0.7

        # Compute axis limits with outlier truncation (accounting for error bars)
        show_error = (metric_key == 'avg_fitness')
        y_min, y_max, truncated_indices = compute_axis_limits_with_outlier_truncation(
            values, std_values=std_values if show_error else None, start_at_zero=start_at_zero
        )

        # Draw bars
        bars = ax.bar(x, values, width, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=1.0)

        # Draw error bars separately with matching darkened colors
        if show_error:
            for i, (val, std, color) in enumerate(zip(values, std_values, colors)):
                if std > 0:
                    dark_color = darken_color(color, factor=0.66)
                    ax.errorbar(x[i], val, yerr=std, fmt='none',
                               capsize=10, capthick=2.0, elinewidth=2.0,
                               ecolor=dark_color)

        # Styling
        ax.set_xlabel('Method')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Set {set_num} ({num_ops} operators)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Add separator line between RL methods and baselines
        ax.axvline(x=5.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        # Set y-axis limits
        ax.set_ylim(bottom=y_min, top=y_max)

    # Add common legend below subplots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DQN_COLORS[1], alpha=0.85, label='DQN'),
        Patch(facecolor=PPO_COLORS[1], alpha=0.85, label='PPO'),
        Patch(facecolor=BASELINE_COLORS[0], alpha=0.85, label='Baselines')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])

    # Save
    filename = f'combined_{metric_key}_all_sets'
    filepath = os.path.join(output_dir, filename)
    save_figure_multi_format(fig, filepath)
    plt.close()

    print(f'  Saved combined {title_prefix} plot')


def main():
    """Main function to create RL algorithm comparison plots."""
    log_dir = 'results/logs/rl_algorithm_logs'
    output_dir = 'results/plots/rl_algorithm_comparison'

    print('='*80)
    print('RL ALGORITHM COMPARISON PLOTTING')
    print('='*80)
    print(f'\nScanning {log_dir}/ for log files...\n')

    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f'Error: Log directory not found: {log_dir}')
        return

    # Parse all logs and organize by set
    data_by_set = organize_by_set_and_algorithm(log_dir)

    if not data_by_set:
        print('\nNo valid data found in log files!')
        return

    # Print summary
    print(f'\nFound {len(data_by_set)} operator sets:\n')
    for set_num in sorted(data_by_set.keys()):
        set_data = data_by_set[set_num]
        print(f'  Set {set_num}: {set_data["num_operators"]} operators')
        print(f'    DQN methods: {len(set_data["DQN"])}')
        print(f'    PPO methods: {len(set_data["PPO"])}')
        print(f'    Baselines: {len(set_data["baselines"])}')

    # Create plots
    print('\n' + '='*80)
    print('CREATING INDIVIDUAL PLOTS')
    print('='*80)
    plot_algorithm_comparison_by_set(data_by_set, output_dir)

    # Create combined plots
    print('\n' + '='*80)
    print('CREATING COMBINED PLOTS')
    print('='*80)

    print('\nCombined Fitness Plot:')
    plot_algorithm_comparison_combined_by_metric(
        data_by_set, output_dir,
        metric_key='avg_fitness',
        ylabel='Average Fitness (lower is better)',
        title_prefix='Fitness',
        start_at_zero=False
    )

    print('\nCombined Time Plot:')
    plot_algorithm_comparison_combined_by_metric(
        data_by_set, output_dir,
        metric_key='avg_time',
        ylabel='Average Time (seconds)',
        title_prefix='Time',
        start_at_zero=True
    )

    print('\n' + '='*80)
    print('ALL PLOTS CREATED SUCCESSFULLY')
    print('='*80)
    print(f'Plots saved to: {output_dir}/')
    print(f'Total plots created: {len(data_by_set) * 2 + 2} ({len(data_by_set) * 2} individual + 2 combined)')
    print(f'Formats: PNG and PDF')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
