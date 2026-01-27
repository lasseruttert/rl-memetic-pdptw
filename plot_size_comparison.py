"""Script to create size comparison plots for DQN RL methods.

This script compares DQN RL methods (OneShot, Roulette, Ranking) against baselines
for two different problem sizes (200 and 400).

Output: 2 plots (fitness and time comparison) with side-by-side size comparison
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def darken_color(color, factor=0.6):
    """Make a color darker by the given factor (0=black, 1=original)."""
    rgb = mcolors.to_rgb(color)
    return tuple(c * factor for c in rgb)


# =============================================================================
# CONFIGURATION - Set log file paths here
# =============================================================================
LOG_PATH_200 = 'logs/training_rl_local_search_dqn_200_greedy_binary_seed100_set2_200_1768812256.log'
LOG_PATH_400 = 'logs/training_rl_local_search_dqn_400_greedy_binary_seed100_set2_400_1768869261.log'
OUTPUT_DIR = 'results/plots/size_comparison'
# =============================================================================

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

# DQN color family (blue shades)
DQN_COLORS = ['#1A5276', '#2E86AB', '#5DADE2']  # Dark blue, Steel blue, Light blue

# Baseline colors (gray shades)
BASELINE_COLORS = ['#5A5A5A', '#7A7A7A', '#9A9A9A', '#BABABA']


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


def parse_log_filename(filename):
    """Parse configuration from log filename.

    Expected format: training_rl_local_search_{algo}_{size}_{acceptance}_{reward}_seed{seed}_set{set}_{size}_{timestamp}.log

    Returns:
        dict with keys: rl_algorithm, problem_size, acceptance_strategy, reward_strategy,
                       use_attention, seed, set_number, timestamp
        or None if parsing fails
    """
    # Pattern to match filename with set number and optional size before timestamp
    # The size may appear again after set number (e.g., _set2_200_timestamp)
    pattern = r'training_rl_local_search_(?P<algo>\w+)_(?P<size>\d+)_(?P<acceptance>[\w_]+)_(?P<reward>[\w_]+)(?P<attention>_attention)?_seed(?P<seed>\d+)_set(?P<set>\d+)(?:_\d+)?_(?P<timestamp>\d+)\.log'

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


def extract_dqn_methods(testing_data):
    """Extract DQN method results from testing data.

    Args:
        testing_data: Parsed testing data from parse_testing_overall_summary

    Returns:
        dict with keys: OneShot, Roulette, Ranking (each containing results)
    """
    dqn_methods = {}

    for model_name, results in testing_data['rl_models'].items():
        # model_name format: "rl_local_search_dqn_..._set1_final.pt (OneShot)"
        match = re.search(r'\((\w+)\)', model_name)
        if match:
            method = match.group(1)  # OneShot, Roulette, or Ranking
            dqn_methods[method] = results

    return dqn_methods


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


def plot_size_comparison(data_200, data_400, output_dir, metric_key, ylabel, title, filename,
                         start_at_zero=False, std_key=None, ylim_200=None, ylim_400=None):
    """Create side-by-side comparison plot for two problem sizes.

    Args:
        data_200: Parsed data for size 200
        data_400: Parsed data for size 400
        output_dir: Output directory
        metric_key: Key for metric value (e.g., 'avg_fitness')
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename (without extension)
        start_at_zero: Whether to start y-axis at 0
        std_key: Key for std value (e.g., 'std_fitness') or None for no error bars
        ylim_200: Tuple (ymin, ymax) for size 200 subplot, or None for auto
        ylim_400: Tuple (ymin, ymax) for size 400 subplot, or None for auto
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Define method order: 3 DQN methods + 4 baselines
    method_order = [
        ('DQN-OneShot', 'OneShot', 'dqn'),
        ('DQN-Roulette', 'Roulette', 'dqn'),
        ('DQN-Ranking', 'Ranking', 'dqn'),
        ('Adaptive', 'Adaptive', 'baseline'),
        ('Naive', 'Naive', 'baseline'),
        ('Naive (best)', 'Naive (best)', 'baseline'),
        ('Random', 'Random', 'baseline'),
    ]

    # Colors: 3 DQN (blue shades) + 4 baselines (gray shades)
    colors = DQN_COLORS + BASELINE_COLORS

    ylims = [ylim_200, ylim_400]
    for ax_idx, (data, size_label) in enumerate([(data_200, 'Size 200'), (data_400, 'Size 400')]):
        ax = axes[ax_idx]

        dqn_methods = extract_dqn_methods(data)
        baselines = data['baselines']

        method_names = []
        values = []
        std_values = []

        for display_name, key, method_type in method_order:
            method_names.append(display_name)
            if method_type == 'dqn':
                results = dqn_methods.get(key)
            else:
                results = baselines.get(key)

            if results:
                values.append(results[metric_key])
                if std_key and std_key in results:
                    std_values.append(results[std_key])
                else:
                    std_values.append(0)
            else:
                values.append(0)
                std_values.append(0)

        x = np.arange(len(method_names))
        width = 0.7

        # Compute axis limits with outlier truncation
        show_error = std_key is not None
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
        ax.set_title(size_label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Add separator line between DQN methods and baselines (after 3 DQN methods)
        ax.axvline(x=2.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        # Set y-axis limits (use custom if provided, otherwise use computed limits)
        if ylims[ax_idx] is not None:
            ax.set_ylim(bottom=ylims[ax_idx][0], top=ylims[ax_idx][1])
        else:
            ax.set_ylim(bottom=y_min, top=y_max)

    # Add common legend below subplots
    legend_elements = [
        Patch(facecolor=DQN_COLORS[1], alpha=0.85, label='DQN'),
        Patch(facecolor=BASELINE_COLORS[0], alpha=0.85, label='Baselines')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray')

    # Overall title
    fig.suptitle(title, fontsize=28, y=0.995)

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])

    # Save
    filepath = os.path.join(output_dir, filename)
    save_figure_multi_format(fig, filepath)
    plt.close()


def main():
    """Main function to create size comparison plots."""
    print('='*80)
    print('SIZE COMPARISON PLOTTING')
    print('='*80)

    # Verify input files exist
    if not os.path.exists(LOG_PATH_200):
        print(f'Error: Size 200 log file not found: {LOG_PATH_200}')
        return

    if not os.path.exists(LOG_PATH_400):
        print(f'Error: Size 400 log file not found: {LOG_PATH_400}')
        return

    # Parse log files
    print(f'\nParsing size 200 log: {LOG_PATH_200}')
    data_200 = parse_testing_overall_summary(LOG_PATH_200)
    if not data_200:
        print('  Error: Failed to parse size 200 log file')
        return
    print(f'  Found {len(data_200["rl_models"])} RL models, {len(data_200["baselines"])} baselines')

    print(f'\nParsing size 400 log: {LOG_PATH_400}')
    data_400 = parse_testing_overall_summary(LOG_PATH_400)
    if not data_400:
        print('  Error: Failed to parse size 400 log file')
        return
    print(f'  Found {len(data_400["rl_models"])} RL models, {len(data_400["baselines"])} baselines')

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'\nCreating plots in {OUTPUT_DIR}/...\n')

    # Plot 1: Fitness Comparison
    print('Creating fitness comparison plot:')
    plot_size_comparison(
        data_200, data_400, OUTPUT_DIR,
        metric_key='avg_fitness',
        ylabel='Average Fitness (lower is better)',
        title='Size Comparison: Fitness',
        filename='size_comparison_fitness',
        start_at_zero=False,
        std_key='std_fitness',
        ylim_200=(0, 12500),
        ylim_400=(0, 100000)
    )

    # Plot 2: Time Comparison
    print('\nCreating time comparison plot:')
    plot_size_comparison(
        data_200, data_400, OUTPUT_DIR,
        metric_key='avg_time',
        ylabel='Average Time (seconds)',
        title='Size Comparison: Time',
        filename='size_comparison_time',
        start_at_zero=True,
        std_key=None
    )

    print('\n' + '='*80)
    print('ALL PLOTS CREATED SUCCESSFULLY')
    print('='*80)
    print(f'Plots saved to: {OUTPUT_DIR}/')
    print('Total plots created: 4 (2 metrics × 2 formats)')
    print('Formats: PNG and PDF')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
