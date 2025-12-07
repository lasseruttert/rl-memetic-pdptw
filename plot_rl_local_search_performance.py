"""
Comprehensive visualization script for RL Local Search Performance Experiment results.
Creates thesis-quality plots for convergence, operator analysis, distributions, and correlations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns
from collections import defaultdict
import argparse
import yaml

# Thesis-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Constants
RESULTS_DIR = "results/rl_local_search_performance"
OUTPUT_DIR = "results/plots/rl_local_search_performance_plots"

# Color schemes
INIT_COLORS = {
    'random': '#2E86AB',  # Blue
    'greedy': '#A23B72'   # Purple
}

OPERATOR_COLORS = {
    0: '#E63946',   # Red
    4: '#F77F00',   # Orange
    12: '#06A77D',  # Green
    13: '#118AB2',  # Blue
    14: '#073B4C'   # Dark blue
}

FAMILY_COLORS = {
    'lc': '#264653',
    'lr': '#2A9D8F',
    'lrc': '#E9C46A',
    'bar': '#F4A261',
    'ber': '#E76F51',
    'nyc': '#6A4C93',
    'poa': '#1982C4'
}

# Operator names (default - will be loaded from config if provided)
OPERATOR_NAMES = {}


def load_operator_names_from_config(config_path: str) -> Dict[int, str]:
    """Load operator names from YAML config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary mapping operator index to name
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    operator_names = {}

    if 'operators' in config and 'custom' in config['operators']:
        for idx, op_config in enumerate(config['operators']['custom']):
            op_type = op_config['type']
            # Remove 'Operator' suffix and format nicely
            name = op_type.replace('Operator', '')
            # Add parameter info for variants if needed
            if 'params' in op_config and op_config['params']:
                params = op_config['params']
                if 'type' in params:
                    name = f"{name}({params['type']})"
                elif 'single_route' in params and params['single_route']:
                    name = f"{name}(SR)"
            operator_names[idx] = name

    return operator_names


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure_multi_format(fig, filepath: str, formats: List[str] = ['png', 'pdf']):
    """Save figure in multiple formats for thesis use.

    Args:
        fig: Matplotlib figure object
        filepath: Base filepath (without extension)
        formats: List of formats to save
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


def setup_plot_style():
    """Configure matplotlib rcParams for thesis quality."""
    # Already configured at module level, but can be called explicitly
    pass


def get_instance_family(instance_name: str) -> str:
    """Extract instance family from name.

    Args:
        instance_name: Instance name (e.g., 'lc101', 'lr203')

    Returns:
        Family prefix (e.g., 'lc', 'lr', 'lrc')
    """
    # Handle compound families like 'lrc'
    if instance_name.startswith('lrc'):
        return 'lrc'
    elif instance_name.startswith('lc'):
        return 'lc'
    elif instance_name.startswith('lr'):
        return 'lr'
    elif instance_name.startswith('bar'):
        return 'bar'
    elif instance_name.startswith('ber'):
        return 'ber'
    elif instance_name.startswith('nyc'):
        return 'nyc'
    elif instance_name.startswith('poa'):
        return 'poa'
    else:
        return 'other'


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_master_summary(results_dir: str) -> Dict:
    """Load master summary JSON file.

    Returns:
        Complete experiment data with all instances
    """
    filepath = Path(results_dir) / "rl_local_search_all_instances_summary.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def load_averaged_metrics_csv(results_dir: str) -> pd.DataFrame:
    """Load CSV summary with averaged metrics per instance.

    Returns:
        DataFrame with columns: Instance, Initialization, Mean_Initial_Fitness, etc.
    """
    filepath = Path(results_dir) / "rl_local_search_all_instances_averaged_metrics.csv"
    return pd.read_csv(filepath)


def load_all_best_run_histories(results_dir: str) -> Dict:
    """Load all best run history files efficiently.

    Returns:
        Nested dict: {instance_name: {init_type: history_data}}
    """
    histories = {}

    # Pattern: rl_local_search_instance_{name}_{init_type}_init_best_run_full_history.json
    history_files = Path(results_dir).glob("*_best_run_full_history.json")

    for file_path in history_files:
        # Parse filename to extract instance name and init type
        filename = file_path.stem

        # Split by underscores
        parts = filename.split('_')

        # Find 'init' keyword to identify initialization type
        try:
            init_idx = parts.index('init')
            init_type = parts[init_idx - 1]  # 'random' or 'greedy'

            # Instance name is between 'instance' and init_type
            instance_start_idx = parts.index('instance') + 1
            instance_name = '_'.join(parts[instance_start_idx:init_idx - 1])

            # Load data
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Store in nested dict
            if instance_name not in histories:
                histories[instance_name] = {}
            histories[instance_name][init_type] = data

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse filename {filename}: {e}")
            continue

    return histories


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def extract_convergence_data(history: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract iterations and fitness trajectory from iteration history.

    Args:
        history: Best run history data

    Returns:
        Tuple of (iterations, fitness_values) arrays
    """
    iteration_history = history['iteration_history']
    iterations = sorted([int(k) for k in iteration_history.keys()])
    fitness_values = np.array([iteration_history[str(i)]['fitness'] for i in iterations])
    return np.array(iterations), fitness_values


def compute_averaged_convergence(all_histories: Dict, init_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and std of fitness trajectories across all instances.

    Args:
        all_histories: All best run histories
        init_type: 'random' or 'greedy'

    Returns:
        Tuple of (iterations, mean_fitness, std_fitness)
    """
    all_trajectories = []

    for instance_name, init_data in all_histories.items():
        if init_type not in init_data:
            continue

        history = init_data[init_type]['iteration_history']

        # Extract fitness trajectory
        iterations = sorted([int(k) for k in history.keys()])
        fitness_trajectory = [history[str(i)]['fitness'] for i in iterations]
        all_trajectories.append(fitness_trajectory)

    if not all_trajectories:
        return np.array([]), np.array([]), np.array([])

    # Convert to numpy array
    trajectories_array = np.array(all_trajectories)

    # Compute statistics
    iterations = np.arange(len(all_trajectories[0]))
    mean_fitness = np.mean(trajectories_array, axis=0)
    std_fitness = np.std(trajectories_array, axis=0)

    return iterations, mean_fitness, std_fitness


def compute_operator_statistics(all_histories: Dict, init_type: str) -> Dict:
    """Compute comprehensive operator statistics.

    Args:
        all_histories: All best run histories
        init_type: 'random' or 'greedy'

    Returns:
        Dict with 'uses', 'successes', 'acceptances', 'success_rates', 'acceptance_rates'
    """
    operator_uses = defaultdict(int)
    operator_successes = defaultdict(int)
    operator_acceptances = defaultdict(int)

    for instance_name, init_data in all_histories.items():
        if init_type not in init_data:
            continue

        history = init_data[init_type]['iteration_history']

        for iter_key, iter_data in history.items():
            action = iter_data['action']

            # Count usage
            operator_uses[action] += 1

            # Count successes (fitness improvement)
            if iter_data['fitness_improvement'] > 0:
                operator_successes[action] += 1

            # Count acceptances
            if iter_data['accepted']:
                operator_acceptances[action] += 1

    # Compute rates
    operator_success_rates = {
        op: operator_successes[op] / operator_uses[op] if operator_uses[op] > 0 else 0
        for op in operator_uses
    }

    operator_acceptance_rates = {
        op: operator_acceptances[op] / operator_uses[op] if operator_uses[op] > 0 else 0
        for op in operator_uses
    }

    return {
        'uses': dict(operator_uses),
        'successes': dict(operator_successes),
        'acceptances': dict(operator_acceptances),
        'success_rates': operator_success_rates,
        'acceptance_rates': operator_acceptance_rates
    }


def identify_top_bottom_instances(df: pd.DataFrame, init_type: str, n: int = 5) -> Tuple[List[str], List[str]]:
    """Identify best/worst performing instances by improvement %.

    Args:
        df: DataFrame with averaged metrics
        init_type: 'random' or 'greedy'
        n: Number of instances to return

    Returns:
        Tuple of (top_instances, bottom_instances) lists
    """
    # Filter by initialization type
    df_filtered = df[df['Initialization'] == init_type].copy()

    # Sort by improvement %
    df_sorted = df_filtered.sort_values('Mean_Improvement_%', ascending=False)

    # Get top and bottom instances
    top_instances = df_sorted.head(n)['Instance'].tolist()
    bottom_instances = df_sorted.tail(n)['Instance'].tolist()

    return top_instances, bottom_instances


# ============================================================================
# CONVERGENCE PLOTTING FUNCTIONS
# ============================================================================

def plot_convergence_trajectories_averaged(all_histories: Dict, output_dir: str):
    """Create averaged convergence trajectories for random vs greedy.

    Args:
        all_histories: All best run histories
        output_dir: Output directory path
    """
    print("  Creating averaged convergence trajectories...")

    # Compute averaged convergence for both initialization types
    iter_random, mean_random, std_random = compute_averaged_convergence(all_histories, 'random')
    iter_greedy, mean_greedy, std_greedy = compute_averaged_convergence(all_histories, 'greedy')

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Random initialization
    ax1.plot(iter_random, mean_random, color=INIT_COLORS['random'], linewidth=2, label='Mean')
    ax1.fill_between(iter_random, mean_random - std_random, mean_random + std_random,
                     color=INIT_COLORS['random'], alpha=0.3, label='± 1 Std Dev')
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Fitness', fontweight='bold')
    ax1.set_title('Random Initialization', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Greedy initialization
    ax2.plot(iter_greedy, mean_greedy, color=INIT_COLORS['greedy'], linewidth=2, label='Mean')
    ax2.fill_between(iter_greedy, mean_greedy - std_greedy, mean_greedy + std_greedy,
                     color=INIT_COLORS['greedy'], alpha=0.3, label='± 1 Std Dev')
    ax2.set_xlabel('Iteration', fontweight='bold')
    ax2.set_ylabel('Fitness', fontweight='bold')
    ax2.set_title('Greedy Initialization', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle('Convergence Trajectories: Random vs Greedy Initialization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'convergence_averaged_random_vs_greedy')
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_convergence_top_bottom_instances(histories: Dict, top_instances: List[str],
                                          bottom_instances: List[str], init_type: str,
                                          output_dir: str):
    """Create convergence plots for top 5 and bottom 5 instances.

    Args:
        histories: All best run histories
        top_instances: List of top 5 instance names
        bottom_instances: List of bottom 5 instance names
        init_type: 'random' or 'greedy'
        output_dir: Output directory path
    """
    print(f"  Creating top/bottom convergence plots for {init_type}...")

    # Create 2x5 subplot grid
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    color = INIT_COLORS[init_type]

    # Plot top 5 instances
    for idx, instance in enumerate(top_instances):
        ax = axes[0, idx]
        if instance in histories and init_type in histories[instance]:
            iterations, fitness = extract_convergence_data(histories[instance][init_type])
            ax.plot(iterations, fitness, color=color, linewidth=1.5)
            ax.set_title(f'{instance}', fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.set_ylabel('Fitness', fontweight='bold')

    # Plot bottom 5 instances
    for idx, instance in enumerate(bottom_instances):
        ax = axes[1, idx]
        if instance in histories and init_type in histories[instance]:
            iterations, fitness = extract_convergence_data(histories[instance][init_type])
            ax.plot(iterations, fitness, color=color, linewidth=1.5)
            ax.set_title(f'{instance}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Iteration')
            if idx == 0:
                ax.set_ylabel('Fitness', fontweight='bold')

    # Overall title
    fig.suptitle(f'Convergence: Top 5 (top) vs Bottom 5 (bottom) - {init_type.capitalize()} Initialization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'convergence_top5_bottom5_{init_type}')
    save_figure_multi_format(fig, filepath)
    plt.close()


# ============================================================================
# OPERATOR SELECTION PLOTTING FUNCTIONS
# ============================================================================

def plot_operator_usage_frequency(operator_stats: Dict, output_dir: str):
    """Create grouped bar chart of operator usage frequency.

    Args:
        operator_stats: Dict with 'random' and 'greedy' operator statistics
        output_dir: Output directory path
    """
    print("  Creating operator usage frequency plot...")

    # Get all unique operators
    all_operators = set()
    all_operators.update(operator_stats['random']['uses'].keys())
    all_operators.update(operator_stats['greedy']['uses'].keys())
    operators_sorted = sorted(all_operators)

    # Prepare data
    random_uses = [operator_stats['random']['uses'].get(op, 0) for op in operators_sorted]
    greedy_uses = [operator_stats['greedy']['uses'].get(op, 0) for op in operators_sorted]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(operators_sorted))
    width = 0.35

    ax.bar(x - width/2, random_uses, width, label='Random', color=INIT_COLORS['random'], alpha=0.8)
    ax.bar(x + width/2, greedy_uses, width, label='Greedy', color=INIT_COLORS['greedy'], alpha=0.8)

    ax.set_xlabel('Operator', fontweight='bold')
    ax.set_ylabel('Usage Frequency', fontweight='bold')
    ax.set_title('Operator Usage Frequency: Random vs Greedy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([OPERATOR_NAMES.get(op, f'Op{op}') for op in operators_sorted])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'operator_usage_frequency')
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_operator_success_rates(operator_stats: Dict, output_dir: str):
    """Create grouped bar chart of operator success rates.

    Args:
        operator_stats: Dict with 'random' and 'greedy' operator statistics
        output_dir: Output directory path
    """
    print("  Creating operator success rates plot...")

    # Get all unique operators
    all_operators = set()
    all_operators.update(operator_stats['random']['success_rates'].keys())
    all_operators.update(operator_stats['greedy']['success_rates'].keys())
    operators_sorted = sorted(all_operators)

    # Prepare data
    random_rates = [operator_stats['random']['success_rates'].get(op, 0) for op in operators_sorted]
    greedy_rates = [operator_stats['greedy']['success_rates'].get(op, 0) for op in operators_sorted]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(operators_sorted))
    width = 0.35

    ax.bar(x - width/2, random_rates, width, label='Random', color=INIT_COLORS['random'], alpha=0.8)
    ax.bar(x + width/2, greedy_rates, width, label='Greedy', color=INIT_COLORS['greedy'], alpha=0.8)

    ax.set_xlabel('Operator', fontweight='bold')
    ax.set_ylabel('Success Rate', fontweight='bold')
    ax.set_title('Operator Success Rates: Random vs Greedy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([OPERATOR_NAMES.get(op, f'Op{op}') for op in operators_sorted])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'operator_success_rates')
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_operator_heatmap(all_histories: Dict, init_type: str, output_dir: str):
    """Create heatmap of operator usage across instances.

    Args:
        all_histories: All best run histories
        init_type: 'random' or 'greedy'
        output_dir: Output directory path
    """
    print(f"  Creating operator heatmap for {init_type}...")

    # Collect operator usage per instance
    instance_operator_usage = {}
    all_operators = set()

    for instance_name, init_data in all_histories.items():
        if init_type not in init_data:
            continue

        history = init_data[init_type]['iteration_history']
        operator_counts = defaultdict(int)

        for iter_key, iter_data in history.items():
            action = iter_data['action']
            operator_counts[action] += 1
            all_operators.add(action)

        instance_operator_usage[instance_name] = operator_counts

    # Create matrix
    instances_sorted = sorted(instance_operator_usage.keys())
    operators_sorted = sorted(all_operators)

    matrix = np.zeros((len(instances_sorted), len(operators_sorted)))
    for i, instance in enumerate(instances_sorted):
        for j, operator in enumerate(operators_sorted):
            matrix[i, j] = instance_operator_usage[instance].get(operator, 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(instances_sorted) * 0.3)))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(operators_sorted)))
    ax.set_yticks(np.arange(len(instances_sorted)))
    ax.set_xticklabels([OPERATOR_NAMES.get(op, f'Op{op}') for op in operators_sorted])
    ax.set_yticklabels(instances_sorted, fontsize=8)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Usage Frequency', rotation=270, labelpad=20, fontweight='bold')

    ax.set_xlabel('Operator', fontweight='bold')
    ax.set_ylabel('Instance', fontweight='bold')
    ax.set_title(f'Operator Usage Heatmap: {init_type.capitalize()} Initialization',
                 fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'operator_heatmap_{init_type}')
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_operator_temporal_usage(all_histories: Dict, init_type: str, output_dir: str):
    """Show operator selection evolution over iterations.

    Args:
        all_histories: All best run histories
        init_type: 'random' or 'greedy'
        output_dir: Output directory path
    """
    print(f"  Creating operator temporal usage plot for {init_type}...")

    # Define iteration bins
    bins = [(0, 50), (51, 100), (101, 150), (151, 200)]
    bin_labels = ['0-50', '51-100', '101-150', '151-200']

    # Collect operator usage per bin
    bin_operator_usage = [defaultdict(int) for _ in bins]
    all_operators = set()

    for instance_name, init_data in all_histories.items():
        if init_type not in init_data:
            continue

        history = init_data[init_type]['iteration_history']

        for iter_key, iter_data in history.items():
            iteration = int(iter_key)
            action = iter_data['action']
            all_operators.add(action)

            # Determine which bin this iteration belongs to
            for bin_idx, (start, end) in enumerate(bins):
                if start <= iteration <= end:
                    bin_operator_usage[bin_idx][action] += 1
                    break

    # Prepare data for stacked bar chart
    operators_sorted = sorted(all_operators)
    bin_data = []

    for bin_idx in range(len(bins)):
        bin_totals = [bin_operator_usage[bin_idx].get(op, 0) for op in operators_sorted]
        bin_data.append(bin_totals)

    # Normalize to proportions
    bin_data_normalized = []
    for bin_totals in bin_data:
        total = sum(bin_totals)
        if total > 0:
            bin_data_normalized.append([val / total for val in bin_totals])
        else:
            bin_data_normalized.append([0] * len(operators_sorted))

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(bin_labels))
    width = 0.6
    bottom = np.zeros(len(bin_labels))

    for op_idx, operator in enumerate(operators_sorted):
        values = [bin_data_normalized[bin_idx][op_idx] for bin_idx in range(len(bins))]
        color = OPERATOR_COLORS.get(operator, plt.cm.tab10(op_idx % 10))
        ax.bar(x, values, width, label=OPERATOR_NAMES.get(operator, f'Op{operator}'),
               bottom=bottom, color=color, alpha=0.8)
        bottom += values

    ax.set_xlabel('Iteration Range', fontweight='bold')
    ax.set_ylabel('Operator Usage Proportion', fontweight='bold')
    ax.set_title(f'Operator Temporal Usage: {init_type.capitalize()} Initialization',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'operator_temporal_{init_type}')
    save_figure_multi_format(fig, filepath)
    plt.close()


# ============================================================================
# DISTRIBUTION PLOTTING FUNCTIONS
# ============================================================================

def plot_fitness_distributions(df: pd.DataFrame, output_dir: str):
    """Create 2x2 subplot grid showing initial and final fitness distributions.

    Args:
        df: DataFrame with averaged metrics
        output_dir: Output directory path
    """
    print("  Creating fitness distributions plot...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Random - Initial Fitness
    ax = axes[0, 0]
    data = df[df['Initialization'] == 'random']['Mean_Initial_Fitness']
    ax.hist(data, bins=20, color=INIT_COLORS['random'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Initial Fitness')
    ax.set_ylabel('Frequency')
    ax.set_title('Random Init: Initial Fitness', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Random - Final Fitness
    ax = axes[0, 1]
    data = df[df['Initialization'] == 'random']['Mean_Final_Fitness']
    ax.hist(data, bins=20, color=INIT_COLORS['random'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Final Fitness')
    ax.set_ylabel('Frequency')
    ax.set_title('Random Init: Final Fitness', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Greedy - Initial Fitness
    ax = axes[1, 0]
    data = df[df['Initialization'] == 'greedy']['Mean_Initial_Fitness']
    ax.hist(data, bins=20, color=INIT_COLORS['greedy'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Initial Fitness')
    ax.set_ylabel('Frequency')
    ax.set_title('Greedy Init: Initial Fitness', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Greedy - Final Fitness
    ax = axes[1, 1]
    data = df[df['Initialization'] == 'greedy']['Mean_Final_Fitness']
    ax.hist(data, bins=20, color=INIT_COLORS['greedy'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Final Fitness')
    ax.set_ylabel('Frequency')
    ax.set_title('Greedy Init: Final Fitness', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Fitness Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'fitness_distributions')
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_time_distributions(df: pd.DataFrame, output_dir: str):
    """Create box plots of execution time.

    Args:
        df: DataFrame with averaged metrics
        output_dir: Output directory path
    """
    print("  Creating time distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    data_random = df[df['Initialization'] == 'random']['Mean_Time_Seconds']
    data_greedy = df[df['Initialization'] == 'greedy']['Mean_Time_Seconds']

    # Create box plot
    bp = ax.boxplot([data_random, data_greedy], labels=['Random', 'Greedy'],
                    patch_artist=True)

    for patch, color in zip(bp['boxes'], [INIT_COLORS['random'], INIT_COLORS['greedy']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
    ax.set_title('Execution Time Distribution: Random vs Greedy Initialization',
                 fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'time_distribution')
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_operator_average_improvement(all_histories: Dict, init_type: str, output_dir: str):
    """Create bar chart of average fitness improvement per operator.

    Args:
        all_histories: All best run histories
        init_type: 'random' or 'greedy'
        output_dir: Output directory path
    """
    print(f"  Creating operator average improvement plot for {init_type}...")

    # Collect improvement data per operator
    operator_improvements = defaultdict(list)

    for instance_name, init_data in all_histories.items():
        if init_type not in init_data:
            continue

        history = init_data[init_type]['iteration_history']

        for iter_key, iter_data in history.items():
            action = iter_data['action']
            improvement = iter_data['fitness_improvement']
            operator_improvements[action].append(improvement)

    # Compute average improvement per operator
    operators_sorted = sorted(operator_improvements.keys())
    avg_improvements = [np.mean(operator_improvements[op]) for op in operators_sorted]
    std_improvements = [np.std(operator_improvements[op]) for op in operators_sorted]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(operators_sorted))
    colors = [OPERATOR_COLORS.get(op, plt.cm.tab10(i % 10)) for i, op in enumerate(operators_sorted)]

    ax.bar(x, avg_improvements, color=colors, alpha=0.8, yerr=std_improvements,
           capsize=5, error_kw={'linewidth': 1.5}, edgecolor='black')

    ax.set_xlabel('Operator', fontweight='bold')
    ax.set_ylabel('Average Fitness Improvement', fontweight='bold')
    ax.set_title(f'Average Fitness Improvement per Operator: {init_type.capitalize()} Initialization',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([OPERATOR_NAMES.get(op, f'Op{op}') for op in operators_sorted])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'operator_average_improvement_{init_type}')
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_operator_transition_matrix(all_histories: Dict, init_type: str, output_dir: str):
    """Create heatmap showing directed operator transitions (which operators follow which).

    Args:
        all_histories: All best run histories
        init_type: 'random' or 'greedy'
        output_dir: Output directory path
    """
    print(f"  Creating directed operator transition matrix for {init_type}...")

    # Collect transition data
    all_operators = set()
    transition_counts = defaultdict(lambda: defaultdict(int))

    for instance_name, init_data in all_histories.items():
        if init_type not in init_data:
            continue

        history = init_data[init_type]['iteration_history']
        iterations = sorted([int(k) for k in history.keys()])

        # Track transitions between consecutive operators
        for i in range(len(iterations) - 1):
            current_action = history[str(iterations[i])]['action']
            next_action = history[str(iterations[i + 1])]['action']

            all_operators.add(current_action)
            all_operators.add(next_action)

            transition_counts[current_action][next_action] += 1

    # Create transition matrix
    operators_sorted = sorted(all_operators)
    n_operators = len(operators_sorted)
    matrix = np.zeros((n_operators, n_operators))

    for i, op_from in enumerate(operators_sorted):
        for j, op_to in enumerate(operators_sorted):
            matrix[i, j] = transition_counts[op_from][op_to]

    # Normalize rows to get transition probabilities
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normalized = np.divide(matrix, row_sums, where=row_sums != 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(matrix_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(n_operators))
    ax.set_yticks(np.arange(n_operators))
    ax.set_xticklabels([OPERATOR_NAMES.get(op, f'Op{op}') for op in operators_sorted])
    ax.set_yticklabels([OPERATOR_NAMES.get(op, f'Op{op}') for op in operators_sorted])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20, fontweight='bold')

    ax.set_xlabel('Next Operator', fontweight='bold')
    ax.set_ylabel('Current Operator', fontweight='bold')
    ax.set_title(f'Directed Operator Transition Matrix: {init_type.capitalize()} Initialization',
                 fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'operator_transition_matrix_directed_{init_type}')
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_operator_cooccurrence_matrix(all_histories: Dict, init_type: str, output_dir: str):
    """Create symmetric heatmap showing which operators appear together (undirected).

    Args:
        all_histories: All best run histories
        init_type: 'random' or 'greedy'
        output_dir: Output directory path
    """
    print(f"  Creating operator co-occurrence matrix for {init_type}...")

    # Collect co-occurrence data (symmetric)
    all_operators = set()
    cooccurrence_counts = defaultdict(lambda: defaultdict(int))

    for instance_name, init_data in all_histories.items():
        if init_type not in init_data:
            continue

        history = init_data[init_type]['iteration_history']
        iterations = sorted([int(k) for k in history.keys()])

        # Track co-occurrences in both directions
        for i in range(len(iterations) - 1):
            op1 = history[str(iterations[i])]['action']
            op2 = history[str(iterations[i + 1])]['action']

            all_operators.add(op1)
            all_operators.add(op2)

            # Count both directions (makes it symmetric)
            cooccurrence_counts[op1][op2] += 1
            cooccurrence_counts[op2][op1] += 1

    # Create symmetric matrix
    operators_sorted = sorted(all_operators)
    n_operators = len(operators_sorted)
    matrix = np.zeros((n_operators, n_operators))

    for i, op1 in enumerate(operators_sorted):
        for j, op2 in enumerate(operators_sorted):
            matrix[i, j] = cooccurrence_counts[op1][op2]

    # Normalize by total co-occurrences per operator
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normalized = np.divide(matrix, row_sums, where=row_sums != 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(matrix_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(n_operators))
    ax.set_yticks(np.arange(n_operators))
    ax.set_xticklabels([OPERATOR_NAMES.get(op, f'Op{op}') for op in operators_sorted])
    ax.set_yticklabels([OPERATOR_NAMES.get(op, f'Op{op}') for op in operators_sorted])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Co-occurrence Frequency', rotation=270, labelpad=20, fontweight='bold')

    ax.set_xlabel('Operator', fontweight='bold')
    ax.set_ylabel('Operator', fontweight='bold')
    ax.set_title(f'Operator Co-occurrence Matrix (Symmetric): {init_type.capitalize()} Initialization',
                 fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'operator_cooccurrence_matrix_{init_type}')
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_operator_usage_heatmap_aggregate(all_histories: Dict, init_type: str, output_dir: str):
    """Create heatmap of operator usage across iteration bins (aggregated across all instances).

    Args:
        all_histories: All best run histories
        init_type: 'random' or 'greedy'
        output_dir: Output directory path
    """
    print(f"  Creating aggregated operator usage heatmap for {init_type}...")

    # Define iteration bins
    bins = [(0, 50), (51, 100), (101, 150), (151, 200)]
    bin_labels = ['0-50', '51-100', '101-150', '151-200']

    # Collect operator usage per bin across all instances
    bin_operator_usage = [defaultdict(int) for _ in bins]
    all_operators = set()

    for instance_name, init_data in all_histories.items():
        if init_type not in init_data:
            continue

        history = init_data[init_type]['iteration_history']

        for iter_key, iter_data in history.items():
            iteration = int(iter_key)
            action = iter_data['action']
            all_operators.add(action)

            # Determine which bin this iteration belongs to
            for bin_idx, (start, end) in enumerate(bins):
                if start <= iteration <= end:
                    bin_operator_usage[bin_idx][action] += 1
                    break

    # Create matrix
    operators_sorted = sorted(all_operators)
    n_operators = len(operators_sorted)
    n_bins = len(bins)
    matrix = np.zeros((n_operators, n_bins))

    for i, operator in enumerate(operators_sorted):
        for j, bin_idx in enumerate(range(n_bins)):
            matrix[i, j] = bin_operator_usage[bin_idx].get(operator, 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, max(6, n_operators * 0.5)))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_operators))
    ax.set_xticklabels(bin_labels)
    ax.set_yticklabels([OPERATOR_NAMES.get(op, f'Op{op}') for op in operators_sorted])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Usage Count', rotation=270, labelpad=20, fontweight='bold')

    ax.set_xlabel('Iteration Range', fontweight='bold')
    ax.set_ylabel('Operator', fontweight='bold')
    ax.set_title(f'Operator Usage Over Time (Aggregated): {init_type.capitalize()} Initialization',
                 fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'operator_usage_aggregate_heatmap_{init_type}')
    save_figure_multi_format(fig, filepath)
    plt.close()


# ============================================================================
# REMOVED: CORRELATION PLOTTING FUNCTIONS
# ============================================================================
# Correlation plots have been removed as requested


# ============================================================================
# SUMMARY COMPARISON PLOTTING FUNCTIONS
# ============================================================================

def plot_random_vs_greedy_comparison(df: pd.DataFrame, output_dir: str):
    """Create bar chart comparing key metrics between random and greedy.

    Args:
        df: DataFrame with averaged metrics
        output_dir: Output directory path
    """
    print("  Creating random vs greedy comparison plot...")

    # Compute means and stds
    metrics = ['Mean_Improvement_%', 'Mean_Time_Seconds', 'Mean_Acceptance_Rate', 'Mean_Final_Fitness']
    metric_labels = ['Improvement %', 'Time (s)', 'Acceptance Rate', 'Final Fitness']

    random_means = [df[df['Initialization'] == 'random'][m].mean() for m in metrics]
    random_stds = [df[df['Initialization'] == 'random'][m].std() for m in metrics]
    greedy_means = [df[df['Initialization'] == 'greedy'][m].mean() for m in metrics]
    greedy_stds = [df[df['Initialization'] == 'greedy'][m].std() for m in metrics]

    # Create subplots for each metric (different scales)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (metric_label, r_mean, r_std, g_mean, g_std) in enumerate(
        zip(metric_labels, random_means, random_stds, greedy_means, greedy_stds)):

        ax = axes[idx]
        x = np.array([0, 1])
        means = [r_mean, g_mean]
        stds = [r_std, g_std]
        colors = [INIT_COLORS['random'], INIT_COLORS['greedy']]

        ax.bar(x, means, width=0.6, color=colors, alpha=0.8, yerr=stds,
               capsize=5, error_kw={'linewidth': 2})
        ax.set_xticks(x)
        ax.set_xticklabels(['Random', 'Greedy'])
        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_title(metric_label, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Random vs Greedy Initialization: Key Metrics Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'random_vs_greedy_comparison')
    save_figure_multi_format(fig, filepath)
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot RL Local Search Performance Experiment results")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file for operator names (e.g., config/experiment_rl_algo/dqn_set4.yaml)")
    args = parser.parse_args()

    print("=" * 80)
    print("RL LOCAL SEARCH PERFORMANCE PLOTTING")
    print("=" * 80)

    # Load operator names from config if provided and determine output directory
    global OPERATOR_NAMES
    if args.config:
        print(f"\nLoading operator names from config: {args.config}")
        OPERATOR_NAMES = load_operator_names_from_config(args.config)
        print(f"  Loaded {len(OPERATOR_NAMES)} operator names:")
        for idx, name in sorted(OPERATOR_NAMES.items()):
            print(f"    {idx}: {name}")

        # Add config name to output directory
        config_name = Path(args.config).stem
        output_dir = f"{OUTPUT_DIR}_{config_name}"
        results_dir = f"{RESULTS_DIR}_{config_name}"
    else:
        print("\nNo config provided - using default operator names")
        print("  (Use --config to specify config file for accurate operator names)")
        output_dir = OUTPUT_DIR
        results_dir = RESULTS_DIR

    # Setup
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print("Loading CSV summary...")
    df = load_averaged_metrics_csv(results_dir)
    print(f"  Loaded {len(df)} records")

    print("Loading best run histories...")
    all_histories = load_all_best_run_histories(results_dir)
    print(f"  Loaded {len(all_histories)} instances")

    # Process data
    print("\n" + "=" * 80)
    print("PROCESSING DATA")
    print("=" * 80)
    print("Computing operator statistics for random initialization...")
    operator_stats_random = compute_operator_statistics(all_histories, 'random')

    print("Computing operator statistics for greedy initialization...")
    operator_stats_greedy = compute_operator_statistics(all_histories, 'greedy')

    print("Identifying top/bottom instances for random initialization...")
    top_random, bottom_random = identify_top_bottom_instances(df, 'random')

    print("Identifying top/bottom instances for greedy initialization...")
    top_greedy, bottom_greedy = identify_top_bottom_instances(df, 'greedy')

    # Generate plots
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)
    plot_convergence_trajectories_averaged(all_histories, output_dir)
    plot_convergence_top_bottom_instances(all_histories, top_random, bottom_random, 'random', output_dir)
    plot_convergence_top_bottom_instances(all_histories, top_greedy, bottom_greedy, 'greedy', output_dir)

    print("\n" + "=" * 80)
    print("OPERATOR SELECTION ANALYSIS")
    print("=" * 80)
    plot_operator_usage_frequency({'random': operator_stats_random, 'greedy': operator_stats_greedy}, output_dir)
    plot_operator_success_rates({'random': operator_stats_random, 'greedy': operator_stats_greedy}, output_dir)
    plot_operator_heatmap(all_histories, 'random', output_dir)
    plot_operator_heatmap(all_histories, 'greedy', output_dir)
    plot_operator_temporal_usage(all_histories, 'random', output_dir)
    plot_operator_temporal_usage(all_histories, 'greedy', output_dir)
    plot_operator_average_improvement(all_histories, 'random', output_dir)
    plot_operator_average_improvement(all_histories, 'greedy', output_dir)
    plot_operator_usage_heatmap_aggregate(all_histories, 'random', output_dir)
    plot_operator_usage_heatmap_aggregate(all_histories, 'greedy', output_dir)
    plot_operator_transition_matrix(all_histories, 'random', output_dir)
    plot_operator_transition_matrix(all_histories, 'greedy', output_dir)
    plot_operator_cooccurrence_matrix(all_histories, 'random', output_dir)
    plot_operator_cooccurrence_matrix(all_histories, 'greedy', output_dir)

    print("\n" + "=" * 80)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 80)
    plot_fitness_distributions(df, output_dir)
    plot_time_distributions(df, output_dir)

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISONS")
    print("=" * 80)
    plot_random_vs_greedy_comparison(df, output_dir)

    print("\n" + "=" * 80)
    print("ALL PLOTS CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"Plots saved to: {output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
