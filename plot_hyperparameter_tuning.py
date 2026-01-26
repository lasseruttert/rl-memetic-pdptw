"""Script to create DQN hyperparameter tuning comparison plots from training logs.

This script parses training logs from hyperparameter tuning experiments and creates
multi-panel comparison plots showing performance across different hyperparameter settings.

Input: Log files from logs/hyperparameter_tuning/
Output:
  - Multi-panel plots (PDF + PNG) in results/plots/hyperparameter_tuning/
  - LaTeX table summarizing all results
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Unified plot style (LaTeX-ready, thesis-optimized for maximum readability)
PLOT_STYLE = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
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

BASELINE_COLOR = '#2E86AB'  # Steel Blue for baseline
NON_BASELINE_COLOR = '#A23B72'  # Plum Purple for non-baseline

# ============================================================================
# HYPERPARAMETER GROUPS CONFIGURATION
# ============================================================================

HYPERPARAMETER_GROUPS = {
    'learning_rate': {
        'display_name': 'Learning Rate',
        'configs': ['lr_1e3', 'lr_5e4', 'baseline', 'lr_5e5', 'lr_1e5'],
        'labels': ['1e-3', '5e-4', '1e-4*', '5e-5', '1e-5'],
        'baseline_idx': 2
    },
    'gamma': {
        'display_name': 'Gamma (γ)',
        'configs': ['gamma_099', 'gamma_095', 'baseline', 'gamma_085'],
        'labels': ['0.99', '0.95', '0.90*', '0.85'],
        'baseline_idx': 2
    },
    'epsilon_decay': {
        'display_name': 'Epsilon Decay',
        'configs': ['eps_decay_slow', 'baseline', 'eps_decay_fast', 'eps_decay_vfast'],
        'labels': ['0.999', '0.9975*', '0.995', '0.99'],
        'baseline_idx': 1
    },
    'batch_size': {
        'display_name': 'Batch Size',
        'configs': ['batch_32', 'baseline', 'batch_128', 'batch_256'],
        'labels': ['32', '64*', '128', '256'],
        'baseline_idx': 1
    },
    'n_step': {
        'display_name': 'N-step',
        'configs': ['nstep_1', 'baseline', 'nstep_5', 'nstep_7'],
        'labels': ['1', '3*', '5', '7'],
        'baseline_idx': 1
    },
    'target_update': {
        'display_name': 'Target Update',
        'configs': ['target_update_25', 'target_update_50', 'baseline', 'target_update_200'],
        'labels': ['25', '50', '100*', '200'],
        'baseline_idx': 2
    },
    'network': {
        'display_name': 'Network Architecture',
        'configs': ['net_small', 'baseline', 'net_wide', 'net_large'],
        'labels': ['[64,64]', '[128,128,64]*', '[256,128,64]', '[256,256,128]'],
        'baseline_idx': 1
    }
}

# Map config suffix to hyperparameter group
CONFIG_TO_GROUP = {}
for group_name, group_info in HYPERPARAMETER_GROUPS.items():
    for config in group_info['configs']:
        CONFIG_TO_GROUP[config] = group_name


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_hp_tuning_log_filename(filename):
    """Parse configuration from hyperparameter tuning log filename.

    Expected format: training_rl_local_search_dqn_100_greedy_binary_seed100_hp_tuning_{config}_{timestamp}.log

    Returns:
        dict with keys: config_name, hp_group, rl_algorithm, problem_size, etc.
        or None if parsing fails
    """
    # Pattern to match hp_tuning filename
    pattern = r'training_rl_local_search_(?P<algo>\w+)_(?P<size>\d+)_(?P<acceptance>[\w_]+)_(?P<reward>[\w_]+)(?P<attention>_attention)?_seed(?P<seed>\d+)_hp_tuning_(?P<config>\w+)_(?P<timestamp>\d+)\.log'

    match = re.match(pattern, filename)
    if not match:
        return None

    config_name = match.group('config')

    # Determine hyperparameter group
    hp_group = CONFIG_TO_GROUP.get(config_name, None)

    return {
        'rl_algorithm': match.group('algo'),
        'problem_size': int(match.group('size')),
        'acceptance_strategy': match.group('acceptance'),
        'reward_strategy': match.group('reward'),
        'use_attention': match.group('attention') is not None,
        'seed': int(match.group('seed')),
        'config_name': config_name,
        'hp_group': hp_group,
        'timestamp': match.group('timestamp')
    }


def parse_testing_overall_summary(log_path):
    """Parse OVERALL SUMMARY ACROSS ALL INSTANCES section from log file.

    Returns:
        dict with config and results, or None if section not found
    """
    filename = os.path.basename(log_path)
    config = parse_hp_tuning_log_filename(filename)

    if config is None:
        return None

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {filename}: {e}")
        return None

    # Find the overall summary section
    overall_pattern = r'={80}\nOVERALL SUMMARY ACROSS ALL INSTANCES\n={80}\nTotal evaluations: (\d+)\nUnique instances tested: (\d+)\n\nAverage initial fitness: ([\d.]+)\n\nRL Models:\n(.*?)\n\nBaseline Methods:\n(.*?)\n\n={80}'

    match = re.search(overall_pattern, content, re.DOTALL)
    if not match:
        return None

    total_evals = int(match.group(1))
    unique_instances = int(match.group(2))
    avg_initial = float(match.group(3))
    rl_models_text = match.group(4)

    # Parse method results
    # Format: "  {name}: {avg} ± {std} (Δ={improvement}, time: {time}s)"
    method_pattern = r'^\s*(.+?):\s+([\d.]+)\s+±\s+([\d.]+)\s+\(Δ=([+-][\d.]+),\s+time:\s+([\d.]+)s\)'

    rl_models = {}
    for line in rl_models_text.strip().split('\n'):
        if not line.strip():
            continue
        m = re.match(method_pattern, line)
        if m:
            name = m.group(1).strip()
            rl_models[name] = {
                'avg_fitness': float(m.group(2)),
                'std_fitness': float(m.group(3)),
                'avg_improvement': float(m.group(4)),
                'avg_time': float(m.group(5))
            }

    # Extract OneShot results (primary metric for comparison)
    oneshot_results = None
    for model_name, results in rl_models.items():
        if '(OneShot)' in model_name:
            oneshot_results = results
            break

    if oneshot_results is None and rl_models:
        # Fallback to first model if OneShot not found
        oneshot_results = list(rl_models.values())[0]

    return {
        'config': config,
        'total_evals': total_evals,
        'unique_instances': unique_instances,
        'avg_initial': avg_initial,
        'rl_models': rl_models,
        'oneshot_results': oneshot_results
    }


def organize_by_hyperparameter(log_dir):
    """Organize parsed testing logs by hyperparameter category.

    Args:
        log_dir: Directory containing log files

    Returns:
        dict: {hp_group: {config_name: results_dict}}
    """
    data_by_hp = defaultdict(dict)

    # Find all log files
    log_files = []
    for filename in os.listdir(log_dir):
        if filename.startswith('training_rl_local_search_') and filename.endswith('.log'):
            if 'hp_tuning' in filename:
                log_files.append(os.path.join(log_dir, filename))

    print(f'Found {len(log_files)} hyperparameter tuning log files\n')

    # Parse each log file
    for log_path in log_files:
        filename = os.path.basename(log_path)
        print(f'  Parsing: {filename}')

        # Parse testing data
        testing_data = parse_testing_overall_summary(log_path)
        if not testing_data:
            print(f'    -> No testing data found')
            continue

        config = testing_data['config']
        config_name = config['config_name']
        hp_group = config['hp_group']

        if hp_group is None:
            print(f'    -> Unknown hyperparameter group for config: {config_name}')
            continue

        print(f'    -> Group: {hp_group}, Config: {config_name}')

        # Store results
        data_by_hp[hp_group][config_name] = {
            'config': config,
            'avg_fitness': testing_data['oneshot_results']['avg_fitness'] if testing_data['oneshot_results'] else None,
            'std_fitness': testing_data['oneshot_results']['std_fitness'] if testing_data['oneshot_results'] else None,
            'avg_improvement': testing_data['oneshot_results']['avg_improvement'] if testing_data['oneshot_results'] else None,
            'avg_time': testing_data['oneshot_results']['avg_time'] if testing_data['oneshot_results'] else None,
            'avg_initial': testing_data['avg_initial'],
            'total_evals': testing_data['total_evals']
        }

    return dict(data_by_hp)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

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


def plot_single_hyperparameter(ax, hp_group, data_by_config, metric='avg_fitness'):
    """Plot a single hyperparameter comparison on given axes.

    Args:
        ax: Matplotlib axes object
        hp_group: Hyperparameter group name
        data_by_config: Dict of {config_name: results_dict}
        metric: Metric to plot ('avg_fitness', 'avg_time', 'avg_improvement')
    """
    group_info = HYPERPARAMETER_GROUPS[hp_group]
    configs = group_info['configs']
    labels = group_info['labels']
    baseline_idx = group_info['baseline_idx']

    values = []
    colors = []
    valid_labels = []

    for i, config in enumerate(configs):
        if config in data_by_config and data_by_config[config][metric] is not None:
            values.append(data_by_config[config][metric])
            valid_labels.append(labels[i])
            # Highlight baseline with different color
            if i == baseline_idx:
                colors.append(BASELINE_COLOR)
            else:
                colors.append(NON_BASELINE_COLOR)

    if not values:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(group_info['display_name'], fontweight='bold')
        return

    x = np.arange(len(values))
    width = 0.7

    bars = ax.bar(x, values, width, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if metric == 'avg_fitness':
            label = f'{val:.1f}'
        elif metric == 'avg_time':
            label = f'{val:.2f}s'
        else:
            label = f'{val:+.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_title(group_info['display_name'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

    # Set y-axis limits with padding
    if metric == 'avg_fitness':
        y_range = max(values) - min(values)
        if y_range > 0:
            ax.set_ylim(bottom=min(values) - 0.1 * y_range,
                       top=max(values) + 0.15 * y_range)
    elif metric == 'avg_time':
        ax.set_ylim(bottom=0, top=max(values) * 1.2)


def plot_all_hyperparameters(data_by_hp, output_dir, metric='avg_fitness'):
    """Create 2x4 grid plot showing all hyperparameter comparisons.

    Args:
        data_by_hp: Dict organized by hyperparameter group
        output_dir: Output directory
        metric: Metric to plot ('avg_fitness', 'avg_time', 'avg_improvement')
    """
    metric_labels = {
        'avg_fitness': ('Fitness', 'lower is better'),
        'avg_time': ('Time (s)', 'lower is better'),
        'avg_improvement': ('Improvement', 'higher is better')
    }

    ylabel, direction = metric_labels.get(metric, ('Value', ''))

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # Plot each hyperparameter group
    hp_groups = ['learning_rate', 'gamma', 'epsilon_decay', 'batch_size',
                 'n_step', 'target_update', 'network']

    for idx, hp_group in enumerate(hp_groups):
        ax = axes[idx]
        if hp_group in data_by_hp:
            plot_single_hyperparameter(ax, hp_group, data_by_hp[hp_group], metric)
            ax.set_ylabel(ylabel if idx % 4 == 0 else '')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(HYPERPARAMETER_GROUPS[hp_group]['display_name'], fontweight='bold')

    # Summary subplot (last panel)
    ax_summary = axes[7]
    plot_summary_comparison(ax_summary, data_by_hp, metric)
    ax_summary.set_title('Best per Category', fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BASELINE_COLOR, alpha=0.85, edgecolor='black', label='Baseline'),
        Patch(facecolor=NON_BASELINE_COLOR, alpha=0.85, edgecolor='black', label='Variant')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray')

    # Overall title
    metric_name = metric_labels.get(metric, ('Metric', ''))[0]
    fig.suptitle(f'DQN Hyperparameter Tuning: {metric_name} Comparison\n(* = baseline value, {direction})',
                 fontsize=28, y=0.995)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save
    filename = f'hp_tuning_{metric}_comparison'
    filepath = os.path.join(output_dir, filename)
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_summary_comparison(ax, data_by_hp, metric='avg_fitness'):
    """Plot summary showing best config for each hyperparameter category.

    Args:
        ax: Matplotlib axes object
        data_by_hp: Dict organized by hyperparameter group
        metric: Metric to compare
    """
    hp_groups = ['learning_rate', 'gamma', 'epsilon_decay', 'batch_size',
                 'n_step', 'target_update', 'network']

    group_labels = []
    best_values = []
    baseline_values = []

    for hp_group in hp_groups:
        if hp_group not in data_by_hp:
            continue

        group_info = HYPERPARAMETER_GROUPS[hp_group]
        data = data_by_hp[hp_group]

        # Get baseline value
        baseline_config = group_info['configs'][group_info['baseline_idx']]
        baseline_val = data.get(baseline_config, {}).get(metric)

        # Find best value
        best_val = None
        for config, results in data.items():
            val = results.get(metric)
            if val is not None:
                if best_val is None:
                    best_val = val
                elif metric == 'avg_improvement':
                    best_val = max(best_val, val)
                else:
                    best_val = min(best_val, val)

        if best_val is not None and baseline_val is not None:
            group_labels.append(group_info['display_name'][:10])
            best_values.append(best_val)
            baseline_values.append(baseline_val)

    if not group_labels:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    x = np.arange(len(group_labels))
    width = 0.35

    ax.bar(x - width/2, baseline_values, width, label='Baseline',
           color=BASELINE_COLOR, alpha=0.85, edgecolor='black')
    ax.bar(x + width/2, best_values, width, label='Best',
           color=NON_BASELINE_COLOR, alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def generate_latex_table(data_by_hp, output_path):
    """Generate comprehensive LaTeX table of hyperparameter tuning results.

    Args:
        data_by_hp: Dict organized by hyperparameter group
        output_path: Output file path for LaTeX table
    """
    lines = []

    # Table header
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{DQN Hyperparameter Tuning Results}')
    lines.append(r'\label{tab:hp_tuning}')
    lines.append(r'\begin{tabular}{llrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Category & Value & Fitness & Std & Impr. & Time (s) \\')
    lines.append(r'\midrule')

    hp_groups = ['learning_rate', 'gamma', 'epsilon_decay', 'batch_size',
                 'n_step', 'target_update', 'network']

    for hp_group in hp_groups:
        if hp_group not in data_by_hp:
            continue

        group_info = HYPERPARAMETER_GROUPS[hp_group]
        data = data_by_hp[hp_group]
        configs = group_info['configs']
        labels = group_info['labels']
        baseline_idx = group_info['baseline_idx']

        # Find best values for bolding
        fitness_values = []
        time_values = []
        impr_values = []

        for config in configs:
            if config in data and data[config]['avg_fitness'] is not None:
                fitness_values.append((config, data[config]['avg_fitness']))
                time_values.append((config, data[config]['avg_time']))
                impr_values.append((config, data[config]['avg_improvement']))

        best_fitness_config = min(fitness_values, key=lambda x: x[1])[0] if fitness_values else None
        best_time_config = min(time_values, key=lambda x: x[1])[0] if time_values else None
        best_impr_config = max(impr_values, key=lambda x: x[1])[0] if impr_values else None

        # Add multirow for category
        num_configs = sum(1 for c in configs if c in data and data[c]['avg_fitness'] is not None)
        if num_configs == 0:
            continue

        first_row = True
        for i, config in enumerate(configs):
            if config not in data or data[config]['avg_fitness'] is None:
                continue

            results = data[config]
            label = labels[i]

            # Format values with bolding for best
            fitness_str = f"{results['avg_fitness']:.1f}"
            std_str = f"{results['std_fitness']:.1f}"
            impr_str = f"{results['avg_improvement']:+.1f}"
            time_str = f"{results['avg_time']:.2f}"

            if config == best_fitness_config:
                fitness_str = r'\textbf{' + fitness_str + '}'
            if config == best_impr_config:
                impr_str = r'\textbf{' + impr_str + '}'
            if config == best_time_config:
                time_str = r'\textbf{' + time_str + '}'

            # Add baseline marker
            if i == baseline_idx:
                label = label.replace('*', r'$^\ast$')

            if first_row:
                category_str = r'\multirow{' + str(num_configs) + '}{*}{' + group_info['display_name'] + '}'
                lines.append(f'  {category_str} & {label} & {fitness_str} & {std_str} & {impr_str} & {time_str} \\\\')
                first_row = False
            else:
                lines.append(f'  & {label} & {fitness_str} & {std_str} & {impr_str} & {time_str} \\\\')

        lines.append(r'\midrule')

    # Remove last midrule and add bottomrule
    if lines[-1] == r'\midrule':
        lines[-1] = r'\bottomrule'

    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'  Saved LaTeX table: {output_path}')


def generate_summary_table(data_by_hp, output_path):
    """Generate summary LaTeX table showing best config per category.

    Args:
        data_by_hp: Dict organized by hyperparameter group
        output_path: Output file path for summary table
    """
    lines = []

    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Best Hyperparameter Values by Category}')
    lines.append(r'\label{tab:hp_tuning_summary}')
    lines.append(r'\begin{tabular}{lllr}')
    lines.append(r'\toprule')
    lines.append(r'Category & Best Value & Baseline & Improvement \\')
    lines.append(r'\midrule')

    hp_groups = ['learning_rate', 'gamma', 'epsilon_decay', 'batch_size',
                 'n_step', 'target_update', 'network']

    for hp_group in hp_groups:
        if hp_group not in data_by_hp:
            continue

        group_info = HYPERPARAMETER_GROUPS[hp_group]
        data = data_by_hp[hp_group]
        configs = group_info['configs']
        labels = group_info['labels']
        baseline_idx = group_info['baseline_idx']

        # Get baseline fitness
        baseline_config = configs[baseline_idx]
        baseline_fitness = data.get(baseline_config, {}).get('avg_fitness')

        # Find best config
        best_config = None
        best_fitness = None
        best_label = None

        for i, config in enumerate(configs):
            if config in data and data[config]['avg_fitness'] is not None:
                fitness = data[config]['avg_fitness']
                if best_fitness is None or fitness < best_fitness:
                    best_fitness = fitness
                    best_config = config
                    best_label = labels[i].replace('*', '')

        if best_fitness is None or baseline_fitness is None:
            continue

        # Calculate improvement over baseline
        improvement = ((baseline_fitness - best_fitness) / baseline_fitness) * 100

        baseline_label = labels[baseline_idx].replace('*', '')

        if best_config == baseline_config:
            best_label = r'\textbf{' + best_label + '} (baseline)'
            improvement_str = '--'
        else:
            best_label = r'\textbf{' + best_label + '}'
            improvement_str = f'{improvement:+.2f}\\%'

        lines.append(f'  {group_info["display_name"]} & {best_label} & {baseline_label} & {improvement_str} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'  Saved summary table: {output_path}')


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to create hyperparameter tuning plots and tables."""
    log_dir = 'logs/hyperparameter_tuning'
    output_dir = 'results/plots/hyperparameter_tuning'

    print('='*80)
    print('DQN HYPERPARAMETER TUNING ANALYSIS')
    print('='*80)
    print(f'\nScanning {log_dir}/ for log files...\n')

    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f'Error: Log directory not found: {log_dir}')
        print('Creating directory for future use...')
        os.makedirs(log_dir, exist_ok=True)
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse all logs and organize by hyperparameter
    data_by_hp = organize_by_hyperparameter(log_dir)

    if not data_by_hp:
        print('\nNo valid hyperparameter tuning data found in log files!')
        print('\nExpected filename format:')
        print('  training_rl_local_search_dqn_100_greedy_binary_seed100_hp_tuning_{config}_{timestamp}.log')
        print('\nWhere {config} is one of:')
        for hp_group, info in HYPERPARAMETER_GROUPS.items():
            print(f'  {hp_group}: {info["configs"]}')
        return

    # Print summary
    print(f'\nFound data for {len(data_by_hp)} hyperparameter groups:\n')
    for hp_group in sorted(data_by_hp.keys()):
        configs = list(data_by_hp[hp_group].keys())
        print(f'  {HYPERPARAMETER_GROUPS[hp_group]["display_name"]}: {len(configs)} configs')
        for config in configs:
            results = data_by_hp[hp_group][config]
            print(f'    - {config}: fitness={results["avg_fitness"]:.1f}, time={results["avg_time"]:.2f}s')

    # Create plots
    print('\n' + '='*80)
    print('CREATING PLOTS')
    print('='*80)

    print('\nFitness Comparison Plot:')
    plot_all_hyperparameters(data_by_hp, output_dir, metric='avg_fitness')

    print('\nTime Comparison Plot:')
    plot_all_hyperparameters(data_by_hp, output_dir, metric='avg_time')

    print('\nImprovement Comparison Plot:')
    plot_all_hyperparameters(data_by_hp, output_dir, metric='avg_improvement')

    # Generate LaTeX tables
    print('\n' + '='*80)
    print('GENERATING LATEX TABLES')
    print('='*80)

    print('\nFull Results Table:')
    generate_latex_table(data_by_hp, os.path.join(output_dir, 'hp_tuning_table.tex'))

    print('\nSummary Table:')
    generate_summary_table(data_by_hp, os.path.join(output_dir, 'hp_tuning_summary.tex'))

    # Final summary
    print('\n' + '='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)
    print(f'\nOutput directory: {output_dir}/')
    print(f'  - 3 comparison plots (fitness, time, improvement) × 2 formats (PNG, PDF)')
    print(f'  - 2 LaTeX tables (full results, summary)')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
