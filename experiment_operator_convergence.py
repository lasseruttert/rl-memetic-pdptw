from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager
from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.best_known_solutions import BestKnownSolutions

from memetic.solution_generators.random_solution import generate_random_solution

from memetic.local_search.naive_local_search import NaiveLocalSearch

from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.merge import MergeOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator
from memetic.solution_operators.shift import ShiftOperator
from memetic.solution_operators.two_opt import TwoOptOperator
from memetic.solution_operators.two_opt_star import TwoOptStarOperator
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator
from memetic.solution_operators.request_shift_within import RequestShiftWithinOperator
from memetic.solution_operators.node_swap_within import NodeSwapWithinOperator

import time
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from collections import OrderedDict
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Problem sizes to test (None = default size from manager, or specific sizes like [100, 200, 400])
PROBLEM_SIZES = [100]

# Number of runs per instance
NUM_RUNS_PER_INSTANCE = 5

# Local search hyperparameters
LOCAL_SEARCH_MAX_ITERATIONS = 200
LOCAL_SEARCH_MAX_NO_IMPROVEMENT = 20
LOCAL_SEARCH_FIRST_IMPROVEMENT = True

# Output file for results
RESULTS_BASE_DIR = "results"
RESULTS_OUTPUT_FILE = "results/operator_convergence_results.json"

# Plotting configuration
PLOTS_OUTPUT_DIR = "results/operator_convergence_plots"

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

# Extended colors for many operators (cycle through with different line styles)
EXTENDED_COLORS = METHOD_COLORS + [
    '#C44536',  # Rust
    '#3A7D44',  # Forest Green
    '#7B68EE',  # Medium Slate Blue
    '#CD853F',  # Peru
    '#4682B4',  # Steel Blue 2
    '#9370DB',  # Medium Purple
    '#20B2AA',  # Light Sea Green
    '#DAA520',  # Goldenrod
]

# Line styles for multi-operator plots
LINE_STYLES = ['-', '--', '-.', ':']  # solid, dashed, dash-dot, dotted

# ============================================================================
# OPERATOR FAMILY GROUPINGS & DISPLAY NAMES
# ============================================================================

def darken_color(color, factor=0.6):
    """Make a color darker by the given factor (0=black, 1=original)."""
    rgb = mcolors.to_rgb(color)
    return tuple(c * factor for c in rgb)


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

    if max_upper > outlier_threshold * second_max_upper:
        truncation_limit = second_max_upper * 1.5
        for i, u in enumerate(upper_bounds):
            if u > truncation_limit:
                truncated_indices.append(i)

        if start_at_zero:
            y_min = 0
            y_max = truncation_limit
        else:
            non_outlier_lower = [lower_bounds[i] for i, u in enumerate(upper_bounds) if u <= truncation_limit]
            if non_outlier_lower:
                y_min = min(non_outlier_lower) * 0.95
                y_max = truncation_limit
            else:
                y_min = 0
                y_max = truncation_limit
    else:
        if start_at_zero:
            y_min = 0
            y_max = max(upper_bounds) * 1.1
        else:
            y_min = min(lower_bounds) * 0.95
            y_max = max(upper_bounds) * 1.1

    return y_min, y_max, truncated_indices


# Operator families: determines x-axis ordering in bar charts
OPERATOR_FAMILIES = OrderedDict([
    ('Reinsert', [
        'Reinsert-nC-Max1-NewV-SameV-nF_SameV',
        'Reinsert-C-Max5-NewV-SameV-nF_SameV',
        'Reinsert-nC-Max1-NewV-SameV-F_SameV',
        'Reinsert-nC-Max1-NewV-NoSameV-nF_SameV',
        'Reinsert-nC-Max1-NoNewV-NoSameV-nF_SameV',
    ]),
    ('RouteElim', [
        'RouteElimination',
    ]),
    ('Flip', [
        'Flip-All-Unlimited',
        'Flip-All-Max5',
        'Flip-Single-Unlimited',
    ]),
    ('Merge', [
        'Merge-random-N2-F-R',
        'Merge-random-N2-F-nR',
        'Merge-min-N2-F-R',
        'Merge-min-N2-F-nR',
    ]),
    ('SwapWithin', [
        'SwapWithin-random-A-MaxNone',
        'SwapWithin-random-A-Max5',
        'SwapWithin-random-S-MaxNone',
        'SwapWithin-best-S-MaxNone',
        'SwapWithin-best-A-MaxNone',
    ]),
    ('SwapBetween', [
        'SwapBetween-random',
        'SwapBetween-best',
    ]),
    ('Transfer', [
        'Transfer-A-Max1',
        'Transfer-S-Max1',
        'Transfer-S-Max5',
    ]),
    ('Shift', [
        'Shift-random-A-Max5-Seg3-Dist3',
        'Shift-random-A-Max5-Seg2-Dist4',
        'Shift-random-A-Max3-Seg4-Dist2',
        'Shift-random-A-Max3-Seg3-Dist5',
        'Shift-best-A-Max5-Seg2-Dist3',
        'Shift-best-A-Max5-Seg3-Dist2',
        'Shift-random-S-Max5-Seg3-Dist3',
    ]),
    ('TwoOpt', [
        'TwoOpt',
    ]),
    ('CLS', [
        'CLS-M1',
        'CLS-M2',
        'CLS-M3',
        'CLS-M4',
    ]),
    ('ReqShift', [
        'RequestShiftWithin',
    ]),
    ('NodeSwap', [
        'NodeSwapWithin-WithPrecedence',
        'NodeSwapWithin',
    ]),
])

# One distinct base color per family
FAMILY_BASE_COLORS = {
    'Reinsert':    '#2E86AB',  # Steel Blue
    'RouteElim':   '#A23B72',  # Plum Purple
    'Flip':        '#1D7874',  # Teal
    'Merge':       '#E8A838',  # Muted Gold
    'SwapWithin':  '#6B4C9A',  # Violet
    'SwapBetween': '#D64550',  # Soft Red
    'Transfer':    '#44AF69',  # Sage Green
    'Shift':       '#8B5E3C',  # Brown
    'TwoOpt':      '#C44536',  # Rust
    'CLS':         '#3A7D44',  # Forest Green
    'ReqShift':    '#7B68EE',  # Medium Slate Blue
    'NodeSwap':    '#CD853F',  # Peru
}

# Short display names — follows Name(params) convention from plot_rl_local_search_performance.py
DISPLAY_NAMES = {
    'Reinsert-nC-Max1-NewV-SameV-nF_SameV':       'Reinsert',
    'Reinsert-C-Max5-NewV-SameV-nF_SameV':         'Reinsert(C,k5)',
    'Reinsert-nC-Max1-NewV-SameV-F_SameV':         'Reinsert(F_SameV)',
    'Reinsert-nC-Max1-NewV-NoSameV-nF_SameV':      'Reinsert(NoSameV)',
    'Reinsert-nC-Max1-NoNewV-NoSameV-nF_SameV':    'Reinsert(NoNewV,NoSameV)',
    'RouteElimination':                             'RouteElim',
    'Flip-All-Unlimited':                           'Flip',
    'Flip-All-Max5':                                'Flip(k5)',
    'Flip-Single-Unlimited':                        'Flip(SR)',
    'Merge-random-N2-F-R':                          'Merge(random)',
    'Merge-random-N2-F-nR':                         'Merge(random,noR)',
    'Merge-min-N2-F-R':                             'Merge(min)',
    'Merge-min-N2-F-nR':                            'Merge(min,noR)',
    'SwapWithin-random-A-MaxNone':                  'SwapW',
    'SwapWithin-random-A-Max5':                     'SwapW(k5)',
    'SwapWithin-random-S-MaxNone':                  'SwapW(SR)',
    'SwapWithin-best-S-MaxNone':                    'SwapW(best,SR)',
    'SwapWithin-best-A-MaxNone':                    'SwapW(best)',
    'SwapBetween-random':                           'SwapB',
    'SwapBetween-best':                             'SwapB(best)',
    'Transfer-A-Max1':                              'Transfer',
    'Transfer-S-Max1':                              'Transfer(SR)',
    'Transfer-S-Max5':                              'Transfer(SR,k5)',
    'Shift-random-A-Max5-Seg3-Dist3':               'Shift(rnd,s3,d3,k5)',
    'Shift-random-A-Max5-Seg2-Dist4':               'Shift(rnd,s2,d4,k5)',
    'Shift-random-A-Max3-Seg4-Dist2':               'Shift(rnd,s4,d2,k3)',
    'Shift-random-A-Max3-Seg3-Dist5':               'Shift(rnd,s3,d5,k3)',
    'Shift-best-A-Max5-Seg2-Dist3':                 'Shift(best,s2,d3)',
    'Shift-best-A-Max5-Seg3-Dist2':                 'Shift(best,s3,d2)',
    'Shift-random-S-Max5-Seg3-Dist3':               'Shift(rnd,SR,s3,d3)',
    'TwoOpt':                                       'TwoOpt',
    'CLS-M1':                                       'CLS-M1',
    'CLS-M2':                                       'CLS-M2',
    'CLS-M3':                                       'CLS-M3',
    'CLS-M4':                                       'CLS-M4',
    'RequestShiftWithin':                           'ReqShift',
    'NodeSwapWithin-WithPrecedence':                'NodeSwap(prec)',
    'NodeSwapWithin':                               'NodeSwap',
}


def _display_name(operator_name):
    """Return the short display name for an operator, falling back to original."""
    return DISPLAY_NAMES.get(operator_name, operator_name)


def _generate_family_shades(base_color, n):
    """Create n shades from light to full saturation for variants within a family."""
    if n == 1:
        return [base_color]
    rgb = mcolors.to_rgb(base_color)
    shades = []
    for i in range(n):
        # Blend from white (light) toward base color
        t = 0.35 + 0.65 * (i / (n - 1))  # range [0.35, 1.0]
        shaded = tuple(1.0 - t * (1.0 - c) for c in rgb)
        shades.append(shaded)
    return shades


def _get_operator_colors(results):
    """Return (ordered_names, colors) grouped by family.

    Only includes operators that are present in the results dict.
    """
    ordered_names = []
    colors = []
    for family, members in OPERATOR_FAMILIES.items():
        present = [m for m in members if m in results]
        if not present:
            continue
        base = FAMILY_BASE_COLORS[family]
        shades = _generate_family_shades(base, len(present))
        ordered_names.extend(present)
        colors.extend(shades)

    # Append any operators not listed in families (safety net)
    for op in results:
        if op not in ordered_names:
            ordered_names.append(op)
            colors.append('#888888')

    return ordered_names, colors


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_experiment():
    """Run convergence experiment for all operators across all instances.

    For each operator, creates a local search that only uses that operator,
    then runs it on each instance and records the convergence data.
    Results are saved incrementally to allow for interrupted runs.
    """
    print("=" * 80)
    print("OPERATOR CONVERGENCE EXPERIMENT")
    print("=" * 80)
    print(f"Problem sizes: {PROBLEM_SIZES}")
    print(f"Runs per instance: {NUM_RUNS_PER_INSTANCE}")
    print(f"Local search max iterations: {LOCAL_SEARCH_MAX_ITERATIONS}")
    print(f"Local search max no improvement: {LOCAL_SEARCH_MAX_NO_IMPROVEMENT}")
    print("=" * 80)

    operators = _create_operators()
    li_lim_manager = LiLimInstanceManager()
    mendeley_manager = MendeleyInstanceManager()
    best_known_solutions = BestKnownSolutions()

    print(f"\nTotal operators to test: {len(operators)}")

    # Store all results
    all_results = {}

    # For each operator, run experiment and save results incrementally
    for op_idx, operator in enumerate(operators):
        print(f"\n[{op_idx+1}/{len(operators)}] Starting operator: {operator.name}")
        print("-" * 80)

        operator_start_time = time.time()

        # Run experiment for this operator
        operator_results = _run_operator_experiment(
            operator,
            li_lim_manager,
            mendeley_manager,
            best_known_solutions
        )

        operator_elapsed = time.time() - operator_start_time

        # Store results
        all_results[operator.name] = operator_results

        # Save results incrementally (in case of crash or interruption)
        _save_results(all_results)

        print(f"  Completed in {operator_elapsed:.2f} seconds")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {RESULTS_OUTPUT_FILE}")
    print("=" * 80)

    return all_results

def _save_results(results):
    """Save results to JSON file.

    Args:
        results: Dictionary of results to save
    """
    # Ensure results directory exists
    Path(RESULTS_BASE_DIR).mkdir(exist_ok=True)

    with open(RESULTS_OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def _extend_convergence_data(times, values, max_time):
    """Extend convergence data to max_time with flat line if needed.

    If the convergence data ends before max_time, this function extends
    the last value as a flat line to the end of the x-axis for better
    visualization.

    Args:
        times: List of time points
        values: List of corresponding values (fitness or num_vehicles)
        max_time: Maximum time to extend to

    Returns:
        tuple: (extended_times, extended_values)
    """
    if not times or not values or len(times) == 0:
        return times, values

    # If already at or past max_time, no extension needed
    if times[-1] >= max_time:
        return times, values

    # Extend with flat line from last value to max_time
    extended_times = list(times) + [max_time]
    extended_values = list(values) + [values[-1]]

    return extended_times, extended_values

def _compute_optimal_x_cutoff(results):
    """Compute optimal x-axis cutoff to handle operators with significantly longer runtime.

    Only applies cutoff if there's a clear cluster of fast operators with 1-2 outliers
    that take significantly longer (e.g., most finish at 0.5s, but 1-2 need 1s).

    Args:
        results: Dictionary containing all experiment results

    Returns:
        tuple: (cutoff_time, has_outliers) where cutoff_time is the optimal max time
               for x-axis (or None if no cutoff needed), and has_outliers indicates
               whether outliers were detected
    """
    if not results:
        return None, False

    # Collect max times for each operator
    operator_max_times = []
    for operator_name, operator_results in results.items():
        operator_times = []
        for instance_name, instance_data in operator_results.items():
            for run_data in instance_data['runs']:
                if run_data['convergence_times']:
                    operator_times.append(run_data['convergence_times'][-1])

        if operator_times:
            operator_max_times.append(max(operator_times))

    if len(operator_max_times) < 3:
        # Not enough data to determine outliers
        return None, False

    # Compute statistics
    median_time = np.median(operator_max_times)
    percentile_90 = np.percentile(operator_max_times, 90)
    max_time = max(operator_max_times)

    # Check if there's a significant outlier:
    # If max is more than 2x the median, we likely have outliers
    if max_time > 2 * median_time:
        # Use 90th percentile as cutoff
        return percentile_90, True
    else:
        # No significant outliers detected
        return None, False

def _compute_average_convergence(runs_data, metric='fitnesses', num_bins=100):
    """Compute average convergence curve from multiple runs.

    Args:
        runs_data: List of run dictionaries containing convergence data
        metric: Either 'fitnesses' or 'num_vehicles'
        num_bins: Number of time bins for aggregation (default 100)

    Returns:
        tuple: (time_points, mean_values, std_values) for plotting
    """
    if not runs_data:
        return [], [], []

    # Determine max time across all runs
    max_time = 0
    for run_data in runs_data:
        if run_data['convergence_times']:
            max_time = max(max_time, run_data['convergence_times'][-1])

    if max_time == 0:
        return [], [], []

    # Extend all runs to max_time
    extended_curves = []
    for run_data in runs_data:
        if metric == 'fitnesses':
            extended_times, extended_values = _extend_convergence_data(
                run_data['convergence_times'],
                run_data['convergence_fitnesses'],
                max_time
            )
        else:  # num_vehicles
            extended_times, extended_values = _extend_convergence_data(
                run_data['convergence_times'],
                run_data['convergence_num_vehicles'],
                max_time
            )
        extended_curves.append({
            'times': extended_times,
            'values': extended_values
        })

    # Create time bins
    time_bins = np.linspace(0, max_time, num_bins)
    mean_values = []
    std_values = []

    # Compute mean and std at each time bin
    for t in time_bins:
        values_at_t = []
        for curve in extended_curves:
            times = np.array(curve['times'])
            values = np.array(curve['values'])
            idx_before = np.searchsorted(times, t, side='right') - 1
            if idx_before >= 0:
                values_at_t.append(values[idx_before])

        if values_at_t:
            mean_values.append(np.mean(values_at_t))
            std_values.append(np.std(values_at_t))
        else:
            mean_values.append(np.nan)
            std_values.append(0)

    return time_bins, mean_values, std_values

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


def _extract_operator_metrics(results):
    """Compute per-operator aggregate metrics across all instances and runs.

    Args:
        results: Dictionary containing all experiment results

    Returns:
        dict: {operator_name: {metric_name: value, ...}}
    """
    metrics = {}
    for operator_name, operator_results in results.items():
        improvements_list = []
        improvement_rate_list = []
        improvement_mag_list = []
        final_fitness_list = []
        time_list = []

        for instance_name, instance_data in operator_results.items():
            for run in instance_data['runs']:
                n_imp = run['num_improvements']
                t = run['total_time']
                init_f = run['initial_fitness']
                final_f = run['final_fitness']

                improvements_list.append(n_imp)
                improvement_rate_list.append(n_imp / t if t > 0 else 0.0)
                improvement_mag_list.append(
                    (init_f - final_f) / max(1, n_imp)
                )
                final_fitness_list.append(final_f)
                time_list.append(t)

        metrics[operator_name] = {
            'avg_improvements': np.mean(improvements_list),
            'std_improvements': np.std(improvements_list),
            'avg_improvement_rate': np.mean(improvement_rate_list),
            'std_improvement_rate': np.std(improvement_rate_list),
            'avg_improvement_magnitude': np.mean(improvement_mag_list),
            'std_improvement_magnitude': np.std(improvement_mag_list),
            'avg_final_fitness': np.mean(final_fitness_list),
            'std_final_fitness': np.std(final_fitness_list),
            'avg_time': np.mean(time_list),
            'std_time': np.std(time_list),
        }

    return metrics


def _plot_operator_bar_chart(results, metrics, metric_key, std_key, ylabel,
                             title, filename, start_at_zero=True, ax=None):
    """Reusable bar chart for operator metrics, family-grouped with separators.

    Args:
        results: Full results dict (used for ordering)
        metrics: Dict from _extract_operator_metrics
        metric_key: Key in metrics for bar heights
        std_key: Key in metrics for error bar values
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename (without extension), or None if using ax
        start_at_zero: Whether y-axis starts at 0
        ax: Optional axes to draw on (for grid plots). If None, creates new figure.

    Returns:
        fig if ax is None, else None
    """
    ordered_names, colors = _get_operator_colors(results)

    values = [metrics[n][metric_key] for n in ordered_names]
    std_values = [metrics[n][std_key] for n in ordered_names]
    display_labels = [_display_name(n) for n in ordered_names]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(22, 10))

    x = np.arange(len(ordered_names))
    width = 0.72

    # Draw bars
    bars = ax.bar(x, values, width, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.0)

    # Error bars with darkened matching colors
    for i, (val, std, col) in enumerate(zip(values, std_values, colors)):
        if std > 0:
            dark = darken_color(col, factor=0.66)
            ax.errorbar(x[i], val, yerr=std, fmt='none',
                        capsize=8, capthick=2.0, elinewidth=2.0,
                        ecolor=dark)

    # Separator lines between families
    pos = 0
    for family, members in OPERATOR_FAMILIES.items():
        present = [m for m in members if m in results]
        if not present:
            continue
        pos += len(present)
        if pos < len(ordered_names):
            ax.axvline(x=pos - 0.5, color='black', linestyle='--',
                       linewidth=1.2, alpha=0.4)

    # Axis limits
    y_min, y_max, _ = compute_axis_limits_with_outlier_truncation(
        values, std_values=std_values, start_at_zero=start_at_zero
    )
    ax.set_ylim(bottom=y_min, top=y_max)

    # Labels
    ax.set_xlabel('Operator')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=60, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

    if standalone:
        plt.tight_layout()
        filepath = Path(PLOTS_OUTPUT_DIR) / filename
        save_figure_multi_format(fig, str(filepath))
        plt.close()
        return fig
    return None


# ---- Thin wrapper plot functions ----

def plot_avg_improvements(results, metrics):
    """Bar chart: average number of improvements per operator."""
    _plot_operator_bar_chart(
        results, metrics,
        metric_key='avg_improvements', std_key='std_improvements',
        ylabel='Avg. Number of Improvements',
        title='Average Number of Improvements by Operator',
        filename='operator_avg_improvements',
        start_at_zero=True,
    )

def plot_improvement_rate(results, metrics):
    """Bar chart: improvements per second per operator."""
    _plot_operator_bar_chart(
        results, metrics,
        metric_key='avg_improvement_rate', std_key='std_improvement_rate',
        ylabel='Improvements per Second',
        title='Improvement Rate by Operator',
        filename='operator_improvement_rate',
        start_at_zero=True,
    )

def plot_avg_improvement_magnitude(results, metrics):
    """Bar chart: average fitness gain per improvement per operator."""
    _plot_operator_bar_chart(
        results, metrics,
        metric_key='avg_improvement_magnitude', std_key='std_improvement_magnitude',
        ylabel='Avg. Fitness Gain per Improvement',
        title='Average Improvement Magnitude by Operator',
        filename='operator_improvement_magnitude',
        start_at_zero=True,
    )

def plot_avg_final_fitness(results, metrics):
    """Bar chart: average final fitness per operator."""
    _plot_operator_bar_chart(
        results, metrics,
        metric_key='avg_final_fitness', std_key='std_final_fitness',
        ylabel='Final Fitness',
        title='Average Final Fitness by Operator',
        filename='operator_avg_final_fitness',
        start_at_zero=True,
    )

def plot_avg_computation_time(results, metrics):
    """Bar chart: average computation time per operator."""
    _plot_operator_bar_chart(
        results, metrics,
        metric_key='avg_time', std_key='std_time',
        ylabel='Time (s)',
        title='Average Computation Time by Operator',
        filename='operator_avg_time',
        start_at_zero=True,
    )


def plot_operator_overview_grid(results, metrics):
    """2x2 grid of key operator metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(28, 20))

    panels = [
        (axes[0, 0], 'avg_improvements', 'std_improvements',
         'Avg. Number of Improvements', 'Average Improvements', True),
        (axes[0, 1], 'avg_improvement_rate', 'std_improvement_rate',
         'Improvements per Second', 'Improvement Rate', True),
        (axes[1, 0], 'avg_final_fitness', 'std_final_fitness',
         'Final Fitness', 'Final Fitness', True),
        (axes[1, 1], 'avg_time', 'std_time',
         'Time (s)', 'Computation Time', True),
    ]

    for ax, mk, sk, yl, ttl, saz in panels:
        _plot_operator_bar_chart(
            results, metrics,
            metric_key=mk, std_key=sk,
            ylabel=yl, title=ttl, filename=None,
            start_at_zero=saz, ax=ax,
        )

    fig.suptitle('Operator Performance Overview', fontsize=28, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    filepath = Path(PLOTS_OUTPUT_DIR) / 'operator_overview_grid'
    save_figure_multi_format(fig, str(filepath))
    plt.close()


def plot_all_results(results):
    """Generate all plots from experiment results.

    Args:
        results: Dictionary containing all experiment results
    """
    # Create output directory (with parents)
    Path(PLOTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # Extract operator metrics for bar charts
    print("Extracting operator metrics...")
    metrics = _extract_operator_metrics(results)

    # 1. Plot aggregated convergence curves (all operators on one plot)
    print("Plotting aggregated convergence curves...")
    plot_aggregated_convergence(results)

    # 2. Plot per-operator average across all instances
    print("Plotting per-operator averages across all instances...")
    plot_per_operator_average_all_instances(results)

    # 3. Bar charts for operator metrics
    print("Plotting average improvements bar chart...")
    plot_avg_improvements(results, metrics)

    print("Plotting improvement rate bar chart...")
    plot_improvement_rate(results, metrics)

    print("Plotting improvement magnitude bar chart...")
    plot_avg_improvement_magnitude(results, metrics)

    print("Plotting average final fitness bar chart...")
    plot_avg_final_fitness(results, metrics)

    print("Plotting average computation time bar chart...")
    plot_avg_computation_time(results, metrics)

    # 4. Overview grid
    print("Plotting operator overview grid...")
    plot_operator_overview_grid(results, metrics)

    print(f"\nAll plots saved to: {PLOTS_OUTPUT_DIR}/")
    print("=" * 80)


def plot_aggregated_convergence(results):
    """Plot aggregated convergence curves across all instances for each operator.

    Args:
        results: Dictionary containing all experiment results
    """
    fig, ax = plt.subplots(figsize=(18, 10))

    # Compute optimal x-axis cutoff
    cutoff_time, has_outliers = _compute_optimal_x_cutoff(results)

    # Determine max_time for the plot
    if cutoff_time is not None:
        max_time = cutoff_time
    else:
        # Use absolute max time across all operators
        max_time = 0
        for operator_name, operator_results in results.items():
            for instance_name, instance_data in operator_results.items():
                for run_data in instance_data['runs']:
                    if run_data['convergence_times']:
                        max_time = max(max_time, run_data['convergence_times'][-1])

    # Use family-grouped colors
    ordered_names, colors = _get_operator_colors(results)

    for idx, operator_name in enumerate(ordered_names):
        operator_results = results[operator_name]
        all_fitness_curves = []

        # Collect all convergence curves and extend them to max_time
        for instance_name, instance_data in operator_results.items():
            for run_data in instance_data['runs']:
                # Extend convergence data to max_time
                extended_times, extended_fitnesses = _extend_convergence_data(
                    run_data['convergence_times'],
                    run_data['convergence_fitnesses'],
                    max_time
                )
                all_fitness_curves.append({
                    'times': extended_times,
                    'fitnesses': extended_fitnesses
                })

        if not all_fitness_curves:
            continue

        # Create time bins for aggregation
        time_bins = np.linspace(0, max_time, 100)
        aggregated_fitness = []

        for t in time_bins:
            fitness_values = []
            for curve in all_fitness_curves:
                # Find the fitness value at time t (or closest before)
                times = np.array(curve['times'])
                fitnesses = np.array(curve['fitnesses'])
                idx_before = np.searchsorted(times, t, side='right') - 1
                if idx_before >= 0:
                    fitness_values.append(fitnesses[idx_before])

            if fitness_values:
                aggregated_fitness.append(np.median(fitness_values))
            else:
                aggregated_fitness.append(np.nan)

        # Get line style (cycle through available styles)
        line_style = LINE_STYLES[idx % len(LINE_STYLES)]

        # Plot median curve with display name
        ax.plot(time_bins, aggregated_fitness, label=_display_name(operator_name),
               color=colors[idx], linestyle=line_style, linewidth=2.5, alpha=0.85)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Median Fitness')
    ax.set_title('Aggregated Convergence Curves (Median across all runs)',
                fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, edgecolor='gray')

    # Add annotation if cutoff was applied
    if has_outliers and cutoff_time is not None:
        ax.text(0.02, 0.98, f'Note: X-axis limited to {cutoff_time:.2f}s\n(some operators continue beyond)',
                transform=ax.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    filepath = Path(PLOTS_OUTPUT_DIR) / 'aggregated_convergence'
    save_figure_multi_format(fig, str(filepath))
    plt.close()

def plot_per_operator_average_all_instances(results):
    """Plot average convergence curve for each operator across all instances.

    Creates one plot per operator showing the average performance across all
    instances and runs, with both fitness and number of vehicles.

    Args:
        results: Dictionary containing all experiment results
    """
    # Create subdirectory for per-operator averages
    per_operator_avg_dir = Path(PLOTS_OUTPUT_DIR) / 'per_operator_averages'
    per_operator_avg_dir.mkdir(parents=True, exist_ok=True)

    # Build operator -> family lookup
    op_to_family = {}
    for family, members in OPERATOR_FAMILIES.items():
        for m in members:
            op_to_family[m] = family

    for operator_name, operator_results in results.items():
        # Collect all runs from all instances for this operator
        all_runs = []
        for instance_name, instance_data in operator_results.items():
            all_runs.extend(instance_data['runs'])

        if not all_runs:
            continue

        # Get family base color for this operator
        family = op_to_family.get(operator_name)
        color_fitness = FAMILY_BASE_COLORS.get(family, METHOD_COLORS[0]) if family else METHOD_COLORS[0]
        color_vehicles = darken_color(color_fitness, factor=0.7)

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Compute average convergence for fitness
        time_bins_fitness, mean_fitness, std_fitness = _compute_average_convergence(
            all_runs, metric='fitnesses'
        )

        # Plot fitness convergence
        ax1 = axes[0]
        if len(time_bins_fitness) > 0:
            # Plot mean line
            ax1.plot(time_bins_fitness, mean_fitness, color=color_fitness,
                    linewidth=3, label='Mean', alpha=0.9)
            # Plot shaded std deviation
            mean_fitness_array = np.array(mean_fitness)
            std_fitness_array = np.array(std_fitness)
            ax1.fill_between(time_bins_fitness,
                            mean_fitness_array - std_fitness_array,
                            mean_fitness_array + std_fitness_array,
                            color=color_fitness, alpha=0.25, label='±1 Std Dev')

        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Average Fitness Convergence Across All Instances',
                     fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(framealpha=0.9, edgecolor='gray')

        # Compute average convergence for number of vehicles
        time_bins_vehicles, mean_vehicles, std_vehicles = _compute_average_convergence(
            all_runs, metric='num_vehicles'
        )

        # Plot number of vehicles convergence
        ax2 = axes[1]
        if len(time_bins_vehicles) > 0:
            # Plot mean line
            ax2.plot(time_bins_vehicles, mean_vehicles, color=color_vehicles,
                    linewidth=3, label='Mean', alpha=0.9)
            # Plot shaded std deviation
            mean_vehicles_array = np.array(mean_vehicles)
            std_vehicles_array = np.array(std_vehicles)
            ax2.fill_between(time_bins_vehicles,
                            mean_vehicles_array - std_vehicles_array,
                            mean_vehicles_array + std_vehicles_array,
                            color=color_vehicles, alpha=0.25, label='±1 Std Dev')

        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Number of Vehicles')
        ax2.set_title('Average Vehicles Convergence Across All Instances',
                     fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.legend(framealpha=0.9, edgecolor='gray')

        # Add suptitle with display name and number of instances/runs
        disp = _display_name(operator_name)
        num_instances = len(operator_results)
        num_runs = len(all_runs)
        plt.suptitle(f'Operator: {disp}\n({num_instances} instances, {num_runs} runs)',
                    fontsize=28, fontweight='bold')
        plt.tight_layout()

        # Save plot
        safe_filename = operator_name.replace('/', '_').replace('\\', '_')
        filepath = per_operator_avg_dir / f'{safe_filename}_avg_all_instances'
        save_figure_multi_format(fig, str(filepath))
        plt.close()

def _create_operators():
    return [
        ReinsertOperator(),
        ReinsertOperator(max_attempts=5,clustered=True),
        ReinsertOperator(force_same_vehicle=True),
        ReinsertOperator(allow_same_vehicle=False),
        ReinsertOperator(allow_same_vehicle=False, allow_new_vehicles=False),
        
        RouteEliminationOperator(),
        
        FlipOperator(),
        FlipOperator(max_attempts=5),
        FlipOperator(single_route=True),
        
        MergeOperator(type="random", num_routes=2),
        MergeOperator(type="random", num_routes=2, reorder=False),
        
        MergeOperator(type="min", num_routes=2),
        MergeOperator(type="min", num_routes=2, reorder=False),
        
        SwapWithinOperator(),
        SwapWithinOperator(max_attempts=5),
        SwapWithinOperator(single_route=True),
        SwapWithinOperator(single_route=True, type="best"),
        SwapWithinOperator(single_route=False, type="best"),

        SwapBetweenOperator(),
        SwapBetweenOperator(type="best"),
        
        TransferOperator(),
        TransferOperator(single_route=True),
        TransferOperator(max_attempts=5,single_route=True),

        ShiftOperator(type="random", segment_length=3, max_shift_distance=3, max_attempts=5),
        ShiftOperator(type="random", segment_length=2, max_shift_distance=4, max_attempts=5),
        ShiftOperator(type="random", segment_length=4, max_shift_distance=2, max_attempts=3),
        ShiftOperator(type="random", segment_length=3, max_shift_distance=5, max_attempts=3),
        ShiftOperator(type="best", segment_length=2, max_shift_distance=3),
        ShiftOperator(type="best", segment_length=3, max_shift_distance=2),
        ShiftOperator(type="random", segment_length=3, max_shift_distance=3, max_attempts=5, single_route=True),

        TwoOptOperator(),
        
        CLSM1Operator(),
        CLSM2Operator(),
        CLSM3Operator(),
        CLSM4Operator(),
        
        RequestShiftWithinOperator(),
        NodeSwapWithinOperator(check_precedence=True),
        NodeSwapWithinOperator(check_precedence=False),
    ]
    
def _run_operator_experiment(operator, li_lim_manager, mendeley_manager, best_known_solutions):
    """Run experiment for a single operator across all instances.

    Args:
        operator: The operator to test
        li_lim_manager: Manager for Li&Lim instances
        mendeley_manager: Manager for Mendeley instances
        best_known_solutions: Best known solutions for comparison

    Returns:
        dict: Results organized by instance name, containing tracking data for each run
    """
    print(f"\nTesting operator: {operator.name}")

    # Create local search with only this operator and tracking enabled
    local_search = NaiveLocalSearch(
        operators=[operator],
        max_no_improvement=LOCAL_SEARCH_MAX_NO_IMPROVEMENT,
        max_iterations=LOCAL_SEARCH_MAX_ITERATIONS,
        first_improvement=LOCAL_SEARCH_FIRST_IMPROVEMENT,
        tracking=True
    )

    # Get all instances from both datasets for the specified sizes
    all_instances = []
    for size in PROBLEM_SIZES:
        all_instances.extend(li_lim_manager.get_all(size=size))
        all_instances.extend(mendeley_manager.get_all(size=size))

    results = {}

    # For each instance
    for idx, instance in enumerate(all_instances):
        instance_name = instance.name
        print(f"  [{idx+1}/{len(all_instances)}] Running on {instance_name}...")

        # Get best known solution for this instance
        try:
            bks_fitness, bks_num_vehicles = best_known_solutions.get_bks_as_tuple(instance)
        except:
            bks_fitness = None
            bks_num_vehicles = None

        instance_results = {
            'instance_name': instance_name,
            'bks_fitness': bks_fitness,
            'bks_num_vehicles': bks_num_vehicles,
            'runs': []
        }

        # Multiple runs per instance
        for run in range(NUM_RUNS_PER_INSTANCE):
            # Set deterministic seed based on instance (all runs for this instance use same seed)
            seed = idx
            random.seed(seed)
            np.random.seed(seed)

            # Generate random initial solution (now deterministic)
            initial_solution = generate_random_solution(instance)

            # Run local search with tracking
            start_time = time.time()
            result = local_search.search(instance, initial_solution)
            elapsed_time = time.time() - start_time

            # Extract results (when tracking=True, returns 5 values)
            best_solution, best_fitness, best_fitnesses, best_num_vehicles, times = result

            # Convert absolute times to relative times (seconds from start)
            relative_times = [t - times[0] for t in times]

            # Store run data
            run_data = {
                'run_number': run,
                'initial_fitness': best_fitnesses[0],
                'initial_num_vehicles': best_num_vehicles[0],
                'final_fitness': best_fitness,
                'final_num_vehicles': best_solution.num_vehicles_used,
                'convergence_fitnesses': best_fitnesses,
                'convergence_num_vehicles': best_num_vehicles,
                'convergence_times': relative_times,
                'total_time': elapsed_time,
                'num_improvements': len(best_fitnesses) - 1  # Number of improvements found
            }

            instance_results['runs'].append(run_data)

        results[instance_name] = instance_results

    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check if results file already exists
    if os.path.exists(RESULTS_OUTPUT_FILE):
        print(f"Results file found: {RESULTS_OUTPUT_FILE}")
        print("Loading existing results and generating plots...")
        try:
            with open(RESULTS_OUTPUT_FILE, 'r') as f:
                results = json.load(f)
            plot_all_results(results)
            print("Done!")
        except Exception as e:
            print(f"Error while plotting: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run full experiment
        try:
            results = run_experiment()
            print(f"\nAll results successfully saved to {RESULTS_OUTPUT_FILE}")

            # Generate plots
            plot_all_results(results)

        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user.")
            print(f"Partial results saved to {RESULTS_OUTPUT_FILE}")

            # Try to plot partial results
            try:
                with open(RESULTS_OUTPUT_FILE, 'r') as f:
                    results = json.load(f)
                if results:
                    print("\nGenerating plots from partial results...")
                    plot_all_results(results)
            except:
                pass

        except Exception as e:
            print(f"\n\nError during experiment: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nPartial results may be saved to {RESULTS_OUTPUT_FILE}")
