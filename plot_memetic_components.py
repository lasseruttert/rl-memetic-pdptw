"""
Visualization Script for Memetic Component-wise Performance Experiment

Generates focused plots for analyzing memetic algorithm component performance:
1. Convergence plots (4 metrics: best_fitness, avg_fitness, num_vehicles, total_distance)
2. Time to best solution
3. Efficiency score: 1 / (final_fitness × time_to_best) - higher is better
4. Time-quality product: final_fitness × time_to_best - lower is better

Each plot type is generated for:
- Overall (all instances)
- Per instance category (lc1, lc2, lr1, lr2, lrc1, lrc2, bar, ber, nyc, poa)

Output formats: PNG (300 DPI) and PDF (vector)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_FOLDER = "results/memetic_component_wise"
OUTPUT_BASE_DIR = "results/memetic_component_plots"

# Plot styling - Thesis quality
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
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

# Color scheme
COMBO_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_instance_name(instance_name):
    """
    Parse instance name to extract category information.

    Args:
        instance_name: Name of the instance (may include path)

    Returns:
        dict: Contains 'benchmark', 'category', 'instance_name'
    """
    # Extract just the filename if full path is provided
    if '\\' in instance_name or '/' in instance_name:
        instance_name = Path(instance_name).stem

    # Check if Mendeley instance (bar, ber, nyc, poa)
    mendeley_match = re.match(r'([a-z]+)-n(\d+)-(\d+)', instance_name)
    if mendeley_match:
        return {
            'benchmark': 'mendeley',
            'category': mendeley_match.group(1),
            'instance_name': instance_name,
        }

    # Check if Li & Lim instance (size 100 format: lc101, lr205, etc.)
    lilim_100_match = re.match(r'(l[rc]{1,2}[12])(\d+)', instance_name)
    if lilim_100_match:
        return {
            'benchmark': 'li_lim',
            'category': lilim_100_match.group(1),
            'instance_name': instance_name,
        }

    # Check if Li & Lim instance (larger sizes format: LC1_2_1, etc.)
    lilim_large_match = re.match(r'(L[RC]{1,2}[12])_(\d+)_(\d+)', instance_name)
    if lilim_large_match:
        return {
            'benchmark': 'li_lim',
            'category': lilim_large_match.group(1).lower(),
            'instance_name': instance_name,
        }

    return {
        'benchmark': 'unknown',
        'category': 'unknown',
        'instance_name': instance_name,
    }

def load_results(folder_path):
    """Load results from all combo JSON files in the specified folder.

    Args:
        folder_path: Path to folder containing combo_*.json files

    Returns:
        dict: Aggregated results with structure {combo_id: combo_data}
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Results folder not found: {folder_path}")

    # Find all combo JSON files
    combo_files = sorted(folder.glob("combo_*.json"))

    if not combo_files:
        raise FileNotFoundError(f"No combo_*.json files found in {folder_path}")

    # Load and aggregate all combo files
    aggregated_results = {}

    for combo_file in combo_files:
        with open(combo_file, 'r') as f:
            combo_data = json.load(f)

        # Extract combo_id from the data itself (more reliable than filename)
        combo_id = combo_data.get('combination_id')

        if combo_id is None:
            # Fallback: extract from filename (e.g., "combo_0_Baseline.json" -> "combo_0")
            filename = combo_file.stem  # "combo_0_Baseline"
            combo_id = filename.split('_')[0] + '_' + filename.split('_')[1]  # "combo_0"

        aggregated_results[combo_id] = combo_data

    print(f"Loaded {len(aggregated_results)} combinations from {folder_path}")

    return aggregated_results

def organize_by_category(results):
    """
    Organize results by instance category.

    Returns:
        dict: {category: {combo_id: {instance_name: instance_data}}}
    """
    by_category = defaultdict(lambda: defaultdict(dict))

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            parsed = parse_instance_name(instance_name)
            category = parsed['category']
            by_category[category][combo_id][instance_name] = instance_data

    return by_category

def save_figure(fig, filepath, formats=['png', 'pdf']):
    """Save figure in multiple formats."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = Path(filepath).with_suffix(f'.{fmt}')
        if fmt == 'pdf':
            fig.savefig(output_path, bbox_inches='tight', format=fmt, backend='pdf')
        else:
            fig.savefig(output_path, bbox_inches='tight', format=fmt)
        print(f"  Saved: {output_path}")

# ============================================================================
# CONVERGENCE PLOTTING FUNCTIONS
# ============================================================================

def plot_convergence_metric(results, output_dir, metric_name, ylabel, title_suffix, category=None):
    """
    Generic function to plot convergence of a specific metric over time.

    Args:
        results: Results dictionary
        output_dir: Output directory path
        metric_name: Name of metric in convergence dict ('best_fitness', 'avg_fitness', 'num_vehicles', 'total_distance')
        ylabel: Y-axis label
        title_suffix: Suffix for plot title
        category: Category to filter by (None for overall)
    """
    category_label = category if category else "overall"
    print(f"  Creating {metric_name} convergence ({category_label})...")

    # Collect convergence data
    combo_convergence = defaultdict(list)

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            if instance_data.get('convergence'):
                # Convert dict to list of (time, metric_value) tuples
                conv_dict = instance_data['convergence']
                conv_list = [(float(t), data.get(metric_name)) for t, data in conv_dict.items() if data.get(metric_name) is not None]
                conv_list.sort(key=lambda x: x[0])  # Sort by time
                if conv_list:
                    combo_convergence[combo_id].append(conv_list)

    if not any(combo_convergence.values()):
        print(f"    No convergence data found for {metric_name} in category {category_label}")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5.5))

    combo_ids = sorted(combo_convergence.keys())
    for i, combo_id in enumerate(combo_ids):
        convergences = combo_convergence[combo_id]
        if not convergences:
            continue

        # Collect all unique time points
        all_times = sorted(set(t for conv in convergences for t, _ in conv))

        # Initialize lists for this combination
        times = []
        metric_means = []
        metric_stds = []

        for t in all_times:
            metric_at_t = []
            for conv in convergences:
                # Find metric value at time t (use last known value if not exact)
                metric_val = None
                for conv_time, conv_metric in conv:
                    if conv_time <= t:
                        metric_val = conv_metric
                    else:
                        break
                if metric_val is not None:
                    metric_at_t.append(metric_val)

            if metric_at_t:
                times.append(t)
                metric_means.append(np.mean(metric_at_t))
                metric_stds.append(np.std(metric_at_t))

        if times:
            color = COMBO_COLORS[i % len(COMBO_COLORS)]
            # Plot mean line
            line, = ax.plot(times, metric_means,
                          label=f"C{combo_id.split('_')[1]} (mean)",
                          color=color, linewidth=2)
            # Plot shaded std dev area
            ax.fill_between(times,
                          np.array(metric_means) - np.array(metric_stds),
                          np.array(metric_means) + np.array(metric_stds),
                          alpha=0.2, color=color,
                          label=f"C{combo_id.split('_')[1]} (±1 std dev)")

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(ylabel)

    if category:
        title = f'{title_suffix} - {category.upper()}'
    else:
        title = f'{title_suffix} - Overall'
    ax.set_title(title)

    # Improved legend with explanation
    ax.legend(loc='best', ncol=2)
    ax.grid(alpha=0.3)

    # Save
    filename = f'convergence_{metric_name}_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

def plot_convergence_best_fitness(results, output_dir, category=None):
    """Plot best fitness convergence over time."""
    plot_convergence_metric(
        results, output_dir,
        metric_name='best_fitness',
        ylabel='Best Fitness',
        title_suffix='Best Fitness Convergence',
        category=category
    )

def plot_convergence_avg_fitness(results, output_dir, category=None):
    """Plot average fitness convergence over time."""
    plot_convergence_metric(
        results, output_dir,
        metric_name='avg_fitness',
        ylabel='Average Fitness',
        title_suffix='Average Fitness Convergence',
        category=category
    )

def plot_convergence_num_vehicles(results, output_dir, category=None):
    """Plot number of vehicles convergence over time."""
    plot_convergence_metric(
        results, output_dir,
        metric_name='num_vehicles',
        ylabel='Number of Vehicles',
        title_suffix='Number of Vehicles Convergence',
        category=category
    )

def plot_convergence_total_distance(results, output_dir, category=None):
    """Plot total distance convergence over time."""
    plot_convergence_metric(
        results, output_dir,
        metric_name='total_distance',
        ylabel='Total Distance',
        title_suffix='Total Distance Convergence',
        category=category
    )

# ============================================================================
# TIME TO BEST PLOTTING FUNCTION
# ============================================================================

def plot_time_to_best(results, output_dir, category=None):
    """Bar chart showing time to reach best solution."""
    category_label = category if category else "overall"
    print(f"  Creating time to best solution chart ({category_label})...")

    # Collect data
    combo_times = defaultdict(list)

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            if instance_data.get('final_evaluation'):
                time_to_best = instance_data['final_evaluation'].get('time_to_final_fitness')
                if time_to_best is not None:
                    combo_times[combo_id].append(time_to_best)

    # Filter out combinations with no data
    combo_ids = sorted([cid for cid in combo_times.keys() if combo_times[cid]])
    if not combo_ids:
        print(f"    No data found for category {category_label}")
        return

    # Calculate averages
    avg_times = [np.mean(combo_times[cid]) for cid in combo_ids]

    # Create plot
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(combo_ids))
    bars = ax.bar(x, avg_times, color=COMBO_COLORS[:len(combo_ids)],
                   alpha=0.8, edgecolor='black')

    ax.set_xlabel('Combination')
    ax.set_ylabel('Average Time to Best (seconds)')

    if category:
        title = f'Time to Best Solution - {category.upper()}'
    else:
        title = 'Time to Best Solution - Overall'
    ax.set_title(title)

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)

    # Save
    filename = f'time_to_best_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

def plot_efficiency_score(results, output_dir, category=None):
    """Bar chart showing efficiency score: 1 / (final_fitness × time_to_best) - higher is better."""
    category_label = category if category else "overall"
    print(f"  Creating efficiency score chart ({category_label})...")

    # Collect data
    combo_scores = defaultdict(list)

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            final_eval = instance_data.get('final_evaluation')
            if final_eval:
                time_to_best = final_eval.get('time_to_final_fitness')
                final_fitness = final_eval.get('final_fitness')

                if time_to_best is not None and final_fitness is not None and time_to_best > 0 and final_fitness > 0:
                    # Calculate efficiency score (higher is better)
                    score = 1.0 / (final_fitness * time_to_best)
                    combo_scores[combo_id].append(score)

    # Filter out combinations with no data
    combo_ids = sorted([cid for cid in combo_scores.keys() if combo_scores[cid]])
    if not combo_ids:
        print(f"    No data found for category {category_label}")
        return

    # Calculate averages
    avg_scores = [np.mean(combo_scores[cid]) for cid in combo_ids]

    # Create plot
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(combo_ids))
    bars = ax.bar(x, avg_scores, color=COMBO_COLORS[:len(combo_ids)],
                   alpha=0.8, edgecolor='black')

    ax.set_xlabel('Combination')
    ax.set_ylabel('Efficiency Score (higher is better)')

    if category:
        title = f'Efficiency Score - {category.upper()}'
    else:
        title = 'Efficiency Score - Overall'
    ax.set_title(title)

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}',
                ha='center', va='bottom', fontsize=8)

    # Save
    filename = f'efficiency_score_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

def plot_time_quality_product(results, output_dir, category=None):
    """Bar chart showing time-quality product: final_fitness × time_to_best - lower is better."""
    category_label = category if category else "overall"
    print(f"  Creating time-quality product chart ({category_label})...")

    # Collect data
    combo_products = defaultdict(list)

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            final_eval = instance_data.get('final_evaluation')
            if final_eval:
                time_to_best = final_eval.get('time_to_final_fitness')
                final_fitness = final_eval.get('final_fitness')

                if time_to_best is not None and final_fitness is not None:
                    # Calculate time-quality product (lower is better)
                    product = final_fitness * time_to_best
                    combo_products[combo_id].append(product)

    # Filter out combinations with no data
    combo_ids = sorted([cid for cid in combo_products.keys() if combo_products[cid]])
    if not combo_ids:
        print(f"    No data found for category {category_label}")
        return

    # Calculate averages
    avg_products = [np.mean(combo_products[cid]) for cid in combo_ids]

    # Create plot
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(combo_ids))
    bars = ax.bar(x, avg_products, color=COMBO_COLORS[:len(combo_ids)],
                   alpha=0.8, edgecolor='black')

    ax.set_xlabel('Combination')
    ax.set_ylabel('Time-Quality Product (lower is better)')

    if category:
        title = f'Time-Quality Product - {category.upper()}'
    else:
        title = 'Time-Quality Product - Overall'
    ax.set_title(title)

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)

    # Save
    filename = f'time_quality_product_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()


def plot_combined_convergence(results, output_dir, category=None):
    """Create combined convergence plot with all 4 metrics in subplots.

    Args:
        results: Results dictionary
        output_dir: Output directory path
        category: Category to filter by (None for overall)
    """
    category_label = category if category else "overall"
    print(f"  Creating combined convergence plot ({category_label})...")

    metrics = [
        ('best_fitness', 'Best Fitness', 'Best Fitness Convergence'),
        ('avg_fitness', 'Average Fitness', 'Average Fitness Convergence'),
        ('num_vehicles', 'Number of Vehicles', 'Number of Vehicles Convergence'),
        ('total_distance', 'Total Distance', 'Total Distance Convergence')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_name, ylabel, _) in enumerate(metrics):
        ax = axes[idx]

        # Collect convergence data for this metric
        combo_convergence = defaultdict(list)

        for combo_id, combo_data in results.items():
            for instance_name, instance_data in combo_data['instances'].items():
                if category:
                    parsed = parse_instance_name(instance_name)
                    if parsed['category'] != category:
                        continue

                if instance_data.get('convergence'):
                    conv_dict = instance_data['convergence']
                    conv_list = [(float(t), data.get(metric_name))
                                for t, data in conv_dict.items()
                                if data.get(metric_name) is not None]
                    conv_list.sort(key=lambda x: x[0])
                    if conv_list:
                        combo_convergence[combo_id].append(conv_list)

        if not any(combo_convergence.values()):
            ax.text(0.5, 0.5, f'No data for {metric_name}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot each combination
        combo_ids = sorted(combo_convergence.keys())
        for i, combo_id in enumerate(combo_ids):
            convergences = combo_convergence[combo_id]
            if not convergences:
                continue

            all_times = sorted(set(t for conv in convergences for t, _ in conv))
            times = []
            metric_means = []

            for t in all_times:
                metric_at_t = []
                for conv in convergences:
                    metric_val = None
                    for conv_time, conv_metric in conv:
                        if conv_time <= t:
                            metric_val = conv_metric
                        else:
                            break
                    if metric_val is not None:
                        metric_at_t.append(metric_val)

                if metric_at_t:
                    times.append(t)
                    metric_means.append(np.mean(metric_at_t))

            if times:
                color = COMBO_COLORS[i % len(COMBO_COLORS)]
                ax.plot(times, metric_means,
                       label=f"C{combo_id.split('_')[1]}",
                       color=color, linewidth=2)

        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

    # Overall title
    if category:
        title = f'Convergence Metrics - {category.upper()}'
    else:
        title = 'Convergence Metrics - Overall'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    filename = f'combined_convergence_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()


def plot_combined_metrics_bars(results, output_dir, category=None):
    """Create combined bar chart with time-to-best, efficiency, and time-quality.

    Args:
        results: Results dictionary
        output_dir: Output directory path
        category: Category to filter by (None for overall)
    """
    category_label = category if category else "overall"
    print(f"  Creating combined metrics bar chart ({category_label})...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Collect data for all three metrics
    combo_times = defaultdict(list)
    combo_scores = defaultdict(list)
    combo_products = defaultdict(list)

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            final_eval = instance_data.get('final_evaluation')
            if final_eval:
                time_to_best = final_eval.get('time_to_final_fitness')
                final_fitness = final_eval.get('final_fitness')

                if time_to_best is not None:
                    combo_times[combo_id].append(time_to_best)

                if (time_to_best is not None and final_fitness is not None
                    and time_to_best > 0 and final_fitness > 0):
                    combo_scores[combo_id].append(1.0 / (final_fitness * time_to_best))
                    combo_products[combo_id].append(final_fitness * time_to_best)

    # Plot 1: Time to Best
    ax = axes[0]
    combo_ids = sorted([cid for cid in combo_times.keys() if combo_times[cid]])
    if combo_ids:
        avg_times = [np.mean(combo_times[cid]) for cid in combo_ids]
        x = np.arange(len(combo_ids))
        ax.bar(x, avg_times, color=COMBO_COLORS[:len(combo_ids)],
               alpha=0.8, edgecolor='black')
        ax.set_xlabel('Combination', fontsize=11)
        ax.set_ylabel('Avg Time (s)', fontsize=11)
        ax.set_title('Time to Best', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids], fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Plot 2: Efficiency Score
    ax = axes[1]
    combo_ids = sorted([cid for cid in combo_scores.keys() if combo_scores[cid]])
    if combo_ids:
        avg_scores = [np.mean(combo_scores[cid]) for cid in combo_ids]
        x = np.arange(len(combo_ids))
        ax.bar(x, avg_scores, color=COMBO_COLORS[:len(combo_ids)],
               alpha=0.8, edgecolor='black')
        ax.set_xlabel('Combination', fontsize=11)
        ax.set_ylabel('Efficiency (higher better)', fontsize=11)
        ax.set_title('Efficiency Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids], fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Plot 3: Time-Quality Product
    ax = axes[2]
    combo_ids = sorted([cid for cid in combo_products.keys() if combo_products[cid]])
    if combo_ids:
        avg_products = [np.mean(combo_products[cid]) for cid in combo_ids]
        x = np.arange(len(combo_ids))
        ax.bar(x, avg_products, color=COMBO_COLORS[:len(combo_ids)],
               alpha=0.8, edgecolor='black')
        ax.set_xlabel('Combination', fontsize=11)
        ax.set_ylabel('Product (lower better)', fontsize=11)
        ax.set_title('Time-Quality Product', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids], fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Overall title
    if category:
        title = f'Performance Metrics - {category.upper()}'
    else:
        title = 'Performance Metrics - Overall'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    filename = f'combined_metrics_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def create_all_plots():
    """Main function to orchestrate all plot generation."""
    print("=" * 80)
    print("MEMETIC COMPONENT VISUALIZATION")
    print("=" * 80)

    # Load results
    print(f"\nLoading results from {RESULTS_FOLDER}...")
    results = load_results(RESULTS_FOLDER)

    # Organize by category
    by_category = organize_by_category(results)
    categories = sorted(by_category.keys())
    print(f"Found {len(categories)} categories: {', '.join(categories)}")

    # Create output directories
    overall_dir = Path(OUTPUT_BASE_DIR) / "overall"
    individual_dir = Path(OUTPUT_BASE_DIR) / "individual"

    # ========================================================================
    # OVERALL PLOTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING OVERALL PLOTS")
    print("=" * 80)

    print("\n[1/9] Best Fitness Convergence")
    plot_convergence_best_fitness(results, overall_dir)

    print("\n[2/9] Average Fitness Convergence")
    plot_convergence_avg_fitness(results, overall_dir)

    print("\n[3/9] Number of Vehicles Convergence")
    plot_convergence_num_vehicles(results, overall_dir)

    print("\n[4/9] Total Distance Convergence")
    plot_convergence_total_distance(results, overall_dir)

    print("\n[5/9] Time to Best Solution")
    plot_time_to_best(results, overall_dir)

    print("\n[6/9] Efficiency Score")
    plot_efficiency_score(results, overall_dir)

    print("\n[7/9] Time-Quality Product")
    plot_time_quality_product(results, overall_dir)

    print("\n[8/9] Combined Convergence")
    plot_combined_convergence(results, overall_dir)

    print("\n[9/9] Combined Metrics")
    plot_combined_metrics_bars(results, overall_dir)

    # ========================================================================
    # PER-CATEGORY PLOTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING PER-CATEGORY PLOTS")
    print("=" * 80)

    for category in categories:
        print(f"\n--- Category: {category.upper()} ---")
        category_dir = individual_dir / category

        # Filter results for this category
        category_results = {}
        for combo_id, combo_data in results.items():
            category_instances = {}
            for instance_name, instance_data in combo_data['instances'].items():
                parsed = parse_instance_name(instance_name)
                if parsed['category'] == category:
                    category_instances[instance_name] = instance_data

            if category_instances:
                category_results[combo_id] = {
                    **combo_data,
                    'instances': category_instances
                }

        if not category_results:
            print(f"  No results for category {category}")
            continue

        print("[1/9] Best Fitness Convergence")
        plot_convergence_best_fitness(category_results, category_dir, category)

        print("[2/9] Average Fitness Convergence")
        plot_convergence_avg_fitness(category_results, category_dir, category)

        print("[3/9] Number of Vehicles Convergence")
        plot_convergence_num_vehicles(category_results, category_dir, category)

        print("[4/9] Total Distance Convergence")
        plot_convergence_total_distance(category_results, category_dir, category)

        print("[5/9] Time to Best Solution")
        plot_time_to_best(category_results, category_dir, category)

        print("[6/9] Efficiency Score")
        plot_efficiency_score(category_results, category_dir, category)

        print("[7/9] Time-Quality Product")
        plot_time_quality_product(category_results, category_dir, category)

        print("[8/9] Combined Convergence")
        plot_combined_convergence(category_results, category_dir, category)

        print("[9/9] Combined Metrics")
        plot_combined_metrics_bars(category_results, category_dir, category)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nPlots saved to: {OUTPUT_BASE_DIR}/")
    print(f"  - Overall plots: {overall_dir}/")
    print(f"  - Individual category plots: {individual_dir}/")
    print(f"\nTotal plots generated: {9 + (9 * len(categories))}")
    print(f"  (9 overall + 9 per category × {len(categories)} categories)")
    print(f"  Each plot saved as PNG and PDF")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        create_all_plots()
        print("\nAll visualizations generated successfully!")
    except FileNotFoundError as e:
        print(f"\nError: Results folder not found or no combo files: {RESULTS_FOLDER}")
        print("Please run the experiment first to generate results.")
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
