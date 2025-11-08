"""
Visualization Script for Memetic Component-wise Performance Experiment

Generates comprehensive plots for analyzing memetic algorithm component performance:
1. Component Performance Comparison (gap to BKS, box plots, heatmaps)
2. Convergence Analysis (curves, time to best, stagnation)
3. Component-wise Impact (isolated effects of each component)

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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_FILE = "results/memetic_component_results.json"
OUTPUT_BASE_DIR = "results/plots"

# Plot styling
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Component names for better labels
COMPONENT_NAMES = {
    'selection': {
        0: 'k=2',
        1: 'k=3',
    },
    'crossover': {
        0: 'SREX',
        1: 'Dummy',
    },
    'mutation': {
        0: 'Naive(iter=1)',
        1: 'Naive(iter=10)',
        2: 'Dummy',
    },
    'local_search': {
        0: 'NaiveLS(FI)',
        1: 'NaiveLS(BI)',
        2: 'RandomLS',
        3: 'Dummy',
    }
}

# Color schemes
COMBO_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_instance_name(instance_name):
    """
    Parse instance name to extract category information.

    Args:
        instance_name: Name of the instance

    Returns:
        dict: Contains 'benchmark', 'category', 'instance_name'
    """
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

def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_gap_to_bks(fitness, bks_fitness):
    """Calculate percentage gap to best known solution."""
    if bks_fitness is None or bks_fitness == 0:
        return None
    return ((fitness - bks_fitness) / bks_fitness) * 100

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

def get_combination_label(combo_data):
    """Generate readable label for a combination."""
    sel = COMPONENT_NAMES['selection'][combo_data['selection_idx']]
    cross = COMPONENT_NAMES['crossover'][combo_data['crossover_idx']]
    mut = COMPONENT_NAMES['mutation'][combo_data['mutation_idx']]
    ls = COMPONENT_NAMES['local_search'][combo_data['local_search_idx']]
    return f"C{combo_data['combination_id'].split('_')[1]}"

def save_figure(fig, filepath, formats=['png', 'pdf']):
    """Save figure in multiple formats."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = Path(filepath).with_suffix(f'.{fmt}')
        fig.savefig(output_path, bbox_inches='tight', format=fmt)
        print(f"  Saved: {output_path}")

# ============================================================================
# PLOT TYPE 1: COMPONENT PERFORMANCE COMPARISON
# ============================================================================

def plot_gap_to_bks_bar(results, output_dir, category=None):
    """Bar chart showing average gap to BKS per combination."""
    category_label = category if category else "overall"
    print(f"  Creating gap to BKS bar chart ({category_label})...")

    # Collect data
    combo_gaps = defaultdict(list)

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            gap = calculate_gap_to_bks(
                instance_data['best_fitness'],
                instance_data['bks_fitness']
            )
            if gap is not None:
                combo_gaps[combo_id].append(gap)

    # Calculate averages
    combo_ids = sorted(combo_gaps.keys())
    avg_gaps = [np.mean(combo_gaps[cid]) if combo_gaps[cid] else 0 for cid in combo_ids]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(combo_ids))
    bars = ax.bar(x, avg_gaps, color=COMBO_COLORS[:len(combo_ids)], alpha=0.8, edgecolor='black')

    ax.set_xlabel('Combination')
    ax.set_ylabel('Average Gap to BKS (%)')
    title = f'Average Gap to BKS - {category.upper()}' if category else 'Average Gap to BKS - Overall'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

    # Save
    filename = f'gap_to_bks_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

def plot_fitness_boxplots(results, output_dir, category=None):
    """Box plots showing distribution of fitness values per combination."""
    category_label = category if category else "overall"
    print(f"  Creating fitness box plots ({category_label})...")

    # Collect data
    combo_fitness = defaultdict(list)

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            gap = calculate_gap_to_bks(
                instance_data['best_fitness'],
                instance_data['bks_fitness']
            )
            if gap is not None:
                combo_fitness[combo_id].append(gap)

    # Create plot
    combo_ids = sorted(combo_fitness.keys())
    data = [combo_fitness[cid] for cid in combo_ids]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=[f"C{cid.split('_')[1]}" for cid in combo_ids],
                     patch_artist=True, showmeans=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], COMBO_COLORS[:len(combo_ids)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('Combination')
    ax.set_ylabel('Gap to BKS (%)')
    title = f'Distribution of Gaps to BKS - {category.upper()}' if category else 'Distribution of Gaps to BKS - Overall'
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Save
    filename = f'fitness_boxplot_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

def plot_performance_heatmap(results, output_dir, category=None):
    """Heatmap showing performance of each combination on each instance."""
    category_label = category if category else "overall"
    print(f"  Creating performance heatmap ({category_label})...")

    # Collect data
    combo_instance_gaps = defaultdict(dict)
    all_instances = set()

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            gap = calculate_gap_to_bks(
                instance_data['best_fitness'],
                instance_data['bks_fitness']
            )
            if gap is not None:
                combo_instance_gaps[combo_id][instance_name] = gap
                all_instances.add(instance_name)

    # Build matrix
    combo_ids = sorted(combo_instance_gaps.keys())
    instance_names = sorted(all_instances)

    if not instance_names:
        print(f"    No instances found for category {category_label}")
        return

    matrix = np.zeros((len(combo_ids), len(instance_names)))
    for i, combo_id in enumerate(combo_ids):
        for j, instance_name in enumerate(instance_names):
            matrix[i, j] = combo_instance_gaps[combo_id].get(instance_name, np.nan)

    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, len(instance_names) * 0.3), max(4, len(combo_ids) * 0.5)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(len(instance_names)))
    ax.set_yticks(np.arange(len(combo_ids)))
    ax.set_xticklabels(instance_names, rotation=45, ha='right')
    ax.set_yticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids])

    # Labels
    ax.set_xlabel('Instance')
    ax.set_ylabel('Combination')
    title = f'Gap to BKS Heatmap - {category.upper()}' if category else 'Gap to BKS Heatmap - Overall'
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Gap to BKS (%)', rotation=270, labelpad=15)

    # Save
    filename = f'performance_heatmap_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

# ============================================================================
# PLOT TYPE 2: CONVERGENCE ANALYSIS
# ============================================================================

def plot_convergence_curves(results, output_dir, category=None):
    """Line plots showing convergence over time for each combination."""
    category_label = category if category else "overall"
    print(f"  Creating convergence curves ({category_label})...")

    # Collect convergence data
    combo_convergence = defaultdict(list)

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            if instance_data.get('convergence'):
                combo_convergence[combo_id].append(instance_data['convergence'])

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    combo_ids = sorted(combo_convergence.keys())
    for i, combo_id in enumerate(combo_ids):
        convergences = combo_convergence[combo_id]
        if not convergences:
            continue

        # Average convergence across instances
        # Find max length
        max_len = max(len(conv) for conv in convergences)

        # Interpolate/pad to same length and average
        times = []
        fitness_means = []
        fitness_stds = []

        # Collect all unique time points
        all_times = sorted(set(t for conv in convergences for t, _ in conv))

        for t in all_times:
            fitness_at_t = []
            for conv in convergences:
                # Find fitness at time t (use last known value if not exact)
                fitness = None
                for conv_time, conv_fitness in conv:
                    if conv_time <= t:
                        fitness = conv_fitness
                    else:
                        break
                if fitness is not None:
                    fitness_at_t.append(fitness)

            if fitness_at_t:
                times.append(t)
                fitness_means.append(np.mean(fitness_at_t))
                fitness_stds.append(np.std(fitness_at_t))

        if times:
            ax.plot(times, fitness_means, label=f"C{combo_id.split('_')[1]}",
                   color=COMBO_COLORS[i % len(COMBO_COLORS)], linewidth=2)
            ax.fill_between(times,
                          np.array(fitness_means) - np.array(fitness_stds),
                          np.array(fitness_means) + np.array(fitness_stds),
                          alpha=0.2, color=COMBO_COLORS[i % len(COMBO_COLORS)])

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Average Fitness')
    title = f'Convergence Curves - {category.upper()}' if category else 'Convergence Curves - Overall'
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    # Save
    filename = f'convergence_curves_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

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

    # Calculate averages
    combo_ids = sorted(combo_times.keys())
    avg_times = [np.mean(combo_times[cid]) if combo_times[cid] else 0 for cid in combo_ids]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(combo_ids))
    bars = ax.bar(x, avg_times, color=COMBO_COLORS[:len(combo_ids)], alpha=0.8, edgecolor='black')

    ax.set_xlabel('Combination')
    ax.set_ylabel('Average Time to Best (seconds)')
    title = f'Time to Best Solution - {category.upper()}' if category else 'Time to Best Solution - Overall'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)

    # Save
    filename = f'time_to_best_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

def plot_stagnation_boxplots(results, output_dir, category=None):
    """Box plots showing longest stagnation periods."""
    category_label = category if category else "overall"
    print(f"  Creating stagnation box plots ({category_label})...")

    # Collect data
    combo_stagnation = defaultdict(list)

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            if instance_data.get('final_evaluation'):
                stagnation = instance_data['final_evaluation'].get('longest_stagnation')
                if stagnation is not None:
                    combo_stagnation[combo_id].append(stagnation)

    # Create plot
    combo_ids = sorted(combo_stagnation.keys())
    data = [combo_stagnation[cid] if combo_stagnation[cid] else [0] for cid in combo_ids]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=[f"C{cid.split('_')[1]}" for cid in combo_ids],
                     patch_artist=True, showmeans=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], COMBO_COLORS[:len(combo_ids)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('Combination')
    ax.set_ylabel('Longest Stagnation (seconds)')
    title = f'Stagnation Periods - {category.upper()}' if category else 'Stagnation Periods - Overall'
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    # Save
    filename = f'stagnation_boxplot_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

# ============================================================================
# PLOT TYPE 3: COMPONENT-WISE IMPACT
# ============================================================================

def plot_component_impact(results, output_dir, category=None):
    """
    Grouped bar charts showing isolated effect of each component type.
    This compares average performance when varying one component while others are fixed.
    """
    category_label = category if category else "overall"
    print(f"  Creating component impact analysis ({category_label})...")

    # Collect gaps by component configuration
    component_gaps = {
        'selection': defaultdict(list),
        'crossover': defaultdict(list),
        'mutation': defaultdict(list),
        'local_search': defaultdict(list),
    }

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            if category:
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

            gap = calculate_gap_to_bks(
                instance_data['best_fitness'],
                instance_data['bks_fitness']
            )
            if gap is not None:
                component_gaps['selection'][combo_data['selection_idx']].append(gap)
                component_gaps['crossover'][combo_data['crossover_idx']].append(gap)
                component_gaps['mutation'][combo_data['mutation_idx']].append(gap)
                component_gaps['local_search'][combo_data['local_search_idx']].append(gap)

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Component-wise Impact - {category.upper()}' if category else 'Component-wise Impact - Overall',
                 fontsize=14, fontweight='bold')

    component_order = ['selection', 'crossover', 'mutation', 'local_search']
    titles = ['Selection Strategy', 'Crossover Operator', 'Mutation Operator', 'Local Search Operator']

    for idx, (component, title) in enumerate(zip(component_order, titles)):
        ax = axes[idx // 2, idx % 2]

        # Get data
        indices = sorted(component_gaps[component].keys())
        labels = [COMPONENT_NAMES[component][i] for i in indices]
        avg_gaps = [np.mean(component_gaps[component][i]) if component_gaps[component][i] else 0 for i in indices]

        # Plot
        x = np.arange(len(indices))
        bars = ax.bar(x, avg_gaps, color=COMBO_COLORS[:len(indices)], alpha=0.8, edgecolor='black')

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Average Gap to BKS (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7)

    plt.tight_layout()

    # Save
    filename = f'component_impact_{category_label}'
    save_figure(fig, output_dir / filename)
    plt.close()

# ============================================================================
# COMBINED CATEGORY PLOTS
# ============================================================================

def plot_all_categories_subplots(results, output_dir, plot_function, plot_name):
    """
    Create a subplot figure with all categories.

    Args:
        results: Full results dict
        output_dir: Output directory
        plot_function: Function to call for each category (must accept ax parameter)
        plot_name: Name for the output file
    """
    print(f"  Creating combined category subplots for {plot_name}...")

    # Get all categories
    categories = set()
    for combo_data in results.values():
        for instance_name in combo_data['instances'].keys():
            parsed = parse_instance_name(instance_name)
            categories.add(parsed['category'])

    categories = sorted(categories)

    # Create subplot grid
    n_cats = len(categories)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(15, 5 * n_rows))

    for idx, category in enumerate(categories):
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

        # Create the specific plot for this category
        if category_results:
            # Note: This requires modifying plot functions to accept an ax parameter
            # For now, we'll generate them separately
            pass

    # For now, just note that combined plots need separate implementation
    # We'll generate individual category plots instead

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def create_all_plots():
    """Main function to orchestrate all plot generation."""
    print("=" * 80)
    print("MEMETIC COMPONENT VISUALIZATION")
    print("=" * 80)

    # Load results
    print(f"\nLoading results from {RESULTS_FILE}...")
    results = load_results(RESULTS_FILE)
    print(f"Loaded {len(results)} combinations")

    # Organize by category
    by_category = organize_by_category(results)
    categories = sorted(by_category.keys())
    print(f"Found {len(categories)} categories: {', '.join(categories)}")

    # Create output directories
    overall_dir = Path(OUTPUT_BASE_DIR) / "overall"
    individual_dir = Path(OUTPUT_BASE_DIR) / "individual"
    by_category_dir = Path(OUTPUT_BASE_DIR) / "by_category"

    # ========================================================================
    # OVERALL PLOTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING OVERALL PLOTS")
    print("=" * 80)

    print("\n[1/3] Component Performance Comparison")
    plot_gap_to_bks_bar(results, overall_dir)
    plot_fitness_boxplots(results, overall_dir)
    plot_performance_heatmap(results, overall_dir)

    print("\n[2/3] Convergence Analysis")
    plot_convergence_curves(results, overall_dir)
    plot_time_to_best(results, overall_dir)
    plot_stagnation_boxplots(results, overall_dir)

    print("\n[3/3] Component-wise Impact")
    plot_component_impact(results, overall_dir)

    # ========================================================================
    # PER-CATEGORY PLOTS (INDIVIDUAL FILES)
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING PER-CATEGORY PLOTS (INDIVIDUAL FILES)")
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

        print("[1/3] Component Performance Comparison")
        plot_gap_to_bks_bar(category_results, category_dir, category)
        plot_fitness_boxplots(category_results, category_dir, category)
        plot_performance_heatmap(category_results, category_dir, category)

        print("[2/3] Convergence Analysis")
        plot_convergence_curves(category_results, category_dir, category)
        plot_time_to_best(category_results, category_dir, category)
        plot_stagnation_boxplots(category_results, category_dir, category)

        print("[3/3] Component-wise Impact")
        plot_component_impact(category_results, category_dir, category)

    # ========================================================================
    # COMBINED SUBPLOTS (ALL CATEGORIES)
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING COMBINED CATEGORY SUBPLOTS")
    print("=" * 80)

    # Create combined subplot for gap to BKS
    print("\n[1/3] Gap to BKS - All Categories")
    create_combined_gap_subplot(results, by_category_dir, categories)

    print("\n[2/3] Convergence - All Categories")
    create_combined_convergence_subplot(results, by_category_dir, categories)

    print("\n[3/3] Component Impact - All Categories")
    create_combined_component_subplot(results, by_category_dir, categories)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nPlots saved to: {OUTPUT_BASE_DIR}/")
    print(f"  - Overall plots: {overall_dir}/")
    print(f"  - Individual category plots: {individual_dir}/")
    print(f"  - Combined category subplots: {by_category_dir}/")

def create_combined_gap_subplot(results, output_dir, categories):
    """Create combined subplot showing gap to BKS for all categories."""
    n_cats = len(categories)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Average Gap to BKS - All Categories', fontsize=16, fontweight='bold')

    for idx, category in enumerate(categories):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Collect data for this category
        combo_gaps = defaultdict(list)
        for combo_id, combo_data in results.items():
            for instance_name, instance_data in combo_data['instances'].items():
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

                gap = calculate_gap_to_bks(
                    instance_data['best_fitness'],
                    instance_data['bks_fitness']
                )
                if gap is not None:
                    combo_gaps[combo_id].append(gap)

        # Plot
        combo_ids = sorted(combo_gaps.keys())
        avg_gaps = [np.mean(combo_gaps[cid]) if combo_gaps[cid] else 0 for cid in combo_ids]

        x = np.arange(len(combo_ids))
        ax.bar(x, avg_gaps, color=COMBO_COLORS[:len(combo_ids)], alpha=0.8, edgecolor='black')
        ax.set_title(category.upper(), fontweight='bold')
        ax.set_xlabel('Combination')
        ax.set_ylabel('Avg Gap (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{cid.split('_')[1]}" for cid in combo_ids])
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Hide unused subplots
    for idx in range(len(categories), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    save_figure(fig, output_dir / 'combined_gap_to_bks')
    plt.close()

def create_combined_convergence_subplot(results, output_dir, categories):
    """Create combined subplot showing convergence for all categories."""
    n_cats = len(categories)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Convergence Curves - All Categories', fontsize=16, fontweight='bold')

    for idx, category in enumerate(categories):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Collect convergence data for this category
        combo_convergence = defaultdict(list)
        for combo_id, combo_data in results.items():
            for instance_name, instance_data in combo_data['instances'].items():
                parsed = parse_instance_name(instance_name)
                if parsed['category'] != category:
                    continue

                if instance_data.get('convergence'):
                    combo_convergence[combo_id].append(instance_data['convergence'])

        # Plot
        combo_ids = sorted(combo_convergence.keys())
        for i, combo_id in enumerate(combo_ids):
            convergences = combo_convergence[combo_id]
            if not convergences:
                continue

            # Average convergence
            all_times = sorted(set(t for conv in convergences for t, _ in conv))
            times = []
            fitness_means = []

            for t in all_times:
                fitness_at_t = []
                for conv in convergences:
                    fitness = None
                    for conv_time, conv_fitness in conv:
                        if conv_time <= t:
                            fitness = conv_fitness
                        else:
                            break
                    if fitness is not None:
                        fitness_at_t.append(fitness)

                if fitness_at_t:
                    times.append(t)
                    fitness_means.append(np.mean(fitness_at_t))

            if times:
                ax.plot(times, fitness_means, label=f"C{combo_id.split('_')[1]}",
                       color=COMBO_COLORS[i % len(COMBO_COLORS)], linewidth=1.5)

        ax.set_title(category.upper(), fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Avg Fitness')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(categories), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    save_figure(fig, output_dir / 'combined_convergence')
    plt.close()

def create_combined_component_subplot(results, output_dir, categories):
    """Create combined subplot showing component impact for all categories."""
    # This would be very complex with 4 components x N categories
    # For now, create one combined plot per component type

    component_types = ['selection', 'crossover', 'mutation', 'local_search']
    titles = ['Selection Strategy', 'Crossover Operator', 'Mutation Operator', 'Local Search Operator']

    for component, title in zip(component_types, titles):
        print(f"    Creating {component} impact across categories...")

        n_cats = len(categories)
        n_cols = 3
        n_rows = (n_cats + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{title} Impact - All Categories', fontsize=16, fontweight='bold')

        for idx, category in enumerate(categories):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Collect data for this category and component
            component_gaps = defaultdict(list)
            for combo_id, combo_data in results.items():
                for instance_name, instance_data in combo_data['instances'].items():
                    parsed = parse_instance_name(instance_name)
                    if parsed['category'] != category:
                        continue

                    gap = calculate_gap_to_bks(
                        instance_data['best_fitness'],
                        instance_data['bks_fitness']
                    )
                    if gap is not None:
                        key = combo_data[f'{component}_idx']
                        component_gaps[key].append(gap)

            # Plot
            indices = sorted(component_gaps.keys())
            labels = [COMPONENT_NAMES[component][i] for i in indices]
            avg_gaps = [np.mean(component_gaps[i]) if component_gaps[i] else 0 for i in indices]

            x = np.arange(len(indices))
            ax.bar(x, avg_gaps, color=COMBO_COLORS[:len(indices)], alpha=0.8, edgecolor='black')
            ax.set_title(category.upper(), fontweight='bold')
            ax.set_xlabel('Config')
            ax.set_ylabel('Avg Gap (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Hide unused subplots
        for idx in range(len(categories), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        save_figure(fig, output_dir / f'combined_{component}_impact')
        plt.close()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        create_all_plots()
        print("\nAll visualizations generated successfully!")
    except FileNotFoundError as e:
        print(f"\nError: Results file not found: {RESULTS_FILE}")
        print("Please run the experiment first to generate results.")
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
