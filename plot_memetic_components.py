"""
Memetic Component Comparison Analysis Script

Generates comprehensive comparison reports (tables + plots in PDF format) for all
memetic algorithm combinations across different instance categories and overall performance.

Input: results/memetic_component_wise/memetic_component_summary.csv
Output: results/memetic_component_comparison/*.pdf

Core metrics compared:
- Average Best Fitness
- Average Time to Best Solution
- Average Gap to BKS (%)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_FILE = "results/memetic_component_summary_400.csv"
OUTPUT_DIR = "results/memetic_component_comparison_400"

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

BASELINE_COLORS = ['#5A5A5A', '#7A7A7A', '#9A9A9A', '#BABABA']


def compute_axis_limits_with_outlier_truncation(values, start_at_zero=False, outlier_threshold=2.0):
    """Compute y-axis limits, truncating outliers that are > threshold * next highest.

    Args:
        values: List of values to plot
        start_at_zero: Whether y-axis should start at 0
        outlier_threshold: Truncate bars > this multiple of next highest value

    Returns:
        tuple: (y_min, y_max, truncated_indices) where truncated_indices are bars to clip
    """
    valid_values = [v for v in values if v > 0]
    if len(valid_values) < 2:
        y_min = 0 if start_at_zero else min(values) * 0.9
        y_max = max(values) * 1.1
        return y_min, y_max, []

    sorted_values = sorted(valid_values, reverse=True)
    max_val = sorted_values[0]
    second_max = sorted_values[1]

    truncated_indices = []

    # Check if max is an outlier (more than threshold * second highest)
    if max_val > outlier_threshold * second_max:
        # Find all outlier values
        truncation_limit = second_max * 1.5  # Show up to 1.5x the second highest
        for i, v in enumerate(values):
            if v > truncation_limit:
                truncated_indices.append(i)

        if start_at_zero:
            y_min = 0
            y_max = truncation_limit  # Bars go right to the edge
        else:
            non_outlier = [values[i] for i in range(len(values)) if values[i] <= truncation_limit]
            if non_outlier:
                y_min = min(non_outlier) * 0.95
                y_max = truncation_limit
            else:
                y_min = 0
                y_max = truncation_limit
    else:
        # No outliers - normal scaling
        if start_at_zero:
            y_min = 0
            y_max = max(values) * 1.1
        else:
            y_min = min(values) * 0.95
            y_max = max(values) * 1.1

    return y_min, y_max, truncated_indices


# Color families for combination categories
# Each category gets a base color, with variations for different methods within

def lighten_color(color, factor=0.3):
    """Make a color lighter by mixing with white."""
    rgb = mcolors.to_rgb(color)
    return tuple(c + (1 - c) * factor for c in rgb)

def darken_color(color, factor=0.3):
    """Make a color darker by the given factor."""
    rgb = mcolors.to_rgb(color)
    return tuple(c * (1 - factor) for c in rgb)

# Base colors for each category
CATEGORY_BASE_COLORS = {
    'Baseline': '#2E86AB',   # Steel Blue family
    'LS_Set1': '#A23B72',    # Plum Purple family
    'LS_Set2': '#1D7874',    # Teal family
    'LS_Set3': '#E8A838',    # Muted Gold family
    'LS_Set4': '#6B4C9A',    # Violet family
    'LS_Set5': '#D64550',    # Soft Red family
    'LS_Set6': '#44AF69',    # Sage Green family
    'LS_Set7': '#8B5E3C',    # Brown family
    'No_Mutation': '#5A5A5A',    # Gray
}

def get_combination_color(combo_name, combo_index_in_category=0):
    """Get color for a combination based on its category.

    Args:
        combo_name: Name of the combination (e.g., 'Baseline_Full', 'LS_Set2_OneShot')
        combo_index_in_category: Index within the category for shade variation

    Returns:
        Color string or tuple
    """
    # Determine category from name
    if combo_name.startswith('Baseline'):
        base_color = CATEGORY_BASE_COLORS['Baseline']
    elif combo_name.startswith('No_Mutation'):
        base_color = CATEGORY_BASE_COLORS['No_Mutation']
    elif combo_name.startswith('LS_Set'):
        # Extract set number (e.g., 'LS_Set2_OneShot' -> 'LS_Set2')
        parts = combo_name.split('_')
        if len(parts) >= 2:
            set_key = f"{parts[0]}_{parts[1]}"  # 'LS_Set2'
            base_color = CATEGORY_BASE_COLORS.get(set_key, METHOD_COLORS[0])
        else:
            base_color = METHOD_COLORS[0]
    else:
        # Fallback
        base_color = METHOD_COLORS[combo_index_in_category % len(METHOD_COLORS)]

    # Apply shade variation based on index within category
    if combo_index_in_category == 0:
        return base_color
    elif combo_index_in_category == 1:
        return lighten_color(base_color, 0.25)
    elif combo_index_in_category == 2:
        return darken_color(base_color, 0.25)
    else:
        return lighten_color(base_color, 0.15 * combo_index_in_category)

def assign_colors_to_combinations(combos):
    """Assign colors to a list of combinations based on their categories.

    Args:
        combos: List of combination names

    Returns:
        List of colors corresponding to each combination
    """
    # Group combinations by category
    category_counts = {}
    colors = []

    for combo in combos:
        # Determine category
        if combo.startswith('Baseline'):
            category = 'Baseline'
        elif combo.startswith('No_Mutation'):
            category = 'No_Mutation'
        elif combo.startswith('LS_Set'):
            parts = combo.split('_')
            if len(parts) >= 2:
                category = f"{parts[0]}_{parts[1]}"
            else:
                category = 'Other'
        else:
            category = 'Other'

        # Get index within this category
        idx = category_counts.get(category, 0)
        category_counts[category] = idx + 1

        colors.append(get_combination_color(combo, idx))

    return colors

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_csv_data(csv_file):
    """Load results from CSV file.

    Args:
        csv_file: Path to CSV file

    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows")
    return df

def parse_instance_category(instance_name):
    """Parse instance name to extract category.

    Adapted from plot_memetic_components.py parse_instance_name().

    Args:
        instance_name: Name of the instance

    Returns:
        str: Category (lc1, lc2, lr1, lr2, lrc1, lrc2, bar, ber, nyc, poa)
    """
    # Extract just the filename if full path is provided
    if '\\' in instance_name or '/' in instance_name:
        instance_name = Path(instance_name).stem

    # Check if Mendeley instance (bar, ber, nyc, poa)
    mendeley_match = re.match(r'([a-z]+)-n(\d+)-(\d+)', instance_name)
    if mendeley_match:
        return mendeley_match.group(1)  # bar, ber, nyc, poa

    # Check if Li & Lim instance (size 100 format: lc101, lr205, etc.)
    lilim_100_match = re.match(r'(l[rc]{1,2}[12])(\d+)', instance_name)
    if lilim_100_match:
        return lilim_100_match.group(1)  # lc1, lc2, lr1, lr2, lrc1, lrc2

    # Check if Li & Lim instance (larger sizes format: LC1_2_1, etc.)
    lilim_large_match = re.match(r'(L[RC]{1,2}[12])_(\d+)_(\d+)', instance_name)
    if lilim_large_match:
        return lilim_large_match.group(1).lower()  # convert to lowercase

    return 'unknown'

def calculate_gap_to_bks(row):
    """Calculate percentage gap to best known solution.

    Args:
        row: DataFrame row with Best_Fitness and BKS_Fitness columns

    Returns:
        float: Gap percentage
    """
    if pd.isna(row['BKS_Fitness']) or row['BKS_Fitness'] == 0:
        return np.nan
    return ((row['Best_Fitness'] - row['BKS_Fitness']) / row['BKS_Fitness']) * 100

def process_data(df):
    """Process raw data: add category and gap columns.

    Args:
        df: Raw DataFrame from CSV

    Returns:
        pandas.DataFrame: Processed DataFrame with Category and Gap_to_BKS columns
    """
    print("\nProcessing data...")

    # Add category column
    df['Category'] = df['Instance'].apply(parse_instance_category)

    # Add gap to BKS column
    df['Gap_to_BKS'] = df.apply(calculate_gap_to_bks, axis=1)

    # Exclude No_LocalSearch combinations
    df = df[~df['Combination_Name'].str.startswith('No_LocalSearch')]

    print(f"  Found {df['Category'].nunique()} categories: {sorted(df['Category'].unique())}")
    print(f"  Found {df['Combination_Name'].nunique()} combinations: {sorted(df['Combination_Name'].unique())}")

    return df

def aggregate_by_category(df, category=None):
    """Aggregate metrics by combination for a specific category or overall.

    Args:
        df: Processed DataFrame
        category: Category to filter (None for overall)

    Returns:
        pandas.DataFrame: Aggregated results
    """
    # Filter by category if specified
    if category:
        df_filtered = df[df['Category'] == category].copy()
    else:
        df_filtered = df.copy()

    # Group by combination and calculate means
    agg_df = df_filtered.groupby('Combination_Name').agg({
        'Best_Fitness': 'mean',
        'Time_To_Best': 'mean',
        'Gap_to_BKS': 'mean'
    }).reset_index()

    # Rename columns for clarity
    agg_df.columns = ['Combination', 'Avg_Fitness', 'Avg_Time', 'Avg_Gap']

    # Sort by average fitness (best first)
    agg_df = agg_df.sort_values('Avg_Fitness')

    return agg_df

# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def escape_latex(text):
    """Escape special LaTeX characters in text.

    Args:
        text: String to escape

    Returns:
        str: Escaped string safe for LaTeX
    """
    replacements = {
        '_': r'\_',
        '%': r'\%',
        '&': r'\&',
        '#': r'\#',
        '$': r'\$',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def generate_latex_comparison_table(agg_df, category_name):
    """Generate LaTeX code for a comparison table.

    Args:
        agg_df: Aggregated DataFrame with columns: Combination, Avg_Fitness, Avg_Time, Avg_Gap
        category_name: Name of category (for caption/label)

    Returns:
        str: LaTeX table code
    """
    # Find best values for highlighting
    best_fitness_idx = agg_df['Avg_Fitness'].idxmin()
    best_time_idx = agg_df['Avg_Time'].idxmin()
    best_gap_idx = agg_df['Avg_Gap'].idxmin()

    # Build table
    if category_name == 'overall':
        caption = 'Component Comparison: Overall'
        label = 'tab:comparison_overall'
    else:
        caption = f'Component Comparison: {category_name.upper()}'
        label = f'tab:comparison_{category_name}'

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\begin{tabular}{lrrr}',
        r'\toprule',
        r'Combination & Avg Fitness & Time to Best Solution (s) & Avg Gap to BKS (\%) \\',
        r'\midrule',
    ]

    for idx, row in agg_df.iterrows():
        combo_escaped = escape_latex(row['Combination'])
        fitness_val = f"{row['Avg_Fitness']:.1f}"
        time_val = f"{row['Avg_Time']:.2f}"
        gap_val = f"{row['Avg_Gap']:.2f}"

        # Apply cell highlighting for best values
        if idx == best_fitness_idx:
            fitness_val = r'\cellcolor{green!20}' + fitness_val
        if idx == best_time_idx:
            time_val = r'\cellcolor{green!20}' + time_val
        if idx == best_gap_idx:
            gap_val = r'\cellcolor{green!20}' + gap_val

        lines.append(f'{combo_escaped} & {fitness_val} & {time_val} & {gap_val} \\\\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def generate_latex_ranking_table(df):
    """Generate LaTeX code for the summary ranking table showing winners by category.

    Args:
        df: Processed DataFrame

    Returns:
        str: LaTeX table code
    """
    categories = ['overall'] + sorted([c for c in df['Category'].unique() if c != 'unknown'])

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Summary Rankings: Winners by Category}',
        r'\label{tab:summary_rankings}',
        r'\begin{tabular}{llll}',
        r'\toprule',
        r'Category & Best Fitness & Fastest & Best Gap to BKS \\',
        r'\midrule',
    ]

    for category in categories:
        agg_df = aggregate_by_category(df, None if category == 'overall' else category)

        if len(agg_df) == 0:
            continue

        best_fitness = agg_df.iloc[0]['Combination']  # Already sorted by fitness
        best_time = agg_df.loc[agg_df['Avg_Time'].idxmin(), 'Combination']
        best_gap = agg_df.loc[agg_df['Avg_Gap'].idxmin(), 'Combination']

        cat_display = category.upper()
        lines.append(
            f'{cat_display} & {escape_latex(best_fitness)} & {escape_latex(best_time)} & {escape_latex(best_gap)} \\\\'
        )

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def generate_latex_rank_summary_table(df):
    """Generate LaTeX table showing average rank of each combination across categories.

    Args:
        df: Processed DataFrame

    Returns:
        str: LaTeX table code
    """
    categories = sorted([c for c in df['Category'].unique() if c != 'unknown'])
    combinations = df['Combination_Name'].unique()

    # Calculate ranks for each combination in each category
    rank_data = {combo: {'fitness': [], 'time': [], 'gap': []} for combo in combinations}

    for category in categories:
        agg_df = aggregate_by_category(df, category)
        if len(agg_df) == 0:
            continue

        # Rank by each metric (1 = best)
        agg_df['Rank_Fitness'] = agg_df['Avg_Fitness'].rank(method='min')
        agg_df['Rank_Time'] = agg_df['Avg_Time'].rank(method='min')
        agg_df['Rank_Gap'] = agg_df['Avg_Gap'].rank(method='min')

        for _, row in agg_df.iterrows():
            combo = row['Combination']
            rank_data[combo]['fitness'].append(row['Rank_Fitness'])
            rank_data[combo]['time'].append(row['Rank_Time'])
            rank_data[combo]['gap'].append(row['Rank_Gap'])

    # Calculate average ranks
    summary_data = []
    for combo in combinations:
        if rank_data[combo]['fitness']:  # Has data
            avg_fitness = np.mean(rank_data[combo]['fitness'])
            avg_time = np.mean(rank_data[combo]['time'])
            avg_gap = np.mean(rank_data[combo]['gap'])
            overall_avg = (avg_fitness + avg_time + avg_gap) / 3
            summary_data.append({
                'Combination': combo,
                'Avg_Rank_Fitness': avg_fitness,
                'Avg_Rank_Time': avg_time,
                'Avg_Rank_Gap': avg_gap,
                'Overall_Avg_Rank': overall_avg
            })

    # Sort by overall average rank
    summary_data.sort(key=lambda x: x['Overall_Avg_Rank'])

    # Find best (lowest) values for highlighting
    best_fitness_rank = min(d['Avg_Rank_Fitness'] for d in summary_data)
    best_time_rank = min(d['Avg_Rank_Time'] for d in summary_data)
    best_gap_rank = min(d['Avg_Rank_Gap'] for d in summary_data)
    best_overall_rank = min(d['Overall_Avg_Rank'] for d in summary_data)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Average Rank Summary (1 = Best)}',
        r'\label{tab:avg_rank_summary}',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Combination & Avg Rank (Fitness) & Avg Rank (Time) & Avg Rank (Gap) & Overall Avg Rank \\',
        r'\midrule',
    ]

    for row in summary_data:
        combo_escaped = escape_latex(row['Combination'])
        fitness_val = f"{row['Avg_Rank_Fitness']:.2f}"
        time_val = f"{row['Avg_Rank_Time']:.2f}"
        gap_val = f"{row['Avg_Rank_Gap']:.2f}"
        overall_val = f"{row['Overall_Avg_Rank']:.2f}"

        # Highlight best values
        if row['Avg_Rank_Fitness'] == best_fitness_rank:
            fitness_val = r'\cellcolor{green!20}' + fitness_val
        if row['Avg_Rank_Time'] == best_time_rank:
            time_val = r'\cellcolor{green!20}' + time_val
        if row['Avg_Rank_Gap'] == best_gap_rank:
            gap_val = r'\cellcolor{green!20}' + gap_val
        if row['Overall_Avg_Rank'] == best_overall_rank:
            overall_val = r'\cellcolor{green!20}' + overall_val

        lines.append(f'{combo_escaped} & {fitness_val} & {time_val} & {gap_val} & {overall_val} \\\\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def generate_latex_win_count_table(df):
    """Generate LaTeX table showing win counts per combination.

    Args:
        df: Processed DataFrame

    Returns:
        str: LaTeX table code
    """
    categories = sorted([c for c in df['Category'].unique() if c != 'unknown'])
    combinations = df['Combination_Name'].unique()

    # Count wins for each combination
    win_counts = {combo: {'fitness': 0, 'time': 0, 'gap': 0} for combo in combinations}

    for category in categories:
        agg_df = aggregate_by_category(df, category)
        if len(agg_df) == 0:
            continue

        # Find winners
        best_fitness = agg_df.loc[agg_df['Avg_Fitness'].idxmin(), 'Combination']
        best_time = agg_df.loc[agg_df['Avg_Time'].idxmin(), 'Combination']
        best_gap = agg_df.loc[agg_df['Avg_Gap'].idxmin(), 'Combination']

        win_counts[best_fitness]['fitness'] += 1
        win_counts[best_time]['time'] += 1
        win_counts[best_gap]['gap'] += 1

    # Build summary data
    summary_data = []
    for combo in combinations:
        total = win_counts[combo]['fitness'] + win_counts[combo]['time'] + win_counts[combo]['gap']
        summary_data.append({
            'Combination': combo,
            'Fitness_Wins': win_counts[combo]['fitness'],
            'Time_Wins': win_counts[combo]['time'],
            'Gap_Wins': win_counts[combo]['gap'],
            'Total_Wins': total
        })

    # Sort by total wins (descending)
    summary_data.sort(key=lambda x: x['Total_Wins'], reverse=True)

    # Find best (highest) values for highlighting
    max_fitness_wins = max(d['Fitness_Wins'] for d in summary_data)
    max_time_wins = max(d['Time_Wins'] for d in summary_data)
    max_gap_wins = max(d['Gap_Wins'] for d in summary_data)
    max_total_wins = max(d['Total_Wins'] for d in summary_data)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Win Count Summary (across all categories)}',
        r'\label{tab:win_count_summary}',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Combination & Fitness Wins & Time Wins & Gap Wins & Total Wins \\',
        r'\midrule',
    ]

    for row in summary_data:
        combo_escaped = escape_latex(row['Combination'])
        fitness_val = str(row['Fitness_Wins'])
        time_val = str(row['Time_Wins'])
        gap_val = str(row['Gap_Wins'])
        total_val = str(row['Total_Wins'])

        # Highlight best values (only if > 0)
        if row['Fitness_Wins'] == max_fitness_wins and max_fitness_wins > 0:
            fitness_val = r'\cellcolor{green!20}' + fitness_val
        if row['Time_Wins'] == max_time_wins and max_time_wins > 0:
            time_val = r'\cellcolor{green!20}' + time_val
        if row['Gap_Wins'] == max_gap_wins and max_gap_wins > 0:
            gap_val = r'\cellcolor{green!20}' + gap_val
        if row['Total_Wins'] == max_total_wins and max_total_wins > 0:
            total_val = r'\cellcolor{green!20}' + total_val

        lines.append(f'{combo_escaped} & {fitness_val} & {time_val} & {gap_val} & {total_val} \\\\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def save_latex_file(content, filepath):
    """Write LaTeX content to a file with proper preamble.

    Args:
        content: LaTeX table content (without document preamble)
        filepath: Output file path
    """
    preamble = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{longtable}

\begin{document}

"""
    postamble = r"""
\end{document}
"""

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(preamble)
        f.write(content)
        f.write(postamble)
    print(f"  Saved: {filepath}")

# ============================================================================
# PLOT GENERATION
# ============================================================================

def add_bar_labels(ax, bars, values, y_min, y_max, truncated_indices, fmt='.1f', suffix=''):
    """Add value labels to bars, handling truncation and positioning.

    Args:
        ax: Matplotlib axes
        bars: Bar container from ax.bar()
        values: List of values
        y_min: Y-axis minimum
        y_max: Y-axis maximum
        truncated_indices: List of bar indices that are truncated
        fmt: Format string for values (e.g., '.1f', '.2f')
        suffix: Suffix to add (e.g., '%')
    """
    y_range = y_max - y_min
    top_threshold = y_max - 0.08 * y_range  # If bar is within 8% of top, put label inside

    for i, bar in enumerate(bars):
        # Skip truncated bars - no label
        if i in truncated_indices:
            continue

        height = bar.get_height()
        label = f'{height:{fmt}}{suffix}'

        # Determine if label should be inside or outside the bar
        if height > top_threshold:
            # Label inside the bar (near top)
            y_pos = height - 0.02 * y_range
            va = 'top'
            color = 'white'
        else:
            # Label above the bar
            y_pos = height + 0.01 * y_range
            va = 'bottom'
            color = 'black'

        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                label, ha='center', va=va, fontsize=14, fontweight='bold',
                color=color, rotation=45)


def create_comparison_plots(agg_df, category_name):
    """Create comparison bar charts.

    Args:
        agg_df: Aggregated DataFrame
        category_name: Name of category (for title)

    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Prepare data
    combos = agg_df['Combination'].tolist()
    x = np.arange(len(combos))

    # Assign colors based on category
    colors = assign_colors_to_combinations(combos)

    # Plot 1: Average Fitness
    ax = axes[0]
    values = agg_df['Avg_Fitness'].tolist()
    y_min, y_max, truncated = compute_axis_limits_with_outlier_truncation(values, start_at_zero=False)
    y_max = y_max + (y_max - y_min) * 0.15  # Extra top padding for labels
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Combination')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=y_min, top=y_max)
    add_bar_labels(ax, bars, values, y_min, y_max, truncated, fmt='.1f')

    # Plot 2: Average Time to Best
    ax = axes[1]
    values = agg_df['Avg_Time'].tolist()
    y_min, y_max, truncated = compute_axis_limits_with_outlier_truncation(values, start_at_zero=True)
    y_max = y_max * 1.15  # Extra top padding for labels
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Combination')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time to Best', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=y_min, top=y_max)
    add_bar_labels(ax, bars, values, y_min, y_max, truncated, fmt='.1f')

    # Plot 3: Average Gap to BKS
    ax = axes[2]
    values = agg_df['Avg_Gap'].tolist()
    y_min, y_max, truncated = compute_axis_limits_with_outlier_truncation(values, start_at_zero=False)
    y_max = y_max + (y_max - y_min) * 0.15  # Extra top padding for labels
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Combination')
    ax.set_ylabel('Gap to BKS (%)')
    ax.set_title('Gap to BKS', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=y_min, top=y_max)
    add_bar_labels(ax, bars, values, y_min, y_max, truncated, fmt='.2f', suffix='%')

    # Overall title
    if category_name == 'overall':
        suptitle = 'Component Comparison: Overall'
    else:
        suptitle = f'Component Comparison: {category_name.upper()}'

    fig.suptitle(suptitle, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    return fig


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def save_pdf(fig, filepath):
    """Save figure as PDF.

    Args:
        fig: Matplotlib figure
        filepath: Output path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, bbox_inches='tight', format='pdf', backend='pdf')
    print(f"  Saved: {filepath}")
    plt.close(fig)

def create_comparison_report():
    """Main function to create all comparison reports."""
    print("=" * 80)
    print("MEMETIC COMPONENT COMPARISON ANALYSIS")
    print("=" * 80)

    # Load and process data
    df = load_csv_data(CSV_FILE)
    df = process_data(df)

    output_dir = Path(OUTPUT_DIR)

    # Collect LaTeX tables
    category_tables = []
    overview_tables = []

    # Generate overall comparison
    print("\n" + "=" * 80)
    print("GENERATING OVERALL COMPARISON")
    print("=" * 80)

    agg_overall = aggregate_by_category(df, category=None)

    print("\nGenerating overall comparison table (LaTeX)...")
    overview_tables.append(generate_latex_comparison_table(agg_overall, 'overall'))

    print("Creating overall comparison plots...")
    fig_plots = create_comparison_plots(agg_overall, 'overall')
    save_pdf(fig_plots, output_dir / 'overall_comparison_plots.pdf')

    # Generate per-category comparisons
    print("\n" + "=" * 80)
    print("GENERATING PER-CATEGORY COMPARISONS")
    print("=" * 80)

    categories = sorted([c for c in df['Category'].unique() if c != 'unknown'])

    for category in categories:
        print(f"\n--- Category: {category.upper()} ---")

        agg_category = aggregate_by_category(df, category)

        if len(agg_category) == 0:
            print(f"  No data for category {category}")
            continue

        print(f"Generating {category} comparison table (LaTeX)...")
        category_tables.append(generate_latex_comparison_table(agg_category, category))

        print(f"Creating {category} comparison plots...")
        fig_plots = create_comparison_plots(agg_category, category)
        save_pdf(fig_plots, output_dir / f'category_{category}_plots.pdf')

    # Generate summary/overview tables
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY TABLES")
    print("=" * 80)

    print("\nGenerating summary ranking table (LaTeX)...")
    overview_tables.append(generate_latex_ranking_table(df))

    print("Generating average rank summary table (LaTeX)...")
    overview_tables.append(generate_latex_rank_summary_table(df))

    print("Generating win count table (LaTeX)...")
    overview_tables.append(generate_latex_win_count_table(df))

    # Save LaTeX files
    print("\n" + "=" * 80)
    print("SAVING LATEX FILES")
    print("=" * 80)

    print("\nSaving category tables...")
    category_content = '\n\n'.join(category_tables)
    save_latex_file(category_content, output_dir / 'category_tables.tex')

    print("Saving overview tables...")
    overview_content = '\n\n'.join(overview_tables)
    save_latex_file(overview_content, output_dir / 'overview_tables.tex')

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}/")
    print(f"  - category_tables.tex: {len(categories)} category comparison tables")
    print(f"  - overview_tables.tex: 4 overview tables (overall comparison, rankings, avg rank, win count)")
    print(f"  - Plot PDFs: {1 + len(categories)} files (overall + per-category)")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        create_comparison_report()
        print("\nAll comparison reports generated successfully!")
    except FileNotFoundError as e:
        print(f"\nError: CSV file not found: {CSV_FILE}")
        print("Please ensure the experiment has been run and the summary CSV exists.")
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()
