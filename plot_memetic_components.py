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
from pathlib import Path
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_FILE = "results/memetic_component_wise/memetic_component_summary.csv"
OUTPUT_DIR = "results/memetic_component_comparison"

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

# Color scheme for combinations (using unified palette)
COMBO_COLORS = {
    'Baseline': METHOD_COLORS[0],       # Steel Blue
    'No_Mutation': METHOD_COLORS[1],    # Plum Purple
    'No_LocalSearch': METHOD_COLORS[2], # Teal
    'LS_Set2_OneShot': METHOD_COLORS[3],# Muted Gold
    'LS_Set2_Ranking': METHOD_COLORS[4],# Violet
    'LS_Set5_OneShot': METHOD_COLORS[5],# Soft Red
    'LS_Set5_Ranking': METHOD_COLORS[6],# Sage Green
}

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
# TABLE GENERATION
# ============================================================================

def create_comparison_table(agg_df, category_name):
    """Create formatted comparison table.

    Args:
        agg_df: Aggregated DataFrame
        category_name: Name of category (for title)

    Returns:
        matplotlib.figure.Figure: Table figure
    """
    fig, ax = plt.subplots(figsize=(14, len(agg_df) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []

    # Header row
    headers = ['Combination', 'Avg Fitness', 'Avg Time (s)', 'Avg Gap to BKS (%)']
    table_data.append(headers)

    # Data rows
    for _, row in agg_df.iterrows():
        table_data.append([
            row['Combination'],
            f"{row['Avg_Fitness']:.1f}",
            f"{row['Avg_Time']:.2f}",
            f"{row['Avg_Gap']:.2f}%"
        ])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.25])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 2.5)

    # Format header row
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Highlight best values in each column (green)
    best_fitness_idx = agg_df['Avg_Fitness'].idxmin()
    best_time_idx = agg_df['Avg_Time'].idxmin()
    best_gap_idx = agg_df['Avg_Gap'].idxmin()

    for idx, row in agg_df.iterrows():
        row_num = list(agg_df.index).index(idx) + 1  # +1 for header

        if idx == best_fitness_idx:
            table[(row_num, 1)].set_facecolor('#C6EFCE')  # Light green
        if idx == best_time_idx:
            table[(row_num, 2)].set_facecolor('#C6EFCE')
        if idx == best_gap_idx:
            table[(row_num, 3)].set_facecolor('#C6EFCE')

    # Title
    if category_name == 'overall':
        title = 'Component Comparison: Overall'
    else:
        title = f'Component Comparison: {category_name.upper()}'

    plt.title(title, fontsize=24, fontweight='bold', pad=20)

    return fig

# ============================================================================
# PLOT GENERATION
# ============================================================================

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
    colors = [COMBO_COLORS.get(c, 'gray') for c in combos]

    # Plot 1: Average Fitness
    ax = axes[0]
    bars = ax.bar(x, agg_df['Avg_Fitness'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Combination')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=14)

    # Plot 2: Average Time to Best
    ax = axes[1]
    bars = ax.bar(x, agg_df['Avg_Time'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Combination')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time to Best', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=14)

    # Plot 3: Average Gap to BKS
    ax = axes[2]
    bars = ax.bar(x, agg_df['Avg_Gap'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Combination')
    ax.set_ylabel('Gap to BKS (%)')
    ax.set_title('Gap to BKS', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=14)

    # Overall title
    if category_name == 'overall':
        suptitle = 'Component Comparison: Overall'
    else:
        suptitle = f'Component Comparison: {category_name.upper()}'

    fig.suptitle(suptitle, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    return fig

# ============================================================================
# RANKING GENERATION
# ============================================================================

def create_ranking_table(df):
    """Create summary ranking table showing winners in each category.

    Args:
        df: Processed DataFrame

    Returns:
        matplotlib.figure.Figure: Ranking table figure
    """
    categories = ['overall'] + sorted([c for c in df['Category'].unique() if c != 'unknown'])

    ranking_data = []

    for category in categories:
        agg_df = aggregate_by_category(df, None if category == 'overall' else category)

        if len(agg_df) == 0:
            continue

        best_fitness = agg_df.iloc[0]['Combination']  # Already sorted by fitness
        best_time = agg_df.loc[agg_df['Avg_Time'].idxmin(), 'Combination']
        best_gap = agg_df.loc[agg_df['Avg_Gap'].idxmin(), 'Combination']

        ranking_data.append({
            'Category': category.upper(),
            'Best Fitness': best_fitness,
            'Best Time': best_time,
            'Best Gap': best_gap
        })

    # Create figure
    fig, ax = plt.subplots(figsize=(16, len(ranking_data) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Category', 'Best Fitness', 'Fastest', 'Best Gap to BKS']
    table_data.append(headers)

    for row in ranking_data:
        table_data.append([
            row['Category'],
            row['Best Fitness'],
            row['Best Time'],
            row['Best Gap']
        ])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.2, 0.27, 0.27, 0.27])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 2.5)

    # Format header row
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Title
    plt.title('Component Comparison: Rankings', fontsize=24, fontweight='bold', pad=20)

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

    # Generate overall comparison
    print("\n" + "=" * 80)
    print("GENERATING OVERALL COMPARISON")
    print("=" * 80)

    agg_overall = aggregate_by_category(df, category=None)

    print("\nCreating overall comparison table...")
    fig_table = create_comparison_table(agg_overall, 'overall')
    save_pdf(fig_table, output_dir / 'overall_comparison_table.pdf')

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

        print(f"Creating {category} comparison table...")
        fig_table = create_comparison_table(agg_category, category)
        save_pdf(fig_table, output_dir / f'category_{category}_table.pdf')

        print(f"Creating {category} comparison plots...")
        fig_plots = create_comparison_plots(agg_category, category)
        save_pdf(fig_plots, output_dir / f'category_{category}_plots.pdf')

    # Generate summary rankings
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY RANKINGS")
    print("=" * 80)

    print("\nCreating summary ranking table...")
    fig_rankings = create_ranking_table(df)
    save_pdf(fig_rankings, output_dir / 'summary_rankings.pdf')

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}/")
    print(f"  - Overall: 2 PDFs (table + plots)")
    print(f"  - Categories: {len(categories)} categories × 2 PDFs each")
    print(f"  - Summary rankings: 1 PDF")
    print(f"\nTotal PDFs generated: {2 + len(categories) * 2 + 1}")

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
