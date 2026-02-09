"""
Visualization for State Archetype Action Distribution Experiment.

Reads the CSV and JSON outputs from experiment_state_archetype_action_distributions.py
and creates thesis-quality plots showing how the RL agent adapts its operator
selection based on solution state archetypes.

Plots:
  1. Grouped bar chart: mean softmax probabilities per archetype
  2. Grouped bar chart: empirical action fractions per archetype
  3. Radar/spider chart: probability profiles per archetype
  4. Heatmap: operators x archetypes (mean probability)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List

# ============================================================================
# PLOT STYLE (matching existing thesis plots)
# ============================================================================

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
    'figure.constrained_layout.use': False,
    'axes.labelpad': 12,
    'xtick.major.pad': 10,
    'ytick.major.pad': 10,
    'lines.linewidth': 3.0,
    'axes.linewidth': 1.5,
}
plt.rcParams.update(PLOT_STYLE)

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = "results/state_archetype_distributions"
OUTPUT_DIR = "results/plots/state_archetype_distributions"

# Archetype colors
ARCHETYPE_COLORS = {
    'Infeasible': '#D64550',              # Soft Red
    'Feasible, High Distance': '#E8A838', # Muted Gold
    'Feasible, Compact': '#2E86AB',       # Steel Blue
    'Many Routes': '#A23B72',             # Plum Purple
    'Few Routes': '#1D7874',              # Teal
    'Imbalanced Routes': '#6B4C9A',       # Violet
    'Balanced Routes': '#44AF69',         # Sage Green
}

# Preferred ordering per grouping
GROUPING_ORDER = {
    'feasibility': ['Infeasible', 'Feasible, High Distance', 'Feasible, Compact'],
    'route_count': ['Many Routes', 'Few Routes'],
    'route_balance': ['Imbalanced Routes', 'Balanced Routes'],
}

GROUPING_TITLES = {
    'feasibility': 'Feasibility & Distance',
    'route_count': 'Route Count',
    'route_balance': 'Route Balance',
}

# Human-readable operator display names (raw name -> short label)
OPERATOR_DISPLAY_NAMES = {
    'Reinsert-nC-Max1-NewV-SameV-nF_SameV':       'Reinsert',
    'Reinsert-C-Max5-NewV-SameV-nF_SameV':         'Reinsert(C,k5)',
    'Reinsert-nC-Max1-NewV-NoSameV-nF_SameV':      'Reinsert(NoSameV)',
    'Reinsert-nC-Max1-NoNewV-NoSameV-nF_SameV':    'Reinsert(NoNewV,NoSameV)',
    'RouteElimination':                             'RouteElim',
    'TwoOpt':                                       'TwoOpt',
    'SwapBetween-best':                             'SwapBetween',
    'Merge-min-N2-F-R':                             'Merge',
    'CLS-M2':                                       'CLS-M2',
    'CLS-M3':                                       'CLS-M3',
    'CLS-M4':                                       'CLS-M4',
}


def get_display_name(raw_name):
    """Get short display name for an operator."""
    return OPERATOR_DISPLAY_NAMES.get(raw_name, raw_name)


def darken_color(color, factor=0.6):
    """Make a color darker by the given factor."""
    rgb = mcolors.to_rgb(color)
    return tuple(c * factor for c in rgb)


def save_figure_multi_format(fig, filepath, formats=('png', 'pdf')):
    """Save figure in multiple formats."""
    for fmt in formats:
        output_path = f"{filepath}.{fmt}"
        fig.savefig(output_path, format=fmt, bbox_inches='tight', dpi=300, pad_inches=0.3)
        print(f'    Saved: {output_path}')


def abbreviate_operator_name(name, max_length=20):
    """Shorten long operator names for axis labels."""
    if len(name) <= max_length:
        return name
    parts = name.split('-')
    if len(parts) > 3:
        abbreviated = list(parts[:2])
        for part in parts[2:]:
            short = ''.join(c for c in part if c.isupper() or c.isdigit())
            abbreviated.append(short if short else part[:3])
        return '-'.join(abbreviated)
    return name[:max_length - 3] + '...'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv(results_dir, grouping_name=""):
    """Load the probability table CSV for a given grouping."""
    suffix = f"_{grouping_name}" if grouping_name else ""
    csv_path = Path(results_dir) / f"state_archetype_probability_table{suffix}.csv"
    return pd.read_csv(csv_path)


def discover_groupings(results_dir):
    """Find all grouping CSVs in the results directory."""
    groupings = []
    for path in Path(results_dir).glob("state_archetype_probability_table_*.csv"):
        # Extract grouping name from filename
        name = path.stem.replace("state_archetype_probability_table_", "")
        groupings.append(name)
    if not groupings:
        # Fall back to legacy single-file format
        legacy = Path(results_dir) / "state_archetype_probability_table.csv"
        if legacy.exists():
            groupings.append("")
    return sorted(groupings)


def load_json(results_dir):
    """Load the full results JSON. Returns None if missing or corrupt."""
    json_path = Path(results_dir) / "state_archetype_full_results.json"
    if not json_path.exists():
        print("  WARNING: JSON results file not found, skipping.")
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON results file is corrupt ({e}), skipping.")
        return None


def get_archetypes_from_csv(df):
    """Extract archetype names from CSV columns."""
    archetypes = []
    for col in df.columns:
        if col.endswith('_mean_prob'):
            name = col.replace('_mean_prob', '')
            archetypes.append(name)
    return archetypes


# ============================================================================
# PLOT 1: GROUPED BAR — MEAN SOFTMAX PROBABILITIES
# ============================================================================

def plot_mean_probabilities(df, archetypes, output_dir, grouping_name=""):
    """Grouped bar chart of mean softmax action probabilities per archetype."""
    print("  Creating mean probability bar chart...")

    operators = df['Operator'].tolist()
    short_names = [get_display_name(op) for op in operators]
    n_ops = len(operators)
    n_arch = len(archetypes)

    fig, ax = plt.subplots(figsize=(max(16, n_ops * 1.5), 10))

    x = np.arange(n_ops)
    total_width = 0.75
    bar_width = total_width / n_arch

    for arch_idx, archetype in enumerate(archetypes):
        means = df[f'{archetype}_mean_prob'].values
        stds = df[f'{archetype}_std_prob'].values
        offset = (arch_idx - (n_arch - 1) / 2) * bar_width
        color = ARCHETYPE_COLORS.get(archetype, plt.cm.tab10(arch_idx))

        bars = ax.bar(
            x + offset, means, bar_width,
            label=archetype,
            color=color, alpha=0.85,
            edgecolor=darken_color(color), linewidth=1.0,
            yerr=stds, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': 'gray'},
        )

    title = GROUPING_TITLES.get(grouping_name, grouping_name or 'All')
    ax.set_xlabel('Operator')
    ax.set_ylabel('Mean Action Probability')
    ax.set_title(f'Softmax Probabilities: {title}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend(framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

    plt.tight_layout(pad=1.5)
    suffix = f"_{grouping_name}" if grouping_name else ""
    save_figure_multi_format(fig, os.path.join(output_dir, f'archetype_mean_probabilities{suffix}'))
    plt.close()


# ============================================================================
# PLOT 2: GROUPED BAR — EMPIRICAL ACTION FRACTIONS
# ============================================================================

def plot_empirical_fractions(df, archetypes, output_dir, grouping_name=""):
    """Grouped bar chart of empirical action fractions per archetype."""
    print("  Creating empirical fractions bar chart...")

    operators = df['Operator'].tolist()
    short_names = [get_display_name(op) for op in operators]
    n_ops = len(operators)
    n_arch = len(archetypes)

    fig, ax = plt.subplots(figsize=(max(16, n_ops * 1.5), 10))

    x = np.arange(n_ops)
    total_width = 0.75
    bar_width = total_width / n_arch

    for arch_idx, archetype in enumerate(archetypes):
        fracs = df[f'{archetype}_empirical_frac'].values
        offset = (arch_idx - (n_arch - 1) / 2) * bar_width
        color = ARCHETYPE_COLORS.get(archetype, plt.cm.tab10(arch_idx))

        ax.bar(
            x + offset, fracs, bar_width,
            label=archetype,
            color=color, alpha=0.85,
            edgecolor=darken_color(color), linewidth=1.0,
        )

    title = GROUPING_TITLES.get(grouping_name, grouping_name or 'All')
    ax.set_xlabel('Operator')
    ax.set_ylabel('Empirical Selection Fraction')
    ax.set_title(f'Empirical Operator Selection: {title}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend(framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

    plt.tight_layout(pad=1.5)
    suffix = f"_{grouping_name}" if grouping_name else ""
    save_figure_multi_format(fig, os.path.join(output_dir, f'archetype_empirical_fractions{suffix}'))
    plt.close()


# ============================================================================
# PLOT 3: RADAR / SPIDER CHART
# ============================================================================

def plot_radar_chart(df, archetypes, output_dir, grouping_name=""):
    """Radar chart showing probability profile per archetype."""
    print("  Creating radar chart...")

    operators = df['Operator'].tolist()
    short_names = [get_display_name(op) for op in operators]
    n_ops = len(operators)

    # Compute angles for radar
    angles = np.linspace(0, 2 * np.pi, n_ops, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    for archetype in archetypes:
        values = df[f'{archetype}_mean_prob'].values.tolist()
        values += values[:1]  # close polygon
        color = ARCHETYPE_COLORS.get(archetype, 'gray')

        ax.plot(angles, values, linewidth=2.5, label=archetype, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    title = GROUPING_TITLES.get(grouping_name, grouping_name or 'All')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_names, fontsize=14)
    ax.set_title(f'Probability Profiles: {title}',
                 fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
              framealpha=0.9, edgecolor='gray')

    plt.tight_layout(pad=2.0)
    suffix = f"_{grouping_name}" if grouping_name else ""
    save_figure_multi_format(fig, os.path.join(output_dir, f'archetype_radar{suffix}'))
    plt.close()


# ============================================================================
# PLOT 4: HEATMAP — OPERATORS x ARCHETYPES
# ============================================================================

def plot_probability_heatmap(df, archetypes, output_dir, grouping_name=""):
    """Heatmap of mean probabilities: operators (rows) x archetypes (columns)."""
    print("  Creating probability heatmap...")

    operators = df['Operator'].tolist()
    short_names = [get_display_name(op) for op in operators]
    n_ops = len(operators)
    n_arch = len(archetypes)

    # Build matrix
    matrix = np.zeros((n_ops, n_arch))
    for j, archetype in enumerate(archetypes):
        matrix[:, j] = df[f'{archetype}_mean_prob'].values

    fig, ax = plt.subplots(figsize=(max(12, n_arch * 4), max(10, n_ops * 0.8)))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # X-axis: archetypes at bottom, rotated for readability
    ax.set_xticks(np.arange(n_arch))
    ax.set_xticklabels(archetypes, fontsize=18, rotation=30, ha='right',
                       rotation_mode='anchor')

    # Y-axis: operator display names
    ax.set_yticks(np.arange(n_ops))
    ax.set_yticklabels(short_names, fontsize=18)

    # Annotate cells with values
    for i in range(n_ops):
        for j in range(n_arch):
            val = matrix[i, j]
            text_color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=16, color=text_color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean Action Probability', rotation=270, labelpad=20)

    title = GROUPING_TITLES.get(grouping_name, grouping_name or 'All')
    ax.set_xlabel('State Archetype')
    ax.set_ylabel('Operator')
    ax.set_title(f'Probability Heatmap: {title}', fontweight='bold')

    plt.tight_layout(pad=2.0)
    suffix = f"_{grouping_name}" if grouping_name else ""
    save_figure_multi_format(fig, os.path.join(output_dir, f'archetype_probability_heatmap{suffix}'))
    plt.close()


# ============================================================================
# PLOT 5: PROBABILITY DIFFERENCE — DEVIATION FROM UNIFORM/OVERALL MEAN
# ============================================================================

def plot_probability_difference(df, archetypes, output_dir, grouping_name=""):
    """Horizontal bar chart showing per-operator probability difference from overall mean."""
    print("  Creating probability difference chart...")

    operators = df['Operator'].tolist()
    short_names = [get_display_name(op) for op in operators]
    n_ops = len(operators)
    n_arch = len(archetypes)

    # Compute overall mean probability per operator (across archetypes)
    overall_mean = np.zeros(n_ops)
    for archetype in archetypes:
        overall_mean += df[f'{archetype}_mean_prob'].values
    overall_mean /= n_arch

    fig, axes = plt.subplots(1, n_arch, figsize=(7 * n_arch, max(10, n_ops * 0.6)),
                             sharey=True)
    if n_arch == 1:
        axes = [axes]

    for arch_idx, archetype in enumerate(archetypes):
        ax = axes[arch_idx]
        diffs = df[f'{archetype}_mean_prob'].values - overall_mean
        color = ARCHETYPE_COLORS.get(archetype, 'gray')

        y = np.arange(n_ops)
        colors = [color if d >= 0 else darken_color(color, 0.5) for d in diffs]
        ax.barh(y, diffs, color=colors, alpha=0.85,
                edgecolor=[darken_color(c) for c in colors], linewidth=1.0)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Probability Difference from Mean')
        ax.set_title(archetype, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')

        if arch_idx == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(short_names)

    title = GROUPING_TITLES.get(grouping_name, grouping_name or 'All')
    plt.suptitle(f'Probability Shift vs. Overall Mean: {title}',
                 fontsize=24, y=1.02)
    plt.tight_layout(pad=1.5)
    suffix = f"_{grouping_name}" if grouping_name else ""
    save_figure_multi_format(fig, os.path.join(output_dir, f'archetype_probability_difference{suffix}'))
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("STATE ARCHETYPE ACTION DISTRIBUTION PLOTS")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load JSON (optional, for sample counts)
    full_results = load_json(RESULTS_DIR)

    # Discover all groupings
    groupings = discover_groupings(RESULTS_DIR)
    print(f"\nFound groupings: {groupings}")

    for grouping_name in groupings:
        display = GROUPING_TITLES.get(grouping_name, grouping_name or 'default')
        print(f"\n{'=' * 80}")
        print(f"GROUPING: {display}")
        print(f"{'=' * 80}")

        df = load_csv(RESULTS_DIR, grouping_name)

        # Determine archetype ordering
        csv_archetypes = get_archetypes_from_csv(df)
        preferred = GROUPING_ORDER.get(grouping_name, [])
        archetypes = [a for a in preferred if a in csv_archetypes]
        for a in csv_archetypes:
            if a not in archetypes:
                archetypes.append(a)

        print(f"  Operators: {len(df)}")
        print(f"  Archetypes: {archetypes}")

        # Print sample counts from JSON if available
        if (full_results and 'groupings' in full_results
                and grouping_name in full_results['groupings']):
            dists = full_results['groupings'][grouping_name].get('distributions', {})
            for name, dist in dists.items():
                print(f"    {name}: n={dist.get('count', '?')}")

        # Generate plots
        plot_mean_probabilities(df, archetypes, OUTPUT_DIR, grouping_name)
        plot_empirical_fractions(df, archetypes, OUTPUT_DIR, grouping_name)
        plot_radar_chart(df, archetypes, OUTPUT_DIR, grouping_name)
        plot_probability_heatmap(df, archetypes, OUTPUT_DIR, grouping_name)
        plot_probability_difference(df, archetypes, OUTPUT_DIR, grouping_name)

    print("\n" + "=" * 80)
    print("ALL PLOTS CREATED SUCCESSFULLY")
    print(f"Plots saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
