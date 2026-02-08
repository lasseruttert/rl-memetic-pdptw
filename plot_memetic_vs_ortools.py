"""
Memetic vs OR-Tools Comparison - LaTeX Table Generator

Generates LaTeX tables comparing Memetic algorithm, OR-Tools solver, and BKS
across 251 PDPTW instances (sizes 100, 200, 400).

Input: results/memetic_vs_ortools_summary.csv
Output: results/memetic_vs_ortools_comparison/
  - per_instance_table.tex
  - category_average_table.tex
  - overall_average_table.tex
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = "results/memetic_vs_ortools_summary.csv"
OUTPUT_DIR = "results/memetic_vs_ortools_comparison"

SIZES = [100, 200, 400]

CATEGORY_ORDER = ['lc1', 'lc2', 'lr1', 'lr2', 'lrc1', 'lrc2', 'bar', 'ber', 'nyc', 'poa']

# ============================================================================
# UTILITY FUNCTIONS (adapted from plot_memetic_components.py)
# ============================================================================

def parse_instance_category(instance_name):
    """Parse instance name to extract category."""
    if '\\' in instance_name or '/' in instance_name:
        instance_name = Path(instance_name).stem

    # Mendeley instances (bar, ber, nyc, poa)
    mendeley_match = re.match(r'([a-z]+)-n(\d+)-(\d+)', instance_name)
    if mendeley_match:
        return mendeley_match.group(1)

    # Li & Lim size-100 (lc101, lr205, etc.)
    lilim_100_match = re.match(r'(l[rc]{1,2}[12])(\d+)', instance_name)
    if lilim_100_match:
        return lilim_100_match.group(1)

    # Li & Lim larger sizes (LC1_2_1, etc.)
    lilim_large_match = re.match(r'(L[RC]{1,2}[12])_(\d+)_(\d+)', instance_name)
    if lilim_large_match:
        return lilim_large_match.group(1).lower()

    return 'unknown'


def escape_latex(text):
    """Escape special LaTeX characters in text."""
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


def save_latex_file(content, filepath):
    """Write LaTeX content to a file with proper preamble."""
    preamble = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{longtable}
\usepackage{geometry}
\geometry{a4paper, margin=1cm, landscape}

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
# DATA LOADING
# ============================================================================

def _compute_gap(solver_veh, solver_dist, bks_veh, bks_dist):
    """Compute gap% to BKS.

    If solver uses a different number of vehicles than BKS, gap is vehicle-based.
    Only when vehicles match is the gap distance-based.
    """
    if pd.isna(solver_veh) or pd.isna(bks_veh):
        return np.nan
    solver_veh = int(round(solver_veh))
    bks_veh = int(round(bks_veh))
    if solver_veh != bks_veh:
        if bks_veh == 0:
            return np.nan
        return (solver_veh - bks_veh) / bks_veh * 100
    # Same vehicles -> distance-based gap
    if pd.isna(solver_dist) or pd.isna(bks_dist) or bks_dist == 0:
        return np.nan
    return (solver_dist - bks_dist) / bks_dist * 100


def load_data(csv_path):
    """Load CSV data, add Category column, compute gaps, handle infeasibility."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")

    # Add category
    df['Category'] = df['Instance'].apply(parse_instance_category)

    # Mask infeasible solutions: set distance/vehicles/fitness to NaN
    memetic_infeasible = df['Memetic_IsFeasible'] == False
    df.loc[memetic_infeasible, ['Memetic_Distance', 'Memetic_Vehicles', 'Memetic_Fitness']] = np.nan

    ortools_infeasible = df['ORTools_IsFeasible'] == False
    df.loc[ortools_infeasible, ['ORTools_Distance', 'ORTools_Vehicles', 'ORTools_Fitness']] = np.nan

    # Compute gaps:
    #   - If solver uses different #vehicles than BKS: gap on vehicles
    #   - If solver uses same #vehicles as BKS: gap on distance
    df['Memetic_Gap'] = df.apply(
        lambda r: _compute_gap(r['Memetic_Vehicles'], r['Memetic_Distance'],
                               r['BKS_Vehicles'], r['BKS_Distance']),
        axis=1
    )
    df['ORTools_Gap'] = df.apply(
        lambda r: _compute_gap(r['ORTools_Vehicles'], r['ORTools_Distance'],
                               r['BKS_Vehicles'], r['BKS_Distance']),
        axis=1
    )

    # Round down gaps below 0.1% to 0
    for col in ['Memetic_Gap', 'ORTools_Gap']:
        df[col] = df[col].where(df[col].abs() >= 0.1, other=0.0).where(df[col].notna())

    return df


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_value(val, fmt='.1f', bold=False):
    """Format a numeric value for LaTeX. NaN -> '--'."""
    if pd.isna(val):
        return '--'
    if fmt == 'd':
        s = f'{int(round(val))}'
    else:
        s = f'{val:{fmt}}'
    if bold:
        s = r'\textbf{' + s + '}'
    return s


def determine_best(val_a, val_b):
    """Determine which value is best (lower is better).

    Returns (a_is_best, b_is_best). Both True on tie, both False if both NaN.
    """
    a_nan = pd.isna(val_a)
    b_nan = pd.isna(val_b)
    if a_nan and b_nan:
        return False, False
    if a_nan:
        return False, True
    if b_nan:
        return True, False
    if abs(val_a - val_b) < 1e-9:
        return True, True
    if val_a < val_b:
        return True, False
    return False, True


# ============================================================================
# TABLE GENERATION
# ============================================================================

def _table_header_row():
    """Return the column header lines used across tables."""
    return (
        r'    Instance'
        r' & \multicolumn{2}{c}{BKS}'
        r' & \multicolumn{3}{c}{Memetic}'
        r' & \multicolumn{3}{c}{OR-Tools} \\'
        '\n'
        r'    \cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-9}'
        '\n'
        r'    & Veh & Dist'
        r' & Veh & Dist & Gap\%'
        r' & Veh & Dist & Gap\% \\'
    )


def _determine_winner(mem_veh, mem_dist, ort_veh, ort_dist):
    """Determine which solver wins using unified comparison.

    Rules (lower is better):
    1. If either solver is infeasible (NaN vehicles), the feasible one wins.
    2. Compare vehicles (rounded to int): fewer wins.
    3. If vehicles tie: compare distance: lower wins.
    4. If both tie: both win.

    Returns (mem_bold, ort_bold) booleans applied to all columns.
    """
    mem_nan = pd.isna(mem_veh)
    ort_nan = pd.isna(ort_veh)
    if mem_nan and ort_nan:
        return False, False
    if mem_nan:
        return False, True
    if ort_nan:
        return True, False

    mem_v = int(round(mem_veh))
    ort_v = int(round(ort_veh))
    if mem_v < ort_v:
        return True, False
    if ort_v < mem_v:
        return False, True

    # Vehicles tied -> break tie on distance
    mem_d_nan = pd.isna(mem_dist)
    ort_d_nan = pd.isna(ort_dist)
    if mem_d_nan and ort_d_nan:
        return True, True
    if mem_d_nan:
        return False, True
    if ort_d_nan:
        return True, False
    if abs(mem_dist - ort_dist) < 1e-9:
        return True, True
    if mem_dist < ort_dist:
        return True, False
    return False, True


def _format_data_row(instance_name, bks_veh, bks_dist,
                     mem_veh, mem_dist, mem_gap,
                     ort_veh, ort_dist, ort_gap,
                     veh_fmt='d'):
    """Format one data row with unified bolding for best solver.

    veh_fmt: format for vehicle columns ('d' for int, '.1f' for decimal averages).
    """
    mem_bold, ort_bold = _determine_winner(mem_veh, mem_dist, ort_veh, ort_dist)

    cols = [
        escape_latex(str(instance_name)),
        format_value(bks_veh, fmt=veh_fmt),
        format_value(bks_dist, fmt='.1f'),
        format_value(mem_veh, fmt=veh_fmt, bold=mem_bold),
        format_value(mem_dist, fmt='.1f', bold=mem_bold),
        format_value(mem_gap, fmt='.2f', bold=mem_bold),
        format_value(ort_veh, fmt=veh_fmt, bold=ort_bold),
        format_value(ort_dist, fmt='.1f', bold=ort_bold),
        format_value(ort_gap, fmt='.2f', bold=ort_bold),
    ]
    return '    ' + ' & '.join(cols) + r' \\'


def _sort_key(row):
    """Sort key: Size -> Category order -> Instance name."""
    cat = row['Category']
    cat_idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (row['Size'], cat_idx, row['Instance'])


def generate_per_instance_table(df):
    """Generate a longtable with all 251 instances."""
    lines = []

    # longtable preamble
    lines.append(r'\begin{longtable}{l rr rrr rrr}')
    lines.append(r'  \caption{Per-instance comparison of Memetic algorithm and OR-Tools against BKS.}')
    lines.append(r'  \label{tab:per_instance} \\')

    # First head
    lines.append(r'  \toprule')
    lines.append(_table_header_row())
    lines.append(r'  \midrule')
    lines.append(r'  \endfirsthead')

    # Continuation head
    lines.append(r'  \multicolumn{9}{c}{\textit{(continued)}} \\')
    lines.append(r'  \toprule')
    lines.append(_table_header_row())
    lines.append(r'  \midrule')
    lines.append(r'  \endhead')

    # Foot
    lines.append(r'  \midrule')
    lines.append(r'  \multicolumn{9}{r}{\textit{continued on next page}} \\')
    lines.append(r'  \endfoot')

    # Last foot
    lines.append(r'  \bottomrule')
    lines.append(r'  \endlastfoot')

    # Sort rows
    sorted_df = df.sort_values(
        by=['Size', 'Category', 'Instance'],
        key=lambda col: col.map(lambda x: CATEGORY_ORDER.index(x) if x in CATEGORY_ORDER else len(CATEGORY_ORDER))
        if col.name == 'Category' else col
    )

    current_size = None
    for _, row in sorted_df.iterrows():
        # Size group header
        if row['Size'] != current_size:
            if current_size is not None:
                lines.append(r'  \midrule')
            current_size = row['Size']
            lines.append(
                r'  \multicolumn{9}{l}{\textbf{Size ' + str(current_size) + r'}} \\'
            )
            lines.append(r'  \midrule')

        lines.append(_format_data_row(
            row['Instance'],
            row['BKS_Vehicles'], row['BKS_Distance'],
            row['Memetic_Vehicles'], row['Memetic_Distance'], row['Memetic_Gap'],
            row['ORTools_Vehicles'], row['ORTools_Distance'], row['ORTools_Gap'],
        ))

    lines.append(r'\end{longtable}')
    return '\n'.join(lines)


def generate_category_average_table(df):
    """Generate a table with category averages grouped by size."""
    lines = []

    lines.append(r'\begin{longtable}{l rr rrr rrr}')
    lines.append(r'  \caption{Category-average comparison of Memetic algorithm and OR-Tools against BKS.}')
    lines.append(r'  \label{tab:category_avg} \\')
    lines.append(r'  \toprule')
    lines.append(_table_header_row())
    lines.append(r'  \midrule')
    lines.append(r'  \endfirsthead')
    lines.append(r'  \multicolumn{9}{c}{\textit{(continued)}} \\')
    lines.append(r'  \toprule')
    lines.append(_table_header_row())
    lines.append(r'  \midrule')
    lines.append(r'  \endhead')
    lines.append(r'  \bottomrule')
    lines.append(r'  \endlastfoot')

    for size in SIZES:
        size_df = df[df['Size'] == size]
        lines.append(
            r'  \multicolumn{9}{l}{\textbf{Size ' + str(size) + r'}} \\'
        )
        lines.append(r'  \midrule')

        # Get categories present in this size
        cats_present = [c for c in CATEGORY_ORDER if c in size_df['Category'].values]

        for cat in cats_present:
            cat_df = size_df[size_df['Category'] == cat]
            agg = cat_df[['BKS_Vehicles', 'BKS_Distance',
                          'Memetic_Vehicles', 'Memetic_Distance', 'Memetic_Gap',
                          'ORTools_Vehicles', 'ORTools_Distance', 'ORTools_Gap']].mean(skipna=True)

            lines.append(_format_data_row(
                cat,
                agg['BKS_Vehicles'], agg['BKS_Distance'],
                agg['Memetic_Vehicles'], agg['Memetic_Distance'], agg['Memetic_Gap'],
                agg['ORTools_Vehicles'], agg['ORTools_Distance'], agg['ORTools_Gap'],
                veh_fmt='.1f',
            ))

        if size != SIZES[-1]:
            lines.append(r'  \midrule')

    lines.append(r'\end{longtable}')
    return '\n'.join(lines)


def generate_overall_average_table(df):
    """Generate a table with overall averages per size."""
    lines = []

    lines.append(r'\begin{tabular}{l rr rrr rrr}')
    lines.append(r'  \toprule')
    lines.append(_table_header_row())
    lines.append(r'  \midrule')

    for size in SIZES:
        size_df = df[df['Size'] == size]
        agg = size_df[['BKS_Vehicles', 'BKS_Distance',
                        'Memetic_Vehicles', 'Memetic_Distance', 'Memetic_Gap',
                        'ORTools_Vehicles', 'ORTools_Distance', 'ORTools_Gap']].mean(skipna=True)

        lines.append(_format_data_row(
            f'Size {size}',
            agg['BKS_Vehicles'], agg['BKS_Distance'],
            agg['Memetic_Vehicles'], agg['Memetic_Distance'], agg['Memetic_Gap'],
            agg['ORTools_Vehicles'], agg['ORTools_Distance'], agg['ORTools_Gap'],
            veh_fmt='.1f',
        ))

    lines.append(r'  \bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Memetic vs OR-Tools - LaTeX Table Generator")
    print("=" * 60)

    df = load_data(CSV_PATH)

    # Print summary stats
    print(f"\n  Categories found: {sorted(df['Category'].unique())}")
    print(f"  Sizes: {sorted(df['Size'].unique())}")
    print(f"  Memetic infeasible: {(~df['Memetic_IsFeasible']).sum()}")
    print(f"  ORTools infeasible (incl. missing): "
          f"{df['ORTools_Distance'].isna().sum()}")

    print(f"\nGenerating tables to {OUTPUT_DIR}/...")

    # Table 1: Per-instance
    tex = generate_per_instance_table(df)
    save_latex_file(tex, Path(OUTPUT_DIR) / 'per_instance_table.tex')

    # Table 2: Category averages
    tex = generate_category_average_table(df)
    save_latex_file(tex, Path(OUTPUT_DIR) / 'category_average_table.tex')

    # Table 3: Overall averages
    tex = generate_overall_average_table(df)
    save_latex_file(tex, Path(OUTPUT_DIR) / 'overall_average_table.tex')

    print("\nDone!")


if __name__ == '__main__':
    main()
