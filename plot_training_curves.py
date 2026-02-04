"""Script to plot training curves from TensorBoard event files.

Compares DQN vs PPO training dynamics using thesis-ready visualization style
matching plot_rl_algorithm_comparison.py.

Output: PDF and PNG plots for episode length, episode reward, and training loss.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


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

# Algorithm colors
DQN_COLOR = '#2E86AB'  # Steel Blue
PPO_COLOR = '#A23B72'  # Plum Purple

# Data paths
DQN_LOG_DIR = 'runs/detail/rl_local_search_dqn_100_greedy_binary_seed100_set2_1764822068'
PPO_LOG_DIR = 'runs/detail/rl_local_search_ppo_100_greedy_binary_seed100_set2_1764807663'

OUTPUT_DIR = 'results/plots/training_curves'


def load_tensorboard_data(log_dir, tag):
    """Load scalar data from TensorBoard event file.

    Args:
        log_dir: Directory containing event file
        tag: Scalar tag to extract (e.g., 'Episode/Length')

    Returns:
        tuple: (steps, values) as numpy arrays
    """
    # Find event file
    event_file = None
    for f in os.listdir(log_dir):
        if f.startswith('events.out.tfevents'):
            event_file = os.path.join(log_dir, f)
            break

    if event_file is None:
        raise FileNotFoundError(f"No event file found in {log_dir}")

    # Load data
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags().get('scalars', []):
        raise ValueError(f"Tag '{tag}' not found in {log_dir}")

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])

    return steps, values


def smooth(values, weight=0.8):
    """Apply exponential moving average smoothing.

    Args:
        values: Array of values to smooth
        weight: Smoothing weight (0=no smoothing, 1=infinite smoothing)

    Returns:
        Smoothed values array
    """
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i-1] + (1 - weight) * values[i]
    return smoothed


def save_figure_multi_format(fig, filepath, formats=['png', 'pdf']):
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
        else:
            fig.savefig(output_path, format='png', bbox_inches='tight',
                       dpi=300)
        print(f'    Saved: {output_path}')


def plot_comparison(dqn_steps, dqn_values, ppo_steps, ppo_values,
                    xlabel, ylabel, title, filename, smoothing=None):
    """Create comparison plot with DQN and PPO data.

    Args:
        dqn_steps, dqn_values: DQN data
        ppo_steps, ppo_values: PPO data
        xlabel, ylabel: Axis labels
        title: Plot title
        filename: Output filename (without extension)
        smoothing: EMA weight for smoothing (None for no smoothing)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot raw data with transparency
    if smoothing is not None:
        # Raw data (faded)
        ax.plot(dqn_steps, dqn_values, color=DQN_COLOR, alpha=0.2, linewidth=1.0)
        ax.plot(ppo_steps, ppo_values, color=PPO_COLOR, alpha=0.2, linewidth=1.0)

        # Smoothed data (prominent)
        dqn_smoothed = smooth(dqn_values, smoothing)
        ppo_smoothed = smooth(ppo_values, smoothing)
        ax.plot(dqn_steps, dqn_smoothed, color=DQN_COLOR, linewidth=3.0, label='DQN')
        ax.plot(ppo_steps, ppo_smoothed, color=PPO_COLOR, linewidth=3.0, label='PPO')

        # Add smoothing annotation
        ax.text(0.98, 0.02, f'EMA smoothing: {smoothing}',
                transform=ax.transAxes, fontsize=14, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    else:
        # No smoothing - just plot the data
        ax.plot(dqn_steps, dqn_values, color=DQN_COLOR, linewidth=3.0, label='DQN')
        ax.plot(ppo_steps, ppo_values, color=PPO_COLOR, linewidth=3.0, label='PPO')

    # Styling
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save
    filepath = os.path.join(OUTPUT_DIR, filename)
    save_figure_multi_format(fig, filepath)
    plt.close()


def plot_single(steps, values, color, xlabel, ylabel, title, filename, smoothing=None):
    """Create a single algorithm plot.

    Args:
        steps, values: Data arrays
        color: Line color
        xlabel, ylabel: Axis labels
        title: Plot title
        filename: Output filename (without extension)
        smoothing: EMA weight for smoothing (None for no smoothing)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if smoothing is not None:
        # Raw data (faded)
        ax.plot(steps, values, color=color, alpha=0.2, linewidth=1.0)
        # Smoothed data (prominent)
        smoothed_values = smooth(values, smoothing)
        ax.plot(steps, smoothed_values, color=color, linewidth=3.0)

        # Add smoothing annotation
        ax.text(0.98, 0.02, f'EMA smoothing: {smoothing}',
                transform=ax.transAxes, fontsize=14, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    else:
        ax.plot(steps, values, color=color, linewidth=3.0)

    # Styling
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Save
    filepath = os.path.join(OUTPUT_DIR, filename)
    save_figure_multi_format(fig, filepath)
    plt.close()


def main():
    """Main function to create training curves plots."""
    print('='*80)
    print('TRAINING CURVES PLOTTING')
    print('='*80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'\nOutput directory: {OUTPUT_DIR}/')

    # Load data
    print('\nLoading TensorBoard data...')

    # Episode Length
    print('\n  Episode Length:')
    dqn_length_steps, dqn_length_values = load_tensorboard_data(DQN_LOG_DIR, 'Episode/Length')
    ppo_length_steps, ppo_length_values = load_tensorboard_data(PPO_LOG_DIR, 'Episode/Length')
    print(f'    DQN: {len(dqn_length_values)} data points')
    print(f'    PPO: {len(ppo_length_values)} data points')

    # Episode Reward (pre-averaged)
    print('\n  Episode Reward:')
    dqn_reward_steps, dqn_reward_values = load_tensorboard_data(DQN_LOG_DIR, 'Episode/Reward_Avg10')
    ppo_reward_steps, ppo_reward_values = load_tensorboard_data(PPO_LOG_DIR, 'Episode/Reward_Avg10')
    print(f'    DQN: {len(dqn_reward_values)} data points')
    print(f'    PPO: {len(ppo_reward_values)} data points')

    # Training Loss (different tags for DQN vs PPO)
    print('\n  Training Loss:')
    dqn_loss_steps, dqn_loss_values = load_tensorboard_data(DQN_LOG_DIR, 'Training/Loss')
    ppo_loss_steps, ppo_loss_values = load_tensorboard_data(PPO_LOG_DIR, 'Training/ValueLoss')
    print(f'    DQN: {len(dqn_loss_values)} data points')
    print(f'    PPO: {len(ppo_loss_values)} data points')

    # Create plots
    print('\n' + '='*80)
    print('CREATING PLOTS')
    print('='*80)

    # Plot 1: Episode Length (smoothing=0.8)
    print('\n  Episode Length plot:')
    plot_comparison(
        dqn_length_steps, dqn_length_values,
        ppo_length_steps, ppo_length_values,
        xlabel='Episode',
        ylabel='Episode Length (steps)',
        title='Episode Length: DQN vs PPO',
        filename='episode_length',
        smoothing=0.8
    )

    # Plot 2: Episode Reward (light smoothing on top of pre-averaged data)
    print('\n  Episode Reward plot:')
    plot_comparison(
        dqn_reward_steps, dqn_reward_values,
        ppo_reward_steps, ppo_reward_values,
        xlabel='Episode',
        ylabel='Reward (10-episode moving avg.)',
        title='Episode Reward: DQN vs PPO',
        filename='episode_reward',
        smoothing=0.8
    )

    # Plot 3: DQN Training Loss (no smoothing)
    print('\n  DQN Training Loss plot:')
    plot_single(
        dqn_loss_steps, dqn_loss_values,
        color=DQN_COLOR,
        xlabel='Episode',
        ylabel='TD Loss',
        title='DQN Training Loss',
        filename='training_loss_dqn',
        smoothing=None
    )

    # Plot 4: PPO Value Loss (no smoothing)
    print('\n  PPO Value Loss plot:')
    plot_single(
        ppo_loss_steps, ppo_loss_values,
        color=PPO_COLOR,
        xlabel='Episode',
        ylabel='Value Loss',
        title='PPO Value Loss',
        filename='training_loss_ppo',
        smoothing=None
    )

    print('\n' + '='*80)
    print('ALL PLOTS CREATED SUCCESSFULLY')
    print('='*80)
    print(f'Plots saved to: {OUTPUT_DIR}/')
    print('Formats: PNG and PDF')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
