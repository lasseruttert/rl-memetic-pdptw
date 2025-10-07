"""Script to summarize training logs from RL local search experiments."""

import os
import re
from pathlib import Path
from collections import defaultdict
import argparse


def parse_log_file(log_path):
    """Parse a single log file and extract key metrics.

    Args:
        log_path: Path to log file

    Returns:
        Dictionary with extracted metrics
    """
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    metrics = {
        'file': os.path.basename(log_path),
        'path': str(log_path),
    }

    # Extract configuration
    acceptance_match = re.search(r'acceptance[_\s]*strategy[:\s]*(\w+)', content, re.IGNORECASE)
    if acceptance_match:
        metrics['acceptance_strategy'] = acceptance_match.group(1)

    reward_match = re.search(r'reward[_\s]*strategy[:\s]*(\w+)', content, re.IGNORECASE)
    if reward_match:
        metrics['reward_strategy'] = reward_match.group(1)

    problem_size_match = re.search(r'size[:\s]*(\d+)', content, re.IGNORECASE)
    if problem_size_match:
        metrics['problem_size'] = int(problem_size_match.group(1))

    # Extract final training metrics
    final_reward_match = re.search(r'Final average reward[:\s]*([-\d.]+)', content, re.IGNORECASE)
    if final_reward_match:
        metrics['final_avg_reward'] = float(final_reward_match.group(1))

    final_fitness_match = re.search(r'Final average fitness[:\s]*([-\d.]+)', content, re.IGNORECASE)
    if final_fitness_match:
        metrics['final_avg_fitness'] = float(final_fitness_match.group(1))

    # Extract test results
    # Look for "Model X: Avg best fitness: Y"
    rl_model_match = re.search(r'Model.*?Avg best fitness[:\s]*([-\d.]+)', content, re.IGNORECASE)
    if rl_model_match:
        metrics['rl_test_fitness'] = float(rl_model_match.group(1))

    # Adaptive
    adaptive_match = re.search(r'Adaptive[:\s]*Avg best fitness[:\s]*([-\d.]+)', content, re.IGNORECASE)
    if adaptive_match:
        metrics['adaptive_test_fitness'] = float(adaptive_match.group(1))

    # Naive
    naive_match = re.search(r'Naive[:\s]*Avg best fitness[:\s]*([-\d.]+)', content, re.IGNORECASE)
    if naive_match:
        metrics['naive_test_fitness'] = float(naive_match.group(1))

    # Naive (best improvement)
    naive_best_match = re.search(r'Naive \(best improvement\)[:\s]*Avg best fitness[:\s]*([-\d.]+)', content, re.IGNORECASE)
    if naive_best_match:
        metrics['naive_best_test_fitness'] = float(naive_best_match.group(1))

    # Random
    random_match = re.search(r'Random[:\s]*Avg best fitness[:\s]*([-\d.]+)', content, re.IGNORECASE)
    if random_match:
        metrics['random_test_fitness'] = float(random_match.group(1))

    # Extract average initial fitness
    avg_initial_match = re.search(r'Average initial fitness[:\s]*([-\d.]+)', content, re.IGNORECASE)
    if avg_initial_match:
        metrics['avg_initial_fitness'] = float(avg_initial_match.group(1))

    return metrics


def find_log_files(directory, extensions=['.log', '.txt', '.out']):
    """Find all log files in directory and subdirectories.

    Args:
        directory: Root directory to search
        extensions: List of file extensions to consider

    Returns:
        List of Path objects
    """
    log_files = []
    for ext in extensions:
        log_files.extend(Path(directory).rglob(f'*{ext}'))
    return log_files


def summarize_logs(log_files):
    """Summarize all log files.

    Args:
        log_files: List of log file paths

    Returns:
        List of metric dictionaries
    """
    results = []

    for log_file in log_files:
        try:
            metrics = parse_log_file(log_file)
            if metrics:  # Only add if we extracted something
                results.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to parse {log_file}: {e}")

    return results


def print_summary(results, sort_by='rl_test_fitness'):
    """Print formatted summary of results.

    Args:
        results: List of metric dictionaries
        sort_by: Key to sort by (default: rl_test_fitness)
    """
    if not results:
        print("No results to summarize.")
        return

    # Sort by specified metric (ascending - lower is better)
    valid_results = [r for r in results if sort_by in r]
    if valid_results:
        valid_results.sort(key=lambda x: x.get(sort_by, float('inf')))
    else:
        valid_results = results

    print("\n" + "="*120)
    print(f"{'File':<40} {'Accept':<20} {'Reward':<30} {'RL Fit':<10} {'Adaptive':<10} {'Naive':<10} {'Random':<10}")
    print("="*120)

    for r in valid_results:
        file_name = r.get('file', 'unknown')[:38]
        accept = r.get('acceptance_strategy', 'N/A')[:18]
        reward = r.get('reward_strategy', 'N/A')[:28]
        rl_fit = f"{r['rl_test_fitness']:.2f}" if 'rl_test_fitness' in r else 'N/A'
        adaptive_fit = f"{r['adaptive_test_fitness']:.2f}" if 'adaptive_test_fitness' in r else 'N/A'
        naive_fit = f"{r['naive_test_fitness']:.2f}" if 'naive_test_fitness' in r else 'N/A'
        random_fit = f"{r['random_test_fitness']:.2f}" if 'random_test_fitness' in r else 'N/A'

        print(f"{file_name:<40} {accept:<20} {reward:<30} {rl_fit:<10} {adaptive_fit:<10} {naive_fit:<10} {random_fit:<10}")

    print("="*120)

    # Print best configuration
    if valid_results and sort_by in valid_results[0]:
        best = valid_results[0]
        print(f"\n🏆 Best Configuration (by {sort_by}):")
        print(f"   File: {best.get('file', 'unknown')}")
        print(f"   Acceptance: {best.get('acceptance_strategy', 'N/A')}")
        print(f"   Reward: {best.get('reward_strategy', 'N/A')}")
        if 'rl_test_fitness' in best:
            print(f"   RL Fitness: {best['rl_test_fitness']:.2f}")
        if 'adaptive_test_fitness' in best:
            print(f"   Adaptive Fitness: {best['adaptive_test_fitness']:.2f}")
        if 'naive_test_fitness' in best:
            print(f"   Naive Fitness: {best['naive_test_fitness']:.2f}")

        # Calculate improvements
        if 'rl_test_fitness' in best and 'naive_test_fitness' in best:
            improvement = ((best['naive_test_fitness'] - best['rl_test_fitness']) / best['naive_test_fitness']) * 100
            print(f"   Improvement over Naive: {improvement:.2f}%")


def print_detailed_summary(results):
    """Print detailed summary with statistics."""
    if not results:
        print("No results to summarize.")
        return

    # Group by acceptance strategy
    by_acceptance = defaultdict(list)
    by_reward = defaultdict(list)

    for r in results:
        if 'rl_test_fitness' in r:
            accept = r.get('acceptance_strategy', 'unknown')
            reward = r.get('reward_strategy', 'unknown')
            by_acceptance[accept].append(r['rl_test_fitness'])
            by_reward[reward].append(r['rl_test_fitness'])

    if by_acceptance:
        print("\n📊 Average Fitness by Acceptance Strategy:")
        print("-" * 60)
        for accept, fitnesses in sorted(by_acceptance.items()):
            avg = sum(fitnesses) / len(fitnesses)
            print(f"   {accept:<30} {avg:.2f} (n={len(fitnesses)})")

    if by_reward:
        print("\n📊 Average Fitness by Reward Strategy:")
        print("-" * 60)
        for reward, fitnesses in sorted(by_reward.items()):
            avg = sum(fitnesses) / len(fitnesses)
            print(f"   {reward:<30} {avg:.2f} (n={len(fitnesses)})")


def main():
    parser = argparse.ArgumentParser(description="Summarize RL training logs")
    parser.add_argument("--dir", type=str, default=".", help="Directory to search for logs")
    parser.add_argument("--extensions", type=str, nargs='+', default=['.log', '.txt', '.out'],
                        help="File extensions to search")
    parser.add_argument("--sort-by", type=str, default="rl_test_fitness",
                        choices=['rl_test_fitness', 'adaptive_test_fitness', 'naive_test_fitness',
                                'random_test_fitness', 'final_avg_fitness'],
                        help="Metric to sort by")
    parser.add_argument("--detailed", action='store_true', help="Show detailed statistics")

    args = parser.parse_args()

    print(f"Searching for log files in: {args.dir}")
    log_files = find_log_files(args.dir, args.extensions)
    print(f"Found {len(log_files)} log files")

    if not log_files:
        print("No log files found. Exiting.")
        return

    results = summarize_logs(log_files)
    print(f"Successfully parsed {len(results)} log files with metrics")

    print_summary(results, sort_by=args.sort_by)

    if args.detailed:
        print_detailed_summary(results)


if __name__ == "__main__":
    main()
