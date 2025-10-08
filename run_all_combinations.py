"""
Run all combinations of acceptance and reward strategies for train_rl_local_search.py
without relying on shell or PowerShell scripts.

This script mirrors run_all_combinations.sh / .ps1 behavior:
- Iterates strategy combinations
- Prints progress indicators
- Writes a timestamped log per run into `logs/`
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_ACCEPTANCE_STRATEGIES = [
    "rising_epsilon_greedy",
    "late_acceptance",
    "epsilon_greedy",
    "greedy",
    "always",
    "simulated_annealing",
]

DEFAULT_REWARD_STRATEGIES = [
    "tanh",
    "distance_baseline_tanh",
    "distance_baseline_normalized",
    "pure_normalized",
    "distance_baseline_asymmetric_tanh",
    "initial_improvement",
    "old_improvement",
    "hybrid_improvement",
    "distance_baseline",
    "log_improvement",
    "binary",
    "distance_baseline_clipped",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all acceptance/reward combinations for RL local search"
    )
    parser.add_argument(
        "--problem_size",
        type=int,
        default=100,
        help="Problem size to pass to train_rl_local_search.py",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to pass to train_rl_local_search.py",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("logs"),
        help="Directory to store logs",
    )
    parser.add_argument(
        "--acceptance_strategies",
        type=str,
        nargs="*",
        default=DEFAULT_ACCEPTANCE_STRATEGIES,
        help="Override list of acceptance strategies",
    )
    parser.add_argument(
        "--reward_strategies",
        type=str,
        nargs="*",
        default=DEFAULT_REWARD_STRATEGIES,
        help="Override list of reward strategies",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use when invoking train_rl_local_search.py",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned runs without executing",
    )
    return parser.parse_args()


def ensure_log_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_once(
    *,
    python_exec: str,
    acceptance: str,
    reward: str,
    problem_size: int,
    num_episodes: int,
    log_path: Path,
) -> int:
    cmd = [
        python_exec,
        "train_rl_local_search.py",
        "--acceptance_strategy",
        acceptance,
        "--reward_strategy",
        reward,
        "--problem_size",
        str(problem_size),
        "--num_episodes",
        str(num_episodes),
    ]

    # Stream output to both console and file (tee-like)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command: " + " ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        proc.wait()
        return proc.returncode


def main() -> None:
    args = parse_args()

    acceptance = args.acceptance_strategies
    reward = args.reward_strategies
    total = len(acceptance) * len(reward)

    ensure_log_dir(args.log_dir)

    print("Starting training for all combinations...")
    print(f"Total combinations: {total}")
    print(f"Problem size: {args.problem_size}")
    print(f"Episodes per run: {args.num_episodes}")
    print("")

    idx = 0
    for a, r in itertools.product(acceptance, reward):
        idx += 1
        print(f"[{idx}/{total}] Training: acceptance={a}, reward={r}")

        log_file = args.log_dir / f"{a}_{r}_{timestamp()}.log"

        if args.dry_run:
            print(f"Dry run: would log to {log_file}")
            print("")
            continue

        try:
            code = run_once(
                python_exec=args.python,
                acceptance=a,
                reward=r,
                problem_size=args.problem_size,
                num_episodes=args.num_episodes,
                log_path=log_file,
            )
            if code == 0:
                print("  ✓ Completed successfully")
            else:
                print(f"  ✗ Failed with exit code {code}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        print("")

    print("All combinations completed!")


if __name__ == "__main__":
    main()

