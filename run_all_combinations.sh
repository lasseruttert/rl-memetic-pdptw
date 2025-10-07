#!/bin/bash

# Bash script to run all combinations of acceptance and reward strategies

# Exit immediately if a command exits with a non-zero status
set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh  # adjust path if necessary
conda activate PDPTW

# Define all acceptance strategies
acceptance_strategies=(
    "rising_epsilon_greedy"
    "late_acceptance"
    "epsilon_greedy"
    "greedy"
    "always"
    "simulated_annealing"
)

# Define all reward strategies
reward_strategies=(
    "tanh"
    "distance_baseline_tanh"
    "distance_baseline_normalized"
    "pure_normalized"
    "distance_baseline_asymmetric_tanh"
    "initial_improvement"
    "old_improvement"
    "hybrid_improvement"
    "distance_baseline"
    "log_improvement"
    "binary"
    "distance_baseline_clipped"
)

# Configuration
problem_size=100
num_episodes=1000

# Calculate total combinations
total_combinations=$(( ${#acceptance_strategies[@]} * ${#reward_strategies[@]} ))
current_combination=0

echo "Starting training for all combinations..."
echo "Total combinations: $total_combinations"
echo "Problem size: $problem_size"
echo "Episodes per run: $num_episodes"
echo ""

# Create log directory if it doesn't exist
log_dir="logs"
mkdir -p "$log_dir"

# Loop through all combinations
for acceptance in "${acceptance_strategies[@]}"; do
    for reward in "${reward_strategies[@]}"; do
        ((current_combination++))
        echo "[$current_combination/$total_combinations] Training: acceptance=$acceptance, reward=$reward"

        timestamp=$(date +"%Y%m%d_%H%M%S")
        log_file="$log_dir/${acceptance}_${reward}_${timestamp}.log"

        # Run training and log output
        if python train_rl_local_search.py \
            --acceptance_strategy "$acceptance" \
            --reward_strategy "$reward" \
            --problem_size "$problem_size" \
            --num_episodes "$num_episodes" \
            2>&1 | tee "$log_file"; then
            echo "  ✓ Completed successfully"
        else
            echo "  ✗ Failed (exit code $?)"
        fi

        echo ""
    done
done

echo "All combinations completed!"
