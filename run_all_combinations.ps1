# PowerShell script to run all combinations of acceptance and reward strategies

# Activate conda environment
conda activate PDPTW

# Define all acceptance strategies
$acceptanceStrategies = @(
    "rising_epsilon_greedy",
    "late_acceptance",
    "epsilon_greedy",
    "greedy",
    "always",
    "simulated_annealing"
)

# Define all reward strategies
$rewardStrategies = @(
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
    "distance_baseline_clipped"
)

# Configuration
$problemSize = 100
$numEpisodes = 1000

# Calculate total combinations
$totalCombinations = $acceptanceStrategies.Count * $rewardStrategies.Count
$currentCombination = 0

Write-Host "Starting training for all combinations..."
Write-Host "Total combinations: $totalCombinations"
Write-Host "Problem size: $problemSize"
Write-Host "Episodes per run: $numEpisodes"
Write-Host ""

# Create log directory if it doesn't exist
$logDir = "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# Loop through all combinations
foreach ($acceptance in $acceptanceStrategies) {
    foreach ($reward in $rewardStrategies) {
        $currentCombination++

        Write-Host "[$currentCombination/$totalCombinations] Training: acceptance=$acceptance, reward=$reward"

        # Create log filename
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $logFile = "$logDir/${acceptance}_${reward}_${timestamp}.log"

        # Run training
        try {
            python train_rl_local_search.py `
                --acceptance_strategy $acceptance `
                --reward_strategy $reward `
                --problem_size $problemSize `
                --num_episodes $numEpisodes `
                2>&1 | Tee-Object -FilePath $logFile

            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✓ Completed successfully" -ForegroundColor Green
            }
            else {
                Write-Host "  ✗ Failed with exit code $LASTEXITCODE" -ForegroundColor Red
            }
        }
        catch {
            Write-Host "  ✗ Error: $_" -ForegroundColor Red
        }

        Write-Host ""
    }
}

Write-Host "All combinations completed!"
