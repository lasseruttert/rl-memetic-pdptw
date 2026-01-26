#!/usr/bin/env python3
# runner.py
import os
import sys
import shlex
import subprocess
from typing import List, Tuple, Union

# ------------------------------------------------------------
# Pflege deine Tasks hier (Strings oder Listen):
#   Beispiele:
# TASKS = [
#     "prep.py --data data.csv",
#     ["train.py", "--epochs", "10", "--lr", "1e-3"],
#     r"eval.py --ckpt C:\tmp\model.pt --verbose", 
# ]
# ------------------------------------------------------------
TASKS: List[Union[str, List[str]]] = [    
    # Hyperparameter tuning experiments
    "train_rl_local_search.py --config config/hyperparameter_tuning/baseline.yaml",
    # Learning rate variations
    "train_rl_local_search.py --config config/hyperparameter_tuning/lr_1e3.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/lr_5e4.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/lr_5e5.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/lr_1e5.yaml",
    # Gamma variations
    "train_rl_local_search.py --config config/hyperparameter_tuning/gamma_099.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/gamma_095.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/gamma_085.yaml",
    # Epsilon decay variations
    "train_rl_local_search.py --config config/hyperparameter_tuning/eps_decay_slow.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/eps_decay_fast.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/eps_decay_vfast.yaml",
    # Batch size variations
    "train_rl_local_search.py --config config/hyperparameter_tuning/batch_32.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/batch_128.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/batch_256.yaml",
    # N-step variations
    "train_rl_local_search.py --config config/hyperparameter_tuning/nstep_1.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/nstep_5.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/nstep_7.yaml",
    # Target update interval variations
    "train_rl_local_search.py --config config/hyperparameter_tuning/target_update_25.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/target_update_50.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/target_update_200.yaml",
    # Network architecture variations
    "train_rl_local_search.py --config config/hyperparameter_tuning/net_small.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/net_wide.yaml",
    "train_rl_local_search.py --config config/hyperparameter_tuning/net_large.yaml",
]


SHLEX_POSIX = os.name != "nt"

def normalize_tasks(raw: List[Union[str, List[str]]]) -> List[List[str]]:
    out: List[List[str]] = []
    for t in raw:
        if isinstance(t, str):
            parts = shlex.split(t, posix=SHLEX_POSIX)
        elif isinstance(t, list):
            parts = [str(x) for x in t]
        else:
            raise TypeError("Jede Task muss String oder Liste von Strings sein.")
        if parts:
            out.append(parts)
    return out

def run_command(cmd_parts: List[str]) -> Tuple[int, str]:
    full_cmd = [sys.executable, *cmd_parts]
    try:
        rc = subprocess.run(
            full_cmd,
            check=False,
            env=os.environ.copy(),
        ).returncode
        return rc, " ".join(full_cmd)
    except Exception as e:
        print(f"[Runner] Unerwarteter Fehler: {e}", file=sys.stderr)
        return 255, " ".join(full_cmd)

def main():
    if not TASKS:
        print(
            "Keine Tasks definiert."
        )
        sys.exit(2)

    tasks = normalize_tasks(TASKS)

    results: List[Tuple[int, str]] = []
    for i, parts in enumerate(tasks, 1):
        print(f"\n[Runner] ({i}/{len(tasks)}) Starte: {parts[0]} {' '.join(parts[1:])}")
        rc, cmd_str = run_command(parts)
        status = "OK" if rc == 0 else f"FEHLER (rc={rc})"
        print(f"[Runner] Beendet: {status}")
        results.append((rc, cmd_str))

    print("\n===== Zusammenfassung =====")
    failed = 0
    for rc, cmd_str in results:
        mark = "OK" if rc == 0 else f"FEHLER (rc={rc})"
        print(f"{mark}  |  {cmd_str}")
        if rc != 0:
            failed += 1

    sys.exit(0 if failed == 0 else min(failed, 255))

if __name__ == "__main__":
    main()
