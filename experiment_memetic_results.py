"""
Benchmark Runner für Li & Lim PDPTW Instanzen mit Memetic Algorithm
"""

from pathlib import Path
import time

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.instance_manager import InstanceManager
from utils.result_logger import JSONResultLogger
from memetic.memetic_algorithm import MemeticSolver


def run_benchmark_suite(
    data_dir: str = 'data',
    results_dir: str = 'results',
    sizes: list[int] = [100],
    categories: list[str] = None,
    algorithm_name: str = "MemeticSolver",
    skip_existing: bool = True,
    **solver_params
):
    """
    Führt Memetic Algorithm auf allen angegebenen Instanzen aus.
    
    Args:
        data_dir: Verzeichnis mit Benchmark-Daten
        results_dir: Verzeichnis für Ergebnisse
        sizes: Liste von Problemgrößen
        categories: Liste von Kategorien (None = alle)
        algorithm_name: Name für Logging
        skip_existing: Überspringe bereits berechnete Instanzen
        **solver_params: Parameter für MemeticSolver (z.B. population_size, max_generations)
    """
    # Setup
    manager = InstanceManager(base_dir=data_dir)
    logger = JSONResultLogger(results_dir=results_dir)
    
    # Statistiken
    total_instances = 0
    solved_instances = 0
    skipped_instances = 0
    failed_instances = 0
    total_runtime = 0.0
    
    print("="*80)
    print(f"BENCHMARK SUITE: {algorithm_name}")
    print(f"Sizes: {sizes}")
    print(f"Categories: {categories if categories else 'ALL'}")
    print(f"Solver params: {solver_params}")
    print("="*80)
    
    # Iteriere über alle Instanzen
    for instance_name, category, size, problem in manager.iterate_all(sizes=sizes, categories=categories):
        total_instances += 1

        
        print(f"\n{'='*80}")
        print(f"Processing: {instance_name} (size={size}, category={category})")
        print(f"Problem: {problem.num_requests} requests, {problem.num_vehicles} vehicles")
        print(f"{'='*80}")
        
        try:
            # Initialisiere Solver
            solver = MemeticSolver(**solver_params)
            
            # Solve
            start_time = time.time()
            solution = solver.solve(problem)
            runtime = time.time() - start_time
            
            total_runtime += runtime
            
            # Log Ergebnis
            print(f"\n✓ Solution found:")
            print(f"  Vehicles: {solution.num_vehicles_used}")
            print(f"  Distance: {solution.total_distance:.2f}")
            print(f"  Feasible: {solution.is_feasible}")
            print(f"  Runtime: {runtime:.2f}s")
            
            # Speichern
            logger.save_result(
                instance_name=instance_name,
                size=size,
                category=category,
                algorithm=algorithm_name,
                solution=solution,
                runtime=runtime,
                evaluations=solver.evaluations if hasattr(solver, 'evaluations') else None,
                hyperparams=solver_params,
                best_known_vehicles=None,
                best_known_distance=None
            )
            
            solved_instances += 1
            
        except Exception as e:
            print(f"\n✗ FAILED: {instance_name}")
            print(f"  Error: {str(e)}")
            failed_instances += 1
            
            # Speichere Fehler
            error_file = Path(results_dir) / f"ERROR_{instance_name}_size{size}.txt"
            with open(error_file, 'w') as f:
                f.write(f"Instance: {instance_name}\n")
                f.write(f"Size: {size}\n")
                f.write(f"Category: {category}\n")
                f.write(f"Error: {str(e)}\n")
                import traceback
                f.write(f"\nTraceback:\n{traceback.format_exc()}")
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total instances: {total_instances}")
    print(f"Solved: {solved_instances}")
    print(f"Skipped: {skipped_instances}")
    print(f"Failed: {failed_instances}")
    print(f"Total runtime: {total_runtime:.2f}s ({total_runtime/60:.2f}min)")
    if solved_instances > 0:
        print(f"Average runtime per instance: {total_runtime/solved_instances:.2f}s")
    print("="*80)
    
    return {
        'total': total_instances,
        'solved': solved_instances,
        'skipped': skipped_instances,
        'failed': failed_instances,
        'total_runtime': total_runtime
    }


def run_single_instance(
    instance_name: str,
    size: int = 100,
    data_dir: str = 'data',
    results_dir: str = 'results',
    algorithm_name: str = "MemeticSolver",
    **solver_params
):
    """
    Führt Solver auf einer einzelnen Instanz aus.
    
    Args:
        instance_name: Name der Instanz (z.B. 'lc101')
        size: Problemgröße
        data_dir: Verzeichnis mit Benchmark-Daten
        results_dir: Verzeichnis für Ergebnisse
        algorithm_name: Name für Logging
        **solver_params: Parameter für MemeticSolver
    """
    manager = InstanceManager(base_dir=data_dir)
    logger = JSONResultLogger(results_dir=results_dir)
    
    # Lade Problem
    manager.jump_to(instance_name).jump_to_size(size)
    problem = manager.current()
    
    # Bestimme Kategorie
    category = None
    for cat, instances in manager.CATEGORIES.items():
        if instance_name in instances:
            category = cat
            break
    
    print(f"Processing: {instance_name} (size={size}, category={category})")
    print(f"Problem: {problem.num_requests} requests, {problem.num_vehicles} vehicles")
    
    # Solve
    solver = MemeticSolver(**solver_params)
    
    start_time = time.time()
    solution = solver.solve(problem)
    runtime = time.time() - start_time
    
    print(f"\nSolution:")
    print(f"  Vehicles: {solution.num_vehicles_used}")
    print(f"  Distance: {solution.total_distance:.2f}")
    print(f"  Feasible: {solution.is_feasible}")
    print(f"  Runtime: {runtime:.2f}s")
    
    # Speichern
    logger.save_result(
        instance_name=instance_name,
        size=size,
        category=category,
        algorithm=algorithm_name,
        solution=solution,
        runtime=runtime,
        evaluations=solver.evaluations if hasattr(solver, 'evaluations') else None,
        hyperparams=solver_params
    )
    
    return solution


if __name__ == "__main__":
    # Konfiguration
    DATA_DIR = 'G:/Meine Ablage/rl-memetic-pdptw/data'
    RESULTS_DIR = 'G:/Meine Ablage/rl-memetic-pdptw/results'
    
    # Solver Parameter
    SOLVER_CONFIG = {
        'population_size': 50,
        'max_generations': 100,
        'max_time_seconds': 300,  # 5 Minuten pro Instanz
        'max_no_improvement': 3,
        'ensure_diversity_interval': 5,
        'evaluation_interval': 10,
        'verbose': True,
        'track_history': False
    }
    
    # Test Mode: Einzelne Instanz
    # print("="*80)
    # print("TEST MODE: Single Instance")
    # print("="*80)
    
    # solution = run_single_instance(
    #     instance_name='lc101',
    #     size=100,
    #     data_dir=DATA_DIR,
    #     results_dir=RESULTS_DIR,
    #     algorithm_name='MemeticSolver_v1',
    #     **SOLVER_CONFIG
    # )
    
    # # Production Mode 1: Einzelne Kategorie
    # print("\n\n" + "="*80)
    # print("PRODUCTION MODE 1: LC1 Category (size 100)")
    # print("="*80)
    
    # stats = run_benchmark_suite(
    #     data_dir=DATA_DIR,
    #     results_dir=RESULTS_DIR,
    #     sizes=[100],
    #     categories=['lc1'],
    #     algorithm_name='MemeticSolver_v1',
    #     skip_existing=True,
    #     **SOLVER_CONFIG
    # )
    
    # Production Mode 2: Alle LC Kategorien
    # stats = run_benchmark_suite(
    #     data_dir=DATA_DIR,
    #     results_dir=RESULTS_DIR,
    #     sizes=[100],
    #     categories=['lc1', 'lc2'],
    #     algorithm_name='MemeticSolver_v1',
    #     skip_existing=True,
    #     **SOLVER_CONFIG
    # )
    
    # Production Mode 3: ALLE Instanzen, ALLE Größen
    stats = run_benchmark_suite(
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        sizes=[200, 400, 600, 1000],
        categories=None,
        algorithm_name='MemeticSolver_v1',
        skip_existing=True,
        **SOLVER_CONFIG
    )