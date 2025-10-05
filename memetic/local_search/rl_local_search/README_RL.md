# RL-Based Local Search

Reinforcement Learning-basierte lokale Suche für PDPTW, die adaptive Sequenzen von Neighborhood-Operatoren lernt.

## Überblick

### Problem
Herkömmliche lokale Suche-Verfahren haben fundamentale Limitationen:
- **Exhaustive Search**: Prohibitive Laufzeit bei vollständiger Operatoranwendung
- **Regelbasierte/Stochastische Strategien**: Random Walk, feste Sequenzen agieren kontextunabhängig
- **Suboptimale Nutzung**: Potenzial verfügbarer Operatoren wird nicht ausgeschöpft

### Lösung: RL-gesteuerte Operatorauswahl
- **Adaptive**: Lernt kontextsensitive Operatorauswahl
- **Effizient**: Keine exhaustive Enumeration notwendig
- **Intelligent**: Priorisiert vielversprechende Operatoren basierend auf Problem- und Lösungsmerkmalen

## Architektur

### 1. LocalSearchEnv (Gymnasium Environment)
Standardisiertes RL-Environment für lokale Suche.

**State Space** (22 Features):
- Problem-Features: num_requests, vehicle_capacity, avg_distance, time_window_tightness
- Lösungs-Features: num_routes, total_distance, route_statistics
- Constraint-Violations: capacity, time_window, precedence violations

**Action Space**:
- Discrete(n_operators): Ein Operator pro Action
- Kein No-Op (immer wird ein Operator gewählt)

**Reward Function** (normalisiert):
```python
fitness_improvement = (fitness_old - fitness_new) / problem.distance_baseline
feasibility_improvement = violation_reduction (proportional, in [0, 1])

reward = α * fitness_improvement + β * feasibility_improvement
```
- **Fitness-Differenz**: Normalisiert durch `problem.distance_baseline` → typisch ~0.001-0.1
  - Positive bei Verbesserung, negative bei Verschlechterung
  - Baseline ist Summe aller Depot-zu-Kunde Distanzen × 2
- **Feasibility-Verbesserung**: Normalisiert in [-∞, 1]
  - 1.0: Transition infeasible → feasible
  - [0, 1]: Proportionale Reduktion von Violations
  - Negativ: Zunahme von Violations

### 2. DQN Network & Agent
**Architektur**: MLP (Multi-Layer Perceptron)
```
Input (22 features) → FC(128) + ReLU + Dropout
                    → FC(128) + ReLU + Dropout
                    → FC(64) + ReLU
                    → Output (Q-values für jeden Operator)
```

**Training**:
- Experience Replay Buffer (capacity: 10k-100k)
- Target Network (update interval: 100 steps)
- ε-greedy Exploration (ε: 1.0 → 0.1)
- Huber Loss für Stabilität
- Gradient Clipping

### 3. RLLocalSearch
Hauptklasse, die von `BaseLocalSearch` erbt.

**Zwei Modi**:
1. **Training**: Lernt Policy durch Interaktion mit Problemen
2. **Inference**: Wendet gelernte Policy für lokale Suche an (für Memetic Algorithm)

## Verwendung

### 1. Training

```python
from memetic.local_search.rl_local_search import RLLocalSearch
from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_generators.random_generator import RandomGenerator
from utils.instance_manager import InstanceManager
import random

# Problem-Generator mit InstanceManager
def create_problem_generator(size=100, categories=['lc1', 'lr1']):
    instance_manager = InstanceManager()

    def generator():
        category = random.choice(categories)
        instance_name = random.choice(instance_manager.CATEGORIES[category])
        return instance_manager.load(instance_name, size)

    return generator

# Lösungs-Generator
def create_solution_generator(problem):
    gen = RandomGenerator()
    return gen.generate(problem, n=1)[0]

# Operatoren definieren
operators = [
    ReinsertOperator(max_attempts=1),
    RouteEliminationOperator(),
    SwapBetweenOperator(),
    # ...
]

# RL Local Search initialisieren
rl_local_search = RLLocalSearch(
    operators=operators,
    hidden_dims=[128, 128, 64],
    learning_rate=1e-3,
    gamma=0.99,
    alpha=1.0,      # Fitness-Gewicht
    beta=10.0,      # Feasibility-Gewicht
    acceptance_strategy="greedy",
    device="cuda",
    verbose=True
)

# Training
problem_gen = create_problem_generator(size=100, categories=['lc1', 'lr1'])

history = rl_local_search.train(
    problem_generator=problem_gen,
    initial_solution_generator=create_solution_generator,
    num_episodes=1000,
    new_instance_interval=10,
    new_solution_interval=5,
    save_path="models/rl_ls_100"
)
```

### 2. Inference (Standalone)

```python
# Modell laden
rl_local_search.load("models/rl_ls_final.pt")

# Lokale Suche anwenden
improved_solution, improved_fitness = rl_local_search.search(
    problem=problem,
    solution=initial_solution,
    max_iterations=50,
    epsilon=0.0  # greedy
)
```

### 3. Integration mit Memetic Algorithm

```python
from memetic.memetic_algorithm import MemeticSolver
from utils.instance_manager import InstanceManager

# Problem laden
instance_manager = InstanceManager()
problem = instance_manager.load('lc101', size=100)

# RL Local Search als local_search_operator verwenden
memetic_solver = MemeticSolver(
    population_size=20,
    max_generations=100,
    local_search_operator=rl_local_search,  # <-- RL statt Naive
    verbose=True
)

best_solution = memetic_solver.solve(problem)
```

### 4. Training für mehrere Instanzgrößen

```python
# Trainiere separate Modelle für verschiedene Größen
sizes = [100, 200, 400, 600, 1000]

for size in sizes:
    problem_gen = create_problem_generator(size=size)
    rl_ls = RLLocalSearch(operators=operators, verbose=True)

    rl_ls.train(
        problem_generator=problem_gen,
        initial_solution_generator=create_solution_generator,
        num_episodes=1000,
        save_path=f"models/rl_ls_size{size}"
    )

    rl_ls.save(f"models/rl_ls_size{size}_final.pt")

# Verwendung: Lade Modell für entsprechende Größe
rl_ls = RLLocalSearch(operators=operators)
rl_ls.load("models/rl_ls_size100_final.pt")
```

## Komponenten

### Dateien
- `rl_local_search.py`: Haupt-Klasse
- `local_search_env.py`: Gymnasium Environment
- `dqn_network.py`: DQN Q-Network & Agent
- `replay_buffer.py`: Experience Replay Buffer
- `rl_utils.py`: Feature Extraction & Constraint Violations

### Beispiele
- `examples/train_rl_local_search.py`: Basis-Training für eine Größe
- `examples/train_multi_size_rl.py`: Multi-Size Training (100, 200, 400, ...)
- `examples/memetic_with_rl_local_search.py`: Integration mit Memetic Algorithm

## Hyperparameter

### Training
- `num_episodes`: 500-2000 (je nach Instanzgröße)
- `new_instance_interval`: 10 (neue Instanz alle 10 Episoden)
- `new_solution_interval`: 5 (neue Lösung alle 5 Episoden)
- `warmup_episodes`: 10 (Start Training nach 10 Episoden)

### RL Agent
- `learning_rate`: 1e-3 (Standard)
- `gamma`: 0.99 (Discount Factor)
- `epsilon_start`: 1.0 → `epsilon_end`: 0.1
- `epsilon_decay`: 0.995
- `target_update_interval`: 100 steps
- `batch_size`: 64

### Reward Function
- `alpha`: 1.0-10.0 (Fitness-Gewicht, höher → mehr Fokus auf Distanz-Optimierung)
- `beta`: 1.0-10.0 (Feasibility-Gewicht, höher → mehr Fokus auf Constraint-Erfüllung)
- **Hinweis**: Rewards sind normalisiert durch `problem.distance_baseline`
  - Fitness-Komponente: typisch ~0.001-0.1
  - Feasibility-Komponente: typisch 0-1
  - Empfohlen: alpha=1.0, beta=1.0 als Startpunkt (beide gleichgewichtet)

### Akzeptanzkriterium
- `"greedy"`: Nur bei Verbesserung akzeptieren (empfohlen)
- `"always"`: Immer akzeptieren (mehr Exploration)

## Generalisierung

**Ein Modell pro Instanzgröße**:
- 100 Requests → `rl_ls_size100_final.pt`
- 200 Requests → `rl_ls_size200_final.pt`
- 400 Requests → `rl_ls_size400_final.pt`
- etc.

**Training**: Mix aus verschiedenen Instanzen derselben Größe
```python
# Nutze InstanceManager für diverse Trainings-Instanzen
problem_gen = create_problem_generator(
    size=100,
    categories=['lc1', 'lc2', 'lr1', 'lr2', 'lrc1', 'lrc2']  # Alle Kategorien
)
```

**Evaluation**: Cross-Instance Testing zur Überprüfung der Generalisierung
```python
# Teste auf ungesehenen Instanzen
test_instances = ['lc102', 'lr103', 'lrc201']
instance_manager = InstanceManager()

for inst_name in test_instances:
    problem = instance_manager.load(inst_name, size=100)
    solution, fitness = rl_ls.search(problem, initial_solution)
```

**Beispiel-Script**: `examples/train_multi_size_rl.py` trainiert und evaluiert automatisch

## Erweiterungen (Optional)

### 1. Graph Neural Networks (GNN)
Für strukturelle Information aus Distanzmatrix:
```python
# In dqn_network.py - neue GNN-Architektur hinzufügen
# Graph: Nodes=Locations, Edges=Distances
```

### 2. LSTM/Transformer
Für Routen-Sequenzen:
```python
# Jede Route als Sequence kodieren
# Nutzt temporale Abhängigkeiten
```

### 3. Prioritized Replay Buffer
Bereits implementiert in `replay_buffer.py`:
```python
from memetic.local_search.replay_buffer import PrioritizedReplayBuffer
```

### 4. PPO (Proximal Policy Optimization)
Alternative zu DQN bei unzureichender Performance:
```python
# Besseres Credit Assignment über längere Episoden
# On-policy vs. Off-policy
```

## Evaluation Metriken

Training History enthält:
- `episode_rewards`: Rewards pro Episode
- `episode_lengths`: Schritte pro Episode
- `episode_best_fitness`: Beste Fitness pro Episode
- `losses`: Training Losses
- `epsilon_values`: Exploration Rate über Zeit

```python
import matplotlib.pyplot as plt

# Plot Training Progress
plt.plot(history['episode_best_fitness'])
plt.xlabel('Episode')
plt.ylabel('Best Fitness')
plt.title('RL Local Search Training')
plt.show()
```

## Troubleshooting

### Problem: Training konvergiert nicht
- **Lösung**: Reduzie Learning Rate, erhöhe Batch Size, mehr Warmup Episodes

### Problem: Policy ist zu greedy
- **Lösung**: Erhöhe epsilon_end, reduze epsilon_decay

### Problem: Zu langsam
- **Lösung**: Reduzie max_steps_per_episode, kleineres Netzwerk, GPU verwenden

### Problem: Overfitting auf Trainings-Instanzen
- **Lösung**: Mehr diverse Trainings-Instanzen, reduze new_instance_interval

## Literatur

Basierend auf etablierten Methoden für RL in kombinatorischer Optimierung:
- DQN: Mnih et al. (2015)
- RL for Local Search: Hottung et al. (2020), Wu et al. (2021)
- Operator Selection Learning: Libralesso et al. (2021)
