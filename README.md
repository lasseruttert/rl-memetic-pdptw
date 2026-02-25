# RL-Memetic-PDPTW
This repository contains the implementation of a memetic algorithm for solving the Pickup and Delivery Problem with Time Windows (PDPTW) using a reinforcement learning guided local search and was developed as part of a bachelor thesis with the title "Reinforcement Learning Guided Memetic Algorithm for the Pickup and Delivery Problem with Time Windows". 

Currently the codebase encompasses the implementation, but also the full Li & Lim and Mendeley datasets, all of my trained models, results and plots. These files (/bks, /data, /models, /results, /logs, /runs) can be safely deleted, if you dont need them.

## Installation
Run the following command to install the required dependencies:

```bash
conda env create -f environment.yml
```

Build the C++ extensions:
```bash
python setup_cpp.py build_ext --inplace --force
```

## Usage
To solve a PDPTW instance, first load the instance:
```python
from utils.li_lim_reader import li_lim_reader
from utils.mendeley_reader import mendeley_reader

instance = li_lim_reader('path_to_instance')
# or
instance = mendeley_reader('path_to_instance')
```
Then, you can solve the instance with the memetic algorithm:
```python
from memetic.memetic_algorithm import MemeticSolver
solver = MemeticSolver()
solution = solver.solve(instance)
```


## Training
To train a local search model, run the following command:
```bash
python train_rl_local_search.py --config <path_to_config>
```
Afterwards, you can find the trained model in the /models folder. You can also find the training logs in the /logs folder and the training runs in the /runs folder.


## Experiments
These experiments can be run simply by running their respective scripts, as long as you have the trained models in the /models folder. 
- `python experiments/experiment_feature_ablation.py` -
- `python experiments/experiment_state_archetype_action_distributions.py` -
- `python experiments/experiment_operator_convergence.py` -
- `python experiments/experiment_memetic_component_wise_performance.py` -
- `python experiments/experiment_memetic_component_wise_performance_200.py` -
- `python experiments/experiment_memetic_component_wise_performance_400.py` -
- `python experiments/experiment_memetic_vs_ortools.py` -

This experiments needs a config file, which can be found in the /configs folder. You can run it with the following command:
```bash
python experiments/experiment_rl_local_search_performance.py --config <path_to_config>
```