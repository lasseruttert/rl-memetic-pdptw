"""Factory for creating operators from configuration."""

from typing import List, Dict, Any
from memetic.solution_operators.base_operator import BaseOperator
from config.presets.operator_presets import OPERATOR_PRESETS

# Import all operators
from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.merge import MergeOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator
from memetic.solution_operators.shift import ShiftOperator
from memetic.solution_operators.two_opt import TwoOptOperator
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator
from memetic.solution_operators.request_shift_within import RequestShiftWithinOperator
from memetic.solution_operators.node_swap_within import NodeSwapWithinOperator

# Operator registry mapping type names to classes
OPERATOR_REGISTRY = {
    'ReinsertOperator': ReinsertOperator,
    'RouteEliminationOperator': RouteEliminationOperator,
    'SwapWithinOperator': SwapWithinOperator,
    'SwapBetweenOperator': SwapBetweenOperator,
    'TransferOperator': TransferOperator,
    'ShiftOperator': ShiftOperator,
    'TwoOptOperator': TwoOptOperator,
    'FlipOperator': FlipOperator,
    'MergeOperator': MergeOperator,
    'CLSM1Operator': CLSM1Operator,
    'CLSM2Operator': CLSM2Operator,
    'CLSM3Operator': CLSM3Operator,
    'CLSM4Operator': CLSM4Operator,
    'RequestShiftWithinOperator': RequestShiftWithinOperator,
    'NodeSwapWithinOperator': NodeSwapWithinOperator,
}


def create_operators_from_config(operator_config: Dict[str, Any]) -> List[BaseOperator]:
    """Create operators from configuration.

    Args:
        operator_config: Operator section from config file

    Returns:
        List of operator instances

    Raises:
        ValueError: If preset or operator type is unknown
    """
    preset = operator_config.get('preset', 'standard')

    if preset == 'custom':
        # Use custom operator list
        custom_ops = operator_config.get('custom', [])
        if not custom_ops:
            raise ValueError("Custom operator preset requires 'custom' list with at least one operator")
        return _create_custom_operators(custom_ops)
    else:
        # Use preset
        if preset not in OPERATOR_PRESETS:
            raise ValueError(f"Unknown operator preset: '{preset}'. Available: {list(OPERATOR_PRESETS.keys())}")

        preset_ops = OPERATOR_PRESETS[preset]
        return _create_custom_operators(preset_ops)


def _create_custom_operators(operator_specs: List[Dict[str, Any]]) -> List[BaseOperator]:
    """Create operators from specifications.

    Args:
        operator_specs: List of operator specifications with 'type' and 'params'

    Returns:
        List of operator instances

    Raises:
        ValueError: If operator type is unknown
    """
    operators = []

    for spec in operator_specs:
        op_type = spec.get('type')
        op_params = spec.get('params', {})

        if not op_type:
            raise ValueError(f"Operator specification missing 'type': {spec}")

        if op_type not in OPERATOR_REGISTRY:
            raise ValueError(f"Unknown operator type: '{op_type}'. Available: {list(OPERATOR_REGISTRY.keys())}")

        operator_class = OPERATOR_REGISTRY[op_type]
        try:
            operator = operator_class(**op_params)
            operators.append(operator)
        except TypeError as e:
            raise ValueError(f"Invalid parameters for {op_type}: {op_params}. Error: {e}")

    return operators
