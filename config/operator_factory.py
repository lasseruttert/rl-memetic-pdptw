"""Operator factory for dynamic operator instantiation from configuration.

This module provides utilities to create operator instances from YAML
configuration specifications, with validation and error handling.
"""

import inspect
from typing import List, Dict, Any

# Import all operator classes
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

from .config_schema import OperatorSpec


# Operator class registry: maps operator type strings to operator classes
OPERATOR_REGISTRY = {
    'Reinsert': ReinsertOperator,
    'RouteElimination': RouteEliminationOperator,
    'Flip': FlipOperator,
    'SwapWithin': SwapWithinOperator,
    'SwapBetween': SwapBetweenOperator,
    'Transfer': TransferOperator,
    'Shift': ShiftOperator,
    'CLSM1': CLSM1Operator,
    'CLSM2': CLSM2Operator,
    'CLSM3': CLSM3Operator,
    'CLSM4': CLSM4Operator,
    'TwoOpt': TwoOptOperator,
    'RequestShiftWithin': RequestShiftWithinOperator,
    'NodeSwapWithin': NodeSwapWithinOperator,
}


def create_operator(operator_type: str, params: Dict[str, Any]):
    """Create operator instance from type and parameters.

    Args:
        operator_type: Operator class name (e.g., "Reinsert", "Shift")
        params: Dictionary of constructor parameters

    Returns:
        Instantiated operator

    Raises:
        ValueError: If operator type is unknown
        TypeError: If parameters are invalid for operator
    """
    if operator_type not in OPERATOR_REGISTRY:
        available = ', '.join(sorted(OPERATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown operator type: '{operator_type}'. "
            f"Available operators: {available}"
        )

    operator_class = OPERATOR_REGISTRY[operator_type]

    try:
        return operator_class(**params)
    except TypeError as e:
        # Get valid parameters for better error message
        sig = inspect.signature(operator_class.__init__)
        valid_params = [
            p for p in sig.parameters.keys()
            if p != 'self'
        ]
        raise TypeError(
            f"Invalid parameters for {operator_type}.\n"
            f"  Provided: {list(params.keys())}\n"
            f"  Valid parameters: {valid_params}\n"
            f"  Error: {e}"
        )


def create_operators_from_config(operator_specs: List[OperatorSpec]) -> List:
    """Create list of operators from configuration specifications.

    Args:
        operator_specs: List of OperatorSpec dataclass instances

    Returns:
        List of instantiated operators

    Raises:
        ValueError: If operator specification is invalid
        TypeError: If operator parameters are invalid
    """
    operators = []
    for i, spec in enumerate(operator_specs):
        try:
            op = create_operator(spec.type, spec.params)
            operators.append(op)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Error creating operator {i} (type='{spec.type}'): {e}"
            ) from e

    return operators


def validate_operator_params(operator_type: str, params: Dict[str, Any]) -> List[str]:
    """Validate parameters for an operator type.

    Uses introspection to check if provided parameters are valid for the
    operator's __init__ method.

    Args:
        operator_type: Operator class name
        params: Parameter dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    if operator_type not in OPERATOR_REGISTRY:
        return [f"Unknown operator type: '{operator_type}'"]

    errors = []
    operator_class = OPERATOR_REGISTRY[operator_type]

    # Get __init__ signature
    sig = inspect.signature(operator_class.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}

    # Check for unknown parameters
    for param_name in params.keys():
        if param_name not in valid_params:
            errors.append(
                f"Unknown parameter '{param_name}' for {operator_type}. "
                f"Valid parameters: {', '.join(sorted(valid_params))}"
            )

    return errors


def get_operator_info(operator_type: str) -> Dict[str, Any]:
    """Get information about an operator type.

    Args:
        operator_type: Operator class name

    Returns:
        Dictionary with operator information:
        - 'name': operator type string
        - 'class': operator class
        - 'parameters': list of parameter names
        - 'signature': full signature string

    Raises:
        ValueError: If operator type is unknown
    """
    if operator_type not in OPERATOR_REGISTRY:
        available = ', '.join(sorted(OPERATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown operator type: '{operator_type}'. "
            f"Available operators: {available}"
        )

    operator_class = OPERATOR_REGISTRY[operator_type]
    sig = inspect.signature(operator_class.__init__)
    params = [p for p in sig.parameters.keys() if p != 'self']

    return {
        'name': operator_type,
        'class': operator_class,
        'parameters': params,
        'signature': str(sig),
    }
