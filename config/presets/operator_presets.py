"""Predefined operator configurations."""

# Standard preset - balanced set of operators (matches current create_operators())
STANDARD_PRESET = [
    {'type': 'ReinsertOperator', 'params': {}},
    {'type': 'ReinsertOperator', 'params': {'max_attempts': 5, 'clustered': True}},
    {'type': 'ReinsertOperator', 'params': {'force_same_vehicle': True}},
    {'type': 'ReinsertOperator', 'params': {'allow_same_vehicle': False}},
    {'type': 'ReinsertOperator', 'params': {'allow_same_vehicle': False, 'allow_new_vehicles': False}},
    {'type': 'RouteEliminationOperator', 'params': {}},
    {'type': 'SwapWithinOperator', 'params': {}},
    {'type': 'SwapBetweenOperator', 'params': {}},
    {'type': 'TransferOperator', 'params': {}},
    {'type': 'ShiftOperator', 'params': {'type': 'random', 'segment_length': 3, 'max_shift_distance': 5, 'max_attempts': 3}},
    {'type': 'TwoOptOperator', 'params': {}},
]

# Aggressive preset - more exploration, higher attempts
AGGRESSIVE_PRESET = [
    {'type': 'ReinsertOperator', 'params': {}},
    {'type': 'ReinsertOperator', 'params': {'max_attempts': 10, 'clustered': True}},
    {'type': 'ReinsertOperator', 'params': {'force_same_vehicle': True}},
    {'type': 'ReinsertOperator', 'params': {'allow_same_vehicle': False}},
    {'type': 'RouteEliminationOperator', 'params': {}},
    {'type': 'SwapWithinOperator', 'params': {'max_attempts': 5}},
    {'type': 'SwapBetweenOperator', 'params': {'type': 'best'}},
    {'type': 'TransferOperator', 'params': {'max_attempts': 5}},
    {'type': 'ShiftOperator', 'params': {'type': 'random', 'segment_length': 4, 'max_shift_distance': 8, 'max_attempts': 5}},
    {'type': 'ShiftOperator', 'params': {'type': 'best', 'segment_length': 3, 'max_shift_distance': 4}},
    {'type': 'TwoOptOperator', 'params': {}},
    {'type': 'FlipOperator', 'params': {}},
]

# Minimal preset - fast, essential operators only
MINIMAL_PRESET = [
    {'type': 'ReinsertOperator', 'params': {}},
    {'type': 'SwapWithinOperator', 'params': {}},
    {'type': 'SwapBetweenOperator', 'params': {}},
    {'type': 'TwoOptOperator', 'params': {}},
]

# CLS preset - Chain of Local Search operators
CLS_PRESET = [
    {'type': 'ReinsertOperator', 'params': {}},
    {'type': 'RouteEliminationOperator', 'params': {}},
    {'type': 'CLSM1Operator', 'params': {}},
    {'type': 'CLSM2Operator', 'params': {}},
    {'type': 'CLSM3Operator', 'params': {}},
    {'type': 'CLSM4Operator', 'params': {}},
]

OPERATOR_PRESETS = {
    'standard': STANDARD_PRESET,
    'aggressive': AGGRESSIVE_PRESET,
    'minimal': MINIMAL_PRESET,
    'cls': CLS_PRESET,
}
