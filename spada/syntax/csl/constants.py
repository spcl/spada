"""
CSL hardware-related constant values.
"""
import os

# Cerebras architecture to use. Options: 'wse2', 'wse3'
ARCH = os.environ.get('WSE_ARCH', 'wse2')

# From the SDK: IDs 29 and 30 should generally be avoided in programs as they are used for system tasks.
# https://sdk.cerebras.net/csl/language/task-ids?highlight=color#activatable-identifiers
# NOTE: We also avoid task ID 28 as we reserve it for ``exit_task``
_CSL_LOCAL_TASK_IDS = {
    'wse2': list(range(8, 21)),
    'wse3': list(range(8, 28)),
}

LOCAL_TASK_IDS = _CSL_LOCAL_TASK_IDS[ARCH]

_CSL_CONTROL_TASK_IDS = {
    'wse2': list(range(0, 64)),
    'wse3': list(range(0, 64)),
}
CONTROL_TASK_IDS = _CSL_CONTROL_TASK_IDS[ARCH]

_CSL_COLORS = {
    'wse2': list(range(0, 21)),  # 21-23(,27-31) reserved by memcpy
    'wse3': list(range(0, 21)),
}
COLORS = _CSL_COLORS[ARCH]
DATA_TASK_IDS = _CSL_COLORS[ARCH]

_MEMCPY_COLORS = {
    'wse2': list(range(21, 24)) + list(range(27, 32)),
    'wse3': list(range(21, 24)) + list(range(27, 32)),
}
MEMCPY_COLORS = _MEMCPY_COLORS[ARCH]

# See https://sdk.cerebras.net/csl/language/dsds#fabric-queues
_INPUT_QUEUE_IDS = {
    'wse2': list(range(0, 2)),  # Ignoring 2-7 as they are smaller in capacity
    'wse3': list(range(0, 8)),  # 0 is better than 1-7
}
INPUT_QUEUE_IDS = _INPUT_QUEUE_IDS[ARCH]

_OUTPUT_QUEUE_IDS = {
    'wse2': list(range(2, 4)),  # Ignoring 0-1,4-5 as they are smaller in capacity
    'wse3': list(range(0, 8)),  # All queues are equivalent
}
OUTPUT_QUEUE_IDS = _OUTPUT_QUEUE_IDS[ARCH]

_HARDWARE_FABRIC_DIMS = {
    'wse2': (757, 996),
    'wse3': (762, 1172),
}
HARDWARE_FABRIC_DIMS = _HARDWARE_FABRIC_DIMS[ARCH]

# Router switches. Each router holds one base route configuration (``.routes``) plus up to three
# switch positions (``.pos1``, ``.pos2``, ``.pos3``) per color.
# See https://sdk.cerebras.ai/csl/language/builtins#switching-configuration-semantics
SWITCH_POSITIONS = 4

# Number of switching command slots a control wavelet carries (``<control>``'s MAX_CMDS).
#
# NOTE: only slot 0 is ever executed. Measured on the simulator, every switch-configured router a
# wavelet reaches applies the command in slot 0; slots 1-7 had no effect in any topology tested
# (the sender's own router, one hop, two hops through a plain relay, and two switch-configured
# routers in sequence). A wavelet therefore cannot advance one router while skipping another on its
# path, which is why ``routing.plan_switch_advances`` requires the routers along a path to agree.
# ``<control>``'s ``encode_payload`` also loops over all eight slots regardless of the array length
# it is given, so it must be passed exactly eight; ``encode_single_payload`` writes slot 0 only and
# is what the compiler emits.
MAX_CONTROL_COMMANDS = 8

# Colors whose routers support switches. WSE-3 only implements switches on a subset of colors.
_SWITCHABLE_COLORS = {
    'wse2': list(range(0, 21)),
    'wse3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 20],
}
SWITCHABLE_COLORS = [color for color in _SWITCHABLE_COLORS[ARCH] if color in COLORS]

# Whether one switch position may change both the receiving and the transmitting direction.
# WSE-2 rejects it ("cannot have both an input and an output in the same switch position"), so a PE
# that receives and then sends on one color cannot be expressed there with a single advance.
_SWITCH_POSITION_ALLOWS_BOTH = {'wse2': False, 'wse3': True}
SWITCH_POSITION_ALLOWS_BOTH = _SWITCH_POSITION_ALLOWS_BOTH[ARCH]
