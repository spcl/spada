"""
CSL hardware-related constant values.
"""
import os

# Cerebras architecture to use. Options: 'wse2', 'wse3'
ARCH = os.environ.get('WSE_ARCH', 'wse2')

# Activatable local-task IDs: 0–30 on WSE-2, 8–30 on WSE-3. 29 is the teardown
# handler and 30 is the timer; memcpy also binds several of these as local tasks
# (``sys_params.csl``: SYS_EN_MAIN=24, SYS_UBLK_C22=27, SYS_EXIT=28,
# SYS_SEND_CTRL=30). On WSE-3, ``memcpyd2h.csl`` additionally aliases color 21 as
# ``LOCAL_MEMCPYD2H_DATA`` to save an entrypoint, which is the collision cslc
# reports as "task ID '21' bound to more than one task".
# https://sdk.cerebras.ai/csl/language/task-ids
# https://sdk.cerebras.ai/tensor-streaming
_RESERVED_LOCAL_TASK_IDS = {
    'wse2': [24, 27, 28, 29, 30],
    'wse3': [21, 24, 27, 28, 29, 30],
}
RESERVED_LOCAL_TASK_IDS = _RESERVED_LOCAL_TASK_IDS[ARCH]

# Program-assignable local-task IDs. WSE-3 skips the memcpy holes; ``exit_task``
# is not in this list and takes the next free ID after the assigned slots.
_CSL_LOCAL_TASK_IDS = {
    'wse2': list(range(8, 21)),
    'wse3': [t for t in range(8, 26) if t not in _RESERVED_LOCAL_TASK_IDS['wse3']],
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
    # On WSE-3 a data task's ID *is* its input queue, and memcpy takes 0 and 1 for its own; binding
    # either of them with ``@initialize_queue`` is rejected as "already been set".
    'wse3': list(range(2, 8)),
}
INPUT_QUEUE_IDS = _INPUT_QUEUE_IDS[ARCH]

_OUTPUT_QUEUE_IDS = {
    'wse2': list(range(2, 4)),  # Ignoring 0-1,4-5 as they are smaller in capacity
    'wse3': list(range(2, 8)),  # All queues are equivalent, but memcpy reserves 0 and 1
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

# Router filters usable per PE, over all colors. The hardware has four, but the memcpy module
# reserves one, so a program may configure three (Schnyder, "Distributed Sorting on the Cerebras
# Wafer-Scale Engine", ch. 7).
#
# NOTE: a filter must not be reconfigured while wavelets it counts are still in flight. Filters are
# therefore set once in the layout and never rewritten between phases.
FILTERS_PER_PE = 3
