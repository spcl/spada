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
    'wse3': list(range(8, 26)),
}

LOCAL_TASK_IDS = [t for t in _CSL_LOCAL_TASK_IDS[ARCH] if t not in RESERVED_LOCAL_TASK_IDS]

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

# Fabric queue IDs available for application communication.
# See https://sdk.cerebras.ai/csl/language/dsds#fabric-queues
_INPUT_QUEUE_IDS = {
    'wse2': list(range(0, 2)),  # Queues 0-1 provide full capacity on WSE-2
    'wse3': list(range(2, 8)),  # Queues 2-7; queues 0-1 are reserved by memcpy
}
INPUT_QUEUE_IDS = _INPUT_QUEUE_IDS[ARCH]

_OUTPUT_QUEUE_IDS = {
    'wse2': list(range(2, 4)),  # Queues 2-3 provide full capacity on WSE-2
    'wse3': list(range(2, 8)),  # Queues 2-7; queues 0-1 are reserved by memcpy
}
OUTPUT_QUEUE_IDS = _OUTPUT_QUEUE_IDS[ARCH]

# Hardware microthread IDs for asynchronous DSD operations.
# On WSE-2, the microthread ID is implicitly tied to the queue ID of the highest-priority
# fabric operand. On WSE-3, microthreads 2-7 can be explicitly assigned via the `.ut_id`
# DSD field (queues 0-1 and their corresponding microthreads are reserved by memcpy).
# See https://sdk.cerebras.ai/csl/language/microthreads_wse3
_MICROTHREAD_IDS = {
    'wse2': [],
    'wse3': list(range(2, 8)),
}
MICROTHREAD_IDS = _MICROTHREAD_IDS[ARCH]

_HARDWARE_FABRIC_DIMS = {
    'wse2': (757, 996),
    'wse3': (762, 1172),
}
HARDWARE_FABRIC_DIMS = _HARDWARE_FABRIC_DIMS[ARCH]

# Router switches. Each router holds one base route configuration (``.routes``) plus up to three
# switch positions (``.pos1``, ``.pos2``, ``.pos3``) per color.
# See https://sdk.cerebras.ai/csl/language/builtins#switching-configuration-semantics
SWITCH_POSITIONS = 4

# Number of switching command slots in a CSL control wavelet (<control>'s MAX_CMDS).
# Hardware routers execute command slot 0 across all traversed switch-configured routers;
# remaining slots are ignored. Control messages therefore advance all routers along their
# path uniformly, using encode_single_payload.
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

# Maximum hardware counter filters configurable per PE across all colors.
# While the hardware provides four filters per PE, the memcpy runtime module reserves one,
# leaving three available for application routing.
# Hardware filters cannot be safely reconfigured while traffic is active, so they are
# initialized at layout time.
FILTERS_PER_PE = 3
