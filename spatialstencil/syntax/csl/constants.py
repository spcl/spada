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
    'wse2': list(range(8, 28)),
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
