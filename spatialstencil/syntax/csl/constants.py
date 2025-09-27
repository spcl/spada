"""
CSL hardware-related constant values.
"""
import os

# Cerebras architecture to use. Options: 'wse2', 'wse3'
ARCH = os.environ.get('WSE_ARCH', 'wse2')

_CSL_DATA_TASK_IDS = {
    'wse2': list(range(0, 24)),
    'wse3': list(range(0, 8)),
}
DATA_TASK_IDS = _CSL_DATA_TASK_IDS[ARCH]

# From the SDK: IDs 29 and 30 should generally be avoided in programs as they are used for system tasks.
# https://sdk.cerebras.net/csl/language/task-ids?highlight=color#activatable-identifiers
# NOTE: We also avoid task ID 28 as we reserve it for ``exit_task``
_CSL_LOCAL_TASK_IDS = {
    'wse2': list(range(10, 28)),
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

_MEMCPY_COLORS = {
    'wse2': list(range(21, 24)) + list(range(27, 32)),
    'wse3': list(range(21, 24)) + list(range(27, 32)),
}
MEMCPY_COLORS = _MEMCPY_COLORS[ARCH]
