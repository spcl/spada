"""
This file contains type hints and stubs for the Cerebras SdkRuntime class and related enums.
"""
from typing import Union
from enum import Enum
import pathlib
import numpy

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class MemcpyDataType(Enum):
    MEMCPY_32BIT = 0
    MEMCPY_16BIT = 1


class MemcpyOrder(Enum):
    ROW_MAJOR = 0
    COL_MAJOR = 1


class Task:
    ...


class SdkRuntime:

    def __init__(self,
                 bindir: Union[pathlib.Path, str],
                 *,
                 cmaddr: str = None,
                 suppress_simfab_trace: bool = False,
                 simfab_numthreads: int = 5,
                 msg_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'WARNING') -> None:
        ...

    def is_task_done(self, task_handle: Task) -> bool:
        ...

    def launch(self, symbol: str, *args, nonblock: bool) -> Task:
        ...

    def load(self) -> None:
        ...

    def get_id(self, symbol: str) -> int:
        ...

    def memcpy_d2h(self, dest: numpy.ndarray, src: int, px: int, py: int, w: int, h: int, elem_per_pe: int, *,
                   streaming: bool, data_type: MemcpyDataType, order: MemcpyOrder, nonblock: bool) -> Task:
        ...

    def memcpy_h2d(self, dest: int, src: numpy.ndarray, px: int, py: int, w: int, h: int, elem_per_pe: int, *,
                   streaming: bool, data_type: MemcpyDataType, order: MemcpyOrder, nonblock: bool) -> Task:
        ...

    def run(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def task_wait(task_handle: Task) -> None:
        ...


class SdkTarget(Enum):
    WSE2 = 0
    WSE3 = 1
