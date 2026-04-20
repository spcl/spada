from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class StencilDirection(Enum):
    """
    Indicates the order in which the z-coordinate of the stencil is traversed
    """
    # The stencil is traversed in parallel
    PARALLEL = 0
    # The stencil is traversed in a forward direction (increasing z)
    FORWARD = 1
    # The stencil is traversed in a backward direction (decreasing z)
    BACKWARD = 2


@dataclass
class Stencil:
    # Represented as a k by 3 array
    # column 0 -> x, column 1 -> y, column 2 -> z
    shape: NDArray[np.int32]
    direction: StencilDirection

    # Check if a stencil is valid
    # A stencil is valid if it is a k times 3 shaped array and all its elements are integers
    def __post_init__(self):
        assert self.shape.shape[1] == 3
        assert self.shape.shape[0] > 0
        assert np.issubdtype(self.shape.dtype, np.integer)

    def __eq__(self, other):
        return np.array_equal(self.shape, other.shape)

    def is_horizontal(self) -> bool:
        """
        A stencil is horizontal if all z values (3rd coordinate) are 0
        :return: bool   True if the stencil is horizontal, False otherwise
        """
        return np.all(self.shape[:, 2] == 0)

    def is_vertical(self) -> bool:
        """
        A stencil is vertical if all x and y values (1st and 2nd coordinates) are 0
        :return: bool   True if the stencil is vertical, False otherwise
        """
        return np.all(self.shape[:, 0:2] == 0)

    def is_3d(self) -> bool:
        """
        A stencil is 3D if it is neither horizontal nor vertical
        That is, it has some non-zero x, y and z values.
        :return:
        """
        return not self.is_horizontal() and not self.is_vertical()
