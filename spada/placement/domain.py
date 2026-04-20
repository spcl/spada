from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FieldDomain:
    """
    Represents the 3D domain of a field

    Represented as a 2 by 3 array
    column 0 -> x, column 1 -> y, column 2 -> z
    The first row is the lower bound of the domain
    The second row is the upper bound of the domain
    """
    domain: NDArray[np.int32]

    def x(self) -> NDArray[np.int32]:
        """
        Returns the x coordinate upper and lower bounds of the domain
        :return:
        """
        return self.domain[:, 0]

    def y(self) -> NDArray[np.int32]:
        """
        Returns the y coordinate upper and lower bounds of the domain
        :return:
        """
        return self.domain[:, 1]

    def z(self) -> NDArray[np.int32]:
        """
        Returns the z coordinate upper and lower bounds of the domain
        :return:
        """
        return self.domain[:, 2]

    # Check if a domain is valid
    # A domain is valid if it is a 2x3 shaped array and all its elements are integers
    # column 0 -> x, column 1 -> y, column 2 -> z
    def __post_init__(self):
        assert self.domain.shape == (2, 3)
        assert np.issubdtype(self.domain.dtype, np.integer)
        # Check that the lower bound is less than or equal to the upper bound
        assert np.all(self.domain[0] <= self.domain[1])

    def __eq__(self, other):
        return np.array_equal(self.domain, other.domain)

    def x_length(self) -> int:
        return self.domain[1][0] - self.domain[0][0]

    def y_length(self) -> int:
        return self.domain[1][1] - self.domain[0][1]

    def z_length(self) -> int:
        return self.domain[1][2] - self.domain[0][2]

    def xy_plane_area(self) -> int:
        return self.x_length() * self.y_length()

    def volume(self) -> int:
        return self.z_length() * self.xy_plane_area()

