"""
Represents a partition of the set of fields into disjoint subsets of fields.
These fields are then placed on the same processing element.
"""
from dataclasses import dataclass
from math import log, ceil

from hilbertcurve.hilbertcurve import HilbertCurve
from numpy.typing import NDArray
import numpy as np
import igraph
from typing import Tuple

from spada.placement.graph import FieldDomain
from spada.placement.mla import linearize_with_random_forest
from spada.placement.placement import Placement



@dataclass
class FieldPartition:
    """
    Represents a partition of the set of fields into disjoint subsets of fields with a spatial preference.
    These fields are meant to placed on the same sets of processing elements.
    """

    # n == number of fields
    # n x 2 array
    # where there is a row for each field
    # and two integers denoting the (x, y) position of the partition it belongs to
    # the number of partitions is implicit in the number of distinct (x, y) values
    # column 0 -> x, column 1 -> y
    part: NDArray[np.int32]

    def __post_init__(self):
        # assert integer type
        assert np.issubdtype(self.part.dtype, np.integer)
        # assert shape
        assert self.part.shape[1] == 2
        assert self.part.shape[0] > 0

    def place_interleaved(self) -> Placement:
        """
        In the interleaving placement strategy, the partitions of fields are placed at small offsets to each other in
        an interleaved fashion. The advantage of this approach is that none of the distances are very large. However,
        the average distance is increased for all stencils, even those horizontal stencils that do not cross the
        partition.

        A vertex at position (x, y) in the partition gets the offset (x, y) in the placement
        the strides are given by creating a rectangle around the (x, y) coordinates
        :return:
        """
        def stride(a: np.ndarray) -> np.ndarray:
            return np.abs(np.max(a) - np.min(a))

        stride_x = stride(self.part[:, 0])
        stride_y = stride(self.part[:, 1])

        offsets = self.part.copy()
        strides = np.zeros_like(offsets)
        strides[:, 0] = stride_x + 1
        strides[:, 1] = stride_y + 1
        return Placement(offsets=offsets, strides=strides)

    def place_blocked(self, domain: FieldDomain) -> Placement:
        """
        In the blocked placement, each partition is assigned to a disjoint area of the chip with stride 1.
        This means that all stencils that do not cross a partition can be executed optimally.
        However, stencils that cross the partition may travel a large distance proportional to the domain side length.

        For example, consider 2 partitions, 2x4 8 PEs and a 2D domain of size 2x2
        Then, we will have the following placement
         0 0 1 1
         0 0 1 1
        For 4x2 PEs, we get the following placement
         0 0
         0 0
         1 1
         1 1
        :return:
        """
        # TODO Assumes all domains are the same
        # If domains are not the same, we might need to broadcast and this affects placement...
        # What if it's a single column? -> then we broadcast it so it's 3D again.
        x_offset_multiplier = domain.x_length()
        y_offset_multiplier = domain.y_length()
        strides = np.ones_like(self.part)
        offsets = self.part.copy() * np.array([x_offset_multiplier, y_offset_multiplier])
        return Placement(offsets=offsets, strides=strides)


    @staticmethod
    def from_mla(g: igraph.Graph,
                 partitions_shape: Tuple[int, int],
                 mla_func=linearize_with_random_forest) -> 'FieldPartition':
        """
        Partitions the fields of the graph using a minimum linear arrangement of the graph.
        1) Compute a minimum linear arrangement of the graph.
        2) Partition the fields of the graph using the minimum linear arrangement, assigning consecutive fields to the same partition.
        Here, each partition is identified with a single integer. There are partitions_shape[0] * partitions_shape[1] partitions.
        3) Map the partition numbers to the 2D grid of processing elements using a hilbert curve.
        :param g:
        :param partitions_shape:
        :param mla_func: function that takes a graph and a list and returns a minimum linear arrangement of the graph
        :return:
        """

        num_parts = partitions_shape[0] * partitions_shape[1]

        if num_parts > 1:

            # 1)
            order = []
            g.vs["original_id"] = [i for i in range(g.vcount())]
            mla_func(g, order, base_size=2)
            order_array = np.array(order, dtype=np.int32)
            # 2)

            partition_id = np.zeros(g.vcount(), dtype=np.int32)
            # We would like to assign the number of vertices as equally as possible among the partitions
            # to deal with rounding errors, some partitions may have one more vertex than others
            # this will be the first k partitions (where k is the remainder of the division)
            # TODO this assumes equal storage for each field
            # TODO In particular merge versions of fields
            fields_per_partition = g.vcount() // num_parts
            remainer = g.vcount() % num_parts

            # For example, if we have 6 fields and 3 partitions
            # with order_array = [5, 4, 3, 2, 1, 0]
            # we get partition_id = [2, 2, 1, 1, 0, 0]
            # if order_array = [5, 1, 2, 3, 4, 0]
            # we get partition_id = [2, 0, 1, 1, 1, 0]
            # Now, an example with 7 fields and 3 partitions
            # with order_array = [6, 5, 4, 3, 2, 1, 0]
            # we get partition_id = [2, 2, 1, 1, 0, 0, 0]

            partition_id[order_array] = np.concatenate([np.full(fields_per_partition + 1, i) if i < remainer else np.full(fields_per_partition, i) for i in range(num_parts)])
            assert np.all(partition_id < num_parts)
            assert np.all(partition_id >= 0)
            assert np.max(partition_id) == num_parts - 1
            assert np.min(partition_id) == 0

            # 3)
            hilbert_p = int(ceil(log(num_parts, 4)))
            curve = HilbertCurve(hilbert_p, 2).points_from_distances(np.arange(4 ** hilbert_p))
            curve_arr = np.asarray(curve, dtype=np.int32)
            # choose those rows where x < partitions_shape[0] and y < partitions_shape[1]
            curve_arr = curve_arr[curve_arr[:, 0] < partitions_shape[0]]
            curve_arr = curve_arr[curve_arr[:, 1] < partitions_shape[1]]

            assert np.all(curve_arr[:, 0] < partitions_shape[0])
            assert np.all(curve_arr[:, 1] < partitions_shape[1])
            assert np.all(curve_arr >= 0)
            assert np.max(curve_arr[:, 0]) == partitions_shape[0] - 1
            assert np.max(curve_arr[:, 1]) == partitions_shape[1] - 1
            # hilbert property
            assert (np.abs(np.diff(curve_arr[:, 0])) + np.abs(np.diff(curve_arr[:, 1]))).max() <= 1

            partition = FieldPartition(part=curve_arr[partition_id])
        else:
            partition = FieldPartition(part=np.zeros((g.vcount(), 2), dtype=np.int32))

        return partition

