"""
Represents a placement of fields on a 2D grid of processing elements.
"""
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np
import igraph

from spada.placement.graph import StencilGraph


@dataclass
class Placement:
    """
    A placement is a mapping from fields / vertices of a StencilGraph
    to offsets and strides in the domain of the device.
    each offset and stride is represented as 1x2 arrays of integers.

    To place a stencil graph on our 2-dimensional PE grid, each field $v$ is associated with:
        * An offset O(v)=(O_x(v), O_y(v))
        * A stride I(v)=(I_x(v), I_y(v))
    We place each cell of a field $v$ onto a PE by placing the $(j, k)$-th column
    of v at position O(v) + (j * I_x(v) , k * I_y(v) ).

    # column 0 -> x, column 1 -> y
    """
    offsets: NDArray[np.int32]
    strides: NDArray[np.int32]

    def __post_init__(self):
        assert self.offsets.shape[1] == 2
        assert self.strides.shape[1] == 2
        assert self.offsets.shape[0] == self.strides.shape[0]
        assert np.issubdtype(self.offsets.dtype, np.integer)
        assert np.issubdtype(self.strides.dtype, np.integer)

    def __eq__(self, other):
        return np.array_equal(self.offsets, other.offsets) and np.array_equal(self.strides, other.strides)

    def unique_offsets(self) -> NDArray[np.int32]:
        return np.unique(self.offsets, axis=0)

    def parts(self) -> NDArray[np.int32]:
        unique_offsets = self.unique_offsets()
        # Find for each offset the index of the occurrence in the unique_offsets array
        # This is the partition number
        return np.array([np.where(np.all(unique_offsets == offset, axis=1))[0][0] for offset in self.offsets])

    def edge_crosses_partition(self, e) -> float:
        """
        Calculate the weight of an edge in the depth calculation.
        An edge that crosses a partition has weight 1, an edge that does not cross a partition has weight 0.
        :param e: edge in the graph
        :param placement: Placement
        :return: float   1 if the edge crosses a partition, 0 otherwise
        """
        head = e.source
        tail = e.target
        offset_head = self.offsets[head]
        offset_tail = self.offsets[tail]
        return 1.0 if not np.array_equal(offset_head, offset_tail) else 0.0

    def contention_of_edge(self, stencil_graph: StencilGraph, e: igraph.Edge) -> int:
        """
        The contention of the edge is the number of elements communicated through the edge.
        This involves the height of the z column in the case of horizontal stencils.
        If a part of a stencil has x==y==0, it only contributes to the contention if crosses a partition.
        :param stencil_graph:
        :param e: edge in the graph
        :return:
        """
        stencil_shape_xy = e[StencilGraph.STENCIL].shape[:, :2]

        if self.edge_crosses_partition(e):
            # If the edge crosses the partition, the entire stencil contributes to the communication
            communication_volume = stencil_shape_xy.shape[0]
        else:
            # Note that [0, 0, z] stencils do not cause communication when the edge does not cross the partition
            # Count the number of nonzero elements in the x-y stencil shape
            nonzero_xy = np.sum(np.any(stencil_shape_xy != 0, axis=1), dtype=np.int32)
            communication_volume = nonzero_xy

        # Every edge sends 1 column across each stencil that crosses a partition
        domain = stencil_graph.graph.vs[e.source][StencilGraph.DOMAIN]
        communication_volume *= domain.z_length()

        return communication_volume

    def communication_volume_of_edge(self, stencil_graph: StencilGraph, e: igraph.Edge) -> int:
        """
        The communication volume of the edge is the number of elements communicated through the edge.
        :param stencil_graph:
        :param e:
        :return:
        """
        # the number of elements communicated by each edge is the number of elements in the stencil shape
        # times the volume of the domain
        communication_volume = e[StencilGraph.STENCIL].shape.shape[0]

        domain = stencil_graph.graph.vs[e.source][StencilGraph.DOMAIN]
        communication_volume *= domain.volume()

        return communication_volume

    def distance_vector_of_edge(self, e: igraph.Edge) -> NDArray[np.int64]:
        """
        Calculate the distance vector of an edge for a given placement.
        This is the distance of each element of the stencil shape of the edge.

        For the case where the strides of two fields $u$ and $v$ are the same,
        we can compute the distance of a dependency edge $(u, v)$ as follows:
        Consider the stencil shape
        S(e)= [[delta_x^(0), delta_y^(0), delta_z^(0)], ]
        and let delta^(i) = [delta_x^(0), delta_y^(0)].
        The distance vector Delta(e)^(i) is given by:
        Delta(e)^(i) = O(v)-O(u) - I(u) * delta^(i)

        Note that this formulation works for both horizontal and vertical stencils.
        delta_z does not influence the computation.

        :param placement:
        :param e: edge in the graph
        :return: An array of distances, of size equal to the number of elements in the stencil
        """
        # Calculate the distance of the edge
        # and store it in the result array
        head = e.source
        tail = e.target

        # Get the offsets and strides of the fields
        offset_head = self.offsets[head]
        offset_tail = self.offsets[tail]
        stride_head = self.strides[head]

        # Assert strides of head and tail match
        assert np.array_equal(self.strides[tail], stride_head)

        # Get the stencil shape [(delta_x, delta_y)] of the edge
        delta_xy = e[StencilGraph.STENCIL].shape[:, :2]

        # Calculate the distance vector
        # Note that the offset and stride are 1x2 arrays
        # and the delta is a kx2 array
        # The result is a kx2 array
        # this works by broadcasting the 1x2 arrays to kx2 arrays
        # and then element-wise multiplication
        delta = offset_tail - offset_head - stride_head * delta_xy

        # Now, we compute the Manhattan distance across axis 1
        # To get a 1D array
        distances = np.sum(np.abs(delta), axis=1, dtype=np.int64)
        assert np.all(distances >= 0)
        return distances
