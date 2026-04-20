"""
Defines placements and the cost model of a placement in the spatial model with contention (SMC)
"""
from dataclasses import dataclass
from typing import Sequence

import igraph
import numpy as np
from numpy.typing import NDArray

from spada.placement.graph import StencilGraph
from spada.placement.placement import Placement


@dataclass
class PlacementCost:
    """
    The cost of a placement in the SMC
    """
    # The contention of the placement
    contention: int
    # The energy term of the placement
    energy_over_links: float
    # The distance of the placement
    distance: int
    # The depth of the placement
    depth: int

    # The overall cost of the placement, which is the
    # (maximum of contention, energy_over_links + distance) + (2 * RAMP_TIME + 1) * depth
    overall: float


class CostModel:
    stencil_graph: StencilGraph

    RAMP_TIME: float = 2.0

    def __init__(self, stencil_graph):
        self.stencil_graph = stencil_graph

    def cost(self, placement: Placement) -> PlacementCost:
        """
        Calculate the cost of a placement using the spatial cost model (with contention)
        :param placement: Placement
        :return: float
        """
        number_of_links = self.number_of_links_of_placement(placement)

        contention = self.contention_of_placement(placement)
        energy_over_links = self.energy_of_placement(placement) / number_of_links
        distance = self.distance_of_placement(self.edge_distance_of_placement(placement))
        depth = self.depth_of_placement(placement)
        overall = max(float(contention), energy_over_links + distance) + (2 * self.RAMP_TIME + 1) * depth

        return PlacementCost(contention,
                             energy_over_links,
                             distance,
                             depth,
                             overall)

    def number_of_links_of_placement(self, placement: Placement) -> int:
        """
        Calculate the number of communication links of a placement
        Note that this is an upper bound that ignores the border of the domain
        and ignores the communication pattern being used.
        :param placement: Placement
        :return: float
        """
        domain = self.stencil_graph.domain()
        number_of_partitions = placement.unique_offsets().shape[0]
        grid_size = domain.xy_plane_area() * number_of_partitions

        # Start with the number of PEs as number of links
        number_of_links = grid_size

        if domain.x_length() > 1 and domain.y_length() > 1:
            # If the domain is larger than kx1 or 1xk, there are 4 neighbors per grid point
            number_of_links *= 4
        else:
            # If the domain is kx1 or 1xk, there are 2 neighbors per grid point
            number_of_links *= 2

        # Add bi-directional links
        number_of_links *= 2

        return number_of_links

    def edge_distance_of_placement(self, placement: Placement) -> NDArray[np.int64]:
        """
        Calculate the distance of each edge for a given placement.
        :param placement: Placement
        :return: for each edge, its distance under the placement
        """
        # Iterate over all edges in order
        # For each edge, calculate the distance
        # Return the distances as a numpy array
        distances = np.zeros(len(self.stencil_graph.edges()), dtype=np.int64)
        for e in self.stencil_graph.edges():
            delta = placement.distance_vector_of_edge(e)
            # Calculate the l1 norm of each row
            # then take the maximum across the columns
            # to get the maximum distance of the edge
            distances[e.index] = np.max(delta)

        return distances

    def distance_of_placement(self, distances: NDArray[np.integer]) -> int:
        """
        Calculate the maximum distance of a placement (longest path in the graph, weighted by distances)

        The longest path in the graph weighted by distance is the maximum distance of the placed stencil program.
        :param distances:
        :return:
        """
        # uses the linear-time dynamic programming approach to finding the longest path in a DAG
        # assert edge weights have correct size
        assert len(distances) == len(self.stencil_graph.edges())
        # get the topological order of the graph
        #  If "in", all vertices come before their ancestors.
        top_order = self.stencil_graph.graph.topological_sorting(mode="IN")

        # initialize the distance array
        max_distance: NDArray[np.int64] = np.zeros(len(self.stencil_graph.graph.vs), dtype=np.int64)

        for v in top_order:
            # get the maximum distance of the incoming edges
            # and add the distance of the current edge
            for e in self.stencil_graph.graph.vs[v].out_edges():
                max_distance[v] = max(max_distance[v], max_distance[e.target] + distances[e.index])

        return np.max(max_distance).item()

    def energy_of_placement(self, placement: Placement) -> int:
        """
        For the case where the strides of every field connected by a dependency edge is the same, we use the same
        distance vector computation and sum all the contributions. Each distance vector is multiplied by the
        communication volume. This equals the size of the domain in the case of horizontal stencils and a single x-y
        plane in the case of vertical stencils.
        Then, we sum all edge costs to get the total energy.

        Note that for the energy computation, effects around the border of the domain are neglected
        This is an approximation that is valid for large domains (compared to the stencil size).

        :param placement:
        :return:
        """
        energy = np.zeros_like(self.stencil_graph.edges())
        for e in self.stencil_graph.edges():
            distances = placement.distance_vector_of_edge(e)

            # Energy is distance times volume,
            # summed over each stencil element
            energy[e.index] = np.sum(distances) * placement.communication_volume_of_edge(self.stencil_graph, e)

        return energy.sum()

    def contention_of_placement(self, placement: Placement) -> int:
        """
        The input contention of a field is the sum of the communication volumes of its incoming edges.
        Similarly, the output contention of a field is the sum of the communication volume of its outgoing edges.
        The contention is the sum of input and output contention.

        We assume that each pair of fields is either mapped to a disjoint or the same set of PEs. Then, to compute
        the most congested PE we need to group the fields according to if they map to the same PEs. This can be done
        by  hashing the fields into buckets based on their top-left coordinate. Then, contention of the
        placed stencil program is the largest contention of the fields in any given bucket.

        :param placement: Placement
        :return: float The contention of the placement
        """
        equivalence_classes = dict()

        # Note that even though the order is not deterministic the result still is because
        # of the use of integers for the contention
        for v in self.stencil_graph.graph.vs:
            offset = placement.offsets[v.index]
            t = (offset[0], offset[1])
            if t not in equivalence_classes:
                equivalence_classes[t] = []
            equivalence_classes[t].append(v)

        contention = 0
        for key in equivalence_classes:
            input_contention = self.input_contention_of_fields(equivalence_classes[key], placement)
            output_contention = self.output_contention_of_fields(equivalence_classes[key], placement)
            contention = max(contention, input_contention + output_contention)

        return contention

    def input_contention_of_fields(self, fields: Sequence[igraph.Vertex], placement: Placement) -> int:
        """
        The input contention of a field is the sum of the communication volumes of its incoming edges.
        :param placement:
        :param fields: list of fields
        :return: int
        """
        contention = 0
        for v in fields:
            for e in self.stencil_graph.in_edges(v):
                assert e.target == v.index
                contention += placement.contention_of_edge(self.stencil_graph, e)
        return contention

    def output_contention_of_fields(self, fields: Sequence[igraph.Vertex], placement: Placement) -> int:
        """
        The output contention of a field is the sum of the communication volume of its outgoing edges.
        :param placement: Placement The placement of the stencil graph
        :param fields: list of fields
        :return: int
        """
        contention = 0
        for v in fields:
            for e in self.stencil_graph.out_edges(v):
                assert e.source == v.index
                contention += placement.contention_of_edge(self.stencil_graph, e)
        return contention

    def depth_of_placement(self, placement: Placement) -> int:
        """
        Calculate the communication depth of the stencil graph
        which is the longest path in the stencilGraph, where all edges have weight 1.
        Use the topological sort of the graph to calculate the depth (using the dynamic programming approach).
        :return: float The depth of the placement
        """
        top_order = self.stencil_graph.graph.topological_sorting(mode="IN")

        # initialize the distance array
        max_distance = np.zeros(len(self.stencil_graph.graph.vs), dtype=np.int32)

        for v in top_order:
            # get the maximum distance of the incoming edges
            # and add the distance of the current edge
            for e in self.stencil_graph.out_edges(v):
                add = 0
                stencil_shape_xy = e[StencilGraph.STENCIL].shape[:, :2]
                if placement.edge_crosses_partition(e):
                    # If the edge crosses the partition, there is always communication
                    add = 1
                elif np.sum(np.any(stencil_shape_xy != 0, axis=1), dtype=np.int32) > 0:
                    # If the edge does not cross the partition,
                    # there is communication if the stencil shape xy-plane is not zero
                    add = 1
                max_distance[v] = max(max_distance[v], max_distance[e.target] + add)

        return np.max(max_distance)
