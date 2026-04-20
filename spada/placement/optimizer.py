from typing import Tuple
import igraph
import numpy as np

from spada.placement.graph import StencilGraph
from spada.placement.placed_graph import PlacedStencilGraph
from spada.placement.model import PlacementCost, CostModel
from spada.placement.partition import FieldPartition


def color_graph(g: StencilGraph):
    """
    Computes a coloring of the graph g (ignores the edge directions) using a greedy algorithm.
    Converts to networks and uses the networkx algorithm.
    :param g:
    :return:
    """
    pass


def best_of_k_placement(g: StencilGraph, k: int, shape: Tuple[int, int]) -> Tuple[PlacedStencilGraph, PlacementCost]:
    """
    Find the best placement of the graph g by trying k different placements and returning the best one.
    :param g:
    :param k:
    :param shape:
    :return:
    """
    assert k > 0
    best_cost = float('inf')
    best_cost_obj = None
    best_placement = None
    best_distances = None
    cost_model = CostModel(g)
    merged_graph = g.merge_versions_of_fields()
    for i in range(k):
        partition_merged = FieldPartition.from_mla(merged_graph.graph, shape)
        # map the partition into a partition of the original graph
        partition_auto_pt = np.zeros((g.graph.vcount(), 2), dtype=np.int32)
        for j in range(g.graph.vcount()):
            partition_auto_pt[j, :] = partition_merged.part[merged_graph.original_field_to_merged[j]]
        partition_auto = FieldPartition(part=partition_auto_pt)

        place_auto = partition_auto.place_interleaved()
        cost = cost_model.cost(place_auto)
        distances = cost_model.edge_distance_of_placement(place_auto)
        print(cost)
        if cost.overall < best_cost or cost.overall == best_cost and cost.energy_over_links < best_cost_obj.energy_over_links:
            best_cost = cost.overall
            best_placement = place_auto
            best_cost_obj = cost
            best_distances = distances
        placed_graph = PlacedStencilGraph(g, place_auto, distances)
        # For debugging, plot the placement
        placed_graph.plot(g.graph['name'] + "_auto_" + str(i) + ".png")

    placed_graph = PlacedStencilGraph(g, best_placement, best_distances)
    return placed_graph, best_cost_obj
