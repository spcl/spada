from typing import Tuple

import numpy as np
import igraph as ig
from numpy.typing import NDArray

from scripts import examples
from spada.placement.graph import Stencil, StencilDirection, FieldDomain, StencilGraph
from spada.placement.placed_graph import PlacedStencilGraph
from spada.placement.mla import linearize_with_ck
from spada.placement.model import CostModel, PlacementCost
from spada.placement.optimizer import best_of_k_placement
from spada.placement.partition import FieldPartition


def demo_graph():

    domain = np.array([[0, 0, 0], [256, 256, 64]], dtype=np.int32)
    domain_type = FieldDomain(domain)

    five_point_stencil = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.int32)

    # one_point_stencil_right = np.array( [[1, 0, 0]], dtype=np.int32)
    one_point_stencil_up = np.array([[0, 1, 0]], dtype=np.int32)

    z_stencil_forward = np.array([[0, 0, -1]], dtype=np.int32)
    z_stencil_backward = np.array([[0, 0, 1]], dtype=np.int32)

    # Create StencilGraph with 8 nodes
    # and an inverted tree like shape
    # u -> w, v-> w, x -> z, y -> z, w -> a, z -> a, a -> b, b -> c
    # as numbers, we get
    # u = 0, v = 1, w = 2, x = 3, y = 4, z = 5, a = 6, b= 7, c = 8
    g = ig.Graph(directed=True, n=9)
    g.add_edges([(0, 2), (1, 2), (3, 5), (4, 5), (2, 6), (5, 6), (6, 7), (7, 8), (0, 8)])
    # Set the names of the nodes
    names = ["u", "v", "w", "x", "y", "z", "a", "b", "c"]
    versions = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # We alternate between horizontal (five point) and vertical stencils
    # so the stencils w-> a and z-> a and b-> c are vertical
    # and the rest are horizontal
    # All horizontal stencils are parallel, while the first vertical stencils are forward and the last is backward

    p = StencilDirection.PARALLEL
    f = StencilDirection.FORWARD
    b = StencilDirection.BACKWARD
    stencils = [Stencil(five_point_stencil, p),
                Stencil(five_point_stencil, p),
                Stencil(five_point_stencil, p),
                Stencil(five_point_stencil, p),
                Stencil(z_stencil_forward, f),
                Stencil(z_stencil_forward, f),
                Stencil(five_point_stencil, p),
                Stencil(z_stencil_backward, b),
                Stencil(one_point_stencil_up, p)]

    # Create the StencilGraph
    stencil_graph = StencilGraph(g, domain_type, [domain_type] * 9, names, versions, stencils)

    return stencil_graph


def demo_placement_interleave():
    # Create a placement with two partitions, one with u, v, w, a, b, c and the other with x, y, z
    # The partitions are placed on the same processing elements
    # as numbers, we get
    # u = 0, v = 1, w = 2, x = 3, y = 4, z = 5, a = 6, b= 7, c = 8
    # we use offset (0, 0) for the first partition
    # and offset (0, 1) for the second partition
    # the stride is (2, 1) for both partitions
    parts = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0]], dtype=np.int32)
    partition = FieldPartition(parts)
    placement = partition.place_interleaved()
    return placement


def demo_placement_separated(domain: FieldDomain):
    """
    Places the nodes in the graph on two partitions of separate processing elements
    :param domain:
    :return:
    """
    parts = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0]], dtype=np.int32)
    partition = FieldPartition(parts)
    # automatic result
    placement = partition.place_blocked(domain)
    return placement


def main():

    np.random.seed(42)
    diffusion = examples.horizontal_diffusion()
    diffusion.plot(diffusion.graph['name'] + ".png")
    diffusion.merge_versions_of_fields().plot(diffusion.graph['name'] + "_merged.png")
    print("H diffusion Automatic placement (interleaved)")
    g_placed, cost = best_of_k_placement(diffusion, 10, (2, 1))
    g_placed.plot(diffusion.graph['name'] + "_auto.png")
    print("\n")

    np.random.seed(43)
    advection = examples.vertical_advection_simplified()
    advection.plot(advection.graph['name'] + ".png")
    advection.merge_versions_of_fields().plot(advection.graph['name'] + "_merged.png")
    print("V advection Automatic placement (interleaved)")
    g_placed, cost = best_of_k_placement(advection, 10, (2, 2))
    g_placed.plot(advection.graph['name'] + "_auto.png")


if __name__ == "__main__":
    main()
