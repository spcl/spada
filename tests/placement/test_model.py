import unittest
import numpy as np
from spatialstencil.placement.domain import FieldDomain
from spatialstencil.placement.model import CostModel
from spatialstencil.placement.partition import FieldPartition
from spatialstencil.placement.stencil import Stencil, StencilDirection
from spatialstencil.placement.graph import StencilGraph
import igraph as ig


class TestModel(unittest.TestCase):

    def demo_graph_3path(self, vertical=True):
        domain = np.array([[0, 0, 0], [4, 6, 10]], dtype=np.int32)
        domain_type = FieldDomain(domain)

        if vertical:
            one_point_stencil_up = np.array([[0, 0, -1]], dtype=np.int32)
            one_point_stencil_down = np.array([[0, 0, 1]], dtype=np.int32)
            stencils = [Stencil(one_point_stencil_up, StencilDirection.FORWARD),
                        Stencil(one_point_stencil_down, StencilDirection.BACKWARD)
                        ]
        else:
            one_point_elementwise_stencil = np.array([[0, 0, 0]], dtype=np.int32)
            stencils = [Stencil(one_point_elementwise_stencil, StencilDirection.PARALLEL),
                        Stencil(one_point_elementwise_stencil, StencilDirection.PARALLEL)
                        ]

        edges = [(0, 1), (1, 2)]

        names = ["u", "v", "w"]
        versions = [0, 0, 0]

        stencil_graph = StencilGraph(edges, domain_type, [domain_type] * 3, names, versions, stencils)
        return stencil_graph

    def demo_graph_wedge(self):
        domain = np.array([[0, 0, 0], [4, 6, 10]], dtype=np.int32)
        domain_type = FieldDomain(domain)

        one_point_stencil_left = np.array([[1, 0, 0]], dtype=np.int32)

        # Create StencilGraph with 3 nodes
        # and an inverted tree like shape
        # u -> w, v-> w
        # as numbers, we get
        # u = 0, v = 1, w = 2
        edges = [(0, 2), (1, 2)]
        # Set the names of the nodes
        names = ["u", "v", "w"]
        versions = [0, 0, 0]

        # We alternate between horizontal (five point) and vertical stencils
        # so the stencils w-> a and z-> a and b-> c are vertical
        # and the rest are horizontal
        # All horizontal stencils are parallel, while the first vertical stencils are forward and the last is backward
        p = StencilDirection.PARALLEL
        stencils = [Stencil(one_point_stencil_left, p),
                    Stencil(one_point_stencil_left, p)
                    ]

        # Create the StencilGraph
        stencil_graph = StencilGraph(edges, domain_type, [domain_type] * 3, names, versions, stencils)

        # stencil_graph.plot("test_stencil_wedge.png")

        return stencil_graph

    def test_distance(self):
        demo_graph = self.demo_graph_wedge()
        domain: FieldDomain = demo_graph.graph[StencilGraph.DOMAIN]

        # Create a partition with 1 part
        partition_array = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        # Create a block placement
        placement = partition.place_blocked(demo_graph.graph[StencilGraph.DOMAIN])

        # Create a cost model
        cost_model = CostModel(demo_graph)
        distances = cost_model.edge_distance_of_placement(placement)

        self.assertEqual(1, distances[0], f"The distances are {distances}")
        self.assertEqual(1, distances[1], f"The distances are {distances}")
        self.assertEqual(1, cost_model.distance_of_placement(distances))

        # Create a partition with 2 parts in the x direction
        partition_array = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        # Create a block placement
        placement = partition.place_blocked(demo_graph.graph[StencilGraph.DOMAIN])

        # Create a cost model
        cost_model = CostModel(demo_graph)
        distances = cost_model.edge_distance_of_placement(placement)

        self.assertEqual(1, distances[0], f"The distances are {distances}")
        self.assertEqual(domain.x_length() + 1, distances[1], f"The distances are {distances}")
        self.assertEqual(domain.x_length() + 1, cost_model.distance_of_placement(distances))

        # Create a partition with 2 parts in the y direction
        partition_array = np.array([[0, 0], [0, 1], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        # Create a block placement
        placement = partition.place_blocked(demo_graph.graph[StencilGraph.DOMAIN])

        # Create a cost model
        cost_model = CostModel(demo_graph)
        distances = cost_model.edge_distance_of_placement(placement)

        self.assertEqual(1, distances[0], f"The distances are {distances}")
        self.assertEqual(domain.y_length() + 1, distances[1], f"The distances are {distances}")
        self.assertEqual(domain.y_length() + 1, cost_model.distance_of_placement(distances))

        # Create an interleaved placement from a partition with 2 parts in the x direction
        partition_array = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        # Create a block placement
        placement = partition.place_interleaved()

        # Create a cost model
        cost_model = CostModel(demo_graph)
        distances = cost_model.edge_distance_of_placement(placement)

        self.assertEqual(2, distances[0], f"The distances are {distances}")
        self.assertEqual(3, distances[1], f"The distances are {distances}")
        self.assertEqual(3, cost_model.distance_of_placement(distances))

    def test_depth(self):
        graph = self.demo_graph_wedge()
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(1, cost_model.depth_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(1, cost_model.depth_of_placement(placement))

        graph = self.demo_graph_3path()
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(2, cost_model.depth_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [1, 0], [1, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(1, cost_model.depth_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(0, cost_model.depth_of_placement(placement))
        # graph.plot("test_stencil_3path_v.png")

    def test_contention(self):
        graph = self.demo_graph_wedge()
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(2 * 2 * graph.domain().z_length(), cost_model.contention_of_placement(placement))

        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)).place_interleaved()
        self.assertEqual(2 * graph.domain().z_length(), cost_model.contention_of_placement(placement))

        graph = self.demo_graph_3path(vertical=False)
        # graph.plot("test_stencil_3path_h.png")
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(0, cost_model.contention_of_placement(placement))

        graph = self.demo_graph_3path(vertical=False)
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(graph.domain().z_length(), cost_model.contention_of_placement(placement))

        graph = self.demo_graph_3path(vertical=False)
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(2 * graph.domain().z_length(), cost_model.contention_of_placement(placement))

        # If all stencils are forward/backward and there is a single partition, there is no contention!
        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        graph = self.demo_graph_3path()
        cost_model = CostModel(graph)
        self.assertEqual(0, cost_model.contention_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(graph.domain().z_length(), cost_model.contention_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int32)).place_blocked(graph.domain())
        self.assertEqual(graph.domain().z_length(), cost_model.contention_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [1, 0], [1, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(graph.domain().z_length(), cost_model.contention_of_placement(placement))

        # here it still 1 because the input and output contention is both 1
        placement = FieldPartition(np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(2 * graph.domain().z_length(), cost_model.contention_of_placement(placement))

    def test_energy(self):
        # Path [0, 0, +-1] stencils
        # If all stencils are forward/backward and there is a single partition, the energy is 0
        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        graph = self.demo_graph_3path()
        cost_model = CostModel(graph)
        self.assertEqual(0, cost_model.energy_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        graph = self.demo_graph_3path()
        cost_model = CostModel(graph)
        self.assertEqual(2 * graph.domain().volume(), cost_model.energy_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [1, 0], [1, 0]], dtype=np.int32)).place_interleaved()
        graph = self.demo_graph_3path()
        cost_model = CostModel(graph)
        self.assertEqual(graph.domain().volume(), cost_model.energy_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int32)).place_blocked(graph.domain())
        graph = self.demo_graph_3path()
        cost_model = CostModel(graph)
        self.assertEqual(2 * graph.domain().volume() * graph.domain().x_length(),
                         cost_model.energy_of_placement(placement))

        placement = FieldPartition(np.array([[0, 1], [0, 0], [0, 0]], dtype=np.int32)).place_blocked(graph.domain())
        graph = self.demo_graph_3path()
        cost_model = CostModel(graph)
        self.assertEqual(graph.domain().volume() * graph.domain().y_length(),
                         cost_model.energy_of_placement(placement))

        # Wedge [1, 0, 0] stencil
        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        graph = self.demo_graph_wedge()
        cost_model = CostModel(graph)
        self.assertEqual(2 * graph.domain().volume(), cost_model.energy_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        graph = self.demo_graph_wedge()
        cost_model = CostModel(graph)
        self.assertEqual(2 * graph.domain().volume(), cost_model.energy_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int32)).place_blocked(domain=graph.domain())
        graph = self.demo_graph_wedge()
        cost_model = CostModel(graph)
        self.assertEqual(graph.domain().volume() * (graph.domain().x_length() + 2), cost_model.energy_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [0, 1], [0, 0]], dtype=np.int32)).place_blocked(domain=graph.domain())
        graph = self.demo_graph_wedge()
        cost_model = CostModel(graph)
        self.assertEqual(graph.domain().volume() * (graph.domain().y_length() + 2), cost_model.energy_of_placement(placement))

        placement = FieldPartition(np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int32)).place_blocked(domain=graph.domain())
        graph = self.demo_graph_wedge()
        cost_model = CostModel(graph)
        self.assertEqual(2 * graph.domain().volume() * (graph.domain().x_length() - 1), cost_model.energy_of_placement(placement))

        # Path [0, 0, 0] stencil
        graph = self.demo_graph_3path(vertical=False)
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(0, cost_model.energy_of_placement(placement))

        graph = self.demo_graph_3path(vertical=False)
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int32)).place_interleaved()
        self.assertEqual(graph.domain().volume(), cost_model.energy_of_placement(placement))

        graph = self.demo_graph_3path(vertical=False)
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int32)).place_blocked(graph.domain())
        self.assertEqual(graph.domain().volume() * graph.domain().x_length(), cost_model.energy_of_placement(placement))

        graph = self.demo_graph_3path(vertical=False)
        cost_model = CostModel(graph)
        placement = FieldPartition(np.array([[0, 0], [0, 0], [0, 1]], dtype=np.int32)).place_blocked(graph.domain())
        self.assertEqual(graph.domain().volume() * graph.domain().y_length(), cost_model.energy_of_placement(placement))


if __name__ == '__main__':
    unittest.main()
