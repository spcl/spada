from typing import Sequence
import igraph as ig
import matplotlib.pyplot as plt

from spada.placement.graph import StencilGraph
from spada.placement.partition import Placement
from spada.placement.stencil import StencilDirection


class PlacedStencilGraph(StencilGraph):
    """
    Represents a stencil graph with a specific placement of the fields on the processing elements.
    """
    def __init__(self,
                 stencil_graph: StencilGraph,
                 placement: Placement,
                 distances: Sequence[int]
                 ) -> None:
        """
        Create a new PlacedStencilGraph that encapsulates a graph with fields and stencils and a specific placement.

        :param stencil_graph: the underlying stencil graph
        :param placement: the placement of the fields
        :param distances: the distances of each of the edges (this caches the results of the model costs)
        """
        super().__init__([e.tuple for e in stencil_graph.edges()],
                         stencil_graph.domain(),
                         stencil_graph.graph.vs[StencilGraph.DOMAIN],
                         stencil_graph.graph.vs[StencilGraph.FIELD_NAME],
                         stencil_graph.graph.vs[StencilGraph.FIELD_VERSION],
                         stencil_graph.stencils())
        assert placement.strides.shape[0] == len(self.graph.vs)
        assert len(distances) == len(self.graph.es)
        self.placement = placement
        self.distances = distances
        self.graph.vs['partition'] = self.placement.parts()

    def placement(self) -> Placement:
        return self.placement

    def plot(self, filename="stencil_graph.png"):
        # Plot the stencil graph
        layout = self.graph.layout_sugiyama()

        edge_label = [f"{d} ;\n {s.shape}" for (d, s) in zip(self.distances, self.graph.es[StencilGraph.STENCIL])]

        # Create a categorical color map for the partitions
        cmap = plt.get_cmap('tab20')
        vertex_color = [cmap(p + 1) for p in self.graph.vs['partition']]

        # Assign vertex labels based on the partition
        vertex_label = [f"{l}#{v}\n{p}" for (l, v, p) in zip(self.graph.vs[self.FIELD_NAME],
                                                             self.graph.vs[self.FIELD_VERSION],
                                                             self.placement.offsets)]

        ig.plot(self.graph,
                layout=layout,
                vertex_label=vertex_label,
                vertex_size=100,
                vertex_color=vertex_color,
                edge_color=["#666" if d.direction == StencilDirection.PARALLEL
                            else "red" if d.direction == StencilDirection.FORWARD
                            else "blue" for d in self.graph.es[StencilGraph.STENCIL]],
                edge_width=1,
                edge_label=edge_label,
                bbox=(250 + len(self.graph.vs) * 60, 250 + len(self.graph.vs) * 60),
                margin=60,
                target=filename)