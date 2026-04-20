from typing import Sequence, List, Dict, Tuple
import igraph as ig

from spada.placement.domain import FieldDomain
from spada.placement.stencil import Stencil, StencilDirection


class StencilGraph:

    DOMAIN = 'domain'
    STENCIL = 'stencil'
    FIELD_NAME = 'name'
    FIELD_VERSION = 'version'

    def __init__(self,
                 edges: Sequence[Tuple[int, int]],
                 domain: FieldDomain,
                 field_domains: Sequence[FieldDomain],
                 field_names: Sequence[str],
                 field_versions: Sequence[int],
                 stencils: Sequence[Stencil],
                 ) -> None:
        """
        Create a new StencilGraph that encapsulates a graph with fields and stencils.
        The graph is assumed to already contain all the vertices and edges.
        Here, we only store auxiliary information with it and provide accessors to it.

        See the ig.Graph documentation for more information on the graph object.

        All sequences refer to the order in which the vertices/edges are represented in the ig.Graph object.

        [TODO: It might be much nicer to encapsulate the ig.Graph object within this class and provide accessors to it,
        so that we can abstract from the implementation details of the graph object.]

        :param edges: The directed edges of the graph. Each edge points from source (input) to target (result).
        :param domain: The overall domain of the fields (must contain all the domains of the fields)
        :param field_domains: The domain of each field
        :param field_names: The name of each field
        :param field_versions: The version of each field
        :param stencils: The stencil pattern of each edge
        """

        n = len(field_names)

        assert len(stencils) == len(edges)
        assert len(field_domains) == n
        assert len(field_names) == n
        assert len(field_versions) == n

        self.graph = ig.Graph(directed=True)
        self.graph.add_vertices(n)
        self.graph.add_edges(edges)
        self.graph.vs[StencilGraph.FIELD_NAME] = field_names
        self.graph.vs[StencilGraph.FIELD_VERSION] = field_versions or ([0] * len(field_names))
        self.graph.vs[StencilGraph.DOMAIN] = field_domains
        self.graph.es[StencilGraph.STENCIL] = stencils
        self.graph[StencilGraph.DOMAIN] = domain

    def domain(self) -> FieldDomain:
        return self.graph[StencilGraph.DOMAIN]

    def stencils(self) -> Sequence[Stencil]:
        return self.graph.es[StencilGraph.STENCIL]

    def edges(self) -> ig.EdgeSeq:
        return self.graph.es

    def out_edges(self, vertex) -> ig.EdgeSeq:
        return self.graph.es.select(_source=vertex)

    def in_edges(self, vertex) -> ig.EdgeSeq:
        return self.graph.es.select(_target=vertex)

    def plot(self, filename="stencil_graph.png"):
        # Plot the stencil graph
        layout = self.graph.layout_sugiyama()

        edge_label = [f"{s.shape}" for s in self.graph.es[StencilGraph.STENCIL]]
        vertex_color = "lightblue"
        vertex_label = [f"{name}#{version}" for (name, version) in zip(list(self.graph.vs[self.FIELD_NAME]),
                                                                       list(self.graph.vs[self.FIELD_VERSION]))]
        ig.plot(self.graph,
                layout=layout,
                vertex_label=vertex_label,
                vertex_size=80,
                vertex_color=vertex_color,
                edge_color=["#666" if d.direction == StencilDirection.PARALLEL
                            else "red" if d.direction == StencilDirection.FORWARD
                            else "blue" for d in self.graph.es[StencilGraph.STENCIL]],
                edge_width=1,
                edge_label=edge_label,
                bbox=(250 + len(self.graph.vs) * 60, 250 + len(self.graph.vs) * 60),
                margin=40,
                target=filename)

    def merge_versions_of_fields(self) -> 'MergedStencilGraph':
        """
        Creates a new graph where all fields with the same name are merged into a single field.
        It maintains a mapping from the original fields to the merged fields (and back).

        Example:
        let's say the original graph has the following fields:
        a, b, c, a, b, a, a
        with the following versions (not that versions are not sorted):
        0, 0, 0, 2, 1, 3, 1

        The merged graph will have the following fields:
        a, b, c
        with the following versions:
        0, 0, 0

        The original fields will be mapped to the merged fields as follows:
        0 -> 0
        1 -> 1
        2 -> 2
        3 -> 0
        4 -> 1
        5 -> 0
        6 -> 0

        The reverse mapping will be:

        0 -> 0, 3, 5, 6
        1 -> 1, 4
        2 -> 2
        :return:
        """

        # Find all fields with the same name
        # do this by creating a dictionary with the field names as keys and the indices as values
        # We would like that the field names are sorted by their first appearance in the graph's vertex list
        field_name_to_indices: Dict[str, List[int]] = {}
        original_index_to_merged_field_index = [0] * self.graph.vcount()
        for i, name in enumerate(self.graph.vs[StencilGraph.FIELD_NAME]):
            if name in field_name_to_indices:
                field_name_to_indices[name].append(i)
                # The index is the same as the one of every other field with the same name
                original_index_to_merged_field_index[i] = original_index_to_merged_field_index[field_name_to_indices[name][0]]
            else:
                field_name_to_indices[name] = [i]
                # the index of the new field is the length of the list - 1
                # because we insert the names in the order of appearance
                original_index_to_merged_field_index[i] = len(field_name_to_indices) - 1

        # sort the dictionary by the first appearance of the field name (i.e. the first index)
        field_name_to_indices_sorted = sorted(field_name_to_indices.items(), key=lambda item: item[1][0])
        # We now get a list of tuples (field_name, [indices])
        merged_to_original_field: List[(str, List[int])] = [(i, indices)
                                                            for i, indices in field_name_to_indices_sorted]

        # then create a new graph with the merged fields
        # and create a mapping from the original fields to the merged fields (and back)
        # Create a new graph with the merged fields
        g_m = ig.Graph(directed=True)
        g_m.add_vertices(len(field_name_to_indices))
        g_m.vs[StencilGraph.FIELD_NAME] = [name for name, indices in field_name_to_indices_sorted]

        # Iterate over the edges of the original graph and add the edges to the new graph
        g_m.add_edges([(original_index_to_merged_field_index[e.source],
                        original_index_to_merged_field_index[e.target]) for e in self.graph.es])

        # merged domains:
        merged_domains = [self.graph.vs[StencilGraph.DOMAIN][indices[0]] for (i, indices) in merged_to_original_field]
        merged_names: List[str] = [name for name, indices in merged_to_original_field]

        merged_stencil_graph = MergedStencilGraph(g_m,
                                                  self.domain(),
                                                  merged_domains,
                                                  merged_names,
                                                  [0] * len(merged_names),
                                                  self.stencils(),
                                                  original_index_to_merged_field_index,
                                                  merged_to_original_field)

        # Return the new graph
        return merged_stencil_graph


class MergedStencilGraph(StencilGraph):
    """
    In a merged stencil graph, all fields with the same name are merged into a single field.
    It maintains a mapping from the original fields to the merged fields (and back).
    """

    def __init__(self, graph: ig.Graph,
                 domain: FieldDomain,
                 field_domains: Sequence[FieldDomain],
                 field_names: Sequence[str],
                 field_versions: Sequence[int],
                 stencils: Sequence[Stencil],
                 original_field_to_merged: Sequence[int],
                 merged_to_original_field: Sequence[Sequence[int]]
                 ) -> None:
        super().__init__([e.tuple for e in graph.es], domain, field_domains, field_names, field_versions, stencils)
        self.original_field_to_merged = original_field_to_merged
        self.merged_to_original_field = merged_to_original_field

    def original_field(self, merged_field: int) -> Sequence[int]:
        return self.merged_to_original_field[merged_field]

    def merged_field(self, original_field: int) -> int:
        return self.original_field_to_merged[original_field]

    def plot(self, filename="stencil_graph.png"):
        # Plot the stencil graph
        layout = self.graph.layout_sugiyama()

        edge_label = ""
        vertex_color = "lightblue"
        vertex_label = [f"{name}" for name in self.graph.vs[self.FIELD_NAME]]

        ig.plot(self.graph,
                layout=layout,
                vertex_label=vertex_label,
                vertex_size=80,
                vertex_color=vertex_color,
                edge_color=["#666" if d.direction == StencilDirection.PARALLEL
                            else "red" if d.direction == StencilDirection.FORWARD
                            else "blue" for d in self.graph.es[StencilGraph.STENCIL]],
                edge_width=1,
                edge_label=edge_label,
                bbox=(250 + len(self.graph.vs) * 60, 250 + len(self.graph.vs) * 60),
                margin=40,
                target=filename)
