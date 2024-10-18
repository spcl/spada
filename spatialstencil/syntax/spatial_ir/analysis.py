"""
Contains analysis functions for Spatial IR, such as task dependency analysis.
"""
from spatialstencil.syntax.spatial_ir import irnodes as spir
import networkx as nx  # TODO: Switch to igraph


def to_task_dag(compute: spir.ComputeBlock) -> nx.DiGraph:
    """
    Converts a compute block to a directed graph of task dependencies,
    as defined in the specifications (local order) and based on completions
    and code order.
    """
    result = nx.DiGraph()

    return result
