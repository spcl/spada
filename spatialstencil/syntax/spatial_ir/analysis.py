"""
Contains analysis functions for Spatial IR, such as task dependency analysis.
"""
from spatialstencil.syntax.spatial_ir import irnodes as spir
from dataclasses import dataclass
from typing import Literal
import networkx as nx  # TODO: Switch to igraph


@dataclass(frozen=True)
class TaskDAGNode:
    """
    Object representing a task DAG node.
    """
    optype: Literal['post', 'wait']
    statement_id: int


def to_task_dag(compute: spir.ComputeBlock) -> nx.DiGraph:
    """
    Converts a compute block to a directed graph of task dependencies,
    as defined in the specifications (local order) and based on completions
    and code order.

    The resulting graph contains ``TaskDAGNode`` objects, which refer to
    whether the node is posting an asynchronous task or waiting for one
    (based on the node's ``optype`` field), and which statement index in
    the compute block's statements it refers to.
    """
    result = nx.DiGraph()

    # Keep track of last node for sequential dependencies in local order
    last_node: TaskDAGNode | None = None
    # Keep track of unawaited completions
    incomplete_completions: dict[str, TaskDAGNode] = {}

    for stmt_id, stmt in enumerate(compute.statements):
        node: TaskDAGNode
        completion_node: TaskDAGNode | None = None

        # awaitall
        if isinstance(stmt, spir.AwaitAllStatement):
            # If there is nothing to wait for, skip node
            if not incomplete_completions:
                continue

            # Connect all previous incomplete nodes to this node
            node = TaskDAGNode('wait', stmt_id)
            result.add_node(node)
            for compnode in incomplete_completions.values():
                result.add_edge(compnode, node)
            incomplete_completions.clear()

        # await completion
        elif isinstance(stmt, spir.AwaitCompletionStatement):
            # Create a completion node and connect it to the poster
            compname = stmt.completion_name.as_ir()
            if compname not in incomplete_completions:
                raise SyntaxError(f'Trying to await completion "{stmt.completion_name.as_ir()}", which does not exist '
                                  'or was already awaited for.')
            node = TaskDAGNode('wait', stmt_id)
            result.add_node(node)
            result.add_edge(incomplete_completions[compname], node)
            del incomplete_completions[compname]

        # Asynchronous nodes (completion comp = ...)
        else:
            # Create poster node
            node = TaskDAGNode('post', stmt_id)
            result.add_node(node)
            completion: spir.Completion | None = stmt.completion_name
            if completion is None:
                # Create another completion node immediately after this one and connect it
                completion_node = TaskDAGNode('wait', stmt_id)
                result.add_node(completion_node)
                result.add_edge(node, completion_node)
            else:
                # Add to unawaited completions
                incomplete_completions[completion.name.as_ir()] = node

        # Potentially connect previous node if not already connected (local code order)
        if last_node is not None and last_node not in result.predecessors(node):
            result.add_edge(last_node, node)

        # Set new last node
        if completion_node is not None:
            last_node = completion_node
        else:
            last_node = node

    return result


def get_identifier_sizes(place: spir.PlaceBlock) -> dict[spir.Identifier, list[int]]:
    """
    Returns a dictionary mapping each identifier to its dimensions, or an empty list if scalar.
    """
    result = {}
    for decl in place.statements:
        if isinstance(decl.dtype, spir.ScalarType):
            result[decl.field_name] = []
        else:  # Array type
            evaluated_shape = []
            for s in decl.dtype.shape:
                if isinstance(s, int):
                    evaluated_shape.append(s)
                else:
                    evaluated_shape.append(s.eval())
            result[decl.field_name] = evaluated_shape
    return result


def get_identifier_types(place: spir.PlaceBlock) -> dict[spir.Identifier, spir.ScalarType]:
    """
    Returns a dictionary mapping each identifier to its data type.
    """
    result = {}
    for decl in place.statements:
        if isinstance(decl.dtype, spir.ScalarType):
            result[decl.field_name] = decl.dtype
        else:  # Array type
            result[decl.field_name] = decl.dtype.base_type.element_type
    return result


class _SendRecvCollector(spir.NodeVisitor):

    def __init__(self):
        super().__init__()
        self.sends: set[spir.Identifier] = set()
        self.receives: set[spir.Identifier] = set()

    def _get_underlying_stream(self, node: spir.Identifier | spir.ArraySlice) -> spir.Identifier:
        if isinstance(node, spir.ArraySlice):
            return node.array
        return node

    def visit_ReceiveStatement(self, node: spir.ReceiveStatement):
        self.receives.add(self._get_underlying_stream(node.stream_name))

    def visit_ReceiveGenerator(self, node: spir.ReceiveGenerator):
        self.receives.add(self._get_underlying_stream(node.stream_name))

    def visit_SendStatement(self, node: spir.SendStatement):
        self.sends.add(self._get_underlying_stream(node.stream_name))


def sends_and_receives(compute: spir.ComputeBlock) -> dict[spir.Identifier, tuple[bool, bool]]:
    """
    Returns, for each stream, whether it is used in a send or receive operation.

    :return: Dictionary mapping stream identifiers to a 2-tuple of (is_sent, is_received).
    """
    collector = _SendRecvCollector()
    collector.visit(compute)
    all_identifiers = {k for k in collector.sends | collector.receives}
    return {k: (k in collector.sends, k in collector.receives) for k in all_identifiers}
