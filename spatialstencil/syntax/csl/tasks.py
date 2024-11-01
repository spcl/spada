"""
Contains a CSL task DAG representation and creation methods.
"""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
import networkx as nx  # TODO: Switch to igraph
from typing import Any, Literal
from spatialstencil.syntax.spatial_ir import irnodes as spir, analysis
from spatialstencil.syntax.csl import constants


class InterTaskEdge(Enum):
    """
    Enumeration representing a task dependency edge type.
    """
    SEQUENCE = auto()
    ACTIVATE = auto()
    UNBLOCK = auto()


@dataclass
class CSLTask:
    """
    Object representing a task DAG node.
    """
    task_id: int
    task_type: Literal['local', 'data']  # We do not generate control tasks at the moment
    statements: list[int]  # Index is the statement's index from the completion DAG
    outgoing: list[tuple[int, InterTaskEdge]]  # For each statement, the next task ID and the dependency type
    blocked: bool  # Whether there is an unblock edge leading to this task


def create_csl_tasks(completion_dag: nx.DiGraph, block: spir.ComputeBlock) -> list[CSLTask]:
    """
    Creates a list of CSL tasks. The nodes are tasks that contain a unique ID
    and the list of statements to include; and the edges are the type of dependency across tasks.
    The algorithm operates as follows.

    Statements can take on different task types, based on the statement type and its contents:

        * Foreach statements may take the form of a CSL data task, if it cannot trivially be represented by a
          single DSD operation (@mov, @fadd*, etc.)
        * Send and receive statements that can be lowered to a ``FabricDSD`` operation, in turn can (and should)
          be nonblocking, or ``async`` in CSL terms. In this lowering pipeline, these live in CSL local tasks.
        * Other statements (e.g. free assignments) are blocking and also live in local tasks.
    
    Given that tasks can ``@activate`` and ``@unblock`` other tasks, and that ``FabricDSD`` operations can also
    do the same, both a task terminator and a nonblocking statement can trigger other tasks. Given that there are
    no other options to trigger tasks, we run a preprocessing pass on the graph to convert nodes with in-degree over 2
    to a series of ``wait`` nodes.

    Subsequently, we traverse the Completion DAG topologically (to ensure proper local order). We then decide to create
    new tasks based on a set of necessary rules in which a new task must be formed:

        1. A node with no predecessors creates a new activated and unblocked task
        2. A node with more than one incoming edge must start a new task (the conditions below thus apply to one
           predecessor)
        3. If a node's predecessor represents one kind of CSL task (e.g., data) and this node represents another
        4. Node pairs with ``wait->wait`` edges create a new task (this also fulfills the condition for the above
           preprocessing pass)
        5. ``post->wait`` node pairs where the post is a nonblocking operation creates a new task for the ``wait`` node
           and sets the nonblocking DSD to ``.activate`` the waiting task, or ``.unblock`` it if there is another edge

    The last (i.e., sink) task is called ``exit_task`` and is built into the generation of rectangle code.

    This means that post->post nodes of nonblocking operations can coexist in the same task.
    """
    result: list[CSLTask] = []

    completion_dag = _canonicalize_dag(completion_dag)
    return result


def _contract_node(g: nx.DiGraph, n: Any):
    if g.out_degree(n) == 0:  # Keep sink node
        return
    for u, _ in g.in_edges(n):
        for _, v in g.out_edges(n):
            g.add_edge(u, v)
    g.remove_node(n)


def _canonicalize_dag(completion_dag: nx.DiGraph) -> nx.DiGraph:
    completion_dag = deepcopy(completion_dag)

    # Reduce in-degree of nodes to up to 2
    _limit_indegree(completion_dag)

    return completion_dag


def _limit_indegree(dag: nx.DiGraph):
    """
    Injects extra wait nodes to completion DAGs where the in-degree of a node is larger than two.

    :param dag: The completion DAG.
    """
    counter = -1
    for node in list(dag.nodes):  # Copy nodes to a list
        if dag.in_degree(node) > 2:
            edges = list(dag.in_edges(node))
            current_node = node
            # Create intermediate wait nodes (the counter changes the statement ID because it has to be unique)
            for u, _ in edges[1:]:
                new_node = analysis.CompletionDAGNode('wait', counter)
                counter -= 1
                dag.remove_edge(u, node)
                dag.add_edge(new_node, current_node)
                dag.add_edge(u, new_node)
                current_node = new_node
