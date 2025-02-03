"""
Contains a CSL task DAG representation and creation methods.
"""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
import networkx as nx  # TODO: Switch to igraph
from typing import Any, Literal, Optional
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


def _get_id(value: spir.ArraySlice | spir.Identifier) -> spir.Identifier:
    if isinstance(value, spir.Expression):
        return _get_id(value.value)
    if isinstance(value, spir.ArraySlice):
        return value.array
    return value


def _get_dtype(dtypes: dict[spir.Identifier, spir.IRType],
               value: spir.Identifier | spir.ArraySlice | spir.ConstantLiteral) -> spir.IRType:
    if isinstance(value, spir.Expression):
        return _get_dtype(dtypes, value.value)
    if isinstance(value, spir.ConstantLiteral):
        return value.dtype
    if isinstance(value, (spir.UnaryOperator, spir.BinaryOperator, spir.TernaryOperator)):
        return None
    return dtypes[_get_id(value)]


def _get_base_dtype(dtypes: dict[str, spir.IRType],
                    value: spir.Identifier | spir.ArraySlice | spir.ConstantLiteral) -> spir.ScalarType:
    dtype = _get_dtype(dtypes, value)
    if dtype is None:
        return dtype
    return dtype.element_type


def get_dsd_op(dtypes: dict[spir.Identifier, spir.IRType],
               stmt: spir.ForeachStatement | spir.MapStatement) -> Optional[str]:
    """
    Returns a DSD op name if a foreach or map statement can be represented by a single DSD operation 
    (@mov, @fadd*, etc.), or None if the body cannot be expressed as a single DSD operation.
    This is used in lowering to CSL to determine whether a DSD operation can be used directly vs. creating
    a data task.
    """
    if len(stmt.body) == 0:
        # No-op
        return ''
    if len(stmt.body) > 1:
        return None
    inner_stmt = stmt.body[0]
    if not isinstance(inner_stmt, spir.AssignmentStatement):
        return None

    dst = _get_id(inner_stmt.destination)
    if dst not in dtypes:
        raise NameError(f'"{dst.as_ir()}" not in recognized data types')
    dtype = _get_base_dtype(dtypes, dst)

    inner_stmt = inner_stmt.source.value

    if isinstance(inner_stmt, spir.UnaryOperator):  # @fneg*
        # NOTE: There is no negation DSD operation for integer types
        if dtype == spir.ScalarType.f16:
            return '@fnegh'
        if dtype == spir.ScalarType.f32:
            return '@fnegs'

    elif isinstance(inner_stmt, spir.BinaryOperator):
        # @add*, @fadd*, @fmul*, @sub*, @fsub*
        source_types = (_get_base_dtype(dtypes, inner_stmt.left), _get_base_dtype(dtypes, inner_stmt.right))
        if inner_stmt.op == '+':
            if (dtype == spir.ScalarType.f16 and source_types[0] == spir.ScalarType.f16 and
                    source_types[1] == spir.ScalarType.f16):
                return '@faddh'
            if (dtype == spir.ScalarType.f32 and source_types[0] == spir.ScalarType.f32 and
                    source_types[1] == spir.ScalarType.f32):
                return '@fadds'
            if (dtype == spir.ScalarType.f32 and
                ((source_types[0] == spir.ScalarType.f16 and source_types[1] == spir.ScalarType.f32) or
                 (source_types[0] == spir.ScalarType.f32 and source_types[1] == spir.ScalarType.f16))):
                return '@faddhs'
            if (dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and
                    source_types[0] in (spir.ScalarType.i16, spir.ScalarType.u16) and
                    source_types[1] in (spir.ScalarType.i16, spir.ScalarType.u16)):
                return '@add16'

        elif inner_stmt.op == '-':
            if (dtype == spir.ScalarType.f16 and source_types[0] == spir.ScalarType.f16 and
                    source_types[1] == spir.ScalarType.f16):
                return '@fsubh'
            if (dtype == spir.ScalarType.f32 and source_types[0] == spir.ScalarType.f32 and
                    source_types[1] == spir.ScalarType.f32):
                return '@fsubs'
            if (dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and
                    source_types[0] in (spir.ScalarType.i16, spir.ScalarType.u16) and
                    source_types[1] in (spir.ScalarType.i16, spir.ScalarType.u16)):
                return '@sub16'

        elif inner_stmt.op == '*':
            # NOTE: There is no @mul*
            if (dtype == spir.ScalarType.f16 and source_types[0] == spir.ScalarType.f16 and
                    source_types[1] == spir.ScalarType.f16):
                return '@fmulh'
            if (dtype == spir.ScalarType.f32 and source_types[0] == spir.ScalarType.f32 and
                    source_types[1] == spir.ScalarType.f32):
                return '@fmuls'

    elif isinstance(inner_stmt, spir.MultiplyAccumulateOperator):  # @fmac*
        # @fmac* only works with scalar/constant values of ``c``
        c_type = _get_dtype(dtypes, inner_stmt.c.value)
        if not isinstance(c_type, spir.ScalarType):
            return None
        a_dtype, b_dtype, c_dtype = (_get_base_dtype(dtypes, inner_stmt.a), _get_base_dtype(dtypes, inner_stmt.b),
                                     _get_base_dtype(dtypes, inner_stmt.c))
        if dtype != a_dtype or dtype != b_dtype:
            # NOTE: Destination type semantics are unclear, supporting only same src/dst dtype for now
            return None
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f16 and c_dtype == spir.ScalarType.f16:
            return '@fmach'
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f16:
            return '@fmachs'  # 16-bit multiplication, 32-bit addition
        if a_dtype == b_dtype and a_dtype == spir.ScalarType.f32 and c_dtype == spir.ScalarType.f32:
            return '@fmacs'

    elif isinstance(inner_stmt, (spir.Identifier, spir.ConstantLiteral)):  # @fmov*, @mov*
        src_dtype = _get_base_dtype(dtypes, inner_stmt)
        # Move statements are valid for operands of the same type
        if src_dtype == dtype:
            if dtype in (spir.ScalarType.i16, spir.ScalarType.u16):
                return '@mov16'
            if dtype in (spir.ScalarType.i32, spir.ScalarType.u32):
                return '@mov32'
            if dtype == spir.ScalarType.f16:
                return '@fmovh'
            if dtype == spir.ScalarType.f32:
                return '@fmovs'
        else:
            if dtype == spir.ScalarType.f16 and src_dtype == spir.ScalarType.f32:
                return '@fs2h'
            if dtype == spir.ScalarType.f32 and src_dtype == spir.ScalarType.f16:
                return '@fh2s'
            if dtype == spir.ScalarType.f16 and src_dtype in (spir.ScalarType.i16, spir.ScalarType.u16):
                return '@xp162fh'
            if dtype == spir.ScalarType.f32 and src_dtype in (spir.ScalarType.i16, spir.ScalarType.u16):
                return '@xp162fs'
            if dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and src_dtype == spir.ScalarType.f16:
                return '@fh2xp16'
            if dtype in (spir.ScalarType.i16, spir.ScalarType.u16) and src_dtype == spir.ScalarType.f32:
                return '@fs2xp16'

    return None


def should_be_asynchronous(dtypes: dict[spir.Identifier, spir.IRType], stmt: spir.Statement) -> bool:
    """
    Returns True if a statement can and should be executed asynchronously in CSL.
    The only statements that apply are DSD operations that have to do with fabric DSDs (e.g., send, receive).
    """
    if isinstance(stmt, (spir.SendStatement, spir.ReceiveStatement)):
        return True
    if isinstance(stmt, spir.ForeachStatement) and stmt.receive_stream:
        return get_dsd_op(dtypes, stmt) is not None

    return False


def create_csl_tasks(completion_dag: nx.DiGraph, block: spir.ComputeBlock, dtypes: dict[spir.Identifier,
                                                                                        spir.IRType]) -> list[CSLTask]:
    """
    Creates a list of CSL tasks. The nodes are tasks that contain a unique ID
    and the list of statements to include; and the edges are the type of dependency across tasks.
    The algorithm operates as follows.

    Statements can take on different task types, based on the statement type and its contents:

        * Foreach statements may take the form of a CSL data task, if they cannot trivially be represented by a
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
        2. A node with more than one incoming edge must start a new task 
        (the conditions below thus apply to the case where a node has one predecessor)
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
