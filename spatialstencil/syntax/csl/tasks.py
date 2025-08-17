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
    UNSET = auto()
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


# _DEBUG_i = 1


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
    # global _DEBUG_i
    # nx.nx_pydot.write_dot(completion_dag, f'canon{_DEBUG_i}.dot')
    # _DEBUG_i += 1

    # Mappings between IR statements and tasks
    cnode: analysis.CompletionDAGNode
    current_task: CSLTask = None
    statement_id_to_task_id: dict[int, int] = {}
    cnode_to_task_id: dict[analysis.CompletionDAGNode, int] = {}

    # Loop over completion DAG to coarsen completions to tasks
    for cnode in nx.topological_sort(completion_dag):
        node = block.statements[cnode.statement_id]

        # Figure out whether this task type is a local task or a data task
        if isinstance(node, spir.ForeachStatement) and get_dsd_op(dtypes, node) is None and cnode.optype == 'post':
            # Only if it is a complex task (i.e., not a DSD operation)
            this_task_type = 'data'
        else:
            this_task_type = 'local'

        task_id = None
        # Look at incoming edges:
        indeg = completion_dag.in_degree(cnode)
        if indeg == 1:  # A node with zero or more than one incoming edge has to start a new task
            pred: analysis.CompletionDAGNode
            pred, _ = next(iter(completion_dag.in_edges(cnode)))
            # If {wait,post}->post and there is one edge, and the previous task is a local task, inherit task ID
            if result[statement_id_to_task_id[pred.statement_id]].task_type == 'local' and this_task_type == 'local':
                if pred.optype == 'post' and cnode.optype == 'post':
                    task_id = statement_id_to_task_id[pred.statement_id]
                elif pred.optype == 'wait' and cnode.optype == 'post':
                    task_id = statement_id_to_task_id[pred.statement_id]
                elif pred.optype == 'post' and cnode.optype == 'wait':
                    # ``post->wait`` node pairs where the post is a nonblocking operation creates a new task for the
                    # ``wait`` node
                    if not should_be_asynchronous(dtypes, block.statements[pred.statement_id]):
                        task_id = statement_id_to_task_id[pred.statement_id]
                # wait->wait will create a new task
            elif result[statement_id_to_task_id[pred.statement_id]].task_type == 'local' and this_task_type == 'data':
                # An empty wait task before a data task can be contracted
                if not result[statement_id_to_task_id[pred.statement_id]].statements:
                    task_id = statement_id_to_task_id[pred.statement_id]

            # Otherwise, we need a new task

        # The one condition in which a wait->wait edge can be contracted is if there is a (post,post)->wait->wait,
        # which can be represented by two edges with unblock and activate.
        # TODO(later): this is a performance optimization that can be done later

        # If task ID is not None, append statement to prior task
        if task_id is not None:
            previous_task: CSLTask = result[task_id]
            cnode_to_task_id[cnode] = task_id

            # Modify task type
            if previous_task.task_type != this_task_type:
                previous_task.task_type = this_task_type

            if cnode.statement_id not in statement_id_to_task_id:
                previous_task.statements.append(cnode.statement_id)
                previous_task.outgoing.append((-1, InterTaskEdge.UNSET))
                statement_id_to_task_id[cnode.statement_id] = task_id

            continue

        # Create a new task
        task_id = len(result)
        cnode_to_task_id[cnode] = task_id
        current_task = CSLTask(task_id, this_task_type, [], [], blocked=(indeg > 1))
        result.append(current_task)
        statement_id_to_task_id[cnode.statement_id] = task_id

        if cnode.optype == 'wait':
            # Nothing to do within the task
            pass
        else:  # 'post'
            current_task.statements.append(cnode.statement_id)
            current_task.outgoing.append((-1, InterTaskEdge.UNSET))

    # For edge type detection
    task_has_activate: set[int] = set()

    # Determine edge types between task statements
    for cnode in nx.topological_sort(completion_dag):
        stmt_task = cnode_to_task_id[cnode]
        if cnode.optype == 'post':
            # Find matching "wait" successor
            succ_task = None
            for succ in completion_dag.successors(cnode):
                if succ.optype == 'wait':
                    succ_task = cnode_to_task_id[succ]
                    break

            assert succ_task is not None  # An asynchronous statement must have a unique successor

            # Find outgoing index within task
            ind = next(i for i, s in enumerate(result[stmt_task].statements) if s == cnode.statement_id)

            # Successor lives within same task, make sequence
            if stmt_task == succ_task:
                etype = InterTaskEdge.SEQUENCE
            else:
                # Check if task already has an activate edge
                if succ_task in task_has_activate:
                    etype = InterTaskEdge.UNBLOCK
                else:
                    etype = InterTaskEdge.ACTIVATE
                    task_has_activate.add(succ_task)
            result[stmt_task].outgoing[ind] = (succ_task, etype)

    # If the last task is local and empty, we can contract it with our exit task
    if len(result) > 0 and not result[-1].statements:
        result = result[:-1]

    # Determine terminators: if a task has a predecessor but no matching activator (outgoing statement),
    # add a terminator statement (@activate or @unblock, depending on other dependencies).
    # We define a terminator as a statement with ID -1
    for cnode in nx.topological_sort(completion_dag):
        preds = completion_dag.predecessors(cnode)
        stmt_task = cnode_to_task_id[cnode]
        for pred in preds:
            pred_task = cnode_to_task_id[pred]
            if pred_task == stmt_task:  # Skip sequential edges
                continue

            has_edge = any(e == stmt_task for e, _ in result[pred_task].outgoing)
            if not has_edge:
                task = result[pred_task]
                task.statements.append(-1)
                if stmt_task in task_has_activate:
                    task.outgoing.append((stmt_task, InterTaskEdge.UNBLOCK))
                else:
                    task.outgoing.append((stmt_task, InterTaskEdge.ACTIVATE))
                    task_has_activate.add(stmt_task)

    # Assign task IDs for local and data tasks
    current_local_task_id = -1
    current_data_task_id = -1
    task_id_to_local_id: dict[int, int] = {}
    task_id_to_data_id: dict[int, int] = {}
    for task_id, task in enumerate(result):
        # Increment the current task ID and add a new task with the specified type
        if task.task_type == 'local':
            current_local_task_id += 1
            task_id_to_local_id[task_id] = current_local_task_id
        else: # 'data'
            current_data_task_id += 1
            task_id_to_data_id[task_id] = current_data_task_id

    # Re-number task IDs and outgoing connections based on CSL IDs
    for task_id, task in enumerate(result):
        # TODO(later): Task IDs can be recycled with a global ``var`` that can be set prior to activating
        #              a task, like a state machine
        if task.task_type == 'data':
            tid = task_id_to_data_id[task_id]
            if tid > len(constants.DATA_TASK_IDS):
                raise ValueError('Too many data tasks')
            task.task_id = constants.DATA_TASK_IDS[tid]
        elif task.task_type == 'local':
            tid = task_id_to_local_id[task_id]
            if tid > len(constants.LOCAL_TASK_IDS):
                raise ValueError('Too many local tasks')
            task.task_id = constants.LOCAL_TASK_IDS[tid]

        for i, (target, e) in enumerate(task.outgoing):
            # Mark exit task explicitly
            if target == len(result):
                task.outgoing[i] = (-1, e)
                continue

            if result[target].task_type == 'local':
                target_id = constants.LOCAL_TASK_IDS[task_id_to_local_id[target]]
            else:
                target_id = constants.DATA_TASK_IDS[task_id_to_data_id[target]]
            task.outgoing[i] = (target_id, e)

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
