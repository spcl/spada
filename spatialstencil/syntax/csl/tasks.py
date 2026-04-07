"""
Contains a CSL task DAG representation and creation methods.
"""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
import networkx as nx  # TODO: Switch to igraph
from typing import Any, Literal, Optional
from spatialstencil.syntax.spatial_ir import irnodes as spir, analysis
from spatialstencil.syntax.csl import constants, dsd_ops, structures as cslstruct

UniqueDSDDict = dict[str, list[tuple[str, cslstruct.DataStructureDescriptor]]]


class TaskCreationBehavior(Enum):
    """
    Enumeration prescribing how tasks should be created.
    """
    NO_TASKS = auto()  # All statements in one task
    FAIL_ON_OVERRUN = auto()  # Error if too many tasks
    STATE_MACHINE_ON_OVERRUN = auto()  # Recycle task IDs with a state machine
    SYNCHRONOUS_ON_OVERRUN = auto()  # Run tasks synchronously if too many


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


def should_be_asynchronous(dtypes: dict[spir.Identifier, spir.IRType], stmt: spir.Statement) -> bool:
    """
    Returns True if a statement can and should be executed asynchronously in CSL.
    The only statements that apply are DSD operations that have to do with fabric DSDs (e.g., send, receive).
    """
    if isinstance(stmt, (spir.SendStatement, spir.ReceiveStatement)):
        return True
    if isinstance(stmt, spir.ForeachStatement) and stmt.receive_stream:
        return dsd_ops.get_dsd_op(dtypes, stmt) is not None

    return False


# _DEBUG_i = 1


def create_csl_tasks(
        completion_dag: nx.DiGraph,
        block: spir.ComputeBlock,
        dtypes: dict[spir.Identifier, spir.IRType],
        task_creation_behavior: TaskCreationBehavior = TaskCreationBehavior.FAIL_ON_OVERRUN) -> list[CSLTask]:
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
    num_local_tasks = 0
    num_data_tasks = 0

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
        if (isinstance(node, spir.ForeachStatement) and dsd_ops.get_dsd_op(dtypes, node) is None and
                cnode.optype == 'post'):
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
                    # ``wait`` node, depending on task creation behavior
                    should_create_task = True
                    if task_creation_behavior == TaskCreationBehavior.NO_TASKS:
                        should_create_task = False  # Always inherit prior task
                    elif task_creation_behavior == TaskCreationBehavior.SYNCHRONOUS_ON_OVERRUN:
                        if num_local_tasks >= len(constants.LOCAL_TASK_IDS):
                            should_create_task = False
                    if should_create_task and not should_be_asynchronous(dtypes, block.statements[pred.statement_id]):
                        should_create_task = False
                    if not should_create_task:
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
        current_task = CSLTask(
            task_id, this_task_type, [], [], blocked=((indeg > 1) or (this_task_type == 'data' and indeg > 0)))
        result.append(current_task)
        if this_task_type == 'local':
            num_local_tasks += 1
        else:
            num_data_tasks += 1
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

        elif cnode.optype == 'wait':  # Set the next task after the await to begin sequentially
            ind = next((i for i, s in enumerate(result[stmt_task].statements) if s == cnode.statement_id), None)
            if ind is None:  # Wait already omitted from task
                continue
            # After canonicalization, there must be one successor for each wait node
            num_successors = len(list(completion_dag.successors(cnode)))
            if num_successors == 1:
                succ_task = next(succ for succ in completion_dag.successors(cnode))
                succ_task = cnode_to_task_id[succ_task]
            elif num_successors > 1:
                node = block.statements[cnode.statement_id]
                raise ValueError('Multiple successors for a wait task should not appear after canonicalization.\n  In '
                                 f'line {node.lineinfo}')
            else:  # No successors
                continue

        # Determine edge type and assign outgoing edge
        # Successor lives within same task, make sequence
        if stmt_task == succ_task:
            etype = InterTaskEdge.SEQUENCE
        else:
            # Check if task already has an activate edge
            if succ_task in task_has_activate or result[succ_task].task_type == 'data':
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
    # We define a terminator as a statement with ID "TERMINATOR"
    for cnode in nx.topological_sort(completion_dag):
        preds = completion_dag.predecessors(cnode)
        stmt_task = cnode_to_task_id[cnode]
        for pred in preds:
            pred_task = cnode_to_task_id[pred]
            if pred_task == stmt_task:  # Skip sequential edges
                continue
            # Predecessor lived on the contracted empty last task; no task slot remains for it.
            if pred_task >= len(result):
                continue

            has_edge = any(e == stmt_task for e, _ in result[pred_task].outgoing)
            if not has_edge:
                task = result[pred_task]
                if task.task_type == "local":
                    task.statements.append("TERMINATOR")
                    # After contracting an empty trailing task, stmt_task may equal len(result), meaning exit.
                    succ_is_data = stmt_task < len(result) and result[stmt_task].task_type == 'data'
                    if stmt_task in task_has_activate or succ_is_data:
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
        else:  # 'data'
            current_data_task_id += 1
            task_id_to_data_id[task_id] = current_data_task_id

    # Re-number task IDs and outgoing connections based on CSL IDs
    for task_id, task in enumerate(result):
        # TODO(later): Task IDs can be recycled with a global ``var`` that can be set prior to activating
        #              a task, like a state machine

        # NOTE: This check happens after task fusion, so it is less likely to trigger
        # if task.task_type == 'data':
        #     tid = task_id_to_data_id[task_id]
        #     if tid >= len(constants.DATA_TASK_IDS) and task_creation_behavior == TaskCreationBehavior.FAIL_ON_OVERRUN:
        #         raise ValueError('Too many data tasks')
        #     task.task_id = constants.DATA_TASK_IDS[tid]
        # elif task.task_type == 'local':
        #     tid = task_id_to_local_id[task_id]
        #     if tid >= len(constants.LOCAL_TASK_IDS) and task_creation_behavior == TaskCreationBehavior.FAIL_ON_OVERRUN:
        #         raise ValueError('Too many local tasks')
        #     task.task_id = constants.LOCAL_TASK_IDS[tid]

        for i, (target, e) in enumerate(task.outgoing):
            # Mark exit task explicitly
            if target == len(result):
                task.outgoing[i] = (-1, e)
                continue

            # Do not modify outgoing task IDs
            # if result[target].task_type == 'local':
            #     target_id = constants.LOCAL_TASK_IDS[task_id_to_local_id[target]]
            # else:
            #     target_id = constants.DATA_TASK_IDS[task_id_to_data_id[target]]
            # task.outgoing[i] = (target_id, e)

    # Add explicit terminator statements for tasks that have no successors
    sink_tasks = [
        t for i, t in enumerate(result) if not any(n != i for n, _ in t.outgoing) or -1 in set(n for n, _ in t.outgoing)
    ]
    if len(sink_tasks) > 2:
        raise ValueError('Too many sink tasks')
    for i, task in enumerate(sink_tasks):
        edge_type = InterTaskEdge.ACTIVATE if i == 0 else InterTaskEdge.UNBLOCK
        if task_creation_behavior == TaskCreationBehavior.SYNCHRONOUS_ON_OVERRUN and num_local_tasks >= len(
                constants.LOCAL_TASK_IDS):
            edge_type = InterTaskEdge.SEQUENCE
        elif task_creation_behavior == TaskCreationBehavior.NO_TASKS:
            edge_type = InterTaskEdge.SEQUENCE
        elif len(sink_tasks) == 1:  # Save on task IDs if there is only one sink task
            edge_type = InterTaskEdge.SEQUENCE

        if task.statements[-1] != "TERMINATOR" and task.outgoing[-1][0] != -1 and task.task_type == "local":
            task.statements.append("TERMINATOR")
            task.outgoing.append((-1, edge_type))
        elif task.outgoing[-1][0] == -1:  # Modify existing UNBLOCK edge if there is more than one sink
            task.outgoing[-1] = (-1, edge_type)

    # Inject local tasks in front of data tasks with more than one input
    to_append = []
    for i, task in enumerate(result):
        if task.task_type != "data":
            continue
        predecessors = set(j for j, t in enumerate(result) if any(o == i for o, _ in t.outgoing))
        if len(predecessors) > 1:
            new_id = len(result) + len(to_append)
            new_task = CSLTask(i, "local", ["TERMINATOR"], [(new_id, InterTaskEdge.UNBLOCK)], blocked=True)
            to_append.append(task)
            result[i] = new_task
            # Make one of the predecessors into an ACTIVATE edge
            tpred = next(iter(predecessors))
            result[tpred].outgoing = [
                (o, InterTaskEdge.ACTIVATE) if o == i else (o, e) for o, e in result[tpred].outgoing
            ]

    result.extend(to_append)

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


def fuse_tasks(tasks: list[CSLTask], dsds: UniqueDSDDict, dtypes: dict[spir.Identifier, spir.IRType], rect,
               use_memcpy_mode: bool, compute: spir.ComputeBlock) -> list[CSLTask]:
    """
    Fuses tasks where possible to reduce the number of tasks.
    
    :param tasks: The list of CSL tasks.
    :param dsds: The unique DSD dictionary.
    :param dtypes: The dictionary of identifier types.
    :param kernel: The spatial IR kernel.
    :param use_memcpy_mode: Whether memcpy mode is used.
    :return: The fused list of CSL tasks.
    """
    fused: set[int] = set()
    removed: set[int] = set()
    redirect: dict[int, int] = {}
    for i, task in enumerate(tasks):
        if i in fused or i in removed:
            continue
        if task.task_type == 'data':
            continue
        if not task.statements:
            # Remove task
            removed.add(i)
            continue
        outgoing_id, _ = task.outgoing[-1]
        if outgoing_id == -1:
            # This is a sink task, nothing to fuse with
            continue
        if outgoing_id in fused or outgoing_id in removed:
            continue
        if any(et == InterTaskEdge.UNBLOCK for t in tasks for n, et in t.outgoing if n == outgoing_id):
            # Cannot fuse if there are multiple predecessors to next task
            continue
        if tasks[outgoing_id].task_type != 'local':
            # Cannot fuse with data tasks
            continue

        last_stmt = task.statements[-1]
        if not isinstance(last_stmt, int):
            # Last statement is a terminator, cannot fuse
            continue

        # Identify fusion opportunities
        next_task = tasks[outgoing_id] if outgoing_id < len(tasks) else None
        if next_task is None:
            continue
        stmt = compute.statements[last_stmt]
        if isinstance(stmt, (spir.SendStatement, spir.ReceiveStatement)):
            stream = dsd_ops._get_id(stmt.stream_name)
            localarr = dsd_ops._get_id(stmt.local_array)
            if stream.as_ir() in dsds and isinstance(dsds[stream.as_ir()][0][1], cslstruct.FabricDSD):
                # Cannot fuse if the last statement is a fabric DSD operation
                continue
            if localarr.as_ir() in dsds and isinstance(dsds[localarr.as_ir()][0][1], cslstruct.FabricDSD):
                # Cannot fuse if the last statement is a fabric DSD operation
                continue
        elif isinstance(stmt, spir.ForeachStatement):
            # Try to fuse synchronous DSD operations
            dsd_op: type[dsd_ops.DSDOp] = dsd_ops.DSD_ASSIGNMENT_MAPPING.get(dsd_ops.get_dsd_op(dtypes, stmt), None)
            if dsd_op is None:
                continue
            # Pure memory DSD operations are synchronous and can be fused
            dsd_stmt = dsd_ops.get_dsd_statement(dtypes, stmt)
            if dsd_stmt is None:
                continue
            dsd_objects = dsd_op().used_dsd_objects(dsd_stmt, dsds)
            if any(isinstance(dsd, cslstruct.FabricDSD) for dsd in dsd_objects):
                continue
            # Also check the foreach receive generator
            stream = dsd_ops._get_id(stmt.receive_stream.stream_name)
            if stream.as_ir() in dsds and isinstance(dsds[stream.as_ir()][0][1], cslstruct.FabricDSD):
                continue
        else:
            # Fusible pattern not found
            continue

        # If we can fuse with the next task, do so
        fused.add(outgoing_id)
        redirect[outgoing_id] = i
        task.outgoing[-1] = (i, InterTaskEdge.SEQUENCE)  # Remove outgoing edge
        task.statements.extend(next_task.statements)
        task.outgoing.extend(next_task.outgoing)

    new_tasks: list[CSLTask] = []
    old_to_new: dict[int, int] = {}
    for idx, task in enumerate(tasks):
        if idx in fused or idx in removed:
            continue
        new_idx = len(new_tasks)
        old_to_new[idx] = new_idx
        new_tasks.append(task)

    def resolve_target(target: int) -> int:
        while target in redirect:
            target = redirect[target]
        return target

    for task in new_tasks:
        for j, (target, et) in enumerate(task.outgoing):
            if target == -1:
                continue
            resolved = resolve_target(target)
            if resolved not in old_to_new:
                raise ValueError(f"Dangling task reference {resolved} after fusion")
            task.outgoing[j] = (old_to_new[resolved], et)

    return new_tasks


def renumber_tasks(tasks: list[CSLTask], task_creation_behavior: TaskCreationBehavior) -> None:
    """
    Renumbers tasks to map to hardware task IDs, based on the task creation behavior.

    :param tasks: The list of CSL tasks to operate in-place on.
    :param task_creation_behavior: The task creation behavior.
    """
    current_local_task_id = -1
    current_data_task_id = -1
    task_id_to_local_id: dict[int, int] = {}
    task_id_to_data_id: dict[int, int] = {}
    for task_id, task in enumerate(tasks):
        # Increment the current task ID and add a new task with the specified type
        if task.task_type == 'local':
            current_local_task_id += 1
            task_id_to_local_id[task_id] = current_local_task_id
        else:  # 'data'
            current_data_task_id += 1
            task_id_to_data_id[task_id] = current_data_task_id

    # Re-number task IDs and outgoing connections based on CSL IDs
    for task_id, task in enumerate(tasks):
        if task.task_type == 'data':
            tid = task_id_to_data_id[task_id]
            if tid >= len(constants.DATA_TASK_IDS) and task_creation_behavior == TaskCreationBehavior.FAIL_ON_OVERRUN:
                raise ValueError('Too many data tasks')
            task.task_id = constants.DATA_TASK_IDS[tid]
        elif task.task_type == 'local':
            tid = task_id_to_local_id[task_id]
            if tid >= len(constants.LOCAL_TASK_IDS) and task_creation_behavior == TaskCreationBehavior.FAIL_ON_OVERRUN:
                raise ValueError('Too many local tasks')
            task.task_id = constants.LOCAL_TASK_IDS[tid]
