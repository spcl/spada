"""
This module plans how logical CSL local tasks can share a smaller set of
hardware local-task IDs when the program contains more local tasks than the
target architecture exposes in :mod:`spatialstencil.syntax.csl.constants`.

Terminology
-----------

Logical task
        A ``tdag.CSLTask`` produced by ``create_csl_tasks``.  These are the tasks
        the completion DAG wants to execute.

Hardware slot
        One reusable local-task ID from ``constants.LOCAL_TASK_IDS`` together with
        one generated CSL task function.  The generated function acts as a
        dispatcher for all logical tasks mapped to that slot.

State
        A dense integer assigned to one logical task within a slot.  Codegen emits
        a slot-local state variable such as ``__task_slot_0_state`` and the shared
        dispatcher selects the active logical task by branching on that value.

Representative task
        The first logical task assigned to a slot.  Codegen binds the generated CSL
        task function once, using this task's logical alias.  All other logical
        tasks in the same slot reuse the same hardware ID through additional
        ``const task_<i>_id = @get_local_task_id(...)`` aliases.

High-level methodology
----------------------

The planner works in three phases.

1. Collect local tasks

     Only ``task.task_type == 'local'`` participates in recycling.  Data tasks
     have their own binding scheme and are not handled here.

2. Decide whether recycling is needed

     If the requested task-creation behavior forbids recycling, or if the number
     of local tasks already fits in the available hardware IDs, the planner emits
     a trivial one-task-per-slot mapping.

3. Assign overflow tasks to slots

    Each logical local task becomes a vertex in a conflict graph.  An edge means
    two tasks must not share a slot.

    Slot assignment is solved by load-balanced greedy graph coloring in
    degeneracy (smallest-last) order.  All tasks are colored together without
    any fixed prefix; among eligible colors the least-loaded one is chosen so
    that tasks are spread evenly across hardware slots, keeping dispatcher
    state machines small.  The algorithm runs in O((V+E) log V) time.

Safety criterion
----------------

Two logical tasks may share a hardware slot only if one of them is guaranteed
to occur before *every possible trigger* of the other.

That is stricter than simple topological order.  A plain topological order only
states that task ``A`` comes before task ``B`` in one legal linearization of the
task DAG.  It does **not** say that the hardware slot formerly used for ``A`` is
dead before ``B`` can become runnable.  In this lowering pipeline a task can be
made runnable by ``@activate`` or ``@unblock`` edges from multiple predecessors,
and those trigger points are what matter for recycling safety.

This module therefore computes two auxiliary relations on the task DAG:

``reachable[A]``
        The set of tasks reachable from ``A`` in the task graph. If ``C`` is in ``reachable[A]`` we say ``A`` is an _ancestor_ of ``C``. Note that every task is its own ancestor.

``trigger_sources[B]``
        The set of tasks that can directly trigger ``B``.  These are the immediate
        predecessors of ``B`` in the task graph after DAG construction.

With those relations, ``A`` is considered to _safely precede_ ``B`` iff ``A`` is an ancestor of every trigger source of ``B``.  
In code this is the predicate
implemented by :func:`_precedes_all_trigger_sources`.

Note that we must consider the trigger sources of ``B``, because a trigger source must be able to activate/unblock ``B``: For this, the task id of ``B`` must not be concurrently used by ``A``.

Two tasks conflict if neither direction holds:

* ``A`` does not safely precede all trigger sources of ``B``, and
* ``B`` does not safely precede all trigger sources of ``A``.

When that happens, the tasks may be simultaneously live from the point of view
of slot reuse and therefore need distinct hardware IDs.

When that happens, the tasks may be simultaneously live from the point of view
of slot reuse and therefore need distinct hardware IDs.

Why blocked tasks need extra handling
-------------------------------------

Some logical tasks start in the blocked state.  With unique hardware IDs this
is easy: codegen binds the task once and emits a static ``@block(task_id)``.

Recycling complicates that story.  A single hardware slot may first represent
an unblocked logical task and later be reused for a blocked logical task.  In
that case the old slot state is no longer valid.  Before the recycled slot is
used as the blocked logical task, codegen must:

* detect that the slot is transitioning to a different logical state,
* issue ``@block(task_id)`` for the logical alias being installed, and
* then write the new state value.

That is exactly what :meth:`TaskBindingPlan.emit_local_transition_preamble`
returns.  The planner itself does not emit CSL, but it defines the contract the
lowering code follows.

Code generation contract
------------------------

The plan returned by :func:`plan_task_bindings` is consumed by the lowering
stage with the following conventions.

* Every logical local task still gets a logical alias named ``task_<i>_id``.
* Tasks that share a slot all point to the same ``hardware_task_id``.
* Every recycled slot gets one state variable.
* Lowering emits one dispatcher task function per slot.
* The dispatcher uses a single ``if`` / ``else if`` chain so only one logical
    branch can run per invocation.
* Before activating or unblocking a recycled local task, lowering emits the
    transition preamble returned by this module.

Determinism
-----------

The planner intentionally uses deterministic ordering so generated code is
stable across runs.

* Local tasks are considered in their original task-list order.
* Prefix-preserving overflow recycling keeps the initial unique slots fixed.
* Colors are turned into slots in sorted color order.
* Tasks within one slot are stored in sorted logical-task order.
* State numbers are assigned by that sorted order.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable

from spatialstencil.syntax.csl import constants
from spatialstencil.syntax.csl import tasks as tdag


@dataclass(frozen=True)
class LocalTaskSlot:
    """
    A single reusable hardware local-task slot.

    ``task_indices`` lists every logical local task assigned to this slot.
    If the tuple contains more than one task, codegen emits one shared CSL task
    function that dispatches between the logical tasks via a slot-state
    variable.
    """

    slot_index: int
    hardware_task_id: int
    task_indices: tuple[int, ...]

    @property
    def representative_task_index(self) -> int:
        """Return the first logical task bound to this slot.

        Lowering uses this task as the stable representative when binding the
        generated CSL task function to a hardware local-task ID.
        """
        return self.task_indices[0]

    @property
    def recycled(self) -> bool:
        """Whether the slot hosts more than one logical task."""
        return len(self.task_indices) > 1


@dataclass(frozen=True)
class TaskBindingPlan:
    """
        Binding information consumed by CSL code generation.

        The plan intentionally separates two concerns:

        * ``local_slots`` describes the physical slot layout that codegen must
            materialize, and
        * the two dictionaries provide fast reverse mappings from logical task index
            to slot number and per-slot state number.
    """

    local_slots: tuple[LocalTaskSlot, ...]
    task_to_local_slot: dict[int, int]
    task_to_local_state: dict[int, int]

    @property
    def uses_recycling(self) -> bool:
        """Whether any hardware slot is shared by multiple logical tasks."""
        return any(slot.recycled for slot in self.local_slots)

    def local_slot(self, task_index: int) -> LocalTaskSlot:
        """Return the slot containing ``task_index``."""
        return self.local_slots[self.task_to_local_slot[task_index]]

    def local_state(self, task_index: int) -> int:
        """Return the per-slot state number assigned to ``task_index``."""
        return self.task_to_local_state[task_index]

    def is_recycled_local_task(self, task_index: int) -> bool:
        """Return whether ``task_index`` shares its slot with another task."""
        return self.local_slot(task_index).recycled

    def state_var(self, task_index: int) -> str:
        """Return the generated CSL state variable name for ``task_index``'s slot."""
        return f'__task_slot_{self.task_to_local_slot[task_index]}_state'

    def invalid_state_literal(self, task_index: int) -> str:
        """Return the sentinel state used to mean "no logical task installed".

        States are numbered densely from ``0`` to ``len(slot.task_indices) - 1``.
        The value ``len(slot.task_indices)`` is therefore guaranteed to be
        outside the valid range and can be used as an initialization/reset value.
        """
        return str(len(self.local_slot(task_index).task_indices))

    def local_function_name(self, slot: LocalTaskSlot) -> str:
        """Return the generated dispatcher task name for ``slot``."""
        return f'task_slot_{slot.slot_index}'

    def emit_local_transition_preamble(
        self,
        task_index: int,
        blocked: bool,
        indent: str = '    ',
    ) -> str:
        """Emit the state-change preamble required before using a recycled slot.

        The returned snippet is inserted by codegen immediately before an
        ``@activate``/``@unblock`` or before generating an asynchronous DSD op
        that targets another local task.

        For recycled blocked tasks we must dynamically restore the blocked state
        whenever the slot changes identity.  Without that re-priming step a slot
        that was previously used for an unblocked logical task could be reused
        as a blocked task while still remaining runnable.

        Non-recycled tasks return an empty string because their hardware state is
        fixed for the entire program.
        """
        if not self.is_recycled_local_task(task_index):
            return ''

        lines: list[str] = []
        state_var = self.state_var(task_index)
        state_value = self.local_state(task_index)
        task_id = f'task_{task_index}_id'
        if blocked:
            lines.append(f'{indent}if ({state_var} != {state_value}) {{')
            lines.append(f'{indent}    @block({task_id});')
            lines.append(f'{indent}}}')
        lines.append(f'{indent}{state_var} = {state_value};')
        return '\n'.join(lines) + '\n'


def plan_task_bindings(
    tasks: list[tdag.CSLTask],
    task_creation_behavior: tdag.TaskCreationBehavior,
) -> TaskBindingPlan:
    """Compute a local-task binding plan for the generated CSL.

    Returns either a trivial one-task-per-slot mapping when recycling is not
    required or not allowed, or a state-machine-compatible sharing plan when
    local-task overrun occurs.

    ``STATE_MACHINE_ON_OVERRUN`` is the only mode that attempts recycling.
    Other modes either keep a unique mapping or raise when the local task count
    exceeds the hardware limit.

    When recycling is needed, all tasks are colored together using
    load-balanced greedy coloring in degeneracy order, distributing tasks
    evenly across hardware slots to minimise dispatcher state machine size.
    """
    local_task_indices = [i for i, task in enumerate(tasks) if task.task_type == 'local']
    if not local_task_indices:
        return TaskBindingPlan((), {}, {})

    if task_creation_behavior in (
        tdag.TaskCreationBehavior.FAIL_ON_OVERRUN,
        tdag.TaskCreationBehavior.SYNCHRONOUS_ON_OVERRUN,
    ):
        if len(local_task_indices) > len(constants.LOCAL_TASK_IDS):
            raise ValueError('Too many local tasks')
        return _plan_unique_slots(local_task_indices)

    if task_creation_behavior == tdag.TaskCreationBehavior.NO_TASKS:
        return _plan_unique_slots(local_task_indices)

    if len(local_task_indices) <= len(constants.LOCAL_TASK_IDS):
        return _plan_unique_slots(local_task_indices)

    max_colors = len(constants.LOCAL_TASK_IDS)
    conflict_graph = _build_conflict_graph(tasks, local_task_indices)
    coloring = greedy_coloring(conflict_graph, local_task_indices, max_colors=max_colors, load_balance=True)
    if coloring is None:
        raise ValueError('Too many concurrently-live local tasks for state-machine recycling')

    return _build_plan_from_coloring(coloring)


def _plan_unique_slots(local_task_indices: list[int]) -> TaskBindingPlan:
    """Build the trivial plan where each local task receives its own slot."""
    coloring = {task_index: color for color, task_index in enumerate(local_task_indices)}
    return _build_plan_from_coloring(coloring)


def _build_conflict_graph(tasks: list[tdag.CSLTask], local_task_indices: Iterable[int]) -> dict[int, set[int]]:
    """Construct the slot-sharing conflict graph for local tasks.

    Vertices are logical local task indices.  An undirected edge ``A -- B``
    means the two tasks must not reuse the same hardware slot.

    The graph is intentionally conservative: if we cannot prove that one task
    precedes all trigger sources of the other, we treat them as conflicting.
    """
    adjacency: dict[int, set[int]] = {i: set() for i in local_task_indices}
    reachable = _compute_reachability(tasks)
    trigger_sources = _trigger_sources(tasks)

    local_task_indices = list(local_task_indices)
    for idx, left in enumerate(local_task_indices):
        for right in local_task_indices[idx + 1:]:
            if _conflicts(left, right, trigger_sources, reachable):
                adjacency[left].add(right)
                adjacency[right].add(left)

    return adjacency


def _compute_reachability(tasks: list[tdag.CSLTask]) -> list[set[int]]:
    """Return transitive reachability sets for the task graph.

    ``reachable[A]`` contains every task that can execute after ``A`` by
    following outgoing inter-task edges, excluding the synthetic exit target
    ``-1`` and self-loops.
    """
    reachable: list[set[int]] = [set() for _ in tasks]
    for task_index in range(len(tasks) - 1, -1, -1):
        for target, _ in tasks[task_index].outgoing:
            if target == -1 or target == task_index:
                continue
            reachable[task_index].add(target)
            reachable[task_index].update(reachable[target])
    return reachable


def _trigger_sources(tasks: list[tdag.CSLTask]) -> dict[int, set[int]]:
    """Return the direct trigger sources for every task.

    A trigger source is a predecessor that can make the target runnable through
    an ``ACTIVATE`` or ``UNBLOCK``-style edge.  These immediate predecessors are
    the boundary we care about for recycling safety.
    """
    sources: dict[int, set[int]] = {i: set() for i in range(len(tasks))}
    for source_index, task in enumerate(tasks):
        for target, _ in task.outgoing:
            if target == -1 or target == source_index:
                continue
            sources[target].add(source_index)
    return sources


def _precedes_all_trigger_sources(
    left: int,
    right: int,
    trigger_sources: dict[int, set[int]],
    reachable: list[set[int]],
) -> bool:
    """Whether ``left`` is guaranteed to happen before ``right`` can be triggered.

    ``right`` is safe to place after ``left`` on the same hardware slot only if
    every direct trigger source of ``right`` is either:

    * ``left`` itself, or
    * reachable from ``left``.

    If ``right`` has no trigger sources, the function returns ``False`` because
    there is no evidence that ``left`` dominates the point where ``right`` can
    become runnable.
    """
    right_sources = trigger_sources[right]
    if not right_sources:
        return False
    return all(source == left or source in reachable[left] for source in right_sources)


def _conflicts(
    left: int,
    right: int,
    trigger_sources: dict[int, set[int]],
    reachable: list[set[int]],
) -> bool:
    """Return whether two logical tasks must be assigned distinct slots.

    Reuse is safe only if one task strictly precedes all trigger sources of the
    other.  If neither direction can be proven, the tasks are treated as
    conflicting and therefore receive different graph colors.
    """
    return not (_precedes_all_trigger_sources(left, right, trigger_sources, reachable) or
                _precedes_all_trigger_sources(right, left, trigger_sources, reachable))


def _degeneracy_order(conflict_graph: dict[int, set[int]], vertices: list[int]) -> list[int]:
    """Compute the degeneracy (smallest-last) ordering for ``vertices``.

    Iteratively removes the vertex with minimum current degree in the subgraph
    induced by ``vertices``, building an elimination sequence.  The coloring
    order returned is the reverse of that sequence, so each vertex has at most
    *d* already-colored neighbors when it is processed (where *d* is the
    degeneracy of the induced subgraph).  Ties are broken by vertex index for
    reproducibility.

    Time complexity: O((|V| + |E|) log |V|).
    """
    remaining: set[int] = set(vertices)
    degree: dict[int, int] = {
        v: sum(1 for u in conflict_graph[v] if u in remaining) for v in vertices
    }

    heap: list[tuple[int, int]] = [(degree[v], v) for v in vertices]
    heapq.heapify(heap)

    elimination: list[int] = []
    done: set[int] = set()

    while heap:
        _, v = heapq.heappop(heap)
        if v in done:
            continue
        done.add(v)
        remaining.discard(v)
        elimination.append(v)
        for u in conflict_graph[v]:
            if u in remaining:
                degree[u] -= 1
                heapq.heappush(heap, (degree[u], u))

    return list(reversed(elimination))


def greedy_coloring(
    conflict_graph: dict[int, set[int]],
    vertices: list[int],
    fixed_coloring: dict[int, int] | None = None,
    max_colors: int | None = None,
    load_balance: bool = False,
) -> dict[int, int] | None:
    """Greedy graph coloring using the degeneracy (smallest-last) vertex order.

    Parameters
    ----------
    conflict_graph:
        Adjacency dictionary for the full conflict graph.  Keys are vertex
        identifiers; values are sets of conflicting vertices.
    vertices:
        All vertices to include in the coloring (both fixed and free).
    fixed_coloring:
        Optional pre-assigned colors for a subset of vertices.  These vertices
        keep their colors and are excluded from the degeneracy ordering.  Their
        colors are respected when assigning colors to free vertices.
    max_colors:
        If given, the function returns ``None`` as soon as any free vertex
        would require a color index ``>= max_colors``, allowing the caller to
        detect infeasibility cheaply without backtracking.
    load_balance:
        When ``True``, among all eligible colors the one with the fewest
        current assignments is chosen (ties broken by color index).  This
        spreads free vertices across existing colors rather than collapsing
        them onto color 0, which is desirable when a fixed prefix already
        occupies every color and the overflow tasks should be distributed
        evenly across hardware slots.  When ``False`` (the default), the
        standard first-fit rule is used (smallest eligible color index),
        which minimises the total number of colors.

    Returns
    -------
    A complete coloring dictionary (fixed + newly assigned), or ``None`` if
    ``max_colors`` is set and the greedy assignment would exceed it.

    Notes
    -----
    The degeneracy order guarantees that the number of colors used is at most
    *d* + 1, where *d* is the degeneracy of the subgraph induced by the free
    vertices.  This is optimal for chordal graphs and a good heuristic in
    general — in practice task conflict graphs are sparse and the bound is
    tight.  Unlike the backtracking solvers this function runs in
    O((|V| + |E|) log |V|) time.
    """
    if fixed_coloring is None:
        fixed_coloring = {}

    free_vertices = [v for v in vertices if v not in fixed_coloring]
    ordering = _degeneracy_order(conflict_graph, free_vertices)

    coloring: dict[int, int] = dict(fixed_coloring)

    color_load: dict[int, int] = {}
    if load_balance:
        for c in fixed_coloring.values():
            color_load[c] = color_load.get(c, 0) + 1

    for v in ordering:
        used = {coloring[u] for u in conflict_graph[v] if u in coloring}
        if load_balance and max_colors is not None:
            eligible = [c for c in range(max_colors) if c not in used]
            if not eligible:
                return None
            color = min(eligible, key=lambda c: (color_load.get(c, 0), c))
        else:
            color = 0
            while color in used:
                color += 1
            if max_colors is not None and color >= max_colors:
                return None
        coloring[v] = color
        if load_balance:
            color_load[color] = color_load.get(color, 0) + 1

    return coloring



def _build_plan_from_coloring(coloring: dict[int, int]) -> TaskBindingPlan:
    """Convert a graph coloring into the stable binding structures used by codegen.

    Each color becomes one ``LocalTaskSlot`` backed by the corresponding
    hardware ID in ``constants.LOCAL_TASK_IDS``.  Within each slot, logical task
    indices are sorted to make state assignment deterministic.  The position of a
    task inside that sorted tuple is its per-slot state number.
    """
    color_to_tasks: dict[int, list[int]] = {}
    for task_index, color in coloring.items():
        color_to_tasks.setdefault(color, []).append(task_index)

    local_slots: list[LocalTaskSlot] = []
    task_to_local_slot: dict[int, int] = {}
    task_to_local_state: dict[int, int] = {}
    for slot_index, color in enumerate(sorted(color_to_tasks)):
        task_indices = tuple(sorted(color_to_tasks[color]))
        slot = LocalTaskSlot(slot_index, constants.LOCAL_TASK_IDS[slot_index], task_indices)
        local_slots.append(slot)
        for state_index, task_index in enumerate(task_indices):
            task_to_local_slot[task_index] = slot_index
            task_to_local_state[task_index] = state_index

    return TaskBindingPlan(tuple(local_slots), task_to_local_slot, task_to_local_state)
