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

    By default the planner keeps the first ``len(constants.LOCAL_TASK_IDS)`` local
    tasks on distinct hardware slots and only recycles the overflow tail.  That
    preserves the early task structure and keeps each dispatcher as small as the
    conflict constraints allow.  For comparison or experimentation, callers can
    opt back into the older whole-graph coloring mode that recolors every local
    task together.

The important part is how the conflict graph is built.

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
        The set of tasks reachable from ``A`` through outgoing inter-task edges.
        If a trigger source is in ``reachable[A]``, then ``A`` must happen before
        that source.

``trigger_sources[B]``
        The set of tasks that can directly trigger ``B``.  These are the immediate
        predecessors of ``B`` in the task graph after DAG construction.

With those relations, ``A`` is considered to safely precede ``B`` iff ``A``
precedes every direct trigger source of ``B``.  In code this is the predicate
implemented by :func:`_precedes_all_trigger_sources`.

Two tasks conflict if neither direction holds:

* ``A`` does not precede all trigger sources of ``B``, and
* ``B`` does not precede all trigger sources of ``A``.

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

This does not try to optimize for minimal state numbers or minimal branching
depth; it optimizes for correctness first and reproducibility second.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    recycle_overflow_only: bool = True,
) -> TaskBindingPlan:
    """
    Compute a local-task binding plan for the generated CSL.

    The function returns either:

    * a trivial one-task-per-slot mapping when recycling is not required or not
      allowed, or
    * a state-machine-compatible sharing plan when local-task overrun occurs.

    The planning process is:

    1. collect all logical local tasks,
    2. fast-path to unique slots when possible,
     3. otherwise either preserve a unique prefix and color only the overflow
         tail, or recolor the full local-task set,
     4. solve the resulting coloring problem, and
    5. convert colors into stable ``LocalTaskSlot`` objects and per-task state
       assignments.

    ``STATE_MACHINE_ON_OVERRUN`` is the only mode that actually attempts
    recycling.  Other modes either keep a unique mapping or raise when the local
    task count exceeds the hardware limit.

     When ``recycle_overflow_only`` is true, the planner keeps the earliest local
     tasks on dedicated hardware slots and only searches for placements for the
     overflow tail.  The search minimizes the largest number of logical tasks per
     hardware slot so generated state machines stay small.
    """
    local_task_indices = [i for i, task in enumerate(tasks) if task.task_type == 'local']
    if not local_task_indices:
        return TaskBindingPlan((), {}, {})

    if task_creation_behavior == tdag.TaskCreationBehavior.FAIL_ON_OVERRUN:
        if len(local_task_indices) > len(constants.LOCAL_TASK_IDS):
            raise ValueError('Too many local tasks')
        return _plan_unique_slots(local_task_indices)

    if task_creation_behavior == tdag.TaskCreationBehavior.SYNCHRONOUS_ON_OVERRUN:
        if len(local_task_indices) > len(constants.LOCAL_TASK_IDS):
            raise ValueError('Too many local tasks')
        return _plan_unique_slots(local_task_indices)

    if task_creation_behavior == tdag.TaskCreationBehavior.NO_TASKS:
        return _plan_unique_slots(local_task_indices)

    if len(local_task_indices) <= len(constants.LOCAL_TASK_IDS):
        return _plan_unique_slots(local_task_indices)

    if recycle_overflow_only:
        overflow_coloring = _overflow_only_coloring(tasks, local_task_indices, len(constants.LOCAL_TASK_IDS))
        if overflow_coloring is None:
            raise ValueError(
                'Too many concurrently-live local tasks for overflow-only state-machine recycling; '
                'disable recycle_overflow_only to allow full recoloring'
            )
        return _build_plan_from_coloring(overflow_coloring)

    conflict_graph = _build_conflict_graph(tasks, local_task_indices)
    topological_local_order = list(local_task_indices)
    initial_coloring = _first_fit_coloring(conflict_graph, list(reversed(topological_local_order)))
    num_colors = 1 + max(initial_coloring.values(), default=-1)

    if num_colors > len(constants.LOCAL_TASK_IDS):
        exact_coloring = _bounded_coloring(conflict_graph, topological_local_order, len(constants.LOCAL_TASK_IDS))
        if exact_coloring is None:
            raise ValueError('Too many concurrently-live local tasks for state-machine recycling')
        initial_coloring = exact_coloring

    return _build_plan_from_coloring(initial_coloring)


def _plan_unique_slots(local_task_indices: list[int]) -> TaskBindingPlan:
    """Build the trivial plan where each local task receives its own slot."""
    coloring = {task_index: color for color, task_index in enumerate(local_task_indices)}
    return _build_plan_from_coloring(coloring)


def _overflow_only_coloring(
    tasks: list[tdag.CSLTask],
    topological_local_order: list[int],
    max_colors: int,
) -> dict[int, int] | None:
    """Color only the overflow tail while keeping the initial prefix fixed.

    The first ``max_colors`` local tasks are pinned to distinct colors in their
    original order.  Remaining tasks are then placed onto those existing colors
    subject to the same slot-sharing conflict graph used by the legacy global
    recoloring mode.

    To keep recycled state machines small, the search finds the smallest
    possible upper bound on the number of logical tasks assigned to any one
    hardware slot.
    """
    fixed_prefix = topological_local_order[:max_colors]
    fixed_coloring = {task_index: color for color, task_index in enumerate(fixed_prefix)}
    if len(topological_local_order) <= max_colors:
        return fixed_coloring

    conflict_graph = _build_conflict_graph(tasks, topological_local_order)
    min_slot_size = (len(topological_local_order) + max_colors - 1) // max_colors
    max_slot_size = len(topological_local_order) - max_colors + 1
    best_coloring: dict[int, int] | None = None
    lower_bound = min_slot_size
    upper_bound = max_slot_size

    while lower_bound <= upper_bound:
        trial_slot_size = (lower_bound + upper_bound) // 2
        candidate = _bounded_coloring_with_fixed_prefix(
            conflict_graph,
            topological_local_order,
            max_colors,
            fixed_coloring,
            trial_slot_size,
        )
        if candidate is None:
            lower_bound = trial_slot_size + 1
            continue
        best_coloring = candidate
        upper_bound = trial_slot_size - 1

    return best_coloring


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


def _first_fit_coloring(conflict_graph: dict[int, set[int]], order: list[int]) -> dict[int, int]:
    """Greedily color the conflict graph.

    This is a cheap first pass used both as the final answer when it fits within
    the hardware limit and as a quick estimate of how many colors are likely to
    be needed before attempting the exact bounded search.
    """
    coloring: dict[int, int] = {}
    for vertex in order:
        used = {coloring[neighbor] for neighbor in conflict_graph[vertex] if neighbor in coloring}
        color = 0
        while color in used:
            color += 1
        coloring[vertex] = color
    return coloring


def _bounded_coloring(
    conflict_graph: dict[int, set[int]],
    topological_local_order: list[int],
    max_colors: int,
) -> dict[int, int] | None:
    """Search for a coloring that uses at most ``max_colors`` colors.

    The planner first tries greedy coloring.  If the greedy result exceeds the
    hardware slot budget, this function performs a backtracking search.  The
    search order prioritizes high-degree vertices first and uses the original
    local-task order as a tie-breaker for reproducibility.

    Returning ``None`` means the conflict graph cannot be colored within the
    available number of hardware local-task IDs, so recycling is fundamentally
    impossible for this task graph.
    """
    vertices = sorted(
        conflict_graph,
        key=lambda vertex: (len(conflict_graph[vertex]), topological_local_order.index(vertex)),
        reverse=True,
    )
    coloring: dict[int, int] = {}

    def backtrack(position: int) -> bool:
        if position == len(vertices):
            return True

        vertex = vertices[position]
        used = {coloring[neighbor] for neighbor in conflict_graph[vertex] if neighbor in coloring}
        for color in range(max_colors):
            if color in used:
                continue
            coloring[vertex] = color
            if backtrack(position + 1):
                return True
            del coloring[vertex]
        return False

    if not backtrack(0):
        return None
    return coloring


def _bounded_coloring_with_fixed_prefix(
    conflict_graph: dict[int, set[int]],
    topological_local_order: list[int],
    max_colors: int,
    fixed_coloring: dict[int, int],
    max_slot_size: int,
) -> dict[int, int] | None:
    """Color the overflow tail with a fixed prefix and bounded slot occupancy.

    This is an exact search.  ``fixed_coloring`` pins the unique non-recycled
    prefix to predetermined colors, and the backtracking search assigns colors
    only to the remaining tasks.  A candidate is accepted only if no slot ends
    up hosting more than ``max_slot_size`` logical tasks.
    """
    vertices = [vertex for vertex in topological_local_order if vertex not in fixed_coloring]
    coloring = dict(fixed_coloring)
    color_loads = [0] * max_colors
    for color in fixed_coloring.values():
        color_loads[color] += 1

    if any(load > max_slot_size for load in color_loads):
        return None

    local_order = {vertex: index for index, vertex in enumerate(topological_local_order)}

    def candidate_colors(vertex: int) -> list[int]:
        used = {coloring[neighbor] for neighbor in conflict_graph[vertex] if neighbor in coloring}
        return [
            color for color in range(max_colors)
            if color not in used and color_loads[color] < max_slot_size
        ]

    def backtrack(uncolored: tuple[int, ...]) -> bool:
        if not uncolored:
            return True

        remaining_capacity = sum(max_slot_size - load for load in color_loads)
        if len(uncolored) > remaining_capacity:
            return False

        candidates_by_vertex = {vertex: candidate_colors(vertex) for vertex in uncolored}
        vertex = min(
            uncolored,
            key=lambda item: (len(candidates_by_vertex[item]), -len(conflict_graph[item]), local_order[item]),
        )
        candidates = candidates_by_vertex[vertex]
        if not candidates:
            return False

        next_uncolored = tuple(item for item in uncolored if item != vertex)
        for color in sorted(candidates, key=lambda item: (color_loads[item], item)):
            coloring[vertex] = color
            color_loads[color] += 1
            if backtrack(next_uncolored):
                return True
            color_loads[color] -= 1
            del coloring[vertex]
        return False

    if not backtrack(tuple(vertices)):
        return None
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
