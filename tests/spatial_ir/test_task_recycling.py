import os
import pytest

from spada.lowering import spatial_ir_to_csl as s2c
from spada.syntax.csl import constants, task_recycling, tasks as tdag
from spada.syntax.spatial_ir import analysis, parser, passes
from spada.syntax.spatial_ir.canonicalization import PEBlock


def _load_sample_kernel():
    sample = os.path.join(
        os.path.dirname(__file__), '..', 'csl_runtime', 'samples', f'task_recycling_merge_{constants.ARCH}.sptl')
    kernel = parser.parse_file(sample)
    return passes.constexpr_propagation(kernel)


def _create_unfused_tasks():
    kernel = _load_sample_kernel()
    place, dataflow, compute = kernel.body
    block = PEBlock(place, dataflow, compute)
    dtypes = s2c._collect_identifier_types(block, kernel.arguments)
    completion_dag = analysis.to_completion_dag(block.compute)
    tasks = tdag.create_csl_tasks(
        completion_dag,
        block.compute,
        dtypes,
        tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN,
    )
    return tasks


def _create_linear_local_tasks(length: int):
    tasks = []
    for task_index in range(length):
        outgoing = []
        if task_index + 1 < length:
            outgoing.append((task_index + 1, tdag.InterTaskEdge.ACTIVATE))
        tasks.append(
            tdag.CSLTask(
                task_id=task_index,
                task_type='local',
                statements=[task_index],
                outgoing=outgoing,
                blocked=False,
            ))
    return tasks


def test_task_recycling_overflow_load_balanced():
    local_task_count = len(constants.LOCAL_TASK_IDS) + 5
    tasks = _create_linear_local_tasks(local_task_count)

    plan = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)

    # All hardware slots are used.
    assert len(plan.local_slots) == len(constants.LOCAL_TASK_IDS)
    # With N+5 conflict-free tasks spread across N slots, each slot holds at most 2 tasks.
    assert all(len(slot.task_indices) <= 2 for slot in plan.local_slots)
    # Every task is assigned to exactly one slot.
    assert {t for slot in plan.local_slots for t in slot.task_indices} == set(range(local_task_count))


def test_task_recycling_all_tasks_assigned():
    tasks = _create_linear_local_tasks(len(constants.LOCAL_TASK_IDS) + 5)

    plan = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)

    # Every logical task appears in exactly one slot.
    assigned = [t for slot in plan.local_slots for t in slot.task_indices]
    assert sorted(assigned) == list(range(len(tasks)))


def test_task_recycling_plan_reuses_local_slots():
    tasks = _create_unfused_tasks()
    local_task_count = sum(1 for task in tasks if task.task_type == 'local')

    assert local_task_count > len(constants.LOCAL_TASK_IDS)

    plan = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)

    assert len(plan.local_slots) <= len(constants.LOCAL_TASK_IDS)
    assert any(slot.recycled for slot in plan.local_slots)

    blocked_recycled = [
        task_index for task_index, task in enumerate(tasks)
        if task.task_type == 'local' and task.blocked and plan.is_recycled_local_task(task_index)
    ]
    assert blocked_recycled

    merge_task = blocked_recycled[0]
    shared_slot = plan.local_slot(merge_task)
    assert any(other < merge_task for other in shared_slot.task_indices if other != merge_task)


# ---------------------------------------------------------------------------
# Conflict graph helpers
# ---------------------------------------------------------------------------


def _create_fork_tasks():
    """Task 0 activates tasks 1 and 2 independently; 1 and 2 are concurrent."""
    return [
        tdag.CSLTask(
            0, 'local', [0], [(1, tdag.InterTaskEdge.ACTIVATE), (2, tdag.InterTaskEdge.ACTIVATE)], blocked=False),
        tdag.CSLTask(1, 'local', [1], [(-1, tdag.InterTaskEdge.SEQUENCE)], blocked=False),
        tdag.CSLTask(2, 'local', [2], [(-1, tdag.InterTaskEdge.SEQUENCE)], blocked=False),
    ]


def _create_diamond_tasks():
    """0 forks into 1 and 2, which join at the blocked task 3."""
    return [
        tdag.CSLTask(
            0, 'local', [0], [(1, tdag.InterTaskEdge.ACTIVATE), (2, tdag.InterTaskEdge.ACTIVATE)], blocked=False),
        tdag.CSLTask(1, 'local', [1], [(3, tdag.InterTaskEdge.ACTIVATE)], blocked=False),
        tdag.CSLTask(2, 'local', [2], [(3, tdag.InterTaskEdge.UNBLOCK)], blocked=False),
        tdag.CSLTask(3, 'local', [3], [(-1, tdag.InterTaskEdge.SEQUENCE)], blocked=True),
    ]


def test_fork_concurrent_arms_conflict():
    """In a fork, the two parallel arms can run simultaneously and must not share a slot."""
    tasks = _create_fork_tasks()
    conflict_graph = task_recycling._build_conflict_graph(tasks, [0, 1, 2])

    assert 2 in conflict_graph[1], 'Tasks 1 and 2 are concurrent and must conflict'
    assert 1 in conflict_graph[2]
    # Task 0 strictly precedes both 1 and 2, so it does not conflict with either.
    assert 1 not in conflict_graph[0]
    assert 2 not in conflict_graph[0]


def test_diamond_source_and_sink_do_not_conflict():
    """In a diamond, the source (0) precedes all trigger sources of the sink (3)."""
    tasks = _create_diamond_tasks()
    conflict_graph = task_recycling._build_conflict_graph(tasks, [0, 1, 2, 3])

    # Arms 1 and 2 are concurrent → conflict.
    assert 2 in conflict_graph[1]
    # Source and sink are sequential through both paths → no conflict.
    assert 3 not in conflict_graph[0]
    assert 0 not in conflict_graph[3]


def test_no_conflicting_tasks_share_slot():
    """Fundamental safety invariant: tasks that conflict must be in different slots."""
    tasks = _create_linear_local_tasks(len(constants.LOCAL_TASK_IDS) + 5)
    local_indices = [i for i, t in enumerate(tasks) if t.task_type == 'local']
    conflict_graph = task_recycling._build_conflict_graph(tasks, local_indices)

    plan = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)

    for slot in plan.local_slots:
        for i, a in enumerate(slot.task_indices):
            for b in slot.task_indices[i + 1:]:
                assert b not in conflict_graph[a], (f'Tasks {a} and {b} conflict but were placed in the same slot')


# ---------------------------------------------------------------------------
# greedy_coloring helper
# ---------------------------------------------------------------------------


def test_greedy_coloring_max_colors_infeasible():
    """A triangle (K3) needs 3 colors; requesting 2 must return None."""
    k3 = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    assert task_recycling.greedy_coloring(k3, [0, 1, 2], max_colors=2) is None


def test_greedy_coloring_max_colors_feasible():
    """A triangle (K3) is colorable with exactly 3 colors."""
    k3 = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    result = task_recycling.greedy_coloring(k3, [0, 1, 2], max_colors=3)
    assert result is not None
    assert len(set(result.values())) == 3
    for v, neighbors in k3.items():
        for u in neighbors:
            assert result[v] != result[u], 'Adjacent vertices must have different colors'


def test_greedy_coloring_fixed_colors_preserved():
    """Fixed-prefix colors must be retained in the returned coloring."""
    no_conflicts = {v: set() for v in range(4)}
    fixed = {0: 7, 1: 3}
    result = task_recycling.greedy_coloring(no_conflicts, [0, 1, 2, 3], fixed_coloring=fixed)
    assert result is not None
    assert result[0] == 7
    assert result[1] == 3
    # Free vertices must also be present.
    assert 2 in result and 3 in result


def test_greedy_coloring_fixed_conflict_honored():
    """A free vertex adjacent to a fixed-color vertex must not receive that color."""
    conflict_graph = {0: {1}, 1: {0}, 2: set()}
    fixed = {0: 5}
    result = task_recycling.greedy_coloring(conflict_graph, [0, 1, 2], fixed_coloring=fixed, max_colors=10)
    assert result is not None
    assert result[0] == 5
    assert result[1] != 5


# ---------------------------------------------------------------------------
# plan_task_bindings mode behavior
# ---------------------------------------------------------------------------


def test_fail_on_overrun_raises():
    tasks = _create_linear_local_tasks(len(constants.LOCAL_TASK_IDS) + 1)
    with pytest.raises(ValueError, match='Too many local tasks'):
        task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.FAIL_ON_OVERRUN)


def test_fail_on_overrun_exact_fit():
    tasks = _create_linear_local_tasks(len(constants.LOCAL_TASK_IDS))
    plan = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.FAIL_ON_OVERRUN)
    assert not plan.uses_recycling
    assert len(plan.local_slots) == len(constants.LOCAL_TASK_IDS)


def test_state_machine_no_recycling_when_tasks_fit():
    """When local task count ≤ hardware slots, every task gets its own unique slot."""
    tasks = _create_linear_local_tasks(len(constants.LOCAL_TASK_IDS))
    plan = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)
    assert not plan.uses_recycling
    assert all(len(slot.task_indices) == 1 for slot in plan.local_slots)


def test_empty_task_list_returns_empty_plan():
    plan = task_recycling.plan_task_bindings([], tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)
    assert plan.local_slots == ()
    assert plan.task_to_local_slot == {}


# ---------------------------------------------------------------------------
# Transition preamble
# ---------------------------------------------------------------------------


def test_transition_preamble_empty_for_non_recycled():
    tasks = _create_linear_local_tasks(len(constants.LOCAL_TASK_IDS))
    plan = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)
    for i in range(len(tasks)):
        assert plan.emit_local_transition_preamble(i, blocked=False) == ''
        assert plan.emit_local_transition_preamble(i, blocked=True) == ''


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_plan_is_deterministic():
    tasks = _create_linear_local_tasks(len(constants.LOCAL_TASK_IDS) + 5)
    plan1 = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)
    plan2 = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)
    assert plan1.task_to_local_slot == plan2.task_to_local_slot
    assert plan1.task_to_local_state == plan2.task_to_local_state
