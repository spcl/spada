import os

from spatialstencil.lowering import spatial_ir_to_csl as s2c
from spatialstencil.syntax.csl import constants, task_recycling, tasks as tdag
from spatialstencil.syntax.spatial_ir import analysis, parser, passes
from spatialstencil.syntax.spatial_ir.canonicalization import PEBlock


def _load_sample_kernel():
    sample = os.path.join(os.path.dirname(__file__), '..', 'csl_runtime', 'samples', 'task_recycling_merge.sptl')
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


def test_task_recycling_overflow_only_keeps_prefix_slots_unique():
    local_task_count = len(constants.LOCAL_TASK_IDS) + 5
    tasks = _create_linear_local_tasks(local_task_count)

    plan = task_recycling.plan_task_bindings(tasks, tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN)

    assert len(plan.local_slots) == len(constants.LOCAL_TASK_IDS)
    assert all(len(slot.task_indices) <= 2 for slot in plan.local_slots)

    for task_index in range(len(constants.LOCAL_TASK_IDS)):
        slot = plan.local_slot(task_index)
        assert slot.slot_index == task_index
        assert plan.local_state(task_index) == 0
        assert slot.task_indices[0] == task_index


def test_task_recycling_can_recolor_all_tasks_when_requested():
    tasks = _create_linear_local_tasks(len(constants.LOCAL_TASK_IDS) + 5)

    plan = task_recycling.plan_task_bindings(
        tasks,
        tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN,
        recycle_overflow_only=False,
    )

    assert len(plan.local_slots) == 1
    assert plan.local_slots[0].task_indices == tuple(range(len(tasks)))


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
