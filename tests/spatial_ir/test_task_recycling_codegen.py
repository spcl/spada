import os
import re

import pytest

from spada.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spada.syntax.csl import task_recycling
from spada.syntax.csl import tasks as tdag
from spada.syntax.spatial_ir import parser, passes

_CSL_RUNTIME_TASK_RECYCLING_SAMPLES = os.path.join(
    os.path.dirname(__file__), '..', 'csl_runtime', 'samples')


def test_task_recycling_codegen_uses_else_if_dispatch_for_recycled_slots():
    sample = os.path.join(
        os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'collectives', 'tree_reduce_2D.sptl')
    kernel = parser.parse_file(sample)
    kernel = passes.concretize_parameters(kernel, LX=8, LY=8, K=16)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel, task_fusion=False)
    code = next(file.code for file in csl_files if file.filename == 'code_0_0.csl')

    task_id_occurrences: dict[str, int] = {}
    for hardware_id in re.findall(r'@get_local_task_id\((\d+)\)', code):
        task_id_occurrences[hardware_id] = task_id_occurrences.get(hardware_id, 0) + 1

    assert any(count > 1 for count in task_id_occurrences.values())
    assert '__task_slot_' in code
    assert 'else if (__task_slot_' in code
    assert re.search(
        r'if \(__task_slot_\d+_state == \d+\) \{.*?\n\s+\}\n\s+else if \(__task_slot_\d+_state == \d+\) \{',
        code,
        re.S,
    )


@pytest.mark.parametrize(
    'filename',
    (
        'task_recycling_merge.sptl',
        'task_recycling_two_stage.sptl',
        'task_recycling_three_stage.sptl',
    ),
)
def test_csl_runtime_task_recycling_sample_lowers(filename: str):
    """Lowering succeeds for kernels under tests/csl_runtime/samples (matches e2e --disable-task-fusion)."""
    path = os.path.join(_CSL_RUNTIME_TASK_RECYCLING_SAMPLES, filename)
    kernel = parser.parse_file(path)
    kernel = passes.constexpr_propagation(kernel)
    csl_files = lower_spatial_ir_to_csl(
        kernel, task_fusion=False, copy_elision=True, prune_memory=True)
    assert csl_files, 'expected at least one generated CSL file'
    combined = '\n'.join(f.code for f in csl_files)
    assert combined.strip(), 'expected non-empty CSL'
    assert '__task_slot_' in combined, 'expected task-ID recycling in generated CSL'


def test_data_tasks_install_the_state_of_a_recycled_successor():
    """A data task handing control to a recycled slot must install that slot's state first.

    Without the assignment the dispatcher runs whichever branch was installed last, which means a
    PE silently skips its comparator and the fabric deadlocks behind the send it never made. The
    bundled Batcher reaches that shape once it has enough phases sharing a colour, at L=4.
    """
    sample = os.path.join(
        os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'sort', 'batcher_oddeven_bundled_1D.sptl')
    kernel = parser.parse_file(sample)
    kernel = passes.concretize_parameters(kernel, L=4)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    checked = 0
    for file in csl_files:
        code = file.code
        hardware_ids: dict[str, list[str]] = {}
        for task_index, hardware_id in re.findall(r'const task_(\d+)_id = @get_local_task_id\((\d+)\)', code):
            hardware_ids.setdefault(hardware_id, []).append(task_index)
        recycled = {task for tasks in hardware_ids.values() if len(tasks) > 1 for task in tasks}

        for body in re.findall(r'task dtask_\d+\([^)]*\) void \{(.*?)\n\}', code, re.S):
            for match in re.finditer(r'@(?:activate|unblock)\(task_(\d+)_id\);', body):
                if match.group(1) not in recycled:
                    continue
                written = [line.strip() for line in body[:match.start()].splitlines() if line.strip()]
                preceding = written[-1] if written else ''
                assert re.fullmatch(r'__task_slot_\d+_state = \d+;', preceding), (
                    f'{file.filename}: @activate(task_{match.group(1)}_id) is not preceded by its '
                    f'slot state assignment, but by "{preceding}"')
                checked += 1

    assert checked, 'sample no longer exercises a data task triggering a recycled local task'


def test_a_reused_channel_binds_one_data_task_that_dispatches_on_its_epoch():
    """Two receives on one channel at one PE share the data task the channel binds.

    A data task's hardware ID is the color, so binding two of them is not merely wasteful
    but rejected by cslc ("task ID '0' bound to more than one task"). The static Batcher
    reaches that shape at L=3, where a PE compares against its neighbour in two phases that
    the sample gives the same channel.
    """
    sample = os.path.join(
        os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'sort', 'batcher_oddeven_1D.sptl')
    kernel = parser.parse_file(sample)
    kernel = passes.concretize_parameters(kernel, L=3)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(kernel)

    shared = 0
    for file in csl_files:
        code = file.code
        colors: dict[str, list[str]] = {}
        for task_index, color in re.findall(r'const dtask_(\d+)_id = @get_data_task_id\(@get_color\((\d+)\)\)', code):
            colors.setdefault(color, []).append(task_index)

        bound = re.findall(r'@bind_data_task\(\w+, dtask_(\d+)_id\);', code)
        assert len(bound) == len(colors), (
            f'{file.filename}: binds {len(bound)} data tasks for {len(colors)} colors')

        for color, task_indices in colors.items():
            if len(task_indices) == 1:
                continue
            shared += 1
            dispatcher = re.search(rf'task dtask_color_{color}\([^)]*\) void \{{(.*?)\n\}}', code, re.S)
            assert dispatcher, f'{file.filename}: color {color} is reused but has no dispatcher'
            body = dispatcher.group(1)
            states = re.findall(r'(?:else )?if \(__dtask_color_' + color + r'_state == (\d+)\)', body)
            assert states == [str(state) for state in range(len(task_indices))], (
                f'{file.filename}: color {color} dispatches on {states} for {len(task_indices)} receives')
            # A branch that keeps its color live would take the next epoch's wavelets as its own.
            for task_index in task_indices:
                assert f'@block(dtask_{task_index}_id);' in body, (
                    f'{file.filename}: dtask_{task_index} does not block color {color} when it is done')

    assert shared, 'sample no longer reuses a channel for two receives at one PE'


def test_data_tasks_sharing_a_channel_must_take_turns():
    """Receives that could run concurrently cannot share a channel's data task."""
    def task(index: int, task_type: str, successor: int) -> tdag.CSLTask:
        edge = tdag.InterTaskEdge.SEQUENCE if successor == -1 else tdag.InterTaskEdge.UNBLOCK
        return tdag.CSLTask(index, task_type, [index], [(successor, edge)], blocked=task_type == 'data')

    # 0 -> 1 (receive) -> 2 -> 3 (receive), with 1 and 3 on the same channel.
    ordered = [task(0, 'local', 1), task(1, 'data', 2), task(2, 'local', 3), task(3, 'data', -1)]

    slots, task_to_slot, task_to_state = task_recycling.plan_data_task_slots(ordered, {1: 5, 3: 5})
    assert [slot.task_indices for slot in slots] == [(1, 3)]
    assert task_to_slot == {1: 0, 3: 0}
    assert task_to_state == {1: 0, 3: 1}

    # Dropping the edge from the first receive to the second one's trigger leaves both live at once.
    concurrent = [task(0, 'local', 1), task(1, 'data', -1), task(2, 'local', 3), task(3, 'data', -1)]
    with pytest.raises(SyntaxError, match='not ordered'):
        task_recycling.plan_data_task_slots(concurrent, {1: 5, 3: 5})


def test_codegen_avoids_local_task_id_color_overlap():
    path = os.path.join(_CSL_RUNTIME_TASK_RECYCLING_SAMPLES, 'task_color_overlap_many_channels.sptl')
    kernel = parser.parse_file(path)
    kernel = passes.constexpr_propagation(kernel)

    csl_files = lower_spatial_ir_to_csl(
        kernel, task_fusion=False, copy_elision=True, prune_memory=True)
    combined = '\n'.join(f.code for f in csl_files)

    local_task_ids = {int(v) for v in re.findall(r'@get_local_task_id\((\d+)\)', combined)}
    colors = {int(v) for v in re.findall(r'@get_color\((\d+)\)', combined)}

    assert 8 in colors, 'sample should force color 8 to be allocated'
    assert local_task_ids
    assert local_task_ids.isdisjoint(colors), (
        f'local task IDs overlap communication colors: ids={sorted(local_task_ids)}, colors={sorted(colors)}'
    )
