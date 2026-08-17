import os
import re

import pytest

from spada.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
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

    Without the assignment the dispatcher runs whichever branch was installed last -- in the
    bundled Batcher at L=3 that meant a PE silently skipped its comparator and the fabric
    deadlocked behind the send it never made.
    """
    sample = os.path.join(
        os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'sort', 'batcher_oddeven_bundled_1D.sptl')
    kernel = parser.parse_file(sample)
    kernel = passes.concretize_parameters(kernel, L=3)
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
