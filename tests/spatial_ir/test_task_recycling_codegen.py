import os
import re

from spatialstencil.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spatialstencil.syntax.spatial_ir import parser, passes


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
