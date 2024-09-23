"""
Converts routed Spatial IR code to Cerebras CSL.
"""

from io import StringIO
from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization
from spatialstencil.syntax.csl import constants as csl
from spatialstencil.syntax.csl.codefile import CodeFile


def lower_spatial_ir_to_csl(kernel: spir.Kernel, rect_offset: tuple[int, int] = (0, 0)) -> list[CodeFile]:
    """
    Lowers a routed Spatial IR kernel into Cerebras CSL code.

    :param kernel: The Spatial IR kernel to lower.
    :param rect_offset: The offset of the output rectangle to use.
    :return: List of code-file objects that can be written to files. See ``write_code_to_files``.
    """
    # Check if virtual rectangles are equal, consolidate, add phase-end remark at end of computation
    kernel = canonicalization.inline_phases(kernel)
    kernel = canonicalization.extend_blocks_to_equivalence_classes(kernel) # Ensure dataflow/compute/place exist

    # Prepare outputs
    header = StringIO()
    current_code = StringIO()
    footer = StringIO()

    # PRECONDITIONS:
    #     * Rectangles of dataflow/compute/place do not intersect (comes from Spatial IR)
    #     * For every place block range, the same range should exist for dataflow and compute. (pass)
    #        * There is no "orphan" block that does not have all matching place/dataflow/compute (pass)
    #     * There are no phases in the code, there may be local phases for each rectangle (pass)

    # Preconditions to verify:
    #     * All Parameter objects have non-None values
    #     * All parameters are inlined
    for param in kernel.parameters:
        if param.value is None:
            raise ValueError(f'Undefined parameter value for "{param.name}"')

    # Create mapping between SpIR blocks and PE rectangles
    #mapping: dict[tuple[int, int], (spir.ComputeBlock | spir.DataflowBlock | spir.PlaceBlock)] = {}

    # For each rectangle, collect metadata:
    #     * Find colors from dataflow blocks
    #     * Collect PE-local arrays from place blocks
    #     * Make (unique) DSDs out of memory accesses in compute blocks
    #     * Generate routing instructions from dataflow blocks
    #     * Make unique colors out of streams, reduce number of streams
    #for rect in rectangles:

    # Convert compute block subgraphs into tasks:
    #    * Make task DAG out of computations
    #    * Any node that has two or more incoming edges (i.e., requires wait) initiates a new task
    #    * Communication inter-task dependency uses ``activate``
    #    * Compute task dependency uses ``unblock``
    #    * Phase end is a task that modifies the current system state and activates next phase's tasks (see below)
    #    * First phase begin is done as part of the kernel function call
    #    * (re)cycle task IDs based on ``csl.{DATA,LOCAL,CONTROL}_TASK_IDS``

    # Convert compute blocks' contents:
    # Preprocessing pass: FMA fusion
    # Communication calls:
    #    * Become async calls
    # ``map``:
    #    * becomes DSD operations as much as possible
    #    * @map as a fallback
    # ``foreach``:
    #   * Try to make DSD operations as much as possible
    #   * If index is requested: before unblocking task, set k; inc at end of task
    #   * Wavelet-triggered task as fallback
    # Rebinding tasks (i.e., recycling IDs) between phases becomes switch-case on the variable that maintains
    # the current phase

    # Generic footer code
    footer.write(f'''
comptime {{
    @bind_data_task(main_task, {csl.DATA_TASK_IDS[0]});
}}
''')

    rect_size = kernel.get_grid_size()

    scalar_argument_types = []
    # Layout file
    layout_code = f'''
layout {{
    // Rectangle and code setup
    @set_rectangle{rect_size};
    for (@range(i16, {rect_size[0]})) |pe_x| {{
        for (@range(i16, {rect_size[1]})) |pe_y| {{
            @set_tile_code(pe_x, pe_y, "code.csl", .{{  }});
        }}
    }}

    // TODO: Routes

    // TODO: Symbol names for arguments and kernel
    // @export_name("", [*]f32, true);
    @export_name("{kernel.name}", fn({", ".join(scalar_argument_types)})void);
}}'''
    layout_cf = CodeFile('layout.csl', layout_code)

    # Finalize code generation by concatenating carets
    main_cf = CodeFile('code.csl', header.getvalue() + '\n' + current_code.getvalue() + '\n' + footer.getvalue())

    # Return all generated code files
    return [main_cf, layout_cf]
