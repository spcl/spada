"""
Converts routed Spatial IR code to Cerebras CSL.
"""

from io import StringIO
from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization
from spatialstencil.syntax.csl import constants as csl
from spatialstencil.syntax.csl.codefile import CodeFile


def lower_spatial_ir_to_csl(kernel: spir.Kernel, rect_size: tuple[int, int] = (64, 64)) -> list[CodeFile]:
    """
    Lowers a routed Spatial IR kernel into Cerebras CSL code.

    :param kernel: The Spatial IR kernel to lower.
    :param rect_size: The size of the output rectangle to use.
    :return: List of code-file objects that can be written to files. See ``write_code_to_files``.
    """
    # Turn single phase code to be nested in a ``phase``
    kernel = canonicalization.canonicalize_phases(kernel)

    # Prepare outputs
    header = StringIO()
    current_code = StringIO()
    footer = StringIO()

    # TODO: Start with single phase, continue to multi-phase
    if len(kernel.body) > 1:
        raise NotImplementedError('Only single-phase kernels are currently supported')

    # Collect metadata:
    #     * Find colors from dataflow blocks
    #     * Collect PE-local arrays from place blocks
    #     * Make (unique) DSDs out of memory accesses in compute blocks
    #     * Generate routing instructions from dataflow blocks

    # Convert compute block subgraphs into tasks:
    #    * Make dataflow DAG out of computations
    #    * Any node that has two or more incoming edges (i.e., requires wait) initiates a new task
    #    * Communication inter-task dependency uses ``activate``
    #    * Compute task dependency uses ``unblock``
    #    * Phase end is a task that modifies the current system state and activates next phase's tasks (see below)
    #    * First phase begin is done as part of the kernel function call
    #    * (re)cycle task IDs based on ``csl.{DATA,LOCAL,CONTROL}_TASK_IDS``

    # Convert compute blocks' contents:
    # Communication calls:
    #    * Become async calls
    # ``phase``:
    #    * Become a PE-local integer that keeps track of current phase
    #    * Rebinding tasks (i.e., recycling IDs) between phases becomes switch-case on the variable that maintains
    #      the current phase
    #    * Phase end behaves as a dependency on all completions contained in it
    # ``map``:
    #    * becomes DSD operations as much as possible
    #    * @map as a fallback
    # ``foreach``:
    #   * Try to make DSD operations as much as possible
    #   * If index is requested: before unblocking task, set k; inc at end of task
    #   * Wavelet-triggered task as fallback

    # Generic footer code
    footer.write(f'''
comptime {{
    @bind_data_task(main_task, {csl.DATA_TASK_IDS[0]});
}}
''')

    # Layout file
    layout_code = f'''
layout {{
    // Rectangle and code setup
    @set_rectangle{rect_size};
    for (@range(i16, {rect_size[0]})) |pe_x| {{
        for (@range(i16, {rect_size[1]})) |pe_y| {{
            @set_tile_code(pe_x, pe_y, "code.csl", .{{ .pe_x = pe_x, .pe_y = pe_y }});
        }}
    }}

    // TODO: Routes

    // TODO: Symbol names for arguments and kernel
    // @export_name("", [*]f32, true);
    @export_name("{kernel.name}", fn()void);
}}'''
    layout_cf = CodeFile('layout.csl', layout_code)

    # Finalize code generation by concatenating carets
    main_cf = CodeFile('code.csl', header.getvalue() + '\n' + current_code.getvalue() + '\n' + footer.getvalue())

    # Return all generated code files
    return [main_cf, layout_cf]
