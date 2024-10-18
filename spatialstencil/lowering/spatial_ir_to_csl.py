"""
Converts routed Spatial IR code to Cerebras CSL.
"""

from io import StringIO
import networkx as nx
from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization, analysis
from spatialstencil.syntax.spatial_ir.canonicalization import PEBlock, Rectangle
from spatialstencil.syntax.csl import constants as csl
from spatialstencil.syntax.csl.codefile import CodeFile


def lower_spatial_ir_to_csl(kernel: spir.Kernel, rect_offset: tuple[int, int] = (0, 0)) -> list[CodeFile]:
    """
    Lowers a routed Spatial IR kernel into Cerebras CSL code.

    :param kernel: The Spatial IR kernel to lower.
    :param rect_offset: The offset of the output rectangle to use.
    :return: List of code-file objects that can be written to files. See ``write_code_to_files``.
    """
    # PRECONDITION: Rectangles of dataflow/compute/place do not intersect (comes from Spatial IR)

    # Verify all parameter objects are either inlined or have defined values
    for param in kernel.parameters:
        if param.value is None:
            raise ValueError(f'Undefined parameter value for "{param.name}"')

    # Transform Spatial IR such that:
    #     * For every place block range, the same range should exist for dataflow and compute. (pass)
    #        * There is no "orphan" block that does not have all matching place/dataflow/compute (pass)
    #     * There are no phases in the code, there may be local phases for each rectangle (pass)

    # Check if virtual rectangles are equal, consolidate, add phase-end remark at end of computation
    kernel = canonicalization.canonicalize_phases(kernel)
    kernel = canonicalization.reduce_streams(kernel)
    kernel = canonicalization.inline_phases(kernel)
    print(kernel.as_ir())

    # Create mapping between SpIR blocks and PE rectangles. Creates empty blocks as necessary
    rectangles = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)

    # Collect scalar argument types
    scalar_argument_types = []
    scalar_arguments = []

    # For each rectangle, collect metadata and generate code
    csl_codes: list[CodeFile] = []
    routing_instructions: list[str] = []
    for rect in rectangles:
        # Create a unique CSL code file based on rectangle offset
        csl_name = f'code_{rect.x_range[0]}_{rect.y_range[0]}.csl'
        rect_code = generate_rectangle(kernel, rect, routing_instructions, scalar_arguments)
        csl_codes.append(CodeFile(csl_name, rect_code))

    # Prepare outputs
    layout_code = StringIO()

    ###############################################
    # Generate main layout file
    grid_rect = kernel.get_grid_rect()
    rect_size = grid_rect[1], grid_rect[3]

    layout_code.write(f'''layout {{
    // Rectangle and code setup
    @set_rectangle{rect_size};''')

    for rect in rectangles:
        xs, xe, ys, ye = rect.x_range[0], rect.x_range[1], rect.y_range[0], rect.y_range[1]
        code_filename = f'code_{xs}_{ys}.csl'
        # Add global offsets as necessary
        xs += rect_offset[0]
        xe += rect_offset[0]
        ys += rect_offset[1]
        ye += rect_offset[1]

        # Emit rectangle code setup
        layout_code.write(f'''
    for (@range(i16, {xs}, {xe}, 1)) |pe_x| {{
        for (@range(i16, {ys}, {ye}, 1)) |pe_y| {{
            @set_tile_code(pe_x, pe_y, "{code_filename}", .{{  }});
        }}
    }}\n''')

    # Emit routing instructions
    layout_code.write('    // Routes\n')
    for rinst in routing_instructions:
        layout_code.write(rinst + '\n')

    # Emit symbol names for arguments and kernel
    for argument in kernel.arguments:
        shape = f'[{", ".join(s.as_ir() for s in argument.dtype.shape)}]' if len(argument.dtype.shape) > 0 else ''
        layout_code.write(f'''
    @export_name("{argument.identifier.name}", {shape}{argument.dtype.element_type.element_type.as_ir()}, 
                 {"false" if argument.writeonly else "true"});''')

    layout_code.write(f'''
    @export_name("{kernel.name}", fn({", ".join(scalar_argument_types)})void);
}}''')
    csl_codes.append(CodeFile('layout.csl', layout_code.getvalue()))

    # Return all generated code files
    return csl_codes


def generate_rectangle(kernel: spir.Kernel, rect: Rectangle[PEBlock], routing_instructions: list[str],
                       scalar_arguments: list[str]):
    # Code generation carets
    header = StringIO()
    current_code = StringIO()
    footer = StringIO()

    # Initialize footer
    footer.write(f'''comptime {{
    @bind_data_task(main_task, {csl.DATA_TASK_IDS[0]});

''')

    # Collect metadata:
    #     * Find colors from dataflow blocks
    #     * Collect PE-local arrays from place blocks
    #     * Make (unique) DSDs out of memory accesses in compute blocks
    #     * Generate routing instructions from dataflow blocks
    #     * Make unique colors out of streams, reduce number of streams
    color_map = _collect_and_allocate_colors(rect.metadata, header)
    _collect_and_generate_arrays(rect.metadata.place, header, footer)  # TODO: Use @export_symbol here
    dsds = _collect_unique_dsds(rect.metadata, header)
    routing_instructions.append(_collect_routes(rect.metadata.dataflow))

    # Convert compute block subgraphs into tasks:
    #    * Make task DAG out of computations
    #    * Any node that has two or more incoming edges (i.e., requires wait) initiates a new task
    #    * Communication inter-task dependency uses ``activate``
    #    * Compute task dependency uses ``unblock``
    #    * Phase end is a task that modifies the current system state and activates next phase's tasks (see below)
    #    * First phase begin is done as part of the kernel function call
    #    * (re)cycle task IDs based on ``csl.{DATA,LOCAL,CONTROL}_TASK_IDS``
    dag = analysis.to_task_dag(rect.metadata.compute)
    task_map = _create_task_ids(dag)

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

    # Write entry point code
    current_code.write(f'''fn {kernel.name}({", ".join(scalar_arguments)}) void {{
  sys_mod.unblock_cmd_stream();
}}''')

    # Finalize footer
    footer.write(f'''
    @export_symbol("{kernel.name}");
}}\n''')

    # Finalize code generation by concatenating carets
    return header.getvalue() + '\n' + current_code.getvalue() + '\n' + footer.getvalue()


def _collect_and_allocate_colors(rect: PEBlock, header: StringIO) -> dict[int, int]:
    """
    Returns a mapping of each channel to a CSL color.
    """
    # TODO
    return {}


def _collect_and_generate_arrays(place: spir.PlaceBlock, header: StringIO, footer: StringIO):
    """
    Generates array allocation and symbol exports from a rectangle's ``place`` block.

    :param place: The ``place`` block to generate from.
    :param header: A code generator stream for a file's header (where the array would be defined).
    :param footer: A code generator stream for a file's footer (where the array would be exported).
    """
    # TODO: Use @export_symbol here
    pass


def _collect_unique_dsds(rect: PEBlock, header: StringIO) -> list[str]:
    """
    Returns a list of DSD descriptors
    """
    # TODO
    return []


def _collect_routes(dataflow: spir.DataflowBlock) -> str:
    """
    Returns a code segement to add to the layout CSL file.
    """
    return '    // route'


def _create_task_ids(task_dag: nx.DiGraph) -> dict[spir.Statement, int]:
    """
    Creates a mapping between tasks and physical task IDs.
    """
    # TODO
    # TODO: Consider explicitly defining data/local/control tasks in return value.
    return {}
