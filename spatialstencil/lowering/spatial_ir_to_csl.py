"""
Converts routed Spatial IR code to Cerebras CSL.
"""

from io import StringIO
import networkx as nx
from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization, analysis
from spatialstencil.syntax.spatial_ir.canonicalization import PEBlock, Rectangle
from spatialstencil.syntax.csl import constants as csl, preprocessing, tasks as tdag
from spatialstencil.syntax.csl.structures import DataStructureDescriptor
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

    # Lower array receives and sends to foreach and for, respectively
    # (maybe unnecessary given that bulk send/receive can be implemented with fabout/fabin)
    # canonicalization.lower_bulk_communication(rectangles)

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

    # Collect unique routes for all rectangles
    routes_per_rectangle = _collect_routes(rectangles)

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
{routes_per_rectangle[(xs, ys)]}
        }}
    }}\n''')

    # Emit routing instructions
    layout_code.write('\n    // Routes\n')
    for rinst in routing_instructions:
        layout_code.write(rinst + '\n')

    # Emit symbol names for arguments and kernel
    layout_code.write('\n    // Arguments\n')
    for argument in kernel.arguments:
        layout_code.write(f'    @export_name("{argument.identifier.name}", {dtype_as_csl(argument.dtype)}, '
                          f'{"false" if argument.writeonly else "true"});\n')

    layout_code.write(f'''
    // Kernel
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
    footer.write('comptime {\n')

    # TODO: Preprocessing passes:
    #     * FMA fusion
    preprocessing.preprocess_rectangle(rect.metadata)

    # Collect metadata:
    #     * Find colors from dataflow blocks
    #     * Collect PE-local arrays from place blocks
    #     * Make (unique) DSDs out of memory accesses in compute blocks
    #     * Generate routing instructions from dataflow blocks
    #     * Make unique colors out of streams, reduce number of streams
    color_map = _collect_and_allocate_colors(rect.metadata, header)
    _collect_and_generate_fields(rect.metadata.place, header, footer)
    dtypes = _collect_identifier_types(rect.metadata)

    # Convert compute block subgraphs into tasks:
    #    * Make task DAG out of computations
    #    * Any node that has two or more incoming edges (i.e., requires wait) initiates a new task
    #    * Communication inter-task dependency uses ``activate`` and then ``unblock``
    #    * Compute task dependency uses ``unblock``
    #    * Phase end is a task that modifies the current system state and activates next phase's tasks (see below)
    #    * First phase begin is done as part of the kernel function call
    #    * (re)cycle task IDs based on ``csl.{DATA,LOCAL,CONTROL}_TASK_IDS``: becomes switch-case on the variable that
    #      maintains the current state
    completion_dag = analysis.to_completion_dag(rect.metadata.compute)
    tasks = tdag.create_csl_tasks(completion_dag, rect.metadata.compute, dtypes)
    dsds = _collect_unique_dsds(tasks, rect.metadata, header, dtypes)

    # TODO: Collect all scalar types for foreach receivers. Every sequential foreach can recycle index var

    # Generate each task
    for task in tasks:
        current_code.write(f'const task_{task.task_id}_id = @get_{task.task_type}_task_id({task.task_id});\n')
        if task.task_type == 'local':
            current_code.write(f'task task_{task.task_id}() void {{\n')
            _generate_task_code(rect.metadata, task, current_code, header, footer, dsds, dtypes, color_map)
            current_code.write(f'}}\n')
        elif task.task_type == 'data':
            _generate_data_task(rect.metadata, task, current_code, header, footer, dsds, dtypes, color_map)

        footer.write(f'    @bind_{task.task_type}_task(task_{task.task_id}, task_{task.task_id}_id);\n')

        # Make sure to block tasks
        if task.blocked:
            footer.write(f'    @block(task_{task.task_id}_id);\n')

    # Write entry point code
    current_code.write(f'''\nfn {kernel.name}({", ".join(scalar_arguments)}) void {{
''')
    non_source_tasks = set(n for t in tasks for n, _ in t.outgoing)
    source_tasks = set(t.task_id for t in tasks) - non_source_tasks
    for task in source_tasks:
        current_code.write(f'    @activate(task_{task}_id);\n')
    current_code.write('}\n')

    current_code.write(f'''
const exit_task_id = @get_local_task_id({csl.EXIT_TASK_ID});
task exit_task() void {{
    // On completion, unblock command stream
    sys_mod.unblock_cmd_stream();
}}''')

    # Finalize footer
    footer.write(f'''
    @export_symbol("{kernel.name}");
}}\n''')

    # Finalize code generation by concatenating carets
    return header.getvalue() + '\n' + current_code.getvalue() + '\n' + footer.getvalue()


def _collect_and_allocate_colors(rect: PEBlock, header: StringIO) -> dict[str, int]:
    """
    Returns a mapping of each stream to a CSL color, and adds an allocation there.

    :param rect: The rectangle to use.
    :param header: A code generator stream for a file's header (where the declarations are).
    :return: Dictionary mapping each stream to its respective color
    """
    result: dict[int, int] = {}

    if rect.dataflow.statements:
        header.write('// Colors\n')

    # Collect colors from streams in dataflow
    for stream_decl in rect.dataflow.statements:
        name = name_to_csl(stream_decl.stream_name)
        if stream_decl.routing is None:
            raise SyntaxError(f'Non-routed stream "{name}". When generating CSL, Spatial IR code must have all streams '
                              'routed.')
        if stream_decl.routing.channel == 'auto':
            raise SyntaxError(f'"auto" stream channel found in stream "{name}". All streams must be concretized prior '
                              'to lowering to CSL')
        if stream_decl.routing.channel not in csl.COLORS:
            raise SyntaxError(f'Too many communication channels allocated for CSL: stream {name} has channel '
                              f'{stream_decl.routing.channel}')

        # Add to mapping
        result[name] = csl.COLORS[stream_decl.routing.channel]
        # Declare color
        header.write(f'const {name}_color: color = @get_color({result[name]});\n')

    if result:
        header.write('\n')

    return result


def _collect_and_generate_fields(place: spir.PlaceBlock, header: StringIO, footer: StringIO):
    """
    Generates array allocation and symbol exports from a rectangle's ``place`` block.

    :param place: The ``place`` block to generate from.
    :param header: A code generator stream for a file's header (where the array would be defined).
    :param footer: A code generator stream for a file's footer (the comptime block where the array would be exported).
    """
    header.write('// Place block\n')
    for field_dec in place.statements:
        name = name_to_csl(field_dec.field_name)
        header.write(f'var {name}: {dtype_as_csl(field_dec.dtype)};\n')

        header.write(f'var __{name}_ptr = &{name};\n')
        footer.write(f'    @export_symbol(__{name}_ptr, "{name}");\n')
    header.write('\n')


def _collect_unique_dsds(tasks: list[tdag.CSLTask], rect: PEBlock, header: StringIO,
                         dtypes: dict[spir.Identifier, spir.IRType]) -> list[tuple[str, DataStructureDescriptor]]:
    """
    Returns a list of DSDs and generates them in the header.
    """
    dsds: list[tuple[str, DataStructureDescriptor]] = []

    # TODO: Find DSDs

    # Generate appropriate header code
    header.write('// DSDs\n')
    for name, dsd in dsds:
        header.write(f'const {name} = {dsd.as_csl()};\n')

    return dsds


def _route_dir(dx: int, dy: int):
    """
    Helper function that returns directions for routing: (source, target).
    """
    assert abs(dx + dy) == 1
    if dx == -1:
        return ('EAST', 'WEST')
    elif dx == 1:
        return ('WEST', 'EAST')
    elif dy == -1:
        return ('SOUTH', 'NORTH')
    elif dy == 1:
        return ('NORTH', 'SOUTH')


def _collect_routes(rectangles: list[Rectangle[PEBlock]]) -> dict[tuple[int, int], str]:
    """
    Creates a parametric version of the Routing Graph (see the Spatial IR specification for more information) and
    returns a dictionary of code segements to add to the layout CSL file based on the streams.

    :param rectangles: All rectangles involved in this kernel.
    :return: A dictionary mapping the starting point of each rectangle to a string representing the layout instructions.
    """
    INDENT = 12 * ' '
    result = {}

    # TODO: Make routing instructions unique
    # Create a routing graph
    for rect in rectangles:
        # Test whether a receive/send statement are called for creating inbound/outbound routes
        sends_recvs = analysis.sends_and_receives(rect.metadata.compute)
        inst = ''

        # For each hop, make a color WEST-EAST/NORTH-SOUTH pair. For the first and last hop, pair with RAMP
        for stream in rect.metadata.dataflow.statements:
            if stream.stream_name not in sends_recvs:  # Skip unused streams
                continue
            color_name = name_to_csl(stream.stream_name) + '_color'

            if len(stream.routing.hops) == 1:  # Inbound and outbound generated together
                route = _route_dir(*stream.routing.hops[0].offset)
                sent, received = sends_recvs[stream.stream_name]
                if sent:
                    inst += INDENT + '@set_color_config(pe_x, pe_y, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                        color_name, 'RAMP', route[1])
                if received:
                    inst += INDENT + '@set_color_config(pe_x, pe_y, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                        color_name, route[0], 'RAMP')
            else:  # Multi-hop
                sent, received = sends_recvs[stream.stream_name]
                if sent:
                    first_hop = stream.routing.hops[0]
                    route = ('RAMP', _route_dir(*first_hop.offset)[1])
                    inst += INDENT + '@set_color_config(pe_x, pe_y, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                        color_name, route[0], route[1])
                    cur_offx = 0
                    cur_offy = 0
                    for hop in stream.routing.hops[1:]:
                        route = _route_dir(*hop.offset)
                        cur_offx += hop.offset[0]
                        cur_offy += hop.offset[1]
                        inst += INDENT + '@set_color_config(pe_x + %d, pe_y + %d, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                            cur_offx, cur_offy, color_name, route[0], route[1])
                if received:
                    cur_offx = 0
                    cur_offy = 0
                    last_hop = stream.routing.hops[-1]
                    route = (_route_dir(*last_hop.offset)[0], 'RAMP')
                    inst += INDENT + '@set_color_config(pe_x + %d, pe_y + %d, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                        cur_offx, cur_offy, color_name, route[0], route[1])
                    cur_offx += last_hop.offset[0]
                    cur_offy += last_hop.offset[1]
                    for hop in reversed(stream.routing.hops[:-1]):
                        route = _route_dir(*hop.offset)
                        inst += INDENT + '@set_color_config(pe_x + %d, pe_y + %d, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                            cur_offx, cur_offy, color_name, route[0], route[1])
                        cur_offx += hop.offset[0]
                        cur_offy += hop.offset[1]

        result[(rect.x_range[0], rect.y_range[0])] = inst

    return result


def _generate_data_task(rect: PEBlock, task: tdag.CSLTask, current_code: StringIO, header: StringIO, footer: StringIO,
                        dsds: list[tuple[str, DataStructureDescriptor]], dtypes: dict[spir.Identifier, spir.IRType],
                        color_map: dict[str, int]):
    """
    Generates a data task from a foreach loop.

    :param rect: The rectangle PE block to generate.
    :param task: The data task to generate.
    :param current_code: The caret to the code generator at the current position (global).
    :param header: A code generator stream for a file's header (where the declarations are).
    :param footer: A code generator stream for a file's footer (the comptime block where the array would be exported).
    :param dsds: A dictionary mapping names to unique data structure descriptor objects.
    :param dtypes: A dictionary mapping identifiers to their defined types.
    :param color_map: Dictionary mapping each stream to its respective color id ({name}_color also works).
    """
    #   * If index is requested: before unblocking task, set k; inc at end of task
    #   * Wavelet-triggered task as fallback
    assert task.task_type == 'data'
    assert len(task.statements) == 1

    stmt = task.statements[0]
    next_task, itedge = task.outgoing[0]
    # print(task)


def _generate_task_code(rect: PEBlock, task: tdag.CSLTask, current_code: StringIO, header: StringIO, footer: StringIO,
                        dsds: list[tuple[str, DataStructureDescriptor]], dtypes: dict[spir.Identifier, spir.IRType],
                        color_map: dict[str, int]):
    """
    Generates a local task from a CSL task.
    This function converts statements to DSD operations or generates appropriate code.

    :param rect: The rectangle PE block to generate.
    :param task: The CSL task to generate.
    :param current_code: The caret to the code generator at the current position (global).
    :param header: A code generator stream for a file's header (where the declarations are).
    :param footer: A code generator stream for a file's footer (the comptime block where the array would be exported).
    :param dsds: A dictionary mapping names to unique data structure descriptor objects.
    :param dtypes: A dictionary mapping identifiers to their defined types.
    :param color_map: Dictionary mapping each stream to its respective color id ({name}_color also works).
    """
    # Convert task contents:
    # Convert receives/sends from/to arguments to memcpy
    # Communication calls become async calls based on task's outgoing field
    # ``map``:
    #    * becomes DSD operations as much as possible
    #    * @map as a fallback
    # if ``foreach``, has to be a DSD operation
    assert task.task_type == 'local'

    for stmt, (next_task, itedge) in zip(task.statements, task.outgoing):
        # TODO: Write op contents, if DSD or asynchronous, continue

        # If not DSD or asynchronous op, activate/unblock must be called after the generated operation code
        if itedge in (tdag.InterTaskEdge.ACTIVATE, tdag.InterTaskEdge.UNBLOCK):
            # Determine task ID
            if next_task == -1:
                task_id = 'exit_task_id'
            else:
                task_id = f'task_{next_task}_id'
            if itedge == tdag.InterTaskEdge.ACTIVATE:
                current_code.write(f'    @activate({task_id});\n')
            elif itedge == tdag.InterTaskEdge.UNBLOCK:
                current_code.write(f'    @unblock({task_id});\n')


def _collect_identifier_types(rect: PEBlock) -> dict[spir.Identifier, spir.IRType]:
    """
    Returns a dictionary mapping all place, dataflow, and compute variables (streams, arrays, scalars) to datatypes.
    """
    result = {}

    # Collect from place blocks
    for fielddec in rect.place.statements:
        result[fielddec.field_name] = fielddec.dtype

    # Collect from dataflow blocks
    for streamdec in rect.dataflow.statements:
        result[streamdec.stream_name] = streamdec.dtype

    # Collect from compute blocks
    for stmt in rect.compute.statements:
        for value in stmt.walk():
            if isinstance(value, spir.TypedIdentifier):
                result[value.identifier] = value.dtype

    return result


def dtype_as_csl(dtype: spir.ScalarType | spir.StreamType | spir.ArrayType) -> str:
    """
    Returns a CSL syntactic equivalent to a Spatial IR data type.

    :param dtype: Spatial IR data type.
    :return: CSL string representing the given data type.
    """
    if isinstance(dtype, spir.ScalarType):
        return dtype.as_ir()
    if isinstance(dtype, spir.StreamType):
        return dtype.element_type.as_ir()
    if isinstance(dtype, spir.ArrayType):
        shape = f'[{", ".join(s.as_ir() for s in dtype.shape)}]' if len(dtype.shape) > 0 else ''
        return shape + dtype_as_csl(dtype.base_type)


def name_to_csl(name: spir.Identifier) -> str:
    """
    Returns a CSL syntactic equivalent to a Spatial IR identifier.

    :param name: Spatial IR identifier.
    :return: Compilable CSL string representing the identifier.
    """
    if name.version == 0:
        return name.name
    else:
        return f'{name.name}__{name.version}'
