"""
Converts routed Spatial IR code to Cerebras CSL.
"""

from collections import defaultdict
from contextlib import nullcontext
import copy
import functools
from io import StringIO
import textwrap
from spatialstencil.syntax.common.types import BIT_WIDTH
from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization, analysis, passes
from spatialstencil.syntax.spatial_ir import copy_elimination
from spatialstencil.syntax.spatial_ir import canonical_subgrids
from spatialstencil.syntax.spatial_ir.canonicalization import PEBlock, Rectangle
from spatialstencil.syntax.csl import constants as csl, preprocessing, tasks as tdag, statements as cslstmt, dsd_ops
from spatialstencil.syntax.csl import benchmarking as cslbench
from spatialstencil.syntax.csl import structures as cslstruct
from spatialstencil.syntax.csl import task_recycling, prune_unused_fields as csl_pruning
from spatialstencil.syntax.csl.codefile import CodeFile
from spatialstencil.syntax.csl.statements import name_to_csl, dtype_as_csl, expr_to_csl

UniqueDSDDict = dict[str, list[tuple[str, cslstruct.DataStructureDescriptor]]]


def canonicalize_kernel(kernel: spir.Kernel) -> spir.Kernel:
    """
    Runs the full canonicalization pipeline required before CSL lowering.

    This is the single source of truth for the pass ordering.  Both
    :func:`lower_spatial_ir_to_csl` and the metadata generation in
    ``compiler.py`` must call this function so that the two pipelines
    always stay in sync.

    :param kernel: A fully concretized Spatial IR kernel.
    :return: The transformed kernel, ready for
             :func:`~spatialstencil.syntax.spatial_ir.canonicalization.consolidate_rectangles_to_equivalence_classes`.
    """
    kernel = canonicalization.inline_metaprogramming(kernel)
    kernel = canonicalization.canonicalize_phases(kernel)
    kernel = canonicalization.reduce_streams(kernel)
    kernel = canonical_subgrids.canonicalize_subgrids(kernel)
    kernel = canonicalization.resolve_auto_hops(kernel)
    kernel = canonicalization.inline_phases(kernel)
    return kernel


def lower_spatial_ir_to_csl(kernel: spir.Kernel,
                            rect_offset: tuple[int, int] = (0, 0),
                            disable_benchmarking: bool = False,
                            sync_benchmarking: bool = False,
                            disable_asynchronous: bool = False,
                            disable_dsd: bool = False,
                            task_fusion: bool = True,
                            copy_elision: bool = True,
                            prune_memory: bool = True,
                            task_id_recycling: bool = True) -> list[CodeFile]:
    """
    Lowers a routed Spatial IR kernel into Cerebras CSL code.

    :param kernel: The Spatial IR kernel to lower.
    :param rect_offset: The offset of the output rectangle to use.
    :param disable_benchmarking: If True, disables benchmarking code generation (and memory overhead).
                                 Use in memory-limited scenarios.
    :param sync_benchmarking: If True, generate sync-assisted benchmarking support for more accurate cycle counts.
    :param disable_asynchronous: If True, disables asynchronous task code generation.
    :param disable_dsd: If True, disables DSD operation detection and code generation.
    :param task_fusion: If True, enables task fusion to reduce number of tasks.
    :param copy_elision: If True, enables copy elision optimization pass.
    :param prune_memory: If True, enables unused field pruning optimization pass.
    :param task_id_recycling: If True, enables task ID recycling pass.
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

    # Run the shared canonicalization pipeline
    kernel = canonicalize_kernel(kernel)

    if disable_benchmarking and sync_benchmarking:
        raise ValueError("Sync benchmarking requires benchmarking support to be enabled.")

    # Check if we are streaming or using memcpy mode
    use_memcpy_mode = analysis.kernel_uses_memcpy_mode(kernel)

    # Create mapping between SpIR blocks and PE rectangles. Creates empty blocks as necessary
    rectangles = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)

    # Detect stream argument extents (mapping e.g., `stream<f32>[N]` to an `Nx1` rectangle, or `stream<f32>` to one PE)
    stream_rects = analysis.detect_stream_argument_extents(rectangles, kernel)

    # Lower array operations to foreach/map iterators as necessary
    try:
        canonicalization.lower_bulk_communication(rectangles)
        canonicalization.lower_array_assignment(rectangles)
    except KeyError as e:
        if e.args and isinstance(e.args[0], spir.Identifier):
            raise ValueError(f"Error in {e.args[0].lineinfo}. Undefined identifier \"{e.args[0].as_ir()}\".")

    # Lower arguments to extern fields/streams
    canonicalization.lower_arguments_to_extern(rectangles, kernel)

    # Perform optimization passes
    if copy_elision:
        copy_elimination.remove_extern_field_copies(rectangles)
        copy_elimination.eliminate_redundant_copies(rectangles)

    # Prune unused fields from place blocks
    if prune_memory:
        copy_elimination.prune_unused_fields(rectangles)

    # Add benchmarking fields
    if not disable_benchmarking:
        _add_benchmarking_fields(rectangles, sync_benchmarking)

    # Collect scalar argument types
    scalar_argument_types = []
    scalar_arguments = []
    for argument in kernel.arguments:
        if isinstance(argument.dtype, spir.ScalarType):
            scalar_argument_types.append(dtype_as_csl(argument.dtype))
            scalar_arguments.append(f'__arg_{name_to_csl(argument.identifier)}: {dtype_as_csl(argument.dtype)}')

    # For each rectangle, collect metadata and generate code
    csl_codes: list[CodeFile] = []
    routing_instructions: list[str] = []
    color_maps = []

    resource_context = cslbench.reserve_codegen_resources(csl) if sync_benchmarking and not disable_benchmarking else nullcontext(None)
    with resource_context as sync_resources:
        channel_to_color = _collect_colors_globally(kernel, rectangles, use_memcpy_mode)

        for rect in rectangles:
            # Create a unique CSL code file based on rectangle offset
            csl_name = f'code_{rect.x_range[0]}_{rect.y_range[0]}.csl'
            rect_code, color_map = generate_rectangle(kernel, rect, routing_instructions, scalar_arguments, use_memcpy_mode,
                                                    stream_rects, channel_to_color, disable_benchmarking, sync_benchmarking,
                                                    disable_asynchronous, disable_dsd, task_fusion,
                                                    task_id_recycling)
            color_maps.append(color_map)
            csl_codes.append(CodeFile(csl_name, rect_code))

        # Prepare outputs
        layout_code = StringIO()

        if sync_resources is not None:
            csl_codes.extend(cslbench.load_sync_assets())

        ###############################################
        # Generate main layout file

        # Compute the tight PE bounding box. kernel.get_grid_rect() now returns tight bounds
        # (last-contained PE + 1) rather than canonicalized stops.
        x0, x1, y0, y1 = kernel.get_grid_rect()
        assert x0 == 0, "PE Grid must start at x=0"
        assert y0 == 0, "PE Grid must start at y=0"
        rect_size = x1 - x0, y1 - y0

        # Collect unique routes for all rectangles
        routes_per_rectangle = _collect_routes(rectangles, color_maps)

        if use_memcpy_mode:
            layout_code.write(f'''
// Memcpy setup
const memcpy = @import_module("<memcpy/get_params>", .{{
.width = {rect_size[0]},
.height = {rect_size[1]},
}});
''')
        else:
            input_args = []
            output_args = []
            for arg in kernel.arguments:
                if arg.compiletime:
                    continue
                if arg.readonly:
                    input_args.append(arg)
                elif arg.writeonly:
                    output_args.append(arg)
                else:
                    input_args.append(arg)
                    output_args.append(arg)

            # Only up to 4 streams in each direction are supported (4 input, 4 output streams)
            if len(input_args) > 4 or len(output_args) > 4:
                raise ValueError('Too many input/output streams: only 4 input and 4 output streams are supported in CSL')

            # Generate streaming DATA_*_ID parameters for each input/output stream
            layout_code.write('// Streaming copy setup\n')
            for i, input_arg in enumerate(input_args):
                layout_code.write(f'''param MEMCPYH2D_DATA_{i}_ID: i16;
const MEMCPYH2D_DATA_{i}: color = @get_color(MEMCPYH2D_DATA_{i}_ID);
''')
            for i, output_arg in enumerate(output_args):
                layout_code.write(f'''param MEMCPYD2H_DATA_{i}_ID: i16;
const MEMCPYD2H_DATA_{i}: color = @get_color(MEMCPYD2H_DATA_{i}_ID);
''')

            layout_code.write(f'''
const memcpy = @import_module("<memcpy/get_params>", .{{
     .width = width,
     .height = height,
''')
            for i, input_arg in enumerate(input_args):
                layout_code.write(f'''    .MEMCPYH2D_{i} = MEMCPYH2D_DATA_{i}_ID,
''')
            for i, output_arg in enumerate(output_args):
                layout_code.write(f'''    .MEMCPYD2H_{i} = MEMCPYD2H_DATA_{i}_ID,
''')
            layout_code.write(f'''
}});
''')

        if sync_resources is not None:
            layout_code.write(cslbench.generate_sync_layout_setup(rect_size[0], rect_size[1], sync_resources))

        layout_code.write(f'''layout {{
    // Rectangle and code setup
    @set_rectangle{rect_size};''')

        # First pass: @set_tile_code for every PE.
        # All tile codes must be established before any @set_color_config call,
        # because multi-hop routing config may reference neighboring PEs that
        # belong to a different rectangle (e.g. pass-through relays).
        sync_tile_binding = cslbench.generate_sync_tile_binding() if sync_resources is not None else ''
        for rect in rectangles:
            xb_pre, xe_pre, xs, yb_pre, ye_pre, ys = *rect.x_range, *rect.y_range
            code_filename = f'code_{xb_pre}_{yb_pre}.csl'
            xb = xb_pre + rect_offset[0]
            xe = xe_pre + rect_offset[0]
            yb = yb_pre + rect_offset[1]
            ye = ye_pre + rect_offset[1]

            layout_code.write(f'''
    for (@range(i16, {xb}, {xe}, {xs})) |pe_x| {{
        for (@range(i16, {yb}, {ye}, {ys})) |pe_y| {{
            @set_tile_code(pe_x, pe_y, "{code_filename}", .{{ .memcpy_params = memcpy.get_params(pe_x), {sync_tile_binding}}});
        }}
    }}\n''')

        # Second pass: routing (@set_color_config).  By emitting these after all
        # @set_tile_code calls, every PE referenced by a multi-hop offset is
        # guaranteed to already have tile code assigned.
        layout_code.write('\n    // Routes\n')
        for rect in rectangles:
            xb_pre, xe_pre, xs, yb_pre, ye_pre, ys = *rect.x_range, *rect.y_range
            xb = xb_pre + rect_offset[0]
            xe = xe_pre + rect_offset[0]
            yb = yb_pre + rect_offset[1]
            ye = ye_pre + rect_offset[1]
            route_code = routes_per_rectangle.get((xb_pre, yb_pre), '')
            if route_code.strip():
                layout_code.write(f'''
    for (@range(i16, {xb}, {xe}, {xs})) |pe_x| {{
        for (@range(i16, {yb}, {ye}, {ys})) |pe_y| {{
{route_code}
        }}
    }}\n''')

        for rinst in routing_instructions:
            layout_code.write(rinst + '\n')

        # Emit symbol names for arguments and kernel
        layout_code.write('\n    // Extern fields\n')
        # Gather extern fields from kernel arguments
        extern_fields: list[spir.FieldDeclaration] = []
        for rect in rectangles:
            place_block = rect.metadata.place
            for field in place_block.statements:
                if field.is_extern:
                    if any(field.field_name == ef.field_name for ef in extern_fields):
                        continue
                    extern_fields.append(field)

        for field in extern_fields:
            dtype = field.dtype
            if isinstance(field.dtype, spir.ArrayType) and isinstance(field.dtype.base_type, spir.StreamType):
                pass
            elif isinstance(field.dtype, spir.StreamType):
                # Support scalar streams
                dtype = spir.ArrayType(field.dtype, [1])

            layout_code.write(f'    @export_name("{field.field_name.name}", {dtype_as_csl(dtype, export=True)}, true);\n')

        if sync_resources is not None:
            layout_code.write(cslbench.generate_sync_layout_exports())

        layout_code.write(f'''
    // Kernel
    @export_name("{kernel.name}", fn({", ".join(scalar_argument_types)})void);
}}''')
        csl_codes.append(CodeFile('layout.csl', layout_code.getvalue()))

    # Return all generated code files
    return csl_codes


def generate_rectangle(kernel: spir.Kernel,
                       rect: Rectangle[PEBlock],
                       routing_instructions: list[str],
                       scalar_arguments: list[str],
                       use_memcpy_mode: bool,
                       stream_extents: analysis.StreamExtents,
                       channel_to_color: dict[int, int],
                       disable_benchmarking: bool = False,
                       sync_benchmarking: bool = False,
                       disable_asynchronous: bool = False,
                       disable_dsd: bool = False,
                       task_fusion: bool = True,
                       task_id_recycling: bool = True) -> tuple[str, dict[str, int]]:
    # Code generation carets
    header = StringIO()
    current_code = StringIO()
    footer = StringIO()

    if disable_dsd:
        dsd_ops.DISABLE_DSD = True

    header.write("""
param memcpy_params: comptime_struct;
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
""")

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
    color_map = _allocate_colors(rect, header, kernel, use_memcpy_mode, stream_extents, channel_to_color)
    dtypes = _collect_identifier_types(rect.metadata, kernel.arguments)

    # Preprocess potential data tasks to convert to loops if possible
    if use_memcpy_mode:
        canonicalization.convert_foreach_data_tasks_to_loops(rect, dtypes)
        dtypes = _collect_identifier_types(rect.metadata, kernel.arguments)

    benchmark_code = _generate_benchmarking_code(header, disable_benchmarking, sync_benchmarking)

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
    if disable_asynchronous:
        task_creation_behavior = tdag.TaskCreationBehavior.NO_TASKS
    elif not task_id_recycling:
        task_creation_behavior = tdag.TaskCreationBehavior.FAIL_ON_OVERRUN
    else:
        task_creation_behavior = tdag.TaskCreationBehavior.STATE_MACHINE_ON_OVERRUN
    tasks = tdag.create_csl_tasks(completion_dag, rect.metadata.compute, dtypes, task_creation_behavior)

    csl_pruning.prune_unused_place_fields_for_csl_codegen(rect.metadata, dtypes)
    _collect_and_generate_fields(rect.metadata.place, header, footer, kernel, use_memcpy_mode)
    dtypes = _collect_identifier_types(rect.metadata, kernel.arguments)

    try:
        dsds = _collect_unique_dsds(tasks, rect.metadata, header, dtypes, kernel, use_memcpy_mode)
    except KeyError as e:
        if e.args and isinstance(e.args[0], spir.Identifier):
            raise ValueError(f"Error in {e.args[0].lineinfo}. Undefined identifier \"{e.args[0].as_ir()}\".")
        raise

    # Fuse tasks as much as possible to reduce number of resources
    if task_fusion:
        orig_len = 0
        len_for_reporting = len(tasks)
        while orig_len != len(tasks):  # Run to a fixed point
            orig_len = len(tasks)
            tasks = tdag.fuse_tasks(tasks, dsds, dtypes, rect, use_memcpy_mode, rect.metadata.compute)

        if len(tasks) != len_for_reporting:
            print(f'P{rect.x_range[0]},{rect.y_range[0]}: Reduced from {len_for_reporting} to {len(tasks)} tasks.')

    task_bindings = task_recycling.plan_task_bindings(tasks, task_creation_behavior, set(color_map.values()))

    place_block_bytes = _place_block_storage_bytes(rect.metadata.place)

    print(f'Stats P{rect.x_range[0]},{rect.y_range[0]}: {place_block_bytes} bytes/PE, '
          f'{sum(1 if t.task_type == "local" else 0 for t in tasks)} local tasks across '
          f'{len(task_bindings.local_slots)} local task IDs, '
          f'{sum(1 if t.task_type == "data" else 0 for t in tasks)} data tasks, '
          f'{len(set(color_map.values()))} colors')

    # Declare each logical local task ID alias.
    for slot in task_bindings.local_slots:
        for task_index in slot.task_indices:
            current_code.write(f'const task_{task_index}_id = @get_local_task_id({slot.hardware_task_id});\n')
        if slot.recycled:
            representative = slot.representative_task_index
            current_code.write(f'var {task_bindings.state_var(representative)}: u16 = '
                               f'{task_bindings.invalid_state_literal(representative)};\n')

    # Declare each data task ID.
    for i, task in enumerate(tasks):
        if task.task_type != "data":
            continue

        stmt = rect.metadata.compute.statements[task.statements[0]]
        assert isinstance(stmt, spir.ForeachStatement)
        sname = stmt.receive_stream.stream_name
        if isinstance(sname, spir.ArraySlice):
            sname = sname.array
        if name_to_csl(sname) + "_H2D" in color_map:
            color = color_map[name_to_csl(sname) + "_H2D"]
        elif name_to_csl(sname) + "_IN" in color_map:
            color = color_map[name_to_csl(sname) + "_IN"]
        else:
            print(color_map)
            raise ValueError(f'Cannot find color for stream "{name_to_csl(sname)}" in data task {i}')
        current_code.write(f'const dtask_{i}_id = @get_data_task_id(@get_color({color}));\n')

    # Generate each local slot as one hardware task.
    for slot in task_bindings.local_slots:
        current_code.write(f'task {task_bindings.local_function_name(slot)}() void {{\n')
        if slot.recycled:
            for branch_index, task_index in enumerate(slot.task_indices):
                branch_kw = 'if' if branch_index == 0 else 'else if'
                current_code.write(f'    {branch_kw} ({task_bindings.state_var(task_index)} == '
                                   f'{task_bindings.local_state(task_index)}) {{\n')
                try:
                    _generate_task_code(
                        rect.metadata,
                        task_index,
                        tasks[task_index],
                        current_code,
                        header,
                        footer,
                        dsds,
                        dtypes,
                        color_map,
                        tasks,
                        task_bindings,
                        benchmark_code.kernel_postamble,
                        indent='        ',
                    )
                except KeyError as e:
                    identifier = e.args[0]
                    if isinstance(identifier, spir.Identifier):
                        if identifier.lineinfo:
                            raise SyntaxError(
                                f'Undefined identifier "{identifier.as_ir()}" in {identifier.lineinfo}') from e
                        raise SyntaxError(
                            f'Undefined identifier "{identifier.as_ir()}" in local task {task_index}') from e
                    raise
                current_code.write('    }\n')
        else:
            task_index = slot.task_indices[0]
            try:
                _generate_task_code(
                    rect.metadata,
                    task_index,
                    tasks[task_index],
                    current_code,
                    header,
                    footer,
                    dsds,
                    dtypes,
                    color_map,
                    tasks,
                    task_bindings,
                    benchmark_code.kernel_postamble,
                    indent='    ',
                )
            except KeyError as e:
                identifier = e.args[0]
                if isinstance(identifier, spir.Identifier):
                    if identifier.lineinfo:
                        raise SyntaxError(
                            f'Undefined identifier "{identifier.as_ir()}" in {identifier.lineinfo}') from e
                    else:
                        raise SyntaxError(
                            f'Undefined identifier "{identifier.as_ir()}" in local task {task_index}') from e
                else:
                    raise
        current_code.write('}\n')
        footer.write(f'    @bind_local_task({task_bindings.local_function_name(slot)}, '
                     f'task_{slot.representative_task_index}_id);\n')
        if not slot.recycled and tasks[slot.representative_task_index].blocked:
            footer.write(f'    @block(task_{slot.representative_task_index}_id);\n')

    # Generate each data task.
    for i, task in enumerate(tasks):
        if task.task_type != 'data':
            continue
        _generate_data_task(rect.metadata, i, task, current_code, header, footer, dsds, dtypes, color_map, tasks)
        footer.write(f'    @bind_data_task(dtask_{i}, dtask_{i}_id);\n')
        if task.blocked:
            footer.write(f'    @block(dtask_{i}_id);\n')

    max_task_id = max((slot.hardware_task_id for slot in task_bindings.local_slots), default=csl.LOCAL_TASK_IDS[0] - 1)

    # Create exit task that unblocks command stream
    exit_task_sequential = all(typ == tdag.InterTaskEdge.SEQUENCE for t in tasks for n, typ in t.outgoing if n == -1)
    exit_task_sequential &= not any(
        t.task_type == 'data' for t in tasks for n, _ in t.outgoing if n == -1)  # No data tasks
    exit_task_blocked = any(n == -1 and typ == tdag.InterTaskEdge.UNBLOCK for t in tasks for n, typ in t.outgoing)

    # Bind exit task
    if not exit_task_sequential:
        footer.write(f'    @bind_local_task(exit_task, exit_task_id);\n')

    if exit_task_blocked:
        footer.write('    @block(exit_task_id);\n')

    # Write entry point code
    if benchmark_code.helpers:
        current_code.write(benchmark_code.helpers + '\n')

    current_code.write(f'''\nfn {kernel.name}({", ".join(scalar_arguments)}) void {{
''')

    # Write benchmarking code
    if benchmark_code.kernel_preamble:
        current_code.write(benchmark_code.kernel_preamble)

    # Copy scalar arguments to local variables
    for argument in scalar_arguments:
        arg_name = argument.split(':')[0].strip().removeprefix('__arg_')
        current_code.write(f'    {arg_name} = __arg_{arg_name};\n')

    # Reset recycled local-slot state.
    for slot in task_bindings.local_slots:
        if not slot.recycled:
            continue
        representative = slot.representative_task_index
        current_code.write(f'    {task_bindings.state_var(representative)} = '
                           f'{task_bindings.invalid_state_literal(representative)};\n')

    # Reset data task counters and re-block dedicated tasks.
    for i, task in enumerate(tasks):
        if task.task_type == "data":
            # Obtain initial parameter value
            stmt_id = task.statements[0]
            stmt = rect.metadata.compute.statements[stmt_id]
            assert isinstance(stmt, spir.ForeachStatement)
            if stmt.parameter_range:
                param_range = stmt.parameter_range[0]
                current_code.write(f'    __num_dtask_{i} = {param_range.start.as_ir()};\n')
        if task.task_type == 'data' and task.blocked:
            current_code.write(f'    @block(dtask_{i}_id);\n')
        if task.task_type == 'local' and task.blocked and not task_bindings.is_recycled_local_task(i):
            current_code.write(f'    @block(task_{i}_id);\n')

    # Activate/unblock all source tasks.
    # Local tasks are started with @activate;
    # data tasks run when data arrives on their color channel.
    non_source_tasks = set(n for i, t in enumerate(tasks) for n, _ in t.outgoing if n != i)
    source_tasks = [t for i, t in enumerate(tasks) if i not in non_source_tasks]
    for task in source_tasks:
        i = next(i for i, t in enumerate(tasks) if task is t)
        if task.task_type == 'data':
            # Data tasks are activated by data
            # And start as unblocked - noop
            pass
        else:
            current_code.write(task_bindings.emit_local_transition_preamble(i, task.blocked, indent='    '))
            current_code.write(f'    @activate(task_{i}_id);\n')
    if not source_tasks:
        current_code.write(benchmark_code.kernel_postamble)
        # Unblock command stream if function is empty
        current_code.write(f'    sys_mod.unblock_cmd_stream();\n')
    current_code.write('}\n')

    if not exit_task_sequential:
        current_code.write(f'''
const exit_task_id = @get_local_task_id({max_task_id + 1});
task exit_task() void {{
    {benchmark_code.kernel_postamble}
    // On completion, unblock command stream
    sys_mod.unblock_cmd_stream();
}}''')

    # Finalize footer
    footer.write(f'''
{benchmark_code.footer_exports}    @export_symbol({kernel.name}, "{kernel.name}");
}}\n''')

    # Finalize code generation by concatenating carets
    return header.getvalue() + '\n' + current_code.getvalue() + '\n' + footer.getvalue(), color_map


def _collect_colors_globally(kernel: spir.Kernel, rectangles: list[Rectangle[PEBlock]],
                             use_memcpy_mode: bool) -> dict[str, int]:
    """
    Returns a mapping of each channel to a CSL color.

    :param kernel: The kernel to use for argument colors.
    :param use_memcpy_mode: Whether to use memcpy mode.
    :return: Dictionary mapping each channel to its respective color.
    """
    channel_to_color: dict[int, int] = {}

    # Collect, for each rectangle, which channels are being read from and written to
    channel_is_read = set()
    channel_is_written = set()
    auto_stream_is_read = set()
    auto_stream_is_written = set()
    for rect in rectangles:
        sends_recvs = analysis.sends_and_receives(rect.metadata.compute)
        for stream_decl in rect.metadata.dataflow.statements:
            if stream_decl.stream_name not in sends_recvs:
                continue  # Unused stream
            outbound, inbound = sends_recvs[stream_decl.stream_name]
            if stream_decl.stream.routing.resolved_channel == "auto":
                if outbound:
                    auto_stream_is_written.add(stream_decl.stream_name)
                if inbound:
                    auto_stream_is_read.add(stream_decl.stream_name)
                continue  # Skip remainder of "auto" channels and assign them below
            if outbound:
                channel_is_written.add(stream_decl.stream.routing.resolved_channel)
            if inbound:
                channel_is_read.add(stream_decl.stream.routing.resolved_channel)

    max_channel = max(channel_is_read.union(channel_is_written), default=-1)

    # Assign all "auto" channels
    for rect in rectangles:
        for stream_decl in rect.metadata.dataflow.statements:
            if stream_decl.stream.routing.resolved_channel == "auto":
                stream_decl.stream.routing.channel = max_channel + 1
                if stream_decl.stream_name in auto_stream_is_written:
                    channel_is_written.add(max_channel + 1)
                if stream_decl.stream_name in auto_stream_is_read:
                    channel_is_read.add(max_channel + 1)
                max_channel += 1

    # Allocate colors for each channel
    color_offset = 0
    for channel in range(max_channel + 1):
        if channel in channel_to_color:
            continue
        if channel not in channel_is_read and channel not in channel_is_written:
            continue  # Unused channel
        if color_offset >= len(csl.COLORS):
            raise SyntaxError(
                f'Too many communication channels allocated for CSL: channel {channel} cannot be assigned a color')
        if channel in channel_is_written:
            channel_to_color[channel] = csl.COLORS[color_offset]
            color_offset += 1
        if channel in channel_is_read:
            if channel not in channel_to_color:
                channel_to_color[channel] = csl.COLORS[color_offset]
                color_offset += 1

    return channel_to_color


def _allocate_colors(rect: Rectangle[PEBlock], header: StringIO, kernel: spir.Kernel, use_memcpy_mode: bool,
                     stream_extents: analysis.StreamExtents, channel_to_color: dict[int, int]) -> dict[str, int]:
    """
    Creates a mapping of each stream to a CSL color, and adds an allocation there.

    :param rect: The rectangle to use.
    :param header: A code generator stream for a file's header (where the declarations are).
    :param kernel: The kernel to use for argument colors.
    :param use_memcpy_mode: Whether to use memcpy mode.
    :param stream_extents: The stream extents to use for argument color assignment.
    :param channel_to_color: A mapping of Spatial IR channels to colors to use for routed streams.
    :return: Dictionary mapping each stream to its respective color
    """
    result: dict[str, int] = {}
    channel_offset: int = 0

    if rect.metadata.dataflow.statements:
        header.write('\n// Colors\n')

    sends_recvs = analysis.sends_and_receives(rect.metadata.compute)
    # Collect colors from streams in dataflow
    for stream_decl in rect.metadata.dataflow.statements:
        name = name_to_csl(stream_decl.stream_name)

        if stream_decl.stream_name not in sends_recvs:
            continue  # Unused stream
        outbound, inbound = sends_recvs[stream_decl.stream_name]
        if stream_decl.stream.routing is None:
            raise SyntaxError(f'Non-routed stream "{name}". When generating CSL, Spatial IR code must have all streams '
                              'routed.')
        resolved = stream_decl.stream.routing.resolved_channel
        if resolved == 'auto':
            raise SyntaxError(f'"auto" stream channel found in stream "{name}". All streams must be concretized prior '
                              'to lowering to CSL')

        if outbound:
            # Look up channel in color map
            this_color = channel_to_color[channel_offset + resolved]

            # Add to mapping
            result[name + "_OUT"] = csl.COLORS[this_color]
            # Declare color
            header.write(f'const {name}_color_out: color = @get_color({result[name + "_OUT"]});\n')

        if inbound:
            # Look up channel in color map
            this_color = channel_to_color[channel_offset + resolved]

            # Add to mapping
            result[name + "_IN"] = csl.COLORS[this_color]
            # Declare color
            header.write(f'const {name}_color_in: color = @get_color({result[name + "_IN"]});\n')

    if result:
        header.write('\n')

    return result


def _collect_and_generate_fields(place: spir.PlaceBlock, header: StringIO, footer: StringIO, kernel: spir.Kernel,
                                 use_memcpy_mode: bool) -> None:
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

    # Add arguments to header and footer
    for field in place.statements:
        if not field.is_extern:
            continue
        if not isinstance(field.dtype, spir.ArrayType):  # Skip scalar arguments
            continue
        name = name_to_csl(field.field_name)
        if isinstance(field.dtype, spir.ArrayType):
            # Ignore array size in arguments, as they are spatially mapped
            ptrtype = dtype_as_csl(field.dtype, export=True)
        else:
            ptrtype = dtype_as_csl(spir.ArrayType(field.dtype, [1]), export=True)

        #header.write(f'var {name}: [{size}]'
        #                f'{dtype_as_csl(field.dtype.element_type.element_type.element_type)};\n')
        header.write(f'var __{name}_ptr: {ptrtype} = &{name};\n')
        footer.write(f'    @export_symbol(__{name}_ptr, "{name}");\n')

    if not use_memcpy_mode:
        # TODO(later): Some scaffolding for streaming indices within rectangle code
        pass

    # Add scalar arguments to header
    for argument in kernel.arguments:
        if not isinstance(argument.dtype, spir.ScalarType):  # Skip non-scalar arguments
            continue
        name = name_to_csl(argument.identifier)
        header.write(f'var {name}: {dtype_as_csl(argument.dtype)};\n')

    header.write('\n')



def _dtype_storage_bits(dtype: spir.ScalarType | spir.StreamType | spir.ArrayType) -> int:
    def _evaluate_extent(extent: int | spir.Expression) -> int:
        if isinstance(extent, int):
            return extent

        value = extent.eval()
        if not isinstance(value, int):
            raise ValueError(f'Expected integer place-block extent, got {value!r}.')
        return value

    if isinstance(dtype, spir.ScalarType):
        return BIT_WIDTH[dtype]
    if isinstance(dtype, spir.StreamType):
        return BIT_WIDTH[dtype.element_type]
    element_count = functools.reduce(lambda total, extent: total * _evaluate_extent(extent), dtype.shape, 1)
    return element_count * _dtype_storage_bits(dtype.base_type)


def _field_storage_bytes(field: spir.FieldDeclaration) -> int:
    bits = _dtype_storage_bits(field.dtype)
    return (bits + 7) // 8


def _place_block_storage_bytes(place: spir.PlaceBlock) -> int:
    return sum(_field_storage_bytes(field) for field in place.statements)


def _dsd_from_array(array_candidates: dict[str, tuple[spir.FieldDeclaration, list[int | spir.Expression]]],
                    node: spir.Identifier | spir.ArraySlice):
    ident = node if isinstance(node, spir.Identifier) else node.array
    _, shape = array_candidates[ident.as_ir()]
    if len(shape) == 1:
        dsd_type = cslstruct.DSDType.mem1d
        extents = [str(s) if isinstance(s, int) else s.as_ir() for s in shape]
    else:
        dsd_type = cslstruct.DSDType.mem4d
        extents = [str(s) if isinstance(s, int) else s.as_ir() for s in shape]

    # Find the index in the array
    def _find_index(ind: spir.Expression) -> spir.Identifier:
        candidates = []
        for n in ind.walk():
            if isinstance(n, spir.Identifier):
                candidates.append(n)
        if len(candidates) > 1:
            raise SyntaxError(
                f'Expected one index variable in array access, got {candidates}.\n  In line {ind.lineinfo}')
        return candidates[0] if candidates else None

    use_index = len(shape) == 1

    if isinstance(node, spir.ArraySlice):
        # Find and replace index with __index
        idxvars = [_find_index(ind) for ind in node.indices if _find_index(ind) is not None]
        if use_index:
            assert len(
                idxvars) == 1, f'Expected one index variable for 1D array, got {idxvars}.\n  In line {node.lineinfo}'
            far = passes.FindAndReplace({idxvars[0]: spir.Identifier('__index', 0)})
        else:
            far = passes.FindAndReplace({
                old: new for old, new in zip(idxvars, [spir.Identifier(f'__index_{i}', 0) for i in range(len(idxvars))])
            })
        idxvars = ['__index' if use_index else f'__index_{i}' for i in range(len(idxvars))]
        indices = [expr_to_csl(far.visit(copy.deepcopy(ind))) for ind in node.indices]
        name = name_to_csl(node.array)
    else:
        # Use __index if 1d
        if use_index:
            idxvars = ['__index']
            indices = ['__index']
        else:
            idxvars = [f'__index_{i}' for i in range(len(shape))]
            indices = [f'__index_{i}' for i in range(len(shape))]
        name = name_to_csl(node)

    return cslstruct.MemoryDSD(
        dsd_type,
        name,
        extents,
        idxvars,
        indices,
    )


def _dsd_from_stream(stream_candidates: dict[str, tuple[spir.StreamDeclaration | spir.KernelArgument,
                                                        int | spir.Expression]],
                     node: spir.Identifier | spir.ArraySlice):
    ident = node if isinstance(node, spir.Identifier) else node.array
    _, shape = stream_candidates[ident.as_ir()]
    dsd_type = cslstruct.DSDType.mem1d
    extents = [str(shape) if isinstance(shape, int) else shape.as_ir()]

    idxvars = ['__index']
    indices = ['__index']

    if isinstance(node, spir.ArraySlice):
        name = name_to_csl(node.array)
    else:
        name = name_to_csl(node)

    return cslstruct.MemoryDSD(dsd_type, name, extents, idxvars, indices)


def _collect_unique_dsds(
    tasks: list[tdag.CSLTask],
    rect: PEBlock,
    header: StringIO,
    dtypes: dict[spir.Identifier, spir.IRType],
    kernel: spir.Kernel,
    memcpy_mode: bool,
) -> UniqueDSDDict:
    """
    Returns a list of DSDs and generates them in the header.
    """
    dsds: UniqueDSDDict = defaultdict(list)

    # Generate appropriate header code
    header.write('// DSDs\n')

    # What defines a DSD?
    # 1. A (used) dataflow stream;
    # 2. A (used) local array in a place block, whose manipulation can use DSD operations; or
    # 3. An argument that is a stream or an array of streams in non memcpy mode, or buffer_size > 1 in memcpy mode.

    # Collect metadata from dataflow and place blocks
    stream_candidates: dict[str, tuple[spir.StreamDeclaration | spir.KernelArgument, int | spir.Expression]] = {}
    array_candidates: dict[str, tuple[spir.FieldDeclaration, list[int | spir.Expression]]] = {}
    stream_args: set[spir.Identifier] = set()
    for df_statement in rect.dataflow.statements:
        if isinstance(df_statement, spir.StreamDeclaration):
            buffer_size = df_statement.dtype.buffer_size or None
            stream_candidates[df_statement.stream_name.as_ir()] = (df_statement, buffer_size)
            if isinstance(df_statement.stream, spir.ExternStreamDeclaration):
                stream_args.add(df_statement.stream_name)
    for place_statement in rect.place.statements:
        if isinstance(place_statement, spir.FieldDeclaration):
            if isinstance(place_statement.dtype, spir.ArrayType):
                try:
                    eval_shape = [s if isinstance(s, int) else s.eval() for s in place_statement.dtype.shape]
                    # If the product of the shape is 1, it is a scalar
                    if not eval_shape or all(s == 1 for s in eval_shape):
                        # Scalar, no DSD
                        continue
                except ValueError:
                    # Dynamic shape, must create a DSD
                    pass

                array_candidates[place_statement.field_name.as_ir()] = (place_statement, place_statement.dtype.shape)

    # Find used DSDs in compute block
    # TODO: Infer input/output queue ID based on concurrency
    input_queue_id_ctr = 0
    output_queue_id_ctr = 0
    for stmt in rect.compute.statements:
        # Find out if compute block uses this stream for receive/send
        if isinstance(stmt, (spir.ReceiveStatement, spir.SendStatement)):
            stream_name = stmt.stream_name.array if isinstance(stmt.stream_name, spir.ArraySlice) else stmt.stream_name
            if memcpy_mode and stream_name in stream_args:
                # If memcpy mode is enabled, the stream contents will have already been copied to the PE
                dsd = _dsd_from_stream(stream_candidates, stream_name)
                dsds[stream_name.as_ir()].append((f"{name_to_csl(stream_name)}_dsd", dsd))
            elif isinstance(stmt, spir.ReceiveStatement) and stream_name.as_ir() in stream_candidates:
                dsd_type = cslstruct.DSDType.fabin
                dsd_name = f'{name_to_csl(stream_name)}_in_dsd'
                extents = stream_candidates[stream_name.as_ir()][1]
                if extents is not None:  # Use buffer size
                    extents = extents if isinstance(extents, int) else extents.eval()
                else:  # Infer from receive count
                    if isinstance(stmt.local_array, spir.ConstantLiteral) or isinstance(
                            dtypes[stmt.local_array], spir.ScalarType):
                        # Scalar receive
                        extents = 1
                    else:
                        extents = functools.reduce(
                            lambda a, b: a * b,
                            [s.eval() if not isinstance(s, int) else s for s in dtypes[stmt.local_array].shape], 1)
                fabric_color = f'{name_to_csl(stream_name)}_color'
                dsd = cslstruct.FabricDSD(dsd_type, fabric_color, extents,
                                          csl.INPUT_QUEUE_IDS[input_queue_id_ctr % len(csl.INPUT_QUEUE_IDS)])
                input_queue_id_ctr += 1
                dsds[stream_name.as_ir()].append((dsd_name, dsd))
            elif isinstance(stmt, spir.SendStatement) and stream_name.as_ir() in stream_candidates:
                dsd_type = cslstruct.DSDType.fabout
                dsd_name = f'{name_to_csl(stream_name)}_out_dsd'
                extents = stream_candidates[stream_name.as_ir()][1]
                if extents is not None:  # Use buffer size
                    extents = extents if isinstance(extents, int) else extents.eval()
                else:  # Infer from send count
                    if isinstance(stmt.local_array, spir.ConstantLiteral) or isinstance(
                            dtypes[stmt.local_array], spir.ScalarType):
                        # Scalar send
                        extents = 1
                    else:
                        extents = functools.reduce(
                            lambda a, b: a * b,
                            [s.eval() if not isinstance(s, int) else s for s in dtypes[stmt.local_array].shape], 1)
                fabric_color = f'{name_to_csl(stream_name)}_color'
                dsd = cslstruct.FabricDSD(dsd_type, fabric_color, extents,
                                          csl.OUTPUT_QUEUE_IDS[output_queue_id_ctr % len(csl.OUTPUT_QUEUE_IDS)])
                output_queue_id_ctr += 1
                dsds[stream_name.as_ir()].append((dsd_name, dsd))

            if isinstance(stmt, spir.SendStatement):
                # If the send statement sends from a local array, create another DSD
                # Do the same for the stream
                # This case does not apply for receive statements, as they would be lowered to foreach statements
                for arr in (stmt.local_array, stmt.stream_name):
                    if arr.as_ir() in array_candidates:
                        _, shape = array_candidates[arr.as_ir()]
                        if len(shape) == 1:
                            dsd_type = cslstruct.DSDType.mem1d
                            extents = [str(s) if isinstance(s, int) else s.as_ir() for s in shape]
                            indices = ['__index']
                        else:
                            dsd_type = cslstruct.DSDType.mem4d
                            extents = [str(s) if isinstance(s, int) else s.as_ir() for s in shape]
                            indices = [f'__index_{i}' for i in range(len(shape))]

                        dsd = cslstruct.MemoryDSD(
                            dsd_type,
                            name_to_csl(arr),
                            extents,
                            indices,
                            indices,
                        )
                        dsds[arr.as_ir()].append((f"{name_to_csl(arr)}_dsd", dsd))

        elif isinstance(stmt, spir.ForeachStatement):
            # If the foreach statement has a stream generator, it is a DSD
            # unless only the receive generator is given (streaming, no range provided).
            stream_name = (
                stmt.receive_stream.stream_name.array
                if isinstance(stmt.receive_stream.stream_name, spir.ArraySlice) else stmt.receive_stream.stream_name)
            if not stmt.parameter_range:
                if stream_name not in stream_args:
                    raise SyntaxError(f'Foreach generator "{stream_name.as_ir()}" without a defined '
                                      f'range must only be used with a kernel argument or extern_stream.'
                                      f'\n  In line {stmt.lineinfo}')
                # A data task will be created instead (handled in _generate_data_task)
            else:
                if stream_name.as_ir() in stream_candidates:
                    if memcpy_mode and stream_name in stream_args:
                        # If memcpy mode is enabled, the stream contents will have already been copied to the PE
                        dsd = _dsd_from_stream(stream_candidates, stream_name)
                        dsds[stream_name.as_ir()].append((f"{name_to_csl(stream_name)}_dsd", dsd))
                    else:
                        dsd_name = f'{name_to_csl(stream_name)}_in_dsd'
                        extents = stream_candidates[stream_name.as_ir()][1]
                        if extents is not None:  # Use buffer size
                            extents = extents if isinstance(extents, int) else extents.eval()
                        else:  # Infer from foreach range
                            if len(stmt.parameter_range) != 1:
                                raise SyntaxError(
                                    f'Expected one-dimensional foreach range for stream "{stream_name.as_ir()}", got {stmt.parameter_range}.\n  In line {stmt.lineinfo}'
                                )
                            start, end, step = stmt.parameter_range[0].start, stmt.parameter_range[
                                0].stop, stmt.parameter_range[0].step
                            extents = (end.eval() - start.eval()) // (step.eval() if step is not None else 1)
                        fabric_color = f'{name_to_csl(stream_name)}_color'
                        dsd = cslstruct.FabricDSD(cslstruct.DSDType.fabin, fabric_color, extents,
                                                  csl.INPUT_QUEUE_IDS[input_queue_id_ctr % len(csl.INPUT_QUEUE_IDS)])
                        dsds[stream_name.as_ir()].append((dsd_name, dsd))
                        input_queue_id_ctr += 1

        def _visit_nested_send(substmt: spir.SendStatement):
            if substmt.stream_name.as_ir() not in stream_candidates:
                return
            # Stream DSD (i.e., await send in a foreach)
            stream_name = substmt.stream_name
            dsd_type = cslstruct.DSDType.fabout
            dsd_name = f'{name_to_csl(stream_name)}_out_dsd'
            extents = stream_candidates[stream_name.as_ir()][1]
            if extents is not None:  # Use buffer size
                extents = extents if isinstance(extents, int) else extents.eval()
            else:  # Infer from send count
                if isinstance(substmt.local_array, (spir.ArraySlice, spir.ConstantLiteral)) or isinstance(
                        dtypes[substmt.local_array], spir.ScalarType):
                    # Scalar send
                    extents = 1
                else:
                    extents = functools.reduce(
                        lambda a, b: a * b,
                        [s.eval() if not isinstance(s, int) else s for s in dtypes[substmt.local_array].shape], 1)
            fabric_color = f'{name_to_csl(stream_name)}_color'
            nonlocal output_queue_id_ctr
            dsd = cslstruct.FabricDSD(dsd_type, fabric_color, extents,
                                      csl.OUTPUT_QUEUE_IDS[output_queue_id_ctr % len(csl.OUTPUT_QUEUE_IDS)])
            output_queue_id_ctr += 1
            dsds[stream_name.as_ir()].append((dsd_name, dsd))

        def _visit_nested_receive(substmt: spir.ReceiveStatement):
            if substmt.stream_name.as_ir() not in stream_candidates:
                return
            # Stream DSD (i.e., await receive in a lowered for loop)
            stream_name = substmt.stream_name
            dsd_type = cslstruct.DSDType.fabin
            dsd_name = f'{name_to_csl(stream_name)}_in_dsd'
            extents = stream_candidates[stream_name.as_ir()][1]
            if extents is not None:  # Use buffer size
                extents = extents if isinstance(extents, int) else extents.eval()
            else:  # Infer from receive count
                if isinstance(substmt.local_array, spir.TypedIdentifier):
                    local_array = substmt.local_array.identifier
                else:
                    local_array = substmt.local_array
                if isinstance(local_array, spir.ArraySlice) or isinstance(dtypes[local_array], spir.ScalarType):
                    # Scalar receive
                    extents = 1
                else:
                    extents = functools.reduce(
                        lambda a, b: a * b,
                        [s.eval() if not isinstance(s, int) else s for s in dtypes[local_array].shape], 1)
            fabric_color = f'{name_to_csl(stream_name)}_color'
            nonlocal input_queue_id_ctr
            dsd = cslstruct.FabricDSD(dsd_type, fabric_color, extents,
                                      csl.INPUT_QUEUE_IDS[input_queue_id_ctr % len(csl.INPUT_QUEUE_IDS)])
            input_queue_id_ctr += 1
            dsds[stream_name.as_ir()].append((dsd_name, dsd))

        def _visit_dsd(substmt, in_scope, in_assignment):
            if isinstance(substmt, spir.SendStatement) and in_scope:
                _visit_nested_send(substmt)
                return
            if isinstance(substmt, spir.ReceiveStatement):
                if in_scope:
                    _visit_nested_receive(substmt)
                return
            if (isinstance(substmt, spir.Identifier) and substmt.as_ir() in array_candidates and
                    substmt.as_ir() not in dsds):
                dsds[substmt.as_ir()].append((f"{name_to_csl(substmt)}_dsd", _dsd_from_array(array_candidates,
                                                                                             substmt)))
                return

            # If the destination is an array, we need to create a DSD
            if not isinstance(substmt, spir.ArraySlice):
                return
            if substmt.array.as_ir() not in array_candidates:
                return
            if not in_scope:
                return

            dsd = _dsd_from_array(array_candidates, substmt)
            dsds[substmt.array.as_ir()].append((f"{name_to_csl(substmt.array)}_dsd", dsd))

        DSDVisitor(_visit_dsd, toplevel=not hasattr(stmt, 'body')).visit(stmt)

    # Make DSD values in the dictionary unique if equivalent
    for key, dsd_list in dsds.items():
        unique_dsds = {}
        for name, dsd in dsd_list:
            if dsd not in unique_dsds:
                unique_dsds[dsd] = name
        dsds[key] = [(v, k) for k, v in unique_dsds.items()]

    # Write DSDs to header
    for dsd_value in dsds.values():
        for name, dsd in dsd_value:
            header.write(f'const {name} = {dsd.as_csl()};\n')

    # TODO(later): This function assumes that DSDs are tied to identifiers. This is a limitation
    # of the dictionary keys, which cannot use arbitrary Spatial IR nodes. This can lead to issues
    # where an identifier is accessed in multiple contexts (e.g., x[i] and x[i+1]).
    # An ideal solution would tie the DSDs to IR nodes (e.g., foreach) and then run a post-processing
    # pass to eliminate duplicates.
    for key, dsd_list in dsds.items():
        if len(dsd_list) > 1:
            if isinstance(dsd_list[0][1], cslstruct.MemoryDSD):
                raise SyntaxError(f"Multiple Memory DSDs for variable {key}, got {[name for name, _ in dsd_list]}.")
            assert isinstance(dsd_list[0][1], cslstruct.FabricDSD) and isinstance(dsd_list[1][1], cslstruct.FabricDSD), \
                f"Expected FabricDSD for key {key}, got {[type(dsd) for _, dsd in dsd_list]}."
            assert len(dsd_list) == 2, f"Expected up to two DSDs for key {key}, got {[name for name, _ in dsd_list]}."

    return dsds


class DSDVisitor(spir.NodeVisitor):

    def __init__(self, callback, toplevel: bool):
        self.callback = callback
        self.toplevel = toplevel
        self.in_foreach = False
        self.in_map = False
        self.in_for = False
        self.in_assignment = False
        super().__init__()

    def visit_ForeachStatement(self, node: spir.ForeachStatement):
        old_scope = self.in_foreach
        self.in_foreach = True
        self.generic_visit(node)
        self.in_foreach = old_scope

    def visit_MapStatement(self, node: spir.MapStatement):
        old_scope = self.in_map
        self.in_map = True
        self.generic_visit(node)
        self.in_map = old_scope

    def visit_ForStatement(self, node: spir.ForStatement):
        old_scope = self.in_for
        self.in_for = True
        self.generic_visit(node)
        self.in_for = old_scope

    def visit_Identifier(self, node: spir.Identifier):
        self.callback(node, self.in_foreach or self.in_map, self.in_assignment)
        return

    def visit_SendStatement(self, node: spir.SendStatement):
        self.callback(node, self.in_foreach or self.in_map, self.in_assignment)
        return

    def visit_ReceiveStatement(self, node: spir.ReceiveStatement):
        self.callback(node, self.in_foreach or self.in_map, self.in_assignment)
        return

    def visit_ArraySlice(self, node: spir.ArraySlice):
        self.callback(node, self.in_foreach or self.in_map, self.in_assignment)
        # Do not visit internal identifier
        return

    def visit_AssignmentStatement(self, node: spir.AssignmentStatement):
        old_assignment = self.in_assignment
        self.in_assignment = True
        if self.toplevel and isinstance(node.destination, spir.ArraySlice):
            self.generic_visit(node.source)  # Do not visit assignment to array slice
        else:
            self.generic_visit(node)
        self.in_assignment = old_assignment
        return


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


def _collect_routes(rectangles: list[Rectangle[PEBlock]], color_maps: list[dict[str,
                                                                                int]]) -> dict[tuple[int, int], str]:
    """
    Creates a parametric version of the Routing Graph (see the Spatial IR specification for more information) and
    returns a dictionary of code segements to add to the layout CSL file based on the streams.

    :param rectangles: All rectangles involved in this kernel.
    :return: A dictionary mapping the starting point of each rectangle to a string representing the layout instructions.
    """
    INDENT = 12 * ' '
    result = {}

    # Create a routing graph
    for rect, color_map in zip(rectangles, color_maps):
        # Test whether a receive/send statement are called for creating inbound/outbound routes
        sends_recvs = analysis.sends_and_receives(rect.metadata.compute)
        inst = ''

        # Make routing instructions unique
        routing_instructions: set[str] = set()

        # For each hop, make a color WEST-EAST/NORTH-SOUTH pair. For the first and last hop, pair with RAMP
        for stream in rect.metadata.dataflow.statements:
            if stream.stream_name not in sends_recvs:  # Skip unused streams
                continue
            sent, received = sends_recvs[stream.stream_name]
            if received:
                color_name_inbound = f'@get_color({color_map[name_to_csl(stream.stream_name) + "_IN"]})'
            if sent:
                color_name_outbound = f'@get_color({color_map[name_to_csl(stream.stream_name) + "_OUT"]})'

            if isinstance(stream.stream, spir.ExternStreamDeclaration):
                continue  # Extern streams do not have on-chip routing

            if isinstance(stream.stream, spir.MulticastRangeStreamDeclaration):
                if sent and received:
                    raise ValueError(
                        f"Multicast stream '{stream.stream_name.as_ir()}' is both sent and received "
                        f"within the same compute rectangle [{rect.x_range[0]}:{rect.x_range[1]}, "
                        f"{rect.y_range[0]}:{rect.y_range[1]}]. "
                        "Sender and receiver compute blocks must be in separate rectangles for multicast streams.")
                if not sent:
                    # All multicast routing is emitted by the rectangle that sends this stream.
                    continue
                rng = stream.stream.multicast_range
                start = int(rng.start.eval())
                stop = int(rng.stop.eval())
                axis = stream.stream.multicast_axis
                is_negative = start < 0

                if axis == 'y':
                    if is_negative:
                        tx_dir, rx_dir = 'NORTH', 'SOUTH'
                    else:
                        tx_dir, rx_dir = 'SOUTH', 'NORTH'

                    def _coord(k):  # noqa: E731
                        if k >= 0:
                            return 'pe_x', f'pe_y + {k}'
                        return 'pe_x', f'pe_y - {-k}'
                else:
                    if is_negative:
                        tx_dir, rx_dir = 'WEST', 'EAST'
                    else:
                        tx_dir, rx_dir = 'EAST', 'WEST'

                    def _coord(k):  # noqa: E731
                        if k >= 0:
                            return f'pe_x + {k}', 'pe_y'
                        return f'pe_x - {-k}', 'pe_y'

                # Sender: inject into fabric toward receivers.
                routing_inst = INDENT + '@set_color_config(pe_x, pe_y, %s, .{ .routes = .{ .rx = .{RAMP}, .tx = .{%s} } });\n' % (
                    color_name_outbound, tx_dir)
                if routing_inst not in routing_instructions:
                    inst += routing_inst
                    routing_instructions.add(routing_inst)

                if is_negative:
                    # Negative multicast: receivers at start, start-1, …, stop+1 (stop exclusive).
                    k_last = stop + 1  # farthest receiver

                    # Gap relay-only PEs between sender and first receiver (when start < -1).
                    for k in range(-1, start, -1):
                        cx, cy = _coord(k)
                        routing_inst = INDENT + '@set_color_config(%s, %s, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                            cx, cy, color_name_outbound, rx_dir, tx_dir)
                        if routing_inst not in routing_instructions:
                            inst += routing_inst
                            routing_instructions.add(routing_inst)

                    # Intermediate receivers: forward toward farthest and deliver to RAMP.
                    for k in range(start, k_last, -1):
                        cx, cy = _coord(k)
                        routing_inst = INDENT + '@set_color_config(%s, %s, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s, RAMP} } });\n' % (
                            cx, cy, color_name_outbound, rx_dir, tx_dir)
                        if routing_inst not in routing_instructions:
                            inst += routing_inst
                            routing_instructions.add(routing_inst)

                    # Last (farthest) receiver: deliver to RAMP only, no forwarding.
                    cx, cy = _coord(k_last)
                    routing_inst = INDENT + '@set_color_config(%s, %s, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{RAMP} } });\n' % (
                        cx, cy, color_name_outbound, rx_dir)
                    if routing_inst not in routing_instructions:
                        inst += routing_inst
                        routing_instructions.add(routing_inst)
                else:
                    # Positive multicast: receivers at start, start+1, …, stop-1 (stop exclusive).
                    # Gap relay-only PEs between sender and first receiver (when start > 1).
                    for k in range(1, start):
                        cx, cy = _coord(k)
                        routing_inst = INDENT + '@set_color_config(%s, %s, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                            cx, cy, color_name_outbound, rx_dir, tx_dir)
                        if routing_inst not in routing_instructions:
                            inst += routing_inst
                            routing_instructions.add(routing_inst)

                    # Intermediate receivers: forward and simultaneously deliver to RAMP.
                    for k in range(start, stop - 1):
                        cx, cy = _coord(k)
                        routing_inst = INDENT + '@set_color_config(%s, %s, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s, RAMP} } });\n' % (
                            cx, cy, color_name_outbound, rx_dir, tx_dir)
                        if routing_inst not in routing_instructions:
                            inst += routing_inst
                            routing_instructions.add(routing_inst)

                    # Last receiver: deliver to RAMP only, no forwarding.
                    k_last = stop - 1
                    cx, cy = _coord(k_last)
                    routing_inst = INDENT + '@set_color_config(%s, %s, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{RAMP} } });\n' % (
                        cx, cy, color_name_outbound, rx_dir)
                    if routing_inst not in routing_instructions:
                        inst += routing_inst
                        routing_instructions.add(routing_inst)
                continue

            if len(stream.stream.routing.hops) == 1:  # Inbound and outbound generated together
                route = _route_dir(*stream.stream.routing.hops[0].offset)
                if sent:
                    routing_inst = INDENT + '@set_color_config(pe_x, pe_y, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                        color_name_outbound, 'RAMP', route[1])
                    if routing_inst not in routing_instructions:
                        inst += routing_inst
                        routing_instructions.add(routing_inst)
                if received:
                    routing_inst = INDENT + '@set_color_config(pe_x, pe_y, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                        color_name_inbound, route[0], 'RAMP')
                    if routing_inst not in routing_instructions:
                        inst += routing_inst
                        routing_instructions.add(routing_inst)
            else:  # Multi-hop
                if sent:
                    first_hop = stream.stream.routing.hops[0]
                    route = ('RAMP', _route_dir(*first_hop.offset)[1])
                    routing_inst = INDENT + '@set_color_config(pe_x, pe_y, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                        color_name_outbound, route[0], route[1])
                    if routing_inst not in routing_instructions:
                        inst += routing_inst
                        routing_instructions.add(routing_inst)
                    cur_offx = 0
                    cur_offy = 0
                    for hop in stream.stream.routing.hops[1:]:
                        route = _route_dir(*hop.offset)
                        cur_offx += hop.offset[0]
                        cur_offy += hop.offset[1]
                        routing_inst = INDENT + '@set_color_config(pe_x + %d, pe_y + %d, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                            cur_offx, cur_offy, color_name_outbound, route[0], route[1])
                        if routing_inst not in routing_instructions:
                            inst += routing_inst
                            routing_instructions.add(routing_inst)
                if received:
                    # The receiver only configures itself (pe_x + 0, pe_y + 0).
                    # Intermediate PEs are configured by the sender block above,
                    # which walks forward through hops[1:] relative to the sender PE.
                    last_hop = stream.stream.routing.hops[-1]
                    route = (_route_dir(*last_hop.offset)[0], 'RAMP')
                    routing_inst = INDENT + '@set_color_config(pe_x, pe_y, %s, .{ .routes = .{ .rx = .{%s}, .tx = .{%s} } });\n' % (
                        color_name_inbound, route[0], route[1])
                    if routing_inst not in routing_instructions:
                        inst += routing_inst
                        routing_instructions.add(routing_inst)

        result[(rect.x_range[0], rect.y_range[0])] = inst

    return result


def _write_indented_block(current_code: StringIO, block: str, indent: str) -> None:
    block = textwrap.dedent(block).strip('\n')
    if not block:
        return
    for line in block.splitlines():
        current_code.write(f'{indent}{line}\n')


def _generate_data_task(
    rect: PEBlock,
    task_index: int,
    task: tdag.CSLTask,
    current_code: StringIO,
    header: StringIO,
    footer: StringIO,
    dsds: UniqueDSDDict,
    dtypes: dict[spir.Identifier, spir.IRType],
    color_map: dict[str, int],
    tasks: list[tdag.CSLTask],
):
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
    :param tasks: A list of all tasks in the kernel.
    """
    #   * If index is requested: before unblocking task, set k; inc at end of task
    #   * Wavelet-triggered task as fallback
    assert task.task_type == 'data'
    assert len(task.statements) == 1

    stmt_id = task.statements[0]
    if isinstance(stmt_id, int) and stmt_id >= 0:
        stmt = rect.compute.statements[stmt_id]
        assert isinstance(stmt, spir.ForeachStatement)
    else:
        return
    next_task, itedge = task.outgoing[0]
    next_task_type = tasks[next_task].task_type if next_task != -1 else 'local'
    itedge_code = 'unblock' if itedge == tdag.InterTaskEdge.UNBLOCK else 'activate'

    # If a range was specified, write counter and add code to execute next task
    if stmt.parameter_range:
        assert len(stmt.parameter_range) == 1, 'Only one-dimensional foreach loops are supported in data tasks'
        if next_task == -1:
            next_task_code = f'@{itedge_code}(exit_task_id);'
        else:
            prefix = "d" if next_task_type == 'data' else ""
            next_task_code = f'@{itedge_code}({prefix}task_{next_task}_id);'

        var_dtype_csl = dtype_as_csl(stmt.variables[0].dtype)
        param_range = stmt.parameter_range[0]
        current_code.write(f"var __num_dtask_{task_index}: {var_dtype_csl} = {param_range.start.as_ir()};\n")

        next_task_code = f"""
    __num_dtask_{task_index} += {1 if param_range.step is None else param_range.step.as_ir()};
    if (__num_dtask_{task_index} == {param_range.stop.as_ir()}) {{
        {next_task_code}
    }}"""
    else:
        next_task_code = ""

    # Write frame for data task
    argtype_csl = dtype_as_csl(stmt.stream_variable.dtype)
    argname = name_to_csl(stmt.stream_variable.identifier)
    current_code.write(f"task dtask_{task_index}({argname}: {argtype_csl}) void {{\n")
    if stmt.variables:
        current_code.write(
            f'    var {name_to_csl(stmt.variables[0].identifier)}: {var_dtype_csl} = __num_dtask_{task_index};\n')

    # Write op contents
    for substmt in stmt.body:
        code = cslstmt.generate_csl_statement(substmt, dsds, dtypes, None, header, in_foreach_or_map=True)

        for line in code.splitlines():
            current_code.write(f'    {line}\n')

    # Write footer
    current_code.write(next_task_code)
    current_code.write(f"\n}}\n")


def _generate_task_code(rect: PEBlock,
                        task_index: int,
                        task: tdag.CSLTask,
                        current_code: StringIO,
                        header: StringIO,
                        footer: StringIO,
                        dsds: UniqueDSDDict,
                        dtypes: dict[spir.Identifier, spir.IRType],
                        color_map: dict[str, int],
                        tasks: list[tdag.CSLTask],
                        task_bindings: task_recycling.TaskBindingPlan,
                        postamble: str,
                        indent: str = '    '):
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
    :param tasks: A list of all tasks in the kernel.
    :param postamble: The code to be added at the end of the kernel.
    """
    # Convert task contents:
    # Convert receives/sends from/to arguments to memcpy
    # Communication calls become async calls based on task's outgoing field
    # ``map``:
    #    * becomes DSD operations as much as possible
    #    * @map as a fallback
    # if ``foreach``, has to be a DSD operation
    assert task.task_type == 'local'

    for stmt_id, (next_task, itedge) in zip(task.statements, task.outgoing):
        skip_activation = False
        task_id = None
        transition_preamble = ''

        if isinstance(stmt_id, int) and stmt_id >= 0:
            # Write op contents
            stmt = rect.compute.statements[stmt_id]
            # If DSD is asynchronous, encode the next task and edge type (activate/unblock)
            # into the DSD operation
            if itedge in (tdag.InterTaskEdge.ACTIVATE, tdag.InterTaskEdge.UNBLOCK):
                if next_task == -1:
                    task_id = 'exit_task_id'
                else:
                    if tasks[next_task].task_type == 'local':
                        transition_preamble = task_bindings.emit_local_transition_preamble(
                            next_task, tasks[next_task].blocked, indent=indent)
                        task_id = f'task_{next_task}_id'
                    else:
                        task_id = f'dtask_{next_task}_id'

                async_target = dsd_ops.AsyncTarget(task_id, itedge.name.lower())
            else:
                async_target = None

            if transition_preamble:
                current_code.write(transition_preamble)

            code = cslstmt.generate_csl_statement(stmt, dsds, dtypes, async_target, header)
            lines = code.splitlines()

            # DSD operation or async call
            if any(dsdop in line for line in lines for dsdop in dsd_ops.DSD_ASSIGNMENT_MAPPING):
                # Asynchronous DSD op. DSD line already contains activation or unblocking
                skip_activation = True

            for line in lines:
                current_code.write(f'{indent}{line}\n')

        if next_task == -1 and itedge == tdag.InterTaskEdge.SEQUENCE:
            # Run exit task directly
            _write_indented_block(current_code, postamble, indent)
            current_code.write(f'{indent}sys_mod.unblock_cmd_stream();\n')
            continue

        if skip_activation:
            continue

        # If not DSD or asynchronous op, activate/unblock must be called after the generated operation code
        if itedge in (tdag.InterTaskEdge.ACTIVATE, tdag.InterTaskEdge.UNBLOCK):
            # Determine task ID
            if next_task == -1:
                task_id = 'exit_task_id'
            else:
                if tasks[next_task].task_type == 'local':
                    if not transition_preamble:
                        current_code.write(
                            task_bindings.emit_local_transition_preamble(
                                next_task, tasks[next_task].blocked, indent=indent))
                    task_id = f'task_{next_task}_id'
                else:
                    task_id = f'dtask_{next_task}_id'
            if itedge == tdag.InterTaskEdge.ACTIVATE:
                current_code.write(f'{indent}@activate({task_id});\n')
            elif itedge == tdag.InterTaskEdge.UNBLOCK:
                current_code.write(f'{indent}@unblock({task_id});\n')


def _generate_benchmarking_code(header: StringIO, disable_benchmarking: bool,
                                sync_benchmarking: bool) -> cslbench.RectangleBenchmarkingCode:
    """
    Generates benchmarking code in the header and current code.
    
    :param header: A code generator stream for a file's header (where the declarations are).
    :return: Benchmarking code fragments to insert into the generated file.
    """
    if disable_benchmarking:
        return cslbench.RectangleBenchmarkingCode()

    benchmark_code = (
        cslbench.generate_sync_rectangle_code() if sync_benchmarking else cslbench.generate_basic_rectangle_code())
    header.write(benchmark_code.header)
    return benchmark_code


def _add_benchmarking_fields(rectangles: list[Rectangle[PEBlock]], sync_benchmarking: bool = False):
    """
    Adds benchmarking variables to the code.
    :param rectangles: The rectangles to modify.
    """
    for rect in rectangles:
        rect.metadata.place.statements.append(
            spir.FieldDeclaration(
                field_name=spir.Identifier('__benchmark_start', 0),
                dtype=spir.ArrayType(spir.ScalarType.u16, [3]),
                is_extern=True,
            ))
        rect.metadata.place.statements.append(
            spir.FieldDeclaration(
                field_name=spir.Identifier('__benchmark_stop', 0),
                dtype=spir.ArrayType(spir.ScalarType.u16, [3]),
                is_extern=True,
            ))
        if sync_benchmarking:
            rect.metadata.place.statements.append(
                spir.FieldDeclaration(
                    field_name=spir.Identifier('__benchmark_refclock', 0),
                    dtype=spir.ArrayType(spir.ScalarType.u16, [3]),
                    is_extern=True,
                ))


def _collect_identifier_types(rect: PEBlock,
                              kernel_args: list[spir.KernelArgument]) -> dict[spir.Identifier, spir.IRType]:
    """
    Returns a dictionary mapping all place, dataflow, and compute variables (streams, arrays, scalars) to datatypes.
    """
    result = {}

    # Collect from kernel arguments
    for arg in kernel_args:
        result[arg.identifier] = arg.dtype

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
