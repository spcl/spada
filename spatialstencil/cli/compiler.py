import click
import itertools
import os
from spatialstencil.lowering import spatial_ir_to_csl as s2c
from spatialstencil.syntax.spatial_ir import parser, passes, analysis, irnodes as spa, canonicalization
from spatialstencil.syntax.csl import constants as csl
from spatialstencil.syntax.common import serialization
import subprocess


@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_folder', type=click.Path(dir_okay=True))
@click.option('--param', '-p', multiple=True, help='Kernel parameters in key=value format')
@click.option('--offset-x', '-x', default=0, type=int, help='Offset for rectangular region in x direction')
@click.option('--offset-y', '-y', default=0, type=int, help='Offset for rectangular region in y direction')
@click.option('--generate-only', '-g', is_flag=True, help='Only generate the output files without compiling them')
@click.option('--disable-benchmarking', is_flag=True, help='Disable benchmarking code generation (and memory overhead)')
@click.option('--sync-benchmarking', is_flag=True, help='Generate sync-assisted benchmarking support')
@click.option('--disable-asynchronous', is_flag=True, help='Disable asynchronous task code generation')
@click.option('--disable-dsd', is_flag=True, help='Disable DSD operation detection and code generation')
@click.option('--disable-map', is_flag=True, help='Disable @map operation detection and code generation')
@click.option('--disable-task-fusion', is_flag=True, help='Disable task fusion optimization')
@click.option('--disable-task-recycling', is_flag=True, help='Disable task ID recycling')
@click.option('--disable-copy-elision', is_flag=True, help='Disable copy elimination optimization pass')
def compile_spatial_ir(input_file: str, output_folder: str, param: list[str], offset_x: int, offset_y: int,
                       generate_only: bool, disable_benchmarking: bool, sync_benchmarking: bool,
                       disable_asynchronous: bool, disable_dsd: bool, disable_map: bool,
                       disable_task_fusion: bool, disable_task_recycling: bool, disable_copy_elision: bool):
    if disable_benchmarking and sync_benchmarking:
        raise click.UsageError("--sync-benchmarking cannot be used with --disable-benchmarking")

    # Parse parameters into dictionary
    kernel_parameters = {}
    for p in param:
        if '=' not in p:
            raise ValueError(f'Invalid parameter format: {p}. Expected key=value')
        key, value = p.split('=', 1)
        # Try to parse as int, then float, otherwise keep as string
        try:
            kernel_parameters[key] = int(value)
        except ValueError:
            try:
                kernel_parameters[key] = float(value)
            except ValueError:
                kernel_parameters[key] = value

    kernel = parser.parse_file(input_file)
    # If there are unconcretized parameters, we need to concretize them
    non_concrete_parameters = {param.name for param in kernel.parameters if param.value is None}
    non_concrete_parameters -= set(kernel_parameters.keys())
    if non_concrete_parameters:
        raise ValueError(f'Kernel has non-concrete parameters: {", ".join(non_concrete_parameters)}.\n'
                         'Please provide values for them using --param option. For example: -p I=128 -p J=128 -p K=80')

    # Concretize parameters and propagate constant expressions
    if kernel_parameters:
        print("Concretizing parameters:", kernel_parameters)
    kernel = passes.concretize_parameters(kernel, **kernel_parameters)
    kernel = passes.constexpr_propagation(kernel)

    # Argument checks
    using_memcpy_mode = None
    argnames = []
    for arg in kernel.arguments:
        argnames.append(arg.identifier.name)
        # Change all shapes to be lists of integers
        if isinstance(arg.dtype, spa.ArrayType):
            arg.dtype.shape = [dim.eval() if isinstance(dim, spa.Expression) else int(dim) for dim in arg.dtype.shape]
        # Check if the argument is a stream and has a memcpy mode
        if isinstance(arg.dtype, spa.ArrayType) and isinstance(arg.dtype.base_type, spa.StreamType):
            if arg.dtype.base_type.buffer_size is not None:
                if using_memcpy_mode is None:
                    using_memcpy_mode = True
                elif not using_memcpy_mode:
                    raise ValueError("Kernel has both memcpy and non-memcpy stream arguments. "
                                     "Please ensure all stream arguments are either memcpy or non-memcpy.")
            else:
                if using_memcpy_mode is None:
                    using_memcpy_mode = False
                elif using_memcpy_mode:
                    raise ValueError("Kernel has both memcpy and non-memcpy stream arguments. "
                                     "Please ensure all stream arguments are either memcpy or non-memcpy.")

    if using_memcpy_mode is None:
        # If no stream arguments are present, default to non-memcpy mode
        using_memcpy_mode = False

    if disable_map:
        from spatialstencil.syntax.csl import statements
        statements.DISABLE_MAPS = True

    # Lower the spatial IR to CSL
    csl_files = s2c.lower_spatial_ir_to_csl(
        kernel,
        disable_benchmarking=disable_benchmarking,
        sync_benchmarking=sync_benchmarking,
        disable_asynchronous=disable_asynchronous,
        disable_dsd=disable_dsd,
        task_fusion=not disable_task_fusion,
        copy_elision=not disable_copy_elision,
        task_id_recycling=not disable_task_recycling,
    )

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    for f in csl_files:
        output_path = os.path.join(output_folder, f.filename)
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_path, 'w') as out_file:
            out_file.write(f.code)

    # Compile the generated CSL files using the cslc command (and change the cwd to the output folder)
    # Get the fabric dimensions from the kernel and offsets from the command line arguments
    # Command: cslc layout.csl --fabric-dims=16,16 --fabric-offsets=0,0 --memcpy --channels=1
    memcpy_channels = 1  # TODO: Determine the number of memcpy channels (1-16) based on the kernel arguments

    # Generate metadata.json file
    # Run on a separate variable so the original kernel is not mutated.
    # Use the same canonicalization pipeline as lower_spatial_ir_to_csl to keep both in sync.
    kernel_meta = s2c.canonicalize_kernel(kernel)
    input_args, output_args = analysis.get_kernel_stream_arguments(kernel_meta)
    rectangles = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel_meta)
    stream_extents = analysis.detect_stream_argument_extents(rectangles, kernel_meta)

    # kernel_meta.get_grid_rect() returns tight bounds (last-contained PE + 1) after
    # canonicalization has expanded all MetaForBlocks.
    xbegin, xend, ybegin, yend = kernel_meta.get_grid_rect()
    kernel_dims = [xend - xbegin, yend - ybegin]

    if memcpy_channels >= 0:
        xbegin += 4  # Memcpy needs 3 extra columns to the left, and 1 extra column for fabric offset
        xend += 4 + 2 + 1  # Memcpy needs 2 extra columns to the right, plus an extra column for fabric offset
        ybegin += 1
        yend += 1 + 1

    for argname, arg in itertools.chain(input_args.items(), output_args.items()):
        if len(arg["shape"]) > 0:  # Ignore scalar arguments
            arg_id = spa.Identifier(argname, 0)
            if arg_id not in stream_extents.extents:

                if arg['shape'] == kernel_dims:
                    # Special case for arguments that match the kernel grid size exactly
                    arg["rect_offset_used"] = [0, 0]
                    arg["column_major"] = False
                    arg["rect_offset"] = [0, 0]
                    continue

                raise ValueError(f"Argument '{argname}' does not have a detected extent. "
                                 "Please ensure the argument is properly defined in the kernel.")
            arg["rect_offset_used"] = [
                stream_extents.extents[arg_id][0].x_range[0], stream_extents.extents[arg_id][0].y_range[0]
            ]
            arg["column_major"] = stream_extents.is_transposed[arg_id]
            arg["rect_offset"] = list(next(iter(stream_extents.offsets[arg_id])))
        else:
            arg["rect_offset_used"] = [0, 0]
            arg["column_major"] = False
            arg["rect_offset"] = [0, 0]

    metadata = {
        "kernel_name": kernel.name,
        "inputs": input_args,
        "outputs": output_args,
        "argument_order": argnames,
        "memcpy_mode": using_memcpy_mode,
        "fabric_dims": [xend - xbegin, yend - ybegin],
        "kernel_dims": kernel_dims,
        "fabric_offsets": [offset_x + xbegin, offset_y + ybegin],
    }
    serialization.save_to_json(metadata, os.path.join(output_folder, 'metadata.json'))

    if generate_only:
        print("Generated output files without compiling.")
        return

    if 'CM_ADDR' in os.environ:
        fabric_width, fabric_height = csl.HARDWARE_FABRIC_DIMS
    else:
        fabric_width, fabric_height = xend, yend

    cslc_command = [
        os.getenv('CSLC', 'cslc'), f'--arch={csl.ARCH}', 'layout.csl', f'--fabric-dims={fabric_width},{fabric_height}',
        f'--fabric-offsets={offset_x + xbegin},{offset_y + ybegin}', '--memcpy', f'--channels={memcpy_channels}'
    ]
    print("Compiling with command:", ' '.join(cslc_command))
    try:
        subprocess.run(cslc_command, cwd=output_folder, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\033[91mCompilation failed with error:\033[0m {e}")
        exit(e.returncode)

    print("\033[92mCompilation successful.\033[0m "
          "To run the program, use the Cerebras SDK python runtime with npy files as arguments:")
    runtime_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runtime", "runtime.py"))
    args_str = ' '.join(f"{arg}.npy" for arg in metadata["argument_order"] if arg in input_args)
    cs_python = os.getenv('CS_PYTHON', 'cs_python')
    print(f"{cs_python} {runtime_path} {output_folder} {args_str}")


if __name__ == '__main__':
    compile_spatial_ir()
