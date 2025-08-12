import click
import json
import os
from spatialstencil.lowering import spatial_ir_to_csl as s2c
from spatialstencil.syntax.spatial_ir import parser, passes, analysis, irnodes as spa
from spatialstencil.syntax.common import serialization
import subprocess


@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_folder', type=click.Path(dir_okay=True))
@click.option('--param', '-p', multiple=True, help='Kernel parameters in key=value format')
@click.option('--offset-x', '-x', default=0, type=int, help='Offset for rectangular region in x direction')
@click.option('--offset-y', '-y', default=0, type=int, help='Offset for rectangular region in y direction')
@click.option('--generate-only', '-g', is_flag=True, help='Only generate the output files without compiling them')
def compile_spatial_ir(input_file: str, output_folder: str, param: list[str], offset_x: int, offset_y: int,
                       generate_only: bool):
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
    print("Concretizing parameters:", kernel_parameters)
    kernel = passes.concretize_parameters(kernel, **kernel_parameters)
    kernel = passes.constexpr_propagation(kernel)

    # Argument checks
    using_memcpy_mode = None
    for arg in kernel.arguments:
        # Change all shapes to be lists of integers
        if isinstance(arg.dtype, spa.ArrayType):
            arg.dtype.shape = [dim.eval() if isinstance(dim, spa.Expression) else int(dim) for dim in arg.dtype.shape]
        # Check if the argument is a stream and has a memcpy mode
        if isinstance(arg.dtype, spa.ArrayType) and isinstance(arg.dtype.base_type, spa.StreamType):
            # Ensure all stream arguments are readonly or writeonly
            if not (arg.readonly or arg.writeonly):
                raise ValueError(f"Argument '{arg.identifier.name}' must be either readonly or writeonly, "
                                 f"but it is neither. Please check the kernel definition.")

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

    # Lower the spatial IR to CSL
    csl_files = s2c.lower_spatial_ir_to_csl(kernel)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    for f in csl_files:
        output_path = os.path.join(output_folder, f.filename)
        with open(output_path, 'w') as out_file:
            out_file.write(f.code)

    # Compile the generated CSL files using the cslc command (and change the cwd to the output folder)
    # Get the fabric dimensions from the kernel and offsets from the command line arguments
    # Command: cslc layout.csl --fabric-dims=16,16 --fabric-offsets=0,0 --memcpy --channels=1
    xbegin, xend, ybegin, yend = kernel.get_grid_rect()
    kernel_dims = [xend - xbegin, yend - ybegin]
    memcpy_channels = 1  # TODO: Determine the number of memcpy channels (1-16) based on the kernel arguments
    if memcpy_channels >= 0:
        xbegin += 4
        xend += 4 * 3
        ybegin += 1
        yend += 1 * 3

    # Generate metadata.json file
    input_args, output_args = analysis.get_kernel_stream_arguments(kernel)
    metadata = {
        "kernel_name": kernel.name,
        "inputs": input_args,
        "outputs": output_args,
        "argument_order": [a.identifier.name for a in kernel.arguments],
        "memcpy_mode": using_memcpy_mode,
        "fabric_dims": [xend - xbegin, yend - ybegin],
        "kernel_dims": kernel_dims,
        "fabric_offsets": [offset_x + xbegin, offset_y + ybegin],
    }
    serialization.save_to_json(metadata, os.path.join(output_folder, 'metadata.json'))

    if generate_only:
        print("Generated output files without compiling.")
        return

    cslc_command = [
        'cslc', 'layout.csl', f'--fabric-dims={xend - xbegin},{yend - ybegin}',
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
    print(f"cs_python {runtime_path} {output_folder} {args_str}")


if __name__ == '__main__':
    compile_spatial_ir()
