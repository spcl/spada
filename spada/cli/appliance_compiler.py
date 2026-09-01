"""
Appliance-mode Spatial IR compiler (``sptlc-appliance``).

Unlike :mod:`spada.cli.compiler`, which invokes ``cslc`` locally inside the containerized SDK, this
frontend dispatches the CSL compilation to a worker node on a Cerebras Wafer-Scale Cluster through
``cerebras.sdk.client.SdkCompiler``. See https://sdk.cerebras.ai/appliance-mode for details.

The generated CSL sources and ``metadata.json`` are identical to the ones produced by ``sptlc``; the
only additional output is ``artifact_path.json``, which records the compiled artifact returned by
the appliance so that :mod:`spada.runtime.appliance_launcher` (``spada-wse-launcher``) can run it.

This module is intentionally *not* imported anywhere else in the package, so that ``cerebras_sdk``
and ``cerebras_appliance`` remain optional dependencies of SpaDA.
"""
import click
import json
import os
from typing import Any, Optional

from spada.cli.compiler import codegen_options, cslc_arguments, generate_program

#: Name of the file recording the appliance artifact, written into the output folder.
ARTIFACT_INFO_FILE = 'artifact_path.json'

#: Directory (relative to the artifact root) that ``cslc`` writes the compiled ELFs into. It must
#: stay in sync with ``spada.runtime.runtime.Program``, which looks for ``<folder>/out``.
ARTIFACT_OUTPUT_DIR = 'out'


def _import_sdk_compiler():
    """Import ``SdkCompiler`` lazily, with an actionable error message if the SDK is missing."""
    try:
        from cerebras.sdk.client import SdkCompiler
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "The Cerebras appliance client was not found. `sptlc-appliance` requires the appliance "
            "packages (cerebras_sdk and cerebras_appliance) to be installed in the current "
            "environment. Use plain `sptlc` for containerized (non-appliance) compilation.") from e
    return SdkCompiler


def compile_on_appliance(program, hardware_fabric: Optional[bool] = None, mgmt_namespace: Optional[str] = None,
                         resource_cpu: Optional[int] = None, resource_mem: Optional[int] = None,
                         disable_version_check: bool = True) -> str:
    """
    Compile an already-generated program on the appliance and return the resulting artifact path.

    :param program: The :class:`~spada.cli.compiler.GeneratedProgram` to compile.
    :param hardware_fabric: Compile for the full hardware fabric rather than the tight kernel
                            rectangle. Defaults to whether ``CM_ADDR`` is set in the environment.
    :param mgmt_namespace: Appliance cluster namespace (defaults to the cluster's default).
    :param resource_cpu: CPU cores for the compile job, in units of 1/1000.
    :param resource_mem: Memory for the compile job, in bytes.
    :param disable_version_check: Ignore version differences between client and appliance.
    :return: The path of the compile artifact on the appliance.
    """
    SdkCompiler = _import_sdk_compiler()

    compile_args = ' '.join(cslc_arguments(program, hardware_fabric=hardware_fabric) +
                            ['-o', ARTIFACT_OUTPUT_DIR])

    kwargs: dict[str, Any] = {'disable_version_check': disable_version_check}
    if mgmt_namespace is not None:
        kwargs['mgmt_namespace'] = mgmt_namespace
    if resource_cpu is not None:
        kwargs['resource_cpu'] = resource_cpu
    if resource_mem is not None:
        kwargs['resource_mem'] = resource_mem

    print("Compiling on the appliance with arguments:", compile_args)
    with SdkCompiler(**kwargs) as compiler:
        artifact_path = compiler.compile(program.output_folder, 'layout.csl', compile_args, program.output_folder)

    return artifact_path


@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_folder', type=click.Path(dir_okay=True))
@codegen_options
@click.option('--generate-only', '-g', is_flag=True,
              help='Only generate the output files without compiling them on the appliance')
@click.option('--simulator', 'simulator', is_flag=True,
              help='Target the appliance simulator: compile for the tight kernel rectangle instead '
                   'of the full hardware fabric, and default the launcher to simulator mode')
@click.option('--mgmt-namespace', default=None, type=str, help='Appliance cluster namespace')
@click.option('--resource-cpu', default=None, type=int, help='CPU cores for the compile job (units of 1/1000)')
@click.option('--resource-mem', default=None, type=int, help='Memory for the compile job, in bytes')
@click.option('--check-version/--disable-version-check', 'check_version', default=False,
              help='Whether to enforce a matching client/appliance SDK version (default: do not)')
def compile_spatial_ir_appliance(input_file: str, output_folder: str, param: list[str], offset_x: int, offset_y: int,
                                 generate_only: bool, simulator: bool, mgmt_namespace: Optional[str],
                                 resource_cpu: Optional[int], resource_mem: Optional[int], check_version: bool,
                                 disable_benchmarking: bool, disable_asynchronous: bool, disable_dsd: bool,
                                 disable_mac_vectorization: bool,
                                 disable_map: bool, disable_task_fusion: bool, disable_task_recycling: bool,
                                 disable_copy_elision: bool, disable_close_elision: bool, disable_switching: bool):
    program = generate_program(
        input_file,
        output_folder,
        param=param,
        offset_x=offset_x,
        offset_y=offset_y,
        disable_benchmarking=disable_benchmarking,
        disable_asynchronous=disable_asynchronous,
        disable_dsd=disable_dsd,
        disable_mac_vectorization=disable_mac_vectorization,
        disable_map=disable_map,
        disable_task_fusion=disable_task_fusion,
        disable_task_recycling=disable_task_recycling,
        disable_copy_elision=disable_copy_elision,
        disable_close_elision=disable_close_elision,
        disable_switching=disable_switching,
    )

    if generate_only:
        print("Generated output files without compiling.")
        return

    # On real hardware the program must be compiled for the whole fabric; only the simulator can use
    # the tight kernel rectangle.
    hardware_fabric = not simulator

    try:
        artifact_path = compile_on_appliance(
            program,
            hardware_fabric=hardware_fabric,
            mgmt_namespace=mgmt_namespace,
            resource_cpu=resource_cpu,
            resource_mem=resource_mem,
            disable_version_check=not check_version,
        )
    except ImportError as e:
        print(f"\033[91m{e}\033[0m")
        exit(1)
    except Exception as e:
        print(f"\033[91mAppliance compilation failed with error:\033[0m {e}")
        exit(1)

    artifact_info = {
        'artifact_path': artifact_path,
        'simulator': simulator,
        'kernel_name': program.metadata['kernel_name'],
        'mgmt_namespace': mgmt_namespace,
    }
    info_path = os.path.join(output_folder, ARTIFACT_INFO_FILE)
    with open(info_path, 'w', encoding='utf8') as f:
        json.dump(artifact_info, f, indent=2)

    print(f"\033[92mAppliance compilation successful.\033[0m Artifact: {artifact_path}")
    print(f"Artifact recorded in {info_path}.")
    print("To run the program, use the SpaDA appliance launcher with npy files as arguments:")
    args_str = ' '.join(f"{arg}.npy" for arg in program.metadata["argument_order"] if arg in program.input_args)
    simulator_flag = ' --simulator' if simulator else ''
    print(f"spada-wse-launcher {output_folder} {args_str}{simulator_flag}")


if __name__ == '__main__':
    compile_spatial_ir_appliance()
