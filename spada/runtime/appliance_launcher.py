"""
Appliance-mode equivalent of :mod:`spada.runtime.runtime` (``spada-wse-launcher``).

In appliance mode the host program cannot talk to the wafer directly: it has to run on a worker node
inside the Cerebras Wafer-Scale Cluster. ``cerebras.sdk.client.SdkLauncher`` does that -- it unpacks
the compile artifact produced by ``sptlc-appliance`` on a worker node, stages extra files next to it,
and runs shell commands there (see https://sdk.cerebras.ai/appliance-mode).

This module therefore does not reimplement the memcpy logic. It stages the regular SpaDA runtime
(``spada/runtime/runtime.py``), the program's ``metadata.json`` and the input ``.npy`` files onto the
worker node, runs

    cs_python runtime.py . <inputs...> --cm-addr %CMADDR%

there, and downloads the resulting ``OUT_*.npy`` files back to the local machine.

Like :mod:`spada.cli.appliance_compiler`, this module is never imported by the rest of SpaDA, so the
appliance packages (``cerebras_sdk``, ``cerebras_appliance``) stay optional dependencies.
"""
import click
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Name of the file written by ``sptlc-appliance`` that records the compile artifact.
ARTIFACT_INFO_FILE = "artifact_path.json"

#: Name the SpaDA runtime is staged under on the worker node.
REMOTE_RUNTIME_SCRIPT = "runtime.py"

#: Working directory of the staged program on the worker node. The compile artifact unpacks the
#: compiled ELFs into ``./out``, and ``metadata.json`` is staged next to it, which is exactly the
#: layout ``spada.runtime.runtime.Program`` expects.
REMOTE_PROGRAM_FOLDER = "."


def _import_sdk_launcher():
    """Import ``SdkLauncher`` lazily, with an actionable error message if the SDK is missing."""
    try:
        from cerebras.sdk.client import SdkLauncher
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "The Cerebras appliance client was not found. `spada-wse-launcher` requires the "
            "appliance packages (cerebras_sdk and cerebras_appliance) to be installed in the "
            "current environment. Use `cs_python spada/runtime/runtime.py` for containerized "
            "(non-appliance) execution.") from e
    return SdkLauncher


def local_runtime_script() -> str:
    """Absolute path of the SpaDA runtime script that gets staged onto the worker node."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "runtime.py"))


class ApplianceProgram:
    """
    A compiled SpaDA program that runs on a Cerebras Wafer-Scale Cluster through ``SdkLauncher``.

    :param program_folder: Folder produced by ``sptlc-appliance`` (holds ``metadata.json`` and
                           ``artifact_path.json``).
    :param artifact_path: Compile artifact to run. Defaults to the one recorded in
                          ``artifact_path.json``.
    :param simulator: Run on the appliance simulator. Defaults to the value recorded at compile time.
    :param output_dir: Local directory the outputs are downloaded into.
    :param cs_python: Name of the ``cs_python`` executable on the worker node.
    :param mgmt_namespace: Appliance cluster namespace (defaults to the cluster's default).
    :param resource_cpu: CPU cores for the launcher job, in units of 1/1000.
    :param resource_mem: Memory for the launcher job, in bytes.
    :param disable_version_check: Ignore version differences between client and appliance.
    """

    def __init__(
        self,
        program_folder: str,
        artifact_path: Optional[str] = None,
        simulator: Optional[bool] = None,
        output_dir: str = "",
        cs_python: Optional[str] = None,
        mgmt_namespace: Optional[str] = None,
        resource_cpu: Optional[int] = None,
        resource_mem: Optional[int] = None,
        disable_version_check: bool = True,
    ):
        self.folder = Path(program_folder)
        self.output_dir = Path(output_dir)
        self.cs_python = cs_python or "cs_python"
        self.disable_version_check = disable_version_check
        self.resource_cpu = resource_cpu
        self.resource_mem = resource_mem

        self.metadata_path = self.folder / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf8") as f:
            self.metadata: Dict[str, Any] = json.load(f)

        artifact_info = self._load_artifact_info()
        self.artifact_path = artifact_path or artifact_info.get("artifact_path")
        if not self.artifact_path:
            raise ValueError(
                f"No compile artifact given and none recorded in {self.folder / ARTIFACT_INFO_FILE}. "
                "Compile the program with `sptlc-appliance`, or pass --artifact-path explicitly.")
        if simulator is None:
            simulator = bool(artifact_info.get("simulator", False))
        self.simulator = simulator
        self.mgmt_namespace = mgmt_namespace if mgmt_namespace is not None else artifact_info.get("mgmt_namespace")

        self.inputs: Dict[str, Any] = self.metadata.get("inputs", {})
        self.outputs: Dict[str, Any] = self.metadata.get("outputs", {})

    def _load_artifact_info(self) -> Dict[str, Any]:
        info_path = self.folder / ARTIFACT_INFO_FILE
        if not info_path.exists():
            return {}
        with open(info_path, "r", encoding="utf8") as f:
            return json.load(f)

    def _launcher_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "simulator": self.simulator,
            "disable_version_check": self.disable_version_check,
        }
        if self.mgmt_namespace is not None:
            kwargs["mgmt_namespace"] = self.mgmt_namespace
        if self.resource_cpu is not None:
            kwargs["resource_cpu"] = self.resource_cpu
        if self.resource_mem is not None:
            kwargs["resource_mem"] = self.resource_mem
        return kwargs

    def expected_output_files(self, benchmark: bool = False, repetitions: int = 1) -> List[str]:
        """
        Names of the files ``spada.runtime.runtime`` writes on the worker node, which are the ones
        downloaded back after the run.
        """
        files = [f"OUT_{name}.npy" for name in self.outputs]
        if benchmark:
            if self.metadata.get("memcpy_mode", False):
                # runtime.py saves one file per repetition, zero-padded to the repetition count.
                num_digits = len(str(repetitions))
                files += [f"perf_cycles_{i:0{num_digits}d}.npy" for i in range(repetitions)]
            else:
                files.append("perf_cycles.npy")
        return files

    def runtime_command(self, input_files: Sequence[str] = (), benchmark: bool = False, randomize: bool = False,
                        repetitions: int = 1) -> str:
        """Build the ``cs_python`` command line that is executed on the worker node."""
        command = [self.cs_python, REMOTE_RUNTIME_SCRIPT, REMOTE_PROGRAM_FOLDER]
        command += [os.path.basename(f) for f in input_files]
        if benchmark:
            command.append("--benchmark")
        if randomize:
            command.append("--randomize")
        if repetitions != 1:
            command += ["--repetitions", str(repetitions)]
        # %CMADDR% is substituted by SdkLauncher with the address of the allocated system.
        command += ["--cm-addr", "%CMADDR%"]
        return " ".join(command)

    def run(
        self,
        input_files: Sequence[str] = (),
        benchmark: bool = False,
        randomize: bool = False,
        repetitions: int = 1,
        stage_files: Sequence[str] = (),
        download_files: Sequence[str] = (),
        download_simlog: bool = False,
        dry_run: bool = False,
    ) -> List[Path]:
        """
        Run the program on the appliance and download its outputs.

        :param input_files: Local ``.npy`` files to stage and pass to the runtime, in argument order.
        :param benchmark: Run in benchmark mode and fetch the cycle counts.
        :param randomize: Let the runtime generate random inputs instead of using ``input_files``.
        :param repetitions: Number of times to rerun the kernel.
        :param stage_files: Additional local files to stage next to the program.
        :param download_files: Additional worker-node files to download after the run.
        :param download_simlog: Also download ``sim.log`` (simulator runs only).
        :param dry_run: Print what would be done, without allocating an appliance job.
        :return: The list of downloaded local paths.
        """
        if not randomize:
            expected_inputs = [name for name in self.metadata.get("argument_order", []) if name in self.inputs]
            if len(input_files) != len(expected_inputs):
                raise ValueError(f"Expected {len(expected_inputs)} input files "
                                 f"({', '.join(expected_inputs)}), got {len(input_files)}")
            for input_file in input_files:
                if not os.path.exists(input_file):
                    raise FileNotFoundError(f"Input file not found: {input_file}")
        else:
            input_files = ()

        command = self.runtime_command(input_files, benchmark=benchmark, randomize=randomize, repetitions=repetitions)
        staged = [local_runtime_script(), str(self.metadata_path)] + [str(f) for f in input_files]
        staged += [str(f) for f in stage_files]
        wanted = list(self.expected_output_files(benchmark, repetitions)) + [str(f) for f in download_files]
        if download_simlog:
            wanted.append("sim.log")

        if dry_run:
            print(f"Artifact:  {self.artifact_path}")
            print(f"Simulator: {self.simulator}")
            print("Staging:   " + ", ".join(staged))
            print("Command:   " + command)
            print("Download:  " + ", ".join(wanted))
            return []

        os.makedirs(self.output_dir, exist_ok=True)

        SdkLauncher = _import_sdk_launcher()
        downloaded: List[Path] = []
        print(f"Allocating appliance job ({'simulator' if self.simulator else 'hardware'})...", flush=True)
        with SdkLauncher(self.artifact_path, **self._launcher_kwargs()) as launcher:
            for path in staged:
                launcher.stage(path)

            print(f"Running: {command}", flush=True)
            response = launcher.run(command)
            if response:
                print(response, flush=True)

            for name in wanted:
                out_path = self.output_dir / os.path.basename(name)
                try:
                    launcher.download_artifact(name, str(out_path))
                except Exception as e:  # Missing artifacts should not lose the ones that do exist
                    print(f"\033[93mWarning:\033[0m could not download '{name}': {e}")
                    continue
                downloaded.append(out_path)

        return downloaded


@click.command()
@click.argument('program_folder', type=click.Path(exists=True, file_okay=False))
@click.argument('input_files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('--simulator/--hardware', 'simulator', default=None,
              help='Run on the appliance simulator or on hardware (default: as compiled)')
@click.option('--artifact-path', default=None, type=str,
              help=f'Compile artifact to run (default: the one recorded in {ARTIFACT_INFO_FILE})')
@click.option('--benchmark', is_flag=True, help='Run in benchmark mode and fetch cycle counts')
@click.option('--randomize', is_flag=True, help='Randomize input data on the worker node instead of staging files')
@click.option('--repetitions', default=1, type=int, help='Number of repetitions to run')
@click.option('--output-dir', default='', type=click.Path(file_okay=False), help='Local directory for the results')
@click.option('--stage', 'stage_files', multiple=True, type=click.Path(exists=True),
              help='Additional local file to stage on the worker node (repeatable)')
@click.option('--download', 'download_files', multiple=True, type=str,
              help='Additional worker-node file to download after the run (repeatable)')
@click.option('--download-simlog', is_flag=True, help='Also download sim.log (simulator runs)')
@click.option('--cs-python', default=None, type=str, help='Name of the cs_python executable on the worker node')
@click.option('--mgmt-namespace', default=None, type=str, help='Appliance cluster namespace')
@click.option('--resource-cpu', default=None, type=int, help='CPU cores for the launcher job (units of 1/1000)')
@click.option('--resource-mem', default=None, type=int, help='Memory for the launcher job, in bytes')
@click.option('--check-version/--disable-version-check', 'check_version', default=False,
              help='Whether to enforce a matching client/appliance SDK version (default: do not)')
@click.option('--dry-run', is_flag=True, help='Print the staging plan and command without running anything')
def launch(program_folder: str, input_files: tuple, simulator: Optional[bool], artifact_path: Optional[str],
           benchmark: bool, randomize: bool, repetitions: int, output_dir: str, stage_files: tuple,
           download_files: tuple, download_simlog: bool, cs_python: Optional[str], mgmt_namespace: Optional[str],
           resource_cpu: Optional[int], resource_mem: Optional[int], check_version: bool, dry_run: bool):
    """Run a program compiled by `sptlc-appliance` on a Cerebras Wafer-Scale Cluster."""
    program = ApplianceProgram(
        program_folder,
        artifact_path=artifact_path,
        simulator=simulator,
        output_dir=output_dir,
        cs_python=cs_python,
        mgmt_namespace=mgmt_namespace,
        resource_cpu=resource_cpu,
        resource_mem=resource_mem,
        disable_version_check=not check_version,
    )

    try:
        downloaded = program.run(
            input_files=input_files,
            benchmark=benchmark,
            randomize=randomize,
            repetitions=repetitions,
            stage_files=stage_files,
            download_files=download_files,
            download_simlog=download_simlog,
            dry_run=dry_run,
        )
    except ImportError as e:
        print(f"\033[91m{e}\033[0m")
        exit(1)

    if dry_run:
        return

    if not downloaded:
        print("\033[93mRun finished, but no output files were downloaded.\033[0m")
        return

    print("\033[92mRun complete.\033[0m Results:")
    for path in downloaded:
        print(f"  {path}")


if __name__ == '__main__':
    launch()
