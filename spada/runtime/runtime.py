import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple, TYPE_CHECKING, Optional
import numpy as np
import numpy.typing as npt
import time

if TYPE_CHECKING:
    from spada.runtime import cerebras_runtime_stub as crt
else:
    try:
        from cerebras.sdk.runtime import sdkruntimepybind as crt
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Cerebras SDK not found. Please install the Cerebras SDK or use `cs_python` to " "execute this script."
        )

########################################################
# Serialization and Type Definitions
########################################################


@dataclass
class ArrayType:
    """Type for array arguments."""

    shape: List[int]
    dtype: str  # One of f32, f16, i32, u32, etc.
    buffer_size: Union[int, None] = None  # Optional buffer size for streams
    rect_offset: List[int] = field(default_factory=lambda: [0, 0])  # Optional rectangle offset for streams
    # Actual rectangle offset used in PEs. Subtract from rect_offset to get the offset within the buffer to copy
    rect_offset_used: List[int] = field(default_factory=lambda: [0, 0])
    column_major: bool = False  # Whether the array should be copied in column-major order


dtype_to_numpy = {
    "i8": np.int8,
    "u8": np.uint8,
    "i16": np.int16,
    "u16": np.uint16,
    "f16": np.float16,
    "i32": np.int32,
    "u32": np.uint32,
    "f32": np.float32,
    "f64": np.float64,
    "bool": np.bool_,
}


@dataclass
class ProgramMetadata:
    """Metadata for a compiled program."""

    kernel_name: str
    inputs: Dict[str, ArrayType]
    outputs: Dict[str, ArrayType]
    argument_order: List[str]
    memcpy_mode: bool
    kernel_dims: List[int]  # Dimensions of the kernel grid
    fabric_dims: List[int]  # Dimensions of the fabric (i.e., with memcpy extras)
    fabric_offsets: List[int]  # Offsets in the fabric for the kernel

    @classmethod
    def from_json(cls, json_data: Union[str, Dict[str, Any]]) -> "ProgramMetadata":
        """
        Create a ProgramMetadata instance from JSON data.

        :param json_data: JSON string or dictionary containing metadata
        :return: ProgramMetadata instance
        """
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON data: {e}")

        return cls(
            kernel_name=json_data.get("kernel_name", ""),
            inputs={k: ArrayType(**v) for k, v in json_data.get("inputs", {}).items()},
            outputs={k: ArrayType(**v) for k, v in json_data.get("outputs", {}).items()},
            argument_order=json_data.get("argument_order", []),
            memcpy_mode=json_data.get("memcpy_mode", False),
            kernel_dims=json_data.get("kernel_dims", []),
            fabric_dims=json_data.get("fabric_dims", []),
            fabric_offsets=json_data.get("fabric_offsets", []),
        )


########################################################
# Copying and Flattening Utilities
########################################################


def flatten_copy(
    name: str, data: np.ndarray, shape: List[int], runtime: crt.SdkRuntime, metadata: ProgramMetadata, benchmark: bool
):
    """
    Copy data to the device, flattening it if necessary.
    This function assumes that the runtime has a method `memcpy_h2d` for copying.

    :param name: Name of the tensor in the device memory
    :param data: Numpy array to copy
    :param shape: Shape of the data to be copied (w, h, elem_per_pe)
    :param runtime: The Cerebras SDK runtime object to perform the copy operation
    :param metadata: Program metadata containing input/output information
    """
    buffer_id = runtime.get_id(name)
    if buffer_id is None:
        raise ValueError(f"Buffer ID for '{name}' not found in program.")

    # The SDK expects A[h][w][elem_per_pe] but our arrays are (w, h, elem_per_pe).
    # Transpose the first two axes to match the SDK layout.
    src = np.ascontiguousarray(data.reshape(shape[0], shape[1], shape[2]).transpose(1, 0, 2))

    runtime.memcpy_h2d(
        buffer_id,
        src.ravel(),
        metadata.inputs[name].rect_offset_used[0],  # PE offset in x direction
        metadata.inputs[name].rect_offset_used[1],  # PE offset in y direction
        shape[0],  # Width (number of PEs in x)
        shape[1],  # Height (number of PEs in y)
        shape[2],
        streaming=not metadata.memcpy_mode,  # Use streaming if not in memcpy mode
        data_type=crt.MemcpyDataType.MEMCPY_32BIT if data.dtype == np.float32 else crt.MemcpyDataType.MEMCPY_16BIT,
        order=crt.MemcpyOrder.ROW_MAJOR if not metadata.inputs[name].column_major else crt.MemcpyOrder.COL_MAJOR,
        nonblock=not benchmark,  # Non-blocking copy if not benchmarking
    )


def copy_unflatten(name: str, data: np.ndarray, shape: List[int], runtime: crt.SdkRuntime, metadata: ProgramMetadata):
    """
    Copy data from the device, unflattening it if necessary.
    This function assumes that the runtime has a method `memcpy_d2h` for copying.

    :param name: Name of the tensor in the device memory
    :param data: Numpy array to copy
    :param shape: Shape of the data to be copied (w, h, elem_per_pe)
    :param runtime: The Cerebras SDK runtime object to perform the copy operation
    :param metadata: Program metadata containing input/output information
    """
    buffer_id = runtime.get_id(name)
    if buffer_id is None:
        raise ValueError(f"Buffer ID for '{name}' not found in program.")

    # The SDK returns A[h][w][elem_per_pe]; allocate a buffer in that layout.
    sdk_buf = np.empty((shape[1], shape[0], shape[2]), dtype=data.dtype)

    runtime.memcpy_d2h(
        sdk_buf.ravel(),
        buffer_id,
        metadata.outputs[name].rect_offset_used[0],  # PE offset in x direction
        metadata.outputs[name].rect_offset_used[1],  # PE offset in y direction
        shape[0],  # Width (number of PEs in x)
        shape[1],  # Height (number of PEs in y)
        shape[2],
        streaming=not metadata.memcpy_mode,  # Use streaming if not in memcpy mode
        data_type=crt.MemcpyDataType.MEMCPY_32BIT if data.dtype == np.float32 else crt.MemcpyDataType.MEMCPY_16BIT,
        order=crt.MemcpyOrder.ROW_MAJOR if not metadata.outputs[name].column_major else crt.MemcpyOrder.COL_MAJOR,
        nonblock=False,  # Blocking copy to ensure data is ready after copy
    )

    # Transpose back from (h, w, elem) to (w, h, elem) to match our convention.
    np.copyto(data, sdk_buf.transpose(1, 0, 2))


def convert_timestamp(hw_timestamp: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint64]:
    # Convert 3x16-bit timestamp to the 48-bit little endian integer
    return (
        hw_timestamp[:, :, 0].astype(np.uint64)
        | (hw_timestamp[:, :, 1].astype(np.uint64) << 16)
        | (hw_timestamp[:, :, 2].astype(np.uint64) << 32)
    ).astype(np.uint64)


def copy_back_benchmark_data(runtime: crt.SdkRuntime, metadata: ProgramMetadata) -> Tuple[np.ndarray, np.ndarray]:
    """
    Copy back benchmarking data from the device.

    :param runtime: The Cerebras SDK runtime object to perform the copy operation
    :param metadata: Program metadata containing input/output information
    :return: A tuple of Numpy arrays containing (cycles at start time, cycles at end time)
    """
    cycle_start = np.zeros(metadata.kernel_dims + [3], dtype=np.uint32)
    cycle_stop = np.zeros(metadata.kernel_dims + [3], dtype=np.uint32)
    runtime.memcpy_d2h(
        cycle_start.ravel(),
        runtime.get_id("__benchmark_start"),
        0,
        0,
        *cycle_start.shape,
        streaming=False,
        data_type=crt.MemcpyDataType.MEMCPY_16BIT,
        order=crt.MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    runtime.memcpy_d2h(
        cycle_stop.ravel(),
        runtime.get_id("__benchmark_stop"),
        0,
        0,
        *cycle_stop.shape,
        streaming=False,
        data_type=crt.MemcpyDataType.MEMCPY_16BIT,
        order=crt.MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    cycle_start = convert_timestamp(cycle_start)
    cycle_stop = convert_timestamp(cycle_stop)
    return cycle_start, cycle_stop


def copy_back_benchmark_cycles(runtime: crt.SdkRuntime, metadata: ProgramMetadata) -> np.ndarray:
    """
    Copy back benchmarking data from the device.

    :param runtime: The Cerebras SDK runtime object to perform the copy operation
    :param metadata: Program metadata containing input/output information
    :return: Numpy array containing cycle counts
    """
    cycle_start, cycle_stop = copy_back_benchmark_data(runtime, metadata)
    return cycle_stop - cycle_start


def print_cycle_counts(label: str, cycle_counts: np.ndarray) -> None:
    """
    Print benchmark data in a compact form for either scalar or per-PE cycle counts.
    """
    values = np.asarray(cycle_counts)
    if values.ndim == 0 or values.size == 1:
        print(f"{label} total time: {int(values.reshape(())):,}")
        return

    print(
        f"{label} stats:\n"
        f"  Min:    {np.min(values):,}\n"
        f"  Max:    {np.max(values):,}\n"
        f"  Median: {np.median(values).astype(np.uint64):,}"
    )


########################################################
# Program Class
########################################################


class Program:
    """A program that can be run on a device."""

    def __init__(
        self,
        folder: str,
        benchmark: bool = False,
        repetitions: int = 1,
        output_dir: str = "",
        cm_addr: Optional[str] = None,
    ):
        """
        Initialize the Program with a folder containing the compiled program.

        :param folder: Path to the folder containing the program files
        :param benchmark: Whether to run in benchmark mode (not implemented)
        :param repetitions: Number of times to rerun the program
        :param output_dir: Where to store all the results
        """
        self.folder = Path(folder)
        self.out_folder = self.folder / "out"
        self.benchmark = benchmark
        self.output_dir = Path(output_dir)
        self.repetitions = repetitions

        # Load metadata
        metadata_path = self.folder / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.metadata = ProgramMetadata.from_json(metadata)

        if not self.output_dir.exists():
            os.makedirs(self.output_dir, exist_ok=True)

        # Initialize SDK runtime
        cmaddr = cm_addr or os.environ.get("CM_ADDR", None)
        self.simulator = cmaddr is None
        print("SIMULATOR?", self.simulator)
        self.runtime = crt.SdkRuntime(str(self.out_folder), suppress_simfab_trace=True, cmaddr=cmaddr)

        # Store input/output information from metadata
        self.inputs = self.metadata.inputs
        self.outputs = self.metadata.outputs

    def has_symbol(self, symbol: str) -> bool:
        return self.runtime.get_id(symbol) is not None

    def has_basic_benchmarking(self) -> bool:
        return self.has_symbol("__benchmark_start") and self.has_symbol("__benchmark_stop")

    def __call__(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Run the program with the provided arguments.

        :param args: Positional arguments for the program
        :param kwargs: Keyword arguments for the program
        :return: Dictionary of output tensors
        """
        # Use argument_order from metadata if available
        if self.metadata.argument_order and len(args) == len(self.inputs):
            if len(kwargs) > 0:
                raise ValueError("Cannot provide both positional and keyword arguments.")
            kwargs = {name: value for name, value in zip(self.metadata.argument_order, args)}
        if len(args) + len(kwargs) < len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} arguments, got {len(args) + len(kwargs)}")

        # Validate inputs
        for input_name in self.inputs:
            if input_name not in kwargs:
                raise ValueError(f"Missing required input: {input_name}")

        # Separate scalar and array arguments
        scalar_kwargs = {k: v for k, v in kwargs.items() if k in self.inputs and len(self.inputs[k].shape) == 0}
        kwargs = {k: v for k, v in kwargs.items() if k not in scalar_kwargs}
        if scalar_kwargs and not self.metadata.argument_order:
            raise ValueError("Scalar arguments provided but no argument order specified.")
        scalar_args = [scalar_kwargs[name] for name in self.metadata.argument_order if name in scalar_kwargs]

        try:
            print("Loading program...", flush=True, end="")
            self.runtime.load()
            self.runtime.run()
            print("done.", flush=True)

            if self.benchmark and not self.has_basic_benchmarking():
                raise ValueError("Benchmarking requested but not enabled in the program.")

            # Copy data to device
            for name, data in kwargs.items():
                if name not in self.inputs:
                    raise ValueError(f"Unexpected input: {name}")

                # Convert to numpy array if needed
                if not isinstance(data, np.ndarray):
                    data = np.array(data, dtype=dtype_to_numpy[self.inputs[name].dtype])

                # Validate shape if specified in metadata
                expected_shape = tuple(self.inputs[name].shape + [self.inputs[name].buffer_size or 1])
                if data.shape != expected_shape:
                    raise ValueError(f"Input {name} has wrong shape. Expected {expected_shape}, got {data.shape}")

                # Use flatten_copy to copy data to device
                flatten_copy(name, data, expected_shape, self.runtime, self.metadata, self.benchmark)

            # Run the program
            for i in range(self.repetitions):
                if self.metadata.memcpy_mode:
                    if self.benchmark and not self.simulator and i == 0:
                        time.sleep(5.0)
                    print("Launching kernel...", flush=True, end="")
                    self.runtime.launch(self.metadata.kernel_name, *scalar_args, nonblock=False)
                    print("kernel launched.", flush=True)

                    if self.benchmark:
                        cycle_counts = copy_back_benchmark_cycles(self.runtime, self.metadata)
                        num_digits = len(str(self.repetitions))
                        np.save(self.output_dir / f"perf_cycles_{i:0{num_digits}d}.npy", cycle_counts)
                        print_cycle_counts(f"Iteration {i} cycle count", cycle_counts)

            # Copy outputs back from device
            results = {}
            for output_name, output_info in self.outputs.items():
                # Get output shape from metadata
                shape = output_info.shape + [output_info.buffer_size or 1]
                dtype = dtype_to_numpy.get(output_info.dtype, np.float32)

                # Allocate buffer for output
                output_data = np.empty(shape, dtype=dtype)

                # Copy data from device
                copy_unflatten(output_name, output_data, shape, self.runtime, self.metadata)
                results[output_name] = output_data

            if not self.metadata.memcpy_mode and self.benchmark:
                self.runtime.launch("f_toc", nonblock=False)

            print("Copy-back complete.", flush=True)

            if self.benchmark and not self.metadata.memcpy_mode:
                cycle_counts = copy_back_benchmark_data(self.runtime, self.metadata)
                np.save(self.output_dir / "perf_cycles.npy", cycle_counts)
                print_cycle_counts("Cycle count", cycle_counts)

            print("Stopping runtime...", flush=True, end="")
        finally:
            self.runtime.stop()
        print("done.", flush=True)

        return results


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a compiled program with numpy array inputs")
    parser.add_argument("program_folder", help="Path to the program folder")
    parser.add_argument("input_files", nargs="*", help="Input .npy files for the program")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode")
    parser.add_argument("--randomize", action="store_true", help="Randomize input data instead of loading from files")
    parser.add_argument("--repetitions", default=1, type=int, help="Number of repetitions to run")
    parser.add_argument("--output-dir", default="", help="Output directory for files")
    parser.add_argument("--cm-addr", default="", help="Cerebras machine address")

    args = parser.parse_args()

    # Load the program
    program = Program(args.program_folder, args.benchmark, args.repetitions, args.output_dir, args.cm_addr)

    # Load input arrays from .npy files
    inputs = []
    if args.randomize:
        for name, info in program.inputs.items():
            shape = info.shape + [info.buffer_size or 1]
            dtype = dtype_to_numpy.get(info.dtype, np.float32)
            print(f"Randomizing input {name} with shape {shape} and dtype {dtype}")
            data = np.random.rand(*shape).astype(dtype)
            if len(info.shape) == 0:  # Scalar input
                data = data.item()  # Convert single-element arrays to scalar
            inputs.append(data)
    else:
        for i, input_file in enumerate(args.input_files):
            data = np.load(input_file)
            if tuple(data.shape) == (1,) or data.ndim == 0:  # Scalar input
                argname = program.metadata.argument_order[i]
                if argname not in program.metadata.inputs:
                    raise ValueError(
                        f"Scalar file argument {input_file} given for {argname} not found in program inputs."
                    )
                arg = program.metadata.inputs[argname].shape
                if len(arg) > 0:
                    raise ValueError(f"Scalar file argument {input_file} given for {argname} which is not a scalar.")
                data = data.item()  # Convert single-element arrays to scalar
            else:  # Array input
                if len(data.shape) not in (2, 3):
                    raise ValueError(f"Input data from {input_file} must be 2D or 3D. Got shape {data.shape}.")
                if len(data.shape) == 2:
                    data = data.reshape((data.shape[0], data.shape[1], 1))  # Ensure at least 3 dimensions
            inputs.append(data)

    # Run the program with loaded inputs
    outputs = program(*inputs)

    # Save outputs to .npy files
    for name, output in outputs.items():
        output_file = f"OUT_{name}.npy"
        np.save(program.output_dir / output_file, output)
        print(f"Output saved to {output_file}")
