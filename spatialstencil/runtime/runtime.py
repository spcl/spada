import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spatialstencil.runtime import cerebras_runtime_stub as crt
else:
    try:
        from cerebras.sdk.runtime import sdkruntimepybind as crt
    except (ImportError, ModuleNotFoundError):
        raise ImportError("Cerebras SDK not found. Please install the Cerebras SDK or use `cs_python` to "
                          "execute this script.")

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
    def from_json(cls, json_data: Union[str, Dict[str, Any]]) -> 'ProgramMetadata':
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
            inputs={
                k: ArrayType(**v) for k, v in json_data.get("inputs", {}).items()
            },
            outputs={
                k: ArrayType(**v) for k, v in json_data.get("outputs", {}).items()
            },
            argument_order=json_data.get("argument_order", []),
            memcpy_mode=json_data.get("memcpy_mode", False),
            kernel_dims=json_data.get("kernel_dims", []),
            fabric_dims=json_data.get("fabric_dims", []),
            fabric_offsets=json_data.get("fabric_offsets", []))


########################################################
# Copying and Flattening Utilities
########################################################


def flatten_copy(name: str, data: np.ndarray, shape: List[int], runtime: crt.SdkRuntime, metadata: ProgramMetadata):
    """
    Copy data to the device, flattening it if necessary.
    This function assumes that the runtime has a method `memcpy_h2d` for copying.

    :param name: Name of the tensor in the device memory
    :param data: Numpy array to copy
    :param shape: Shape of the data to be copied
    :param runtime: The Cerebras SDK runtime object to perform the copy operation
    :param metadata: Program metadata containing input/output information
    """
    buffer_id = runtime.get_id(name)
    if buffer_id is None:
        raise ValueError(f"Buffer ID for '{name}' not found in program.")

    runtime.memcpy_h2d(
        buffer_id,
        data.ravel(),
        metadata.inputs[name].rect_offset_used[0],  # PE offset in x direction
        metadata.inputs[name].rect_offset_used[1],  # PE offset in y direction
        shape[0],  # Width is the second dimension
        shape[1],  # Height is the first dimension
        shape[2],
        streaming=not metadata.memcpy_mode,  # Use streaming if not in memcpy mode
        data_type=crt.MemcpyDataType.MEMCPY_32BIT if data.dtype == np.float32 else crt.MemcpyDataType.MEMCPY_16BIT,
        order=crt.MemcpyOrder.ROW_MAJOR if not metadata.inputs[name].column_major else crt.MemcpyOrder.COL_MAJOR,
        nonblock=True,  # Non-blocking copy
    )


def copy_unflatten(name: str, data: np.ndarray, shape: List[int], runtime: crt.SdkRuntime, metadata: ProgramMetadata):
    """
    Copy data from the device, unflattening it if necessary.
    This function assumes that the runtime has a method `memcpy_d2h` for copying.

    :param name: Name of the tensor in the device memory
    :param data: Numpy array to copy
    :param shape: Shape of the data to be copied
    :param runtime: The Cerebras SDK runtime object to perform the copy operation
    :param metadata: Program metadata containing input/output information
    """
    buffer_id = runtime.get_id(name)
    if buffer_id is None:
        raise ValueError(f"Buffer ID for '{name}' not found in program.")

    runtime.memcpy_d2h(
        data.ravel(),
        buffer_id,
        metadata.outputs[name].rect_offset_used[0],  # PE offset in x direction
        metadata.outputs[name].rect_offset_used[1],  # PE offset in y direction
        shape[0],  # Width is the second dimension
        shape[1],  # Height is the first dimension
        shape[2],
        streaming=not metadata.memcpy_mode,  # Use streaming if not in memcpy mode
        data_type=crt.MemcpyDataType.MEMCPY_32BIT if data.dtype == np.float32 else crt.MemcpyDataType.MEMCPY_16BIT,
        order=crt.MemcpyOrder.ROW_MAJOR if not metadata.outputs[name].column_major else crt.MemcpyOrder.COL_MAJOR,
        nonblock=False,  # Blocking copy to ensure data is ready after copy
    )


def copy_back_benchmark_data(runtime: crt.SdkRuntime, metadata: ProgramMetadata) -> np.ndarray:
    """
    Copy back benchmarking data from the device.
    
    :param runtime: The Cerebras SDK runtime object to perform the copy operation
    :param metadata: Program metadata containing input/output information
    :return: Numpy array containing cycle counts
    """
    cycle_start = np.zeros(metadata.kernel_dims + [3], dtype=np.uint32)
    cycle_stop = np.zeros(metadata.kernel_dims + [3], dtype=np.uint32)
    cycle_counts = np.zeros(metadata.kernel_dims, dtype=np.uint64)
    runtime.memcpy_d2h(
        cycle_start.ravel(),
        runtime.get_id("__benchmark_start"),
        0,
        0,
        *cycle_start.shape,
        streaming=False,
        data_type=crt.MemcpyDataType.MEMCPY_16BIT,
        order=crt.MemcpyOrder.ROW_MAJOR,
        nonblock=False)
    runtime.memcpy_d2h(
        cycle_stop.ravel(),
        runtime.get_id("__benchmark_stop"),
        0,
        0,
        *cycle_stop.shape,
        streaming=False,
        data_type=crt.MemcpyDataType.MEMCPY_16BIT,
        order=crt.MemcpyOrder.ROW_MAJOR,
        nonblock=False)
    # Convert 3x16-bit timestamp to the 48-bit little endian integer
    cycle_start = (
        cycle_start[:, :, 0].astype(np.uint64) | (cycle_start[:, :, 1].astype(np.uint64) << 16) |
        (cycle_start[:, :, 2].astype(np.uint64) << 32))
    cycle_stop = (
        cycle_stop[:, :, 0].astype(np.uint64) | (cycle_stop[:, :, 1].astype(np.uint64) << 16) |
        (cycle_stop[:, :, 2].astype(np.uint64) << 32))
    cycle_counts = cycle_stop - cycle_start
    return cycle_counts


########################################################
# Program Class
########################################################


class Program:
    """A program that can be run on a device."""

    def __init__(self, folder: str, benchmark: bool = False):
        """
        Initialize the Program with a folder containing the compiled program.

        :param folder: Path to the folder containing the program files
        :param benchmark: Whether to run in benchmark mode (not implemented)
        """
        self.folder = Path(folder)
        self.out_folder = self.folder / "out"
        self.benchmark = benchmark

        # Load metadata
        metadata_path = self.folder / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.metadata = ProgramMetadata.from_json(metadata)

        # Initialize SDK runtime
        cmaddr = os.environ.get('CM_ADDR', None)
        self.runtime = crt.SdkRuntime(str(self.out_folder), suppress_simfab_trace=True, cmaddr=cmaddr)

        # Store input/output information from metadata
        self.inputs = self.metadata.inputs
        self.outputs = self.metadata.outputs

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

        try:
            print("Loading program...", flush=True, end='')
            self.runtime.load()
            self.runtime.run()
            print("done.", flush=True)

            if self.benchmark:
                if self.runtime.get_id("f_tic") is None or self.runtime.get_id("f_toc") is None:
                    raise ValueError("Benchmarking requested but not enabled in the program.")

            if not self.metadata.memcpy_mode and self.benchmark:
                self.runtime.launch("f_tic", nonblock=False)

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
                flatten_copy(name, data, expected_shape, self.runtime, self.metadata)

            # Run the program
            if self.metadata.memcpy_mode:
                print("Launching kernel...", flush=True, end='')
                if self.benchmark:
                    self.runtime.launch("f_tic", nonblock=False)
                self.runtime.launch(self.metadata.kernel_name, nonblock=False)
                if self.benchmark:
                    self.runtime.launch("f_toc", nonblock=False)
                print("kernel launched.", flush=True)

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

            if self.benchmark:
                cycle_counts = copy_back_benchmark_data(self.runtime, self.metadata)
                np.save("perf_cycles.npy", cycle_counts)
                # Print min, max, median cycle counts in a more readable format
                print(f"Cycle count stats:\n"
                      f"  Min:    {np.min(cycle_counts):,}\n"
                      f"  Max:    {np.max(cycle_counts):,}\n"
                      f"  Median: {np.median(cycle_counts).astype(np.uint64):,}")

            print("Stopping runtime...", flush=True, end='')
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

    args = parser.parse_args()

    # Load the program
    program = Program(args.program_folder, args.benchmark)

    # Load input arrays from .npy files
    inputs = []
    if args.randomize:
        for name, info in program.inputs.items():
            shape = info.shape + [info.buffer_size or 1]
            dtype = dtype_to_numpy.get(info.dtype, np.float32)
            print(f"Randomizing input {name} with shape {shape} and dtype {dtype}")
            data = np.random.rand(*shape).astype(dtype)
            inputs.append(data)
    else:
        for input_file in args.input_files:
            data = np.load(input_file)
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
        np.save(output_file, output)
        print(f"Output saved to {output_file}")
