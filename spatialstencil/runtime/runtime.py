from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np

########################################################
# Serialization and Type Definitions
########################################################


@dataclass
class ArrayType:
    """Type for array arguments."""
    shape: List[int]
    dtype: str  # One of f32, f16, i32, u32, etc.


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
            argument_order=json_data.get("argument_order", []))


########################################################
# Copying and Flattening Utilities
########################################################


def flatten_copy(name: str, data: np.ndarray, shape: List[int], runtime):
    """
    Copy data to the device, flattening it if necessary.
    This function assumes that the runtime has a method `memcpy_h2d` for copying.

    :param name: Name of the tensor in the device memory
    :param data: Numpy array to copy
    :param shape: Shape of the data to be copied
    :param runtime: The Cerebras SDK runtime object to perform the copy operation
    """
    # runtime.memcpy_h2d(name, data, ...)
    pass


def copy_unflatten(name: str, data: np.ndarray, shape: List[int], runtime):
    """
    Copy data from the device, unflattening it if necessary.
    This function assumes that the runtime has a method `memcpy_d2h` for copying.

    :param name: Name of the tensor in the device memory
    :param data: Numpy array to copy
    :param shape: Shape of the data to be copied
    :param runtime: The Cerebras SDK runtime object to perform the copy operation
    """
    # runtime.memcpy_d2h(name, data, ...)
    pass


########################################################
# Program Class
########################################################


class Program:
    """A program that can be run on a device."""

    def __init__(self, folder: str):
        """
        Initialize the Program with a folder containing the compiled program.
        
        :param folder: Path to the folder containing the program files
        """
        self.folder = Path(folder)
        self.out_folder = self.folder / "out"

        # Load metadata
        metadata_path = self.folder / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.metadata = ProgramMetadata.from_json(metadata)

        # Initialize SDK runtime
        try:
            from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
        except (ImportError, ModuleNotFoundError):
            raise ImportError("Cerebras SDK not found. Please install the Cerebras SDK or use `cs_python` to "
                              "execute this script.")
        self.runtime = SdkRuntime(str(self.out_folder))

        # Store input/output information from metadata
        self.inputs = self.metadata.get("inputs", {})
        self.outputs = self.metadata.get("outputs", {})

    def __call__(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Run the program with the provided arguments.
        
        :param args: Positional arguments for the program
        :param kwargs: Keyword arguments for the program
        :return: Dictionary of output tensors
        """
        from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder  # pylint: disable=no-name-in-module

        # Use argument_order from metadata if available
        if self.metadata.argument_order and len(args) == len(self.metadata.argument_order):
            if len(kwargs) > 0:
                raise ValueError("Cannot provide both positional and keyword arguments.")
            kwargs = {name: value for name, value in zip(self.metadata.argument_order, args)}
        if len(args) + len(kwargs) < len(self.metadata.argument_order):
            raise ValueError(f"Expected {len(self.metadata.argument_order)} arguments, got {len(args) + len(kwargs)}")

        # Validate inputs
        for input_name in self.inputs:
            if input_name not in kwargs:
                raise ValueError(f"Missing required input: {input_name}")

        self.runtime.load()
        self.runtime.run()

        # Copy data to device
        for name, data in kwargs.items():
            if name not in self.inputs:
                raise ValueError(f"Unexpected input: {name}")

            # Convert to numpy array if needed
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=dtype_to_numpy[self.inputs[name]["dtype"]])

            # Validate shape if specified in metadata
            if "shape" in self.inputs[name]:
                expected_shape = tuple(self.inputs[name]["shape"])
                if data.shape != expected_shape:
                    raise ValueError(f"Input {name} has wrong shape. Expected {expected_shape}, got {data.shape}")

            # Use flatten_copy to copy data to device
            shape = self.inputs[name].shape if isinstance(self.inputs[name], ArrayType) else []
            if not shape:
                shape = list(data.shape)

            # Copy data to device
            flatten_copy(name, data, shape, self.runtime)

        # Run the program
        func_name = self.metadata.get("function_name", "main")
        self.runtime.launch(func_name, nonblock=False)

        # Copy outputs back from device
        results = {}
        for output_name, output_info in self.outputs.items():
            # Get output shape from metadata
            shape = tuple(output_info.get("shape", []))
            dtype = dtype_to_numpy.get(output_info["dtype"], np.float32)

            # Allocate buffer for output
            output_data = np.zeros(shape, dtype=dtype)

            # Copy data from device
            copy_unflatten(output_name, output_data, self.runtime)
            results[output_name] = output_data

        self.runtime.stop()

        return results


if __name__ == "__main__":
    # Example usage
    program = Program("bla")

    # Prepare input data
    a = np.random.rand(256, 256, 80).astype(np.float32)
    b = np.random.rand(256, 256, 80).astype(np.float32)

    # Run the program
    outputs = program(a, b)  # Or use keyword arguments: program(a=a, b=b)

    # Print output shapes
    for name, output in outputs.items():
        print(f"{name}: {output.shape}")
