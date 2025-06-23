from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np

from spatialstencil.syntax.common.serialization import load_from_json
from spatialstencil.syntax.spatial_ir import irnodes as spa


@dataclass
class ProgramMetadata:
    """Metadata for a compiled program."""
    kernel_name: str
    inputs: Dict[str, Union[spa.ArrayType, spa.ScalarType, spa.StreamType]]
    outputs: Dict[str, Union[spa.ArrayType, spa.ScalarType, spa.StreamType]]
    argument_order: List[str]


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

        self.metadata = load_from_json(ProgramMetadata, metadata_path)

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

            # TODO: Use flatten_copy
            # Convert to numpy array if needed
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            # Validate shape if specified in metadata
            if "shape" in self.inputs[name]:
                expected_shape = tuple(self.inputs[name]["shape"])
                if data.shape != expected_shape:
                    raise ValueError(f"Input {name} has wrong shape. Expected {expected_shape}, got {data.shape}")

            # Copy data to device
            self.runtime.memcpy_h2d(name, data, ...)

        # Run the program
        func_name = self.metadata.get("function_name", "main")
        self.runtime.launch(func_name, nonblock=False)

        # Copy outputs back from device
        results = {}
        for output_name, output_info in self.outputs.items():
            # Get output shape from metadata
            shape = tuple(output_info.get("shape", []))
            total_size = np.prod(shape)
            dtype = output_info.get("dtype", "float32")

            # TODO: Use copy_unflatten
            # Allocate buffer for output
            output_data = np.zeros(shape, dtype=dtype)

            # Copy data from device
            self.runtime.memcpy_d2h(output_data, output_name, ...)
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
