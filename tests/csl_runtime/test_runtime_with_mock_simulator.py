import unittest
import numpy as np
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from enum import Enum
from spatialstencil.runtime.cerebras_runtime_stub import MemcpyDataType, MemcpyOrder

# Mock the Cerebras SDK module BEFORE any imports from spatialstencil
# This needs to be done at the very beginning to prevent ImportError

class MockSdkRuntime:
    """Mock implementation of the Cerebras SDK runtime for testing."""

    def __init__(self, bindir: str, **kwargs):
        self.bindir = bindir
        self.loaded = False
        self.running = False
        self.data_buffers = {}  # Store data by buffer ID
        self.buffer_names = {}  # Map names to buffer IDs
        self.next_buffer_id = 1
        self.mock_kernel_func = None
        self.input_data = {}
        self.output_data = {}

    def set_mock_kernel(self, kernel_func):
        """Set the mock kernel function to use."""
        self.mock_kernel_func = kernel_func

    def load(self):
        """Mock load operation."""
        self.loaded = True

    def run(self):
        """Mock run operation."""
        if not self.loaded:
            raise RuntimeError("Program not loaded")
        self.running = True

    def stop(self):
        """Mock stop operation."""
        self.running = False

    def get_id(self, symbol: str) -> int:
        """Get buffer ID for a symbol, creating one if it doesn't exist."""
        if symbol not in self.buffer_names:
            self.buffer_names[symbol] = self.next_buffer_id
            self.next_buffer_id += 1
        return self.buffer_names[symbol]

    def memcpy_h2d(self, dest: int, src: np.ndarray, px: int, py: int, w: int, h: int, elem_per_pe: int, *,
                   streaming: bool, data_type, order, nonblock: bool):
        """Mock host-to-device memory copy."""
        # Store the data in our mock buffer
        self.data_buffers[dest] = src.copy()
        # Also store by name for kernel execution
        for name, buffer_id in self.buffer_names.items():
            if buffer_id == dest:
                self.input_data[name] = src.copy()
                break

    def memcpy_d2h(self, dest: np.ndarray, src: int, px: int, py: int, w: int, h: int, elem_per_pe: int, *,
                   streaming: bool, data_type, order, nonblock: bool):
        """Mock device-to-host memory copy."""
        if src in self.data_buffers:
            dest[:] = self.data_buffers[src]
        else:
            # If no data in buffer, fill with zeros
            dest.fill(0)

    def launch(self, symbol: str, nonblock: bool = False):
        """Mock kernel launch - execute the mock kernel function."""
        if self.mock_kernel_func and len(self.input_data) >= 2:
            # Assume we have inputs 'a' and 'b' and output 'out'
            input_names = list(self.input_data.keys())
            if len(input_names) >= 2:
                a = self.input_data[input_names[0]]
                b = self.input_data[input_names[1]]

                # Create output array with same shape as input
                out = np.zeros_like(a)

                # Execute mock kernel
                self.mock_kernel_func(a, b, out)

                # Store result in output buffer
                out_buffer_id = self.get_id('out')
                self.data_buffers[out_buffer_id] = out


# Create the full mock module structure
mock_crt = MagicMock()
mock_crt.MemcpyDataType = MemcpyDataType
mock_crt.MemcpyOrder = MemcpyOrder
mock_crt.SdkRuntime = MockSdkRuntime

# Mock the cerebras module hierarchy
sys.modules['cerebras'] = MagicMock()
sys.modules['cerebras.sdk'] = MagicMock()
sys.modules['cerebras.sdk.runtime'] = MagicMock()
sys.modules['cerebras.sdk.runtime.sdkruntimepybind'] = mock_crt

# End of mocking the Cerebras SDK

# Now we can safely import the runtime classes
from spatialstencil.runtime.runtime import Program, ProgramMetadata


def mock_kernel(a, b, out):
    """Mock kernel function that adds two arrays."""
    out[:] = a + b


class TestProgramWithMockRuntime(unittest.TestCase):
    """Test the Program class with a mock runtime."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.program_dir = Path(self.temp_dir) / "test_program"
        self.out_dir = self.program_dir / "out"

        # Create directories
        self.program_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Create mock metadata
        self.metadata = {
            "kernel_name": "test_kernel",
            "inputs": {
                "a": {
                    "shape": [4, 4],
                    "dtype": "f32",
                    "buffer_size": 1
                },
                "b": {
                    "shape": [4, 4],
                    "dtype": "f32",
                    "buffer_size": 1
                }
            },
            "outputs": {
                "out": {
                    "shape": [4, 4],
                    "dtype": "f32",
                    "buffer_size": 1
                }
            },
            "argument_order": ["a", "b"],
            "memcpy_mode": True,
            "kernel_dims": [4, 4],
            "fabric_dims": [4, 4],
            "fabric_offsets": [0, 0]
        }

        # Write metadata to file
        metadata_path = self.program_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

        # Create mock runtime instance
        self.mock_runtime = MockSdkRuntime(str(self.out_dir))
        self.mock_runtime.set_mock_kernel(mock_kernel)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('spatialstencil.runtime.runtime.crt.SdkRuntime')
    def test_program_initialization(self, mock_sdk_runtime_class):
        """Test that Program initializes correctly with metadata."""
        mock_sdk_runtime_class.return_value = self.mock_runtime

        program = Program(str(self.program_dir))

        # Check metadata was loaded correctly
        self.assertEqual(program.metadata.kernel_name, "test_kernel")
        self.assertEqual(len(program.inputs), 2)
        self.assertEqual(len(program.outputs), 1)
        self.assertIn("a", program.inputs)
        self.assertIn("b", program.inputs)
        self.assertIn("out", program.outputs)

    def test_program_execution_with_positional_args(self):
        """Test program execution with positional arguments."""
        # Patch the SdkRuntime class directly in the runtime module
        with patch('spatialstencil.runtime.runtime.crt.SdkRuntime', return_value=self.mock_runtime):
            program = Program(str(self.program_dir))

            # Create test input data
            a = np.ones((4, 4, 1), dtype=np.float32) * 2.0
            b = np.ones((4, 4, 1), dtype=np.float32) * 3.0

            # Execute program
            results = program(a, b)

            # Check results
            self.assertIn("out", results)
            expected = a + b
            np.testing.assert_array_equal(results["out"], expected)

    def test_program_execution_with_keyword_args(self):
        """Test program execution with keyword arguments."""
        with patch('spatialstencil.runtime.runtime.crt.SdkRuntime', return_value=self.mock_runtime):
            program = Program(str(self.program_dir))

            # Create test input data
            a = np.ones((4, 4, 1), dtype=np.float32) * 5.0
            b = np.ones((4, 4, 1), dtype=np.float32) * 7.0

            # Execute program with keyword arguments
            results = program(a=a, b=b)

            # Check results
            self.assertIn("out", results)
            expected = a + b
            np.testing.assert_array_equal(results["out"], expected)

    def test_program_shape_validation(self):
        """Test that program validates input shapes correctly."""
        with patch('spatialstencil.runtime.runtime.crt.SdkRuntime', return_value=self.mock_runtime):
            program = Program(str(self.program_dir))

            # Create test input data with wrong shape
            a = np.ones((3, 3, 1), dtype=np.float32)  # Wrong shape
            b = np.ones((4, 4, 1), dtype=np.float32)

            # Should raise ValueError for wrong shape
            with self.assertRaises(ValueError):
                program(a, b)

    def test_program_missing_input(self):
        """Test that program raises error for missing inputs."""
        with patch('spatialstencil.runtime.runtime.crt.SdkRuntime', return_value=self.mock_runtime):
            program = Program(str(self.program_dir))

            # Create test input data - only provide one input
            a = np.ones((4, 4, 1), dtype=np.float32)

            # Should raise ValueError for missing input
            with self.assertRaises(ValueError):
                program(a=a)  # Missing 'b'

    def test_mock_kernel_execution_verification(self):
        """Test that our mock kernel is actually being executed with correct data."""
        with patch('spatialstencil.runtime.runtime.crt.SdkRuntime', return_value=self.mock_runtime):
            program = Program(str(self.program_dir))

            # Create specific test input data to verify kernel execution
            a_val = 10.0
            b_val = 5.0
            a = np.full((4, 4, 1), a_val, dtype=np.float32)
            b = np.full((4, 4, 1), b_val, dtype=np.float32)

            # Execute program
            results = program(a, b)

            # Verify the mock kernel performed the correct addition
            expected = np.full((4, 4, 1), a_val + b_val, dtype=np.float32)
            np.testing.assert_array_equal(results["out"], expected)

            # Verify the exact values
            self.assertEqual(results["out"][0, 0, 0], a_val + b_val)
            self.assertTrue(np.all(results["out"] == (a_val + b_val)))

    def test_program_unexpected_input(self):
        """Test that program raises error for unexpected inputs."""
        with patch('spatialstencil.runtime.runtime.crt.SdkRuntime', return_value=self.mock_runtime):
            program = Program(str(self.program_dir))

            # Create test input data
            a = np.ones((4, 4, 1), dtype=np.float32)
            b = np.ones((4, 4, 1), dtype=np.float32)
            c = np.ones((4, 4, 1), dtype=np.float32)  # Unexpected input

            # Should raise ValueError for unexpected input
            with self.assertRaises(ValueError):
                program(a=a, b=b, c=c)


class TestProgramMetadata(unittest.TestCase):
    """Test the ProgramMetadata class."""

    def test_metadata_from_json_dict(self):
        """Test creating ProgramMetadata from dictionary."""
        data = {
            "kernel_name": "test_kernel",
            "inputs": {
                "x": {
                    "shape": [10, 10],
                    "dtype": "f32"
                }
            },
            "outputs": {
                "y": {
                    "shape": [10, 10],
                    "dtype": "f32"
                }
            },
            "argument_order": ["x"],
            "memcpy_mode": True,
            "kernel_dims": [10, 10],
            "fabric_dims": [10, 10],
            "fabric_offsets": [0, 0]
        }

        metadata = ProgramMetadata.from_json(data)

        self.assertEqual(metadata.kernel_name, "test_kernel")
        self.assertEqual(len(metadata.inputs), 1)
        self.assertEqual(len(metadata.outputs), 1)
        self.assertTrue(metadata.memcpy_mode)

    def test_metadata_from_json_string(self):
        """Test creating ProgramMetadata from JSON string."""
        data = {
            "kernel_name": "test_kernel",
            "inputs": {
                "x": {
                    "shape": [5, 5],
                    "dtype": "f32"
                }
            },
            "outputs": {
                "y": {
                    "shape": [5, 5],
                    "dtype": "f32"
                }
            },
            "argument_order": ["x"],
            "memcpy_mode": False,
            "kernel_dims": [5, 5],
            "fabric_dims": [5, 5],
            "fabric_offsets": [0, 0]
        }

        json_string = json.dumps(data)
        metadata = ProgramMetadata.from_json(json_string)

        self.assertEqual(metadata.kernel_name, "test_kernel")
        self.assertFalse(metadata.memcpy_mode)


if __name__ == '__main__':
    unittest.main()
