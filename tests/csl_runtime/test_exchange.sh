#!/bin/sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Compile the spatial stencil program
FOLDER_NAME="xchg_sptl"
sptlc "$SCRIPT_DIR/../../samples/spatial/neighbor_exchange.sptl" "$FOLDER_NAME" -p K=2

python <<EOF
import numpy as np
a = np.random.rand(2, 1, 2).astype(np.float32)
np.save('a.npy', a)
EOF

# Run the compiled program with the Python runtime and the simulator
timeout -s 9 120 cs_python "$SCRIPT_DIR/../../spatialstencil/runtime/runtime.py" "$FOLDER_NAME" a.npy --benchmark

# Check if the output file matches the expected output
python <<EOF
import numpy as np
# Load the arrays
a = np.load('a.npy')
# Build expected result: exchange a[0] with a[1]
ref = a.copy()
ref[0, ...], ref[1, ...] = a[1, ...], a[0, ...]
output = np.load('OUT_out.npy')
# Check if the output is correct
if not np.allclose(output, ref):
    print("Test failed: Output does not match expected result.")
    print("Expected:")
    print(ref)
    print("Got:")
    print(output)
    exit(1)
else:
    print("Test passed: Output matches expected result.")
EOF

# Clean up generated files
rm -rf "$FOLDER_NAME"
rm -f a.npy OUT_out.npy
