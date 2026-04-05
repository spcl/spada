#!/bin/sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOLDER_NAME="task_recycling_merge_sptl"

sptlc "$SCRIPT_DIR/samples/task_recycling_merge.sptl" "$FOLDER_NAME" --disable-task-fusion

python <<EOF
import numpy as np
input_data = np.arange(1.0, 15.0, dtype=np.float32).reshape(1, 1, 14)
np.save('input.npy', input_data)
EOF

timeout -s 9 120 cs_python "$SCRIPT_DIR/../../spatialstencil/runtime/runtime.py" "$FOLDER_NAME" input.npy --benchmark

python <<EOF
import numpy as np
input_data = np.load('input.npy')
expected = np.sum(input_data, axis=2, keepdims=True)
output = np.load('OUT_output.npy')
if not np.allclose(output, expected):
    print('Test failed: Output does not match expected result.')
    print('Expected:')
    print(expected)
    print('Got:')
    print(output)
    raise SystemExit(1)
print('Test passed: Output matches expected result.')
EOF

rm -rf "$FOLDER_NAME"
rm -f input.npy OUT_output.npy