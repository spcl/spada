#!/bin/sh
# Compile-only check for multi-stage task-recycling samples.
# Verifies that lowering (including greedy task-slot coloring) succeeds for
# task_recycling_two_stage.sptl and task_recycling_three_stage.sptl without
# invoking the simulator or cs_python.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLES="$SCRIPT_DIR/samples"

compile_one() {
    name=$1
    src=$2
    out=$3
    rm -rf "$out"
    echo "Compiling: $name"
    sptlc "$src" "$out" --disable-task-fusion
    rm -rf "$out"
}

compile_one task_recycling_two_stage \
    "$SAMPLES/task_recycling_two_stage.sptl" \
    "$SCRIPT_DIR/_compile_check_two_stage_sptl"

compile_one task_recycling_three_stage \
    "$SAMPLES/task_recycling_three_stage.sptl" \
    "$SCRIPT_DIR/_compile_check_three_stage_sptl"

echo "Test passed: both multi-stage task-recycling samples compile."
