#!/bin/sh
# Hand-written prototype of the counted switch used by 1D shift bundles.
#
# Stage 1 runs without filters and checks the order words arrive in, which is what proves the
# send-order hand-over: every receiver must see the senders in descending order, including the
# case where senders relay for each other.
#
# Stage 2 turns the counter filters on and checks that each receiver keeps only its own block.
# Together they pin down the hardware contract the compiler relies on:
#
#   * a sender's own SWITCH_ADV advances its own router, and over-advancing a router that is
#     already on its last position is harmless,
#   * a receiver transmitting to RAMP and EAST duplicates rather than consumes,
#   * a filter withholds a wavelet from the compute element without removing it from the
#     network, and a wavelet nobody keeps is dropped by the terminating router,
#   * the counter delivers iff counter <= max_counter, counting modulo limit1 + 1 from
#     init_counter.
#
# The probe runs against whichever generation WSE_ARCH selects, so the same contract can be
# confirmed on both.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/handwritten/shift_bundle"
OUT="shift_bundle_hw"

compile_and_run() {
    m=$1
    k=$2
    filter=$3
    width=$((2 * m))

    rm -rf "$OUT"
    cslc --arch="${WSE_ARCH:-wse2}" "$SRC/layout.csl" -o "$OUT" \
        --fabric-dims=$((7 + width)),3 --fabric-offsets=4,1 --memcpy --channels=1 \
        --params=M:$m,K:$k,FILTER:$filter
    timeout -s 9 240 cs_python "$SRC/run.py" "$OUT" --M "$m" --K "$k" --filter "$filter"
}

for case in "3 1" "4 1" "3 2"; do
    set -- $case
    echo "--- stage 1: arrival order, M=$1 K=$2 (no filters) ---"
    compile_and_run "$1" "$2" 0
    echo "--- stage 2: filtered delivery, M=$1 K=$2 ---"
    compile_and_run "$1" "$2" 1
done

rm -rf "$OUT"
echo "Passed: counted switch by send order with filtered delivery."
