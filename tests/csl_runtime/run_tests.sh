#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Array to store failed tests
declare -a FAILED_TESTS

REAL_SPTLC="$(command -v sptlc || true)"
if [ -z "$REAL_SPTLC" ]; then
    echo -e "${RED}ERROR${NC}: sptlc not found on PATH"
    exit 1
fi

SYNC_SPTLC_WRAPPER_DIR="$(mktemp -d)"
cleanup_wrapper() {
    rm -rf "$SYNC_SPTLC_WRAPPER_DIR"
}
trap cleanup_wrapper EXIT

cat >"$SYNC_SPTLC_WRAPPER_DIR/sptlc" <<EOF
#!/bin/sh
real_sptlc="$REAL_SPTLC"
for arg in "\$@"; do
    if [ "\$arg" = "--sync-benchmarking" ]; then
        exec "\$real_sptlc" "\$@"
    fi
done
exec "\$real_sptlc" "\$@" --sync-benchmarking
EOF
chmod +x "$SYNC_SPTLC_WRAPPER_DIR/sptlc"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Running Test Suite${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Find all .sh files in the current directory
shopt -s nullglob
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_FILES=("$SCRIPT_DIR"/*.sh)

# Check if there are any test files
if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}No test files (*.sh) found in current directory${NC}"
    exit 0
fi

# Scripts that are not test cases and should be excluded from the run.
NON_TEST_SCRIPTS=("run_tests.sh" "run-in-lima.sh" "sptlc" "_lib.sh")

is_non_test() {
    local name="$1"
    for skip in "${NON_TEST_SCRIPTS[@]}"; do
        [ "$name" = "$skip" ] && return 0
    done
    return 1
}

# Run each test
for test_file in "${TEST_FILES[@]}"; do
    test_path=$test_file
    test_file=$(basename "$test_file")
    if is_non_test "$test_file"; then
        continue
    fi
    
    for mode in "normal" "sync"; do
        TOTAL=$((TOTAL + 1))

        if [ "$mode" = "normal" ]; then
            mode_label="without sync benchmarking"
            echo -e "${BLUE}Running:${NC} $test_file (${mode_label})"
            echo "----------------------------------------"
            bash "$test_path"
            EXIT_CODE=$?
        else
            mode_label="with sync benchmarking"
            echo -e "${BLUE}Running:${NC} $test_file (${mode_label})"
            echo "----------------------------------------"
            PATH="$SYNC_SPTLC_WRAPPER_DIR:$PATH" bash "$test_path"
            EXIT_CODE=$?
        fi

        echo "----------------------------------------"

        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}✓ PASSED${NC}: $test_file (${mode})"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}✗ FAILED${NC}: $test_file (${mode}, exit code: $EXIT_CODE)"
            FAILED=$((FAILED + 1))
            FAILED_TESTS+=("$test_file (${mode}, exit code: $EXIT_CODE)")
        fi

        echo ""
    done
done

# Print summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Total:  $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

# List failed tests if any
if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed tests:${NC}"
    for failed_test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}✗${NC} $failed_test"
    done
    exit 1
else
    echo ""
    echo -e "${GREEN}All tests passed! 🎉${NC}"
    exit 0
fi
