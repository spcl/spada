#!/bin/bash

set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR/../../samples/benchmarks"
RUNTIME="$SCRIPT_DIR/../../spatialstencil/runtime/runtime.py"
OUTPUT_DIR="$SCRIPT_DIR/benchmark"

TOTAL=0
PASSED=0
FAILED=0
FAILED_TESTS=()

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Running Benchmark Suite${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

shopt -s nullglob
BENCHMARK_FILES=("$BENCHMARK_DIR"/*_4_4_4.sptl)
shopt -u nullglob

if [ ${#BENCHMARK_FILES[@]} -eq 0 ]; then
	echo -e "${YELLOW}No benchmark files matching *_4_4_4.sptl found in ${BENCHMARK_DIR}${NC}"
	exit 0
fi

for benchmark_path in "${BENCHMARK_FILES[@]}"; do
	benchmark_file="$(basename "$benchmark_path")"
	benchmark_name="${benchmark_file%.*}"

	TOTAL=$((TOTAL + 1))

	echo -e "${BLUE}Running:${NC} ${benchmark_file}"
	echo "----------------------------------------"

	rm -rf "$OUTPUT_DIR"
	mkdir -p "$OUTPUT_DIR"

	compile_output=$(sptlc "$benchmark_path" "$OUTPUT_DIR" 2>&1)
	compile_status=$?

	if [ $compile_status -ne 0 ]; then
		echo -e "${RED}Compilation failed${NC}"
		echo "$compile_output"
		FAILED=$((FAILED + 1))
		FAILED_TESTS+=("${benchmark_file} (compile exit code: ${compile_status})")
		echo "----------------------------------------"
		echo ""
		rm -rf "$OUTPUT_DIR"
		continue
	else
		echo -e "${GREEN}Compilation succeeded${NC}"
	fi

	timeout_output=$(timeout -s 9 120 cs_python "$RUNTIME" "$OUTPUT_DIR" --benchmark --randomize 2>&1)
	runtime_status=$?

	if [ $runtime_status -ne 0 ]; then
		echo -e "${RED}Execution failed${NC}"
		echo "$timeout_output"
		FAILED=$((FAILED + 1))
		FAILED_TESTS+=("${benchmark_file} (run exit code: ${runtime_status})")
	else
		echo "$timeout_output"
		echo -e "${GREEN}✓ PASSED${NC}: ${benchmark_file}"
		PASSED=$((PASSED + 1))
	fi

	echo "----------------------------------------"
	echo ""

	rm -rf "$OUTPUT_DIR"
done

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Benchmark Summary${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Total:  $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
	echo ""
	echo -e "${RED}Failed benchmarks:${NC}"
	for failed_test in "${FAILED_TESTS[@]}"; do
		echo -e "  ${RED}✗${NC} $failed_test"
	done
	exit 1
else
	echo ""
	echo -e "${GREEN}All benchmarks passed! 🎉${NC}"
	exit 0
fi
