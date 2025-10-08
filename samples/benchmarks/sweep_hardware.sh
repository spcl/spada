#!/bin/bash

set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BENCHMARK_DIR="samples/benchmarks"
RUNTIME="spatialstencil/runtime/runtime.py"
OUTPUT_DIR="benchmark_results"

mkdir $OUTPUT_DIR

TOTAL=0
PASSED=0
FAILED=0
FAILED_TESTS=()
RUNS_PER_EXPERIMENT=100

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Running Benchmark Suite${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

shopt -s nullglob
BENCHMARK_FILES=("$BENCHMARK_DIR"/copy.sptl "$BENCHMARK_DIR"/reduce.sptl "$BENCHMARK_DIR"/reduce_pipelined.sptl "$BENCHMARK_DIR"/*_128_128_80.sptl)
shopt -u nullglob

if [ ${#BENCHMARK_FILES[@]} -eq 0 ]; then
	echo -e "${YELLOW}No benchmark files matching *_128_128_80.sptl found in ${BENCHMARK_DIR}${NC}"
	exit 0
fi

for benchmark_path in "${BENCHMARK_FILES[@]}"; do
	benchmark_file="$(basename "$benchmark_path")"
	benchmark_name="${benchmark_file%.*}"
	benchmark_dir="${benchmark_file}_sptl"

	TOTAL=$((TOTAL + 1))

	echo -e "${BLUE}Running:${NC} ${benchmark_name}"
	echo "----------------------------------------"

	mkdir -p $OUTPUT_DIR/$benchmark_name

	if [[ $benchmark_name == *"_128_128_80"* ]]; then
		EXTRA_FLAGS=""
    else
		echo "Parameterizing with N=128, K=80"
		EXTRA_FLAGS="-p N=128 -p K=80"
	fi

	compile_output=$(sptlc "$benchmark_path" "$benchmark_dir" $EXTRA_FLAGS $* |& tee -a $OUTPUT_DIR/$benchmark_name/compile.log)
	compile_status=$?

	if [ $compile_status -ne 0 ]; then
		echo -e "${RED}Compilation failed${NC}"
		echo "$compile_output"
		FAILED=$((FAILED + 1))
		FAILED_TESTS+=("${benchmark_name} (compile exit code: ${compile_status})")
		echo "----------------------------------------"
		echo ""
		continue
	else
		echo -e "${GREEN}Compilation succeeded${NC}"
	fi

	THIS_TEST_FAILED="0"
	timeout_output=$(timeout -s 9 300 cs_python "$RUNTIME" "$benchmark_dir" --benchmark --randomize --repetitions $RUNS_PER_EXPERIMENT --output-dir $OUTPUT_DIR/$benchmark_name |& tee -a $OUTPUT_DIR/$benchmark_name/run.log)
	runtime_status=$?

	if [ $runtime_status -ne 0 ]; then
		echo -e "${RED}Execution failed${NC}"
		echo "$timeout_output"
		FAILED=$((FAILED + 1))
		FAILED_TESTS+=("${benchmark_file} (run exit code: ${runtime_status})")
		THIS_TEST_FAILED="1"
		break
	else
		echo "$timeout_output"
		cp perf_cycles.npy $OUTPUT_DIR/$benchmark_name/run${i}_cycles.npy
		echo -e "${GREEN}✓ PASSED${NC}: ${benchmark_file} (${i}/${RUNS_PER_EXPERIMENT})"
	fi

    if [ $THIS_TEST_FAILED = "0" ]; then
		PASSED=$((PASSED + 1))
	fi

	echo "----------------------------------------"
	echo ""
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
