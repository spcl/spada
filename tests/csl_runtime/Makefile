# Resolve paths relative to this Makefile, regardless of where make is invoked.
THIS_DIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(THIS_DIR)/../..)

# SDK download / extraction paths (relative to this directory).
SDK_TAR    := $(THIS_DIR)cerebras-sdk.tar.gz
SDK_DIR    := $(THIS_DIR)cerebras-sdk

# Optional override: path to an already-extracted SDK directory.
# Defaults to SDK_DIR (the directory populated by setup-sdk / extract-sdk).
# Set CSL_SDK_DIR on the command line to point at a different location.
CSL_SDK_DIR ?= $(SDK_DIR)
SDK_PATH_PREFIX := $(abspath $(CSL_SDK_DIR)):

TEST_ENV = PATH="$(SDK_PATH_PREFIX)$$PATH" \
	PYTHONPATH="$(REPO_ROOT)$${PYTHONPATH:+:$$PYTHONPATH}" \
	SINGULARITY_BIND="$(REPO_ROOT)$${SINGULARITY_BIND:+,$$SINGULARITY_BIND}"

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "CSL runtime test targets"
	@echo ""
	@echo "SDK setup (run once):"
	@echo "  make -C tests/csl_runtime setup-sdk CSL_SDK_URL=<url>"
	@echo ""
	@echo "Testing:"
	@echo "  make -C tests/csl_runtime check-sdk  [CSL_SDK_DIR=/path/to/sdk]"
	@echo "  make -C tests/csl_runtime test        [CSL_SDK_DIR=/path/to/sdk]"
	@echo "  make -C tests/csl_runtime test-one TEST=test_add.sh [CSL_SDK_DIR=/path/to/sdk]"
	@echo "  make -C tests/csl_runtime shell       [CSL_SDK_DIR=/path/to/sdk]"
	@echo "  make -C tests/csl_runtime smoke-sdk SDK_EXAMPLES_DIR=/path/to/csl-extras-*"
	@echo ""
	@echo "Cleanup:"
	@echo "  make -C tests/csl_runtime clean"
	@echo "  make -C tests/csl_runtime clean-sdk"
	@echo ""
	@echo "Notes:"
	@echo "  - On Apple Silicon macOS, use run-in-lima.sh instead:"
	@echo "      tests/csl_runtime/run-in-lima.sh --sdk-url <url>"
	@echo "  - CSL_SDK_DIR defaults to tests/csl_runtime/cerebras-sdk/ (populated by setup-sdk)."

# ── SDK download and extraction ───────────────────────────────────────────────
# Download the SDK tarball.  Requires CSL_SDK_URL to be set:
#   make -C tests/csl_runtime download-sdk CSL_SDK_URL=<url>
.PHONY: download-sdk
download-sdk:
	@if [ -z "$(CSL_SDK_URL)" ]; then \
	    echo "ERROR: CSL_SDK_URL is not set."; \
	    echo "  export CSL_SDK_URL=<url>  or pass it as a make variable"; \
	    exit 1; \
	fi
	@echo "==> Downloading Cerebras SDK to $(SDK_TAR) ..."
	@if command -v curl >/dev/null 2>&1; then \
	    curl -L --progress-bar -o "$(SDK_TAR)" "$(CSL_SDK_URL)"; \
	elif command -v wget >/dev/null 2>&1; then \
	    wget -q --show-progress -O "$(SDK_TAR)" "$(CSL_SDK_URL)"; \
	else \
	    echo "ERROR: neither curl nor wget found."; exit 1; \
	fi
	@echo "==> Downloaded: $(SDK_TAR)"

# Extract an already-downloaded tarball into SDK_DIR.
.PHONY: extract-sdk
extract-sdk: $(SDK_TAR)
	@echo "==> Extracting SDK to $(SDK_DIR) ..."
	mkdir -p "$(SDK_DIR)"
	tar xf "$(SDK_TAR)" -C "$(SDK_DIR)" --strip-components=1 2>/dev/null || \
	    tar xf "$(SDK_TAR)" -C "$(SDK_DIR)"
	@echo "==> Extracted: $(SDK_DIR)"

# One-shot: download and extract.
.PHONY: setup-sdk
setup-sdk: download-sdk extract-sdk
	@echo "==> SDK ready at $(SDK_DIR)"

# Clean up downloaded tarball and extracted SDK.
.PHONY: clean-sdk
clean-sdk:
	rm -f "$(SDK_TAR)"
	rm -rf "$(SDK_DIR)"

.PHONY: check-host
check-host:
	@if [ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
	    echo "ERROR: Apple Silicon macOS must run the Cerebras SDK inside an x86_64 Lima VM."; \
	    echo "Use the helper script from the repo root:"; \
	    echo "  tests/csl_runtime/run-in-lima.sh --sdk-url <url>"; \
	    echo "  tests/csl_runtime/run-in-lima.sh --sdk /path/to/cs_sdk  (if already extracted)"; \
	    exit 1; \
	fi

.PHONY: check-python
check-python:
	@python3 -m pip install --quiet -r "$(REPO_ROOT)/requirements-ci.txt" && \
	    python3 -m pip install --no-deps --quiet -e "$(REPO_ROOT)" || { \
	    echo "ERROR: Failed to install Python dependencies."; \
	    echo "Run: python3 -m pip install -r $(REPO_ROOT)/requirements-ci.txt && python3 -m pip install --no-deps -e $(REPO_ROOT)"; \
	    exit 1; \
	}

.PHONY: check-sdk
check-sdk: check-host check-python
	@$(TEST_ENV) bash -lc ' \
	    missing=0; \
	    for cmd in singularity cslc cs_python sdk_debug_shell python3; do \
	        if ! command -v "$$cmd" >/dev/null 2>&1; then \
	            echo "ERROR: $$cmd not found on PATH."; \
	            missing=1; \
	        fi; \
	    done; \
	    if [ "$$missing" -ne 0 ]; then \
	        echo ""; \
	        echo "If the SDK is installed but not on PATH, rerun with:"; \
	        echo "  make -C tests/csl_runtime $@ CSL_SDK_DIR=/absolute/path/to/cs_sdk"; \
	        exit 1; \
	    fi; \
	    echo "Using singularity: $$(command -v singularity)"; \
	    echo "Using cslc: $$(command -v cslc)"; \
	    echo "Using cs_python: $$(command -v cs_python)"; \
	    echo "Using sdk_debug_shell: $$(command -v sdk_debug_shell)"; \
	    singularity --version; \
	    cslc --help >/dev/null; \
	    cs_python --help >/dev/null 2>&1 || true'

.PHONY: smoke-sdk
smoke-sdk: check-sdk
	@if [ -z "$(SDK_EXAMPLES_DIR)" ]; then \
	    echo "Usage: make -C tests/csl_runtime smoke-sdk SDK_EXAMPLES_DIR=/path/to/csl-extras-*"; \
	    exit 1; \
	fi
	@$(TEST_ENV) bash -lc 'sdk_debug_shell smoke "$(SDK_EXAMPLES_DIR)"'

.PHONY: test
test: check-sdk
	@$(TEST_ENV) bash "$(THIS_DIR)/run_tests.sh"

.PHONY: test-one
test-one: check-sdk
	@if [ -z "$(TEST)" ]; then \
	    echo "Usage: make -C tests/csl_runtime test-one TEST=test_add.sh"; \
	    exit 1; \
	fi
	@$(TEST_ENV) bash -lc ' \
	    test_path="$(TEST)"; \
	    case "$$test_path" in \
	        /*) ;; \
	        *) test_path="$(THIS_DIR)$$test_path" ;; \
	    esac; \
	    if [ ! -f "$$test_path" ]; then \
	        echo "ERROR: test script not found: $$test_path"; \
	        exit 1; \
	    fi; \
	    bash "$$test_path"'

.PHONY: shell
shell: check-sdk
	@$(TEST_ENV) bash

.PHONY: clean
clean:
	rm -rf "$(THIS_DIR)"*_sptl
	rm -f "$(THIS_DIR)"*.npy
	rm -f "$(THIS_DIR)"sim_stats.json "$(THIS_DIR)"simconfig.json "$(THIS_DIR)"wio_flow.json
	rm -rf "$(THIS_DIR)"simfab_traces

.PHONY: clean-all
clean-all: clean clean-sdk
