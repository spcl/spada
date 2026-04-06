#!/usr/bin/env bash
# run-in-lima.sh — Run CSL runtime tests inside an x86_64 Lima VM.
#
# On Apple Silicon macOS the Cerebras SDK requires an x86_64 environment.
# This script creates the Lima VM on first use, installs Python dependencies
# inside it, and then delegates to the Makefile targets.
#
# Usage (run from anywhere inside the repo):
#
#   tests/csl_runtime/run-in-lima.sh --sdk /path/to/cs_sdk
#   tests/csl_runtime/run-in-lima.sh --sdk /path/to/cs_sdk --test test_add.sh
#   tests/csl_runtime/run-in-lima.sh --sdk /path/to/cs_sdk --smoke /path/to/csl-extras-*
#   tests/csl_runtime/run-in-lima.sh --sdk /path/to/cs_sdk --shell
#   tests/csl_runtime/run-in-lima.sh --sdk /path/to/cs_sdk --check
#
# The SDK directory and the repository must both be under your Mac home
# directory ($HOME), which Lima mounts automatically.
#
# Prerequisites (install once):
#   brew install lima qemu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VM_NAME="cs-sdk"
LIMA_CONFIG="$SCRIPT_DIR/lima-ubuntu-x86_64.yaml"

# ── Argument parsing ──────────────────────────────────────────────────────────
SDK_DIR=""
SDK_URL=""
TEST_NAME=""
SMOKE_DIR=""
MODE="test"   # test | test-one | smoke | shell | check

usage() {
    cat <<'EOF'
run-in-lima.sh — Run CSL runtime tests inside an x86_64 Lima VM.

On Apple Silicon macOS the Cerebras SDK requires an x86_64 environment.
This script creates the Lima VM on first use, installs Python dependencies
inside it, and then delegates to the Makefile targets.

SDK options (mutually exclusive):
  --sdk-url <url>   Download and extract the SDK automatically (simplest).
                    The tarball is saved to tests/csl_runtime/cerebras-sdk.tar.gz
                    and extracted to tests/csl_runtime/cerebras-sdk/.
                    Re-download is skipped if the tarball already exists.
  --sdk <dir>       Path to an already-extracted SDK directory.

Usage (run from the repo root):

  tests/csl_runtime/run-in-lima.sh --sdk-url <url>
  tests/csl_runtime/run-in-lima.sh --sdk-url <url> --test test_add.sh
  tests/csl_runtime/run-in-lima.sh --sdk-url <url> --smoke /path/to/csl-extras-*
  tests/csl_runtime/run-in-lima.sh --sdk-url <url> --shell
  tests/csl_runtime/run-in-lima.sh --sdk-url <url> --check

  tests/csl_runtime/run-in-lima.sh --sdk /path/to/cs_sdk
  (same --test / --smoke / --shell / --check flags work with --sdk too)

The repository must reside under $HOME, which Lima mounts automatically.

Prerequisites (install once):
  brew install lima qemu lima-additional-guestagents
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sdk)        SDK_DIR="$(cd "$2" && pwd)"; shift 2 ;;
        --sdk-url)    SDK_URL="$2"; shift 2 ;;
        --test)       MODE="test-one"; TEST_NAME="$2"; shift 2 ;;
        --smoke)      MODE="smoke"; SMOKE_DIR="$(cd "$2" && pwd)"; shift 2 ;;
        --shell)      MODE="shell"; shift ;;
        --check)      MODE="check"; shift ;;
        -h|--help)    usage ;;
        *)            echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$SDK_DIR" && -z "$SDK_URL" ]]; then
    echo "ERROR: one of --sdk or --sdk-url is required."
    echo ""
    usage
fi
if [[ -n "$SDK_DIR" && -n "$SDK_URL" ]]; then
    echo "ERROR: --sdk and --sdk-url are mutually exclusive."
    echo ""
    usage
fi

# ── Validate paths are under $HOME ────────────────────────────────────────────
check_under_home() {
    local path="$1" label="$2"
    case "$path" in
        "$HOME"/*|"$HOME") ;;
        *) echo "ERROR: $label must be under \$HOME ($HOME) for Lima to access it."; exit 1 ;;
    esac
}
check_under_home "$REPO_ROOT" "repository"
if [[ -n "$SDK_DIR" ]]; then
    check_under_home "$SDK_DIR" "--sdk path"
fi
if [[ "$MODE" == "smoke" ]]; then
    check_under_home "$SMOKE_DIR" "--smoke path"
fi

# ── Check prerequisites ───────────────────────────────────────────────────────
need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: $1 not found. Install it with: brew install $2"
        exit 1
    }
}
need limactl lima
need qemu-system-x86_64 qemu
# lima-additional-guestagents provides the Linux-x86_64 guest agent binary
# required when running x86_64 VMs on Apple Silicon.
if [[ ! -f "$(brew --prefix lima 2>/dev/null)/share/lima/lima-guestagent.Linux-x86_64.gz" ]] 2>/dev/null; then
    if ! brew list lima-additional-guestagents >/dev/null 2>&1; then
        echo "ERROR: lima-additional-guestagents is required for x86_64 VMs on Apple Silicon."
        echo "Install it with: brew install lima-additional-guestagents"
        exit 1
    fi
fi

# ── SDK download and extraction (--sdk-url path) ──────────────────────────────
if [[ -n "$SDK_URL" ]]; then
    SDK_TAR="$SCRIPT_DIR/cerebras-sdk.tar.gz"
    SDK_DIR="$SCRIPT_DIR/cerebras-sdk"

    if [[ -f "$SDK_TAR" ]]; then
        echo "==> Tarball already present: $SDK_TAR (skipping download)."
        echo "    Delete it and re-run to force a fresh download."
    else
        echo "==> Downloading Cerebras SDK ..."
        if command -v curl >/dev/null 2>&1; then
            curl -L --progress-bar -o "$SDK_TAR" "$SDK_URL"
        elif command -v wget >/dev/null 2>&1; then
            wget -q --show-progress -O "$SDK_TAR" "$SDK_URL"
        else
            echo "ERROR: neither curl nor wget found."; exit 1
        fi
        echo "==> Downloaded: $SDK_TAR"
    fi

    if [[ ! -d "$SDK_DIR" ]] || [[ -z "$(ls -A "$SDK_DIR" 2>/dev/null)" ]]; then
        echo "==> Extracting SDK to $SDK_DIR ..."
        mkdir -p "$SDK_DIR"
        # Try --strip-components=1 first; fall back for tarballs without a top-level dir.
        tar xf "$SDK_TAR" -C "$SDK_DIR" --strip-components=1 2>/dev/null || \
            tar xf "$SDK_TAR" -C "$SDK_DIR"
        echo "==> Extracted: $SDK_DIR"
    else
        echo "==> SDK already extracted at $SDK_DIR"
    fi
fi

# ── Create or start the VM ────────────────────────────────────────────────────
vm_status() { limactl list --format '{{.Name}} {{.Status}}' 2>/dev/null | awk "\$1==\"$VM_NAME\"{print \$2}"; }

status="$(vm_status)"
if [[ -z "$status" ]]; then
    echo "==> Creating Lima VM '$VM_NAME' (first-time setup, ~5–10 min) ..."
    limactl start "$LIMA_CONFIG" --name "$VM_NAME"
elif [[ "$status" != "Running" ]]; then
    echo "==> Starting Lima VM '$VM_NAME' ..."
    limactl start "$VM_NAME"
else
    echo "==> Lima VM '$VM_NAME' is already running."
fi

# ── Helper: run a command inside the VM ──────────────────────────────────────
vm() { limactl shell "$VM_NAME" -- bash -lc "$*"; }

# ── One-time setup inside VM (idempotent) ────────────────────────────────────
echo "==> Checking VM environment ..."
# Ensure `python` resolves to python3 (test scripts use bare `python`).
vm "command -v python >/dev/null 2>&1 || \
    sudo update-alternatives --install /usr/bin/python python \"\$(command -v python3)\" 1"
echo "==> Installing Python dependencies inside VM ..."
# Older or partially-provisioned VMs may have python3 without pip.
vm "if ! python3 -m pip --version >/dev/null 2>&1; then \
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y && \
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip; \
    fi && \
    python3 -m pip install --quiet -r '$REPO_ROOT/requirements-ci.txt' && \
    python3 -m pip install --no-deps --quiet -e '$REPO_ROOT'"

# ── Delegate to the Makefile ──────────────────────────────────────────────────
MAKE_ARGS="CSL_SDK_DIR=$SDK_DIR"

case "$MODE" in
    check)
        echo "==> Checking SDK toolchain ..."
        vm "make -C '$REPO_ROOT/tests/csl_runtime' check-sdk $MAKE_ARGS"
        ;;
    test)
        echo "==> Running full CSL test suite ..."
        vm "make -C '$REPO_ROOT/tests/csl_runtime' test $MAKE_ARGS"
        ;;
    test-one)
        echo "==> Running test: $TEST_NAME ..."
        vm "make -C '$REPO_ROOT/tests/csl_runtime' test-one TEST='$TEST_NAME' $MAKE_ARGS"
        ;;
    smoke)
        echo "==> Running SDK smoke test ..."
        vm "make -C '$REPO_ROOT/tests/csl_runtime' smoke-sdk SDK_EXAMPLES_DIR='$SMOKE_DIR' $MAKE_ARGS"
        ;;
    shell)
        echo "==> Dropping into VM shell (SDK and repo on PATH/PYTHONPATH) ..."
        limactl shell "$VM_NAME" -- bash -lc \
            "export PATH='$SDK_DIR:\$PATH'; \
             export PYTHONPATH='$REPO_ROOT\${PYTHONPATH:+:\$PYTHONPATH}'; \
             cd '$REPO_ROOT'; exec bash"
        ;;
esac
