#!/bin/bash

# Detect package manager and set run command
if [ -f "uv.lock" ]; then
    echo "Detected uv.lock ... running in uv-managed mode"
    RUN_CMD="uv run"
    LIST_CMD="uv pip list"
    ADD_CMD="uv add --dev"
else
    echo "Error: uv.lock not found. Please ensure project uses UV for dependency management"
    exit 1
fi

set -euo pipefail

# Never check pip for new version while running this script
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Check if mypy is installed globally
if ! ($LIST_CMD | grep "mypy" > /dev/null); then
    echo
    echo "Installing 'mypy' in current project"
    echo
    $ADD_CMD "mypy"
fi

# Install types if not installed
#  - do not use > /dev/null 2>&1 - this will show which packages should by in pyproject.toml
$RUN_CMD mypy --install-types --non-interactive

PLUGIN_NAME="MyPy"
echo
echo "Running Plugin $PLUGIN_NAME..."
$RUN_CMD mypy .
mypy_status=$? # Capture the exit status

# Optional: Add logging based on status
if [ $mypy_status -ne 0 ]; then
    echo "Plugin $PLUGIN_NAME failed with status $mypy_status" >&2
fi

exit $mypy_status # Exit with the actual status of the mypy command 