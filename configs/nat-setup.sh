#!/bin/bash
# Setup script for NeMo Agent Toolkit + Dynamo benchmarks
# This script runs on the head node before workers start
#
# Usage: Specify in your recipe with:
#   setup_script: "nat-setup.sh"

set -e

echo "=== NAT Setup Script ==="

# Configuration (override via environment variables)
NAT_REPO="${NAT_REPO:-https://github.com/NVIDIA/NeMo-Agent-Toolkit.git}"
NAT_BRANCH="${NAT_BRANCH:-main}"
NAT_DIR="/workspace/NeMo-Agent-Toolkit"
CUSTOM_DYNAMO_DIR="/workspace/custom_dynamo"

# Clone NeMo-Agent-Toolkit if not present
if [ ! -d "$NAT_DIR" ]; then
    echo "Cloning NeMo-Agent-Toolkit from $NAT_REPO (branch: $NAT_BRANCH)..."
    git clone --depth 1 --branch "$NAT_BRANCH" "$NAT_REPO" "$NAT_DIR"
else
    echo "NeMo-Agent-Toolkit already exists at $NAT_DIR"
fi

# Link custom Dynamo components (router, processor, frontend)
if [ ! -d "$CUSTOM_DYNAMO_DIR" ]; then
    echo "Linking custom Dynamo components..."
    ln -sf "$NAT_DIR/external/dynamo/generalized" "$CUSTOM_DYNAMO_DIR"
else
    echo "Custom Dynamo components already linked at $CUSTOM_DYNAMO_DIR"
fi

# Fix Dynamo 0.8.0 API compatibility
echo "Patching Dynamo API compatibility..."
# Remove static= parameter (was removed in 0.8.0)
sed -i 's/@dynamo_worker(static=False)/@dynamo_worker()/g' "$NAT_DIR/external/dynamo/generalized/router.py"
sed -i 's/@dynamo_worker(static=False)/@dynamo_worker()/g' "$NAT_DIR/external/dynamo/generalized/processor.py"
sed -i 's/@dynamo_worker(static=False)/@dynamo_worker()/g' "$NAT_DIR/external/dynamo/generalized/frontend.py"
# Remove create_service() calls (method was removed in 0.8.0)
sed -i '/await component.create_service()/d' "$NAT_DIR/external/dynamo/generalized/router.py"
sed -i '/await component.create_service()/d' "$NAT_DIR/external/dynamo/generalized/processor.py"
sed -i '/await component.create_service()/d' "$NAT_DIR/external/dynamo/generalized/frontend.py"
# Change frontend port from 8099 to 8000 (srtslurm health checks use 8000)
sed -i 's/port: int = 8099/port: int = 8000/g' "$NAT_DIR/external/dynamo/generalized/frontend.py"

# Download agent leaderboard data if not present
DATA_DIR="$NAT_DIR/examples/dynamo_integration/data"
if [ ! -f "$DATA_DIR/agent_leaderboard_v2_test_subset.json" ]; then
    echo "Downloading agent leaderboard data..."
    cd "$NAT_DIR"
    python3 examples/dynamo_integration/scripts/download_agent_leaderboard_v2.py || {
        echo "Warning: Failed to download data. You may need to run this manually."
    }
else
    echo "Agent leaderboard data already present"
fi

echo "=== NAT Setup Complete ==="
