#!/bin/bash
# InternEvo + torchft Fault Tolerance Demo Launch Script
#
# This script demonstrates running InternEvo with torchft fault tolerance.
# It starts a Lighthouse server and launches multiple replica groups.
#
# Usage:
#   bash run_ft_demo.sh
#
# Prerequisites:
#   - Activated conda env with InternEvo + torchft installed
#   - GPUs available (at least 2 for meaningful FT demo)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INTERNEVO_DIR="${SCRIPT_DIR}"
CONFIG="${INTERNEVO_DIR}/configs/ft_demo.py"
LOG_DIR="${INTERNEVO_DIR}/logs/ft_demo"

mkdir -p "${LOG_DIR}"

# === Step 1: Start Lighthouse Server ===
echo "=== Starting Lighthouse Server ==="
torchft_lighthouse \
    --min_replicas 1 \
    --quorum_tick_ms 100 \
    --join_timeout_ms 10000 \
    --bind "[::]:29510" \
    > "${LOG_DIR}/lighthouse.log" 2>&1 &
LIGHTHOUSE_PID=$!
echo "Lighthouse PID: ${LIGHTHOUSE_PID}"
sleep 2

# Check lighthouse is running
if ! kill -0 ${LIGHTHOUSE_PID} 2>/dev/null; then
    echo "ERROR: Lighthouse failed to start. Check ${LOG_DIR}/lighthouse.log"
    exit 1
fi
echo "Lighthouse running on port 29510"

export TORCHFT_LIGHTHOUSE="http://$(hostname):29510"

# === Step 2: Launch Replica Group 0 (1 GPU) ===
echo ""
echo "=== Launching Replica Group 0 ==="
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_port=29600 \
    --nnodes=1 \
    --node_rank=0 \
    "${INTERNEVO_DIR}/train.py" \
    --config "${CONFIG}" \
    --launcher torch \
    > "${LOG_DIR}/replica0.log" 2>&1 &
REPLICA0_PID=$!
echo "Replica 0 PID: ${REPLICA0_PID}"

# === Step 3: Launch Replica Group 1 (1 GPU) ===
echo "=== Launching Replica Group 1 ==="
CUDA_VISIBLE_DEVICES=1 torchrun \
    --nproc_per_node=1 \
    --master_port=29601 \
    --nnodes=1 \
    --node_rank=0 \
    "${INTERNEVO_DIR}/train.py" \
    --config "${CONFIG}" \
    --launcher torch \
    > "${LOG_DIR}/replica1.log" 2>&1 &
REPLICA1_PID=$!
echo "Replica 1 PID: ${REPLICA1_PID}"

echo ""
echo "=== All processes started ==="
echo "Lighthouse: PID ${LIGHTHOUSE_PID}"
echo "Replica 0:  PID ${REPLICA0_PID}"
echo "Replica 1:  PID ${REPLICA1_PID}"
echo ""
echo "Log files in: ${LOG_DIR}"
echo ""
echo "=== To test fault tolerance ==="
echo "1. Wait for training to start (check replica logs)"
echo "2. Kill one replica: kill ${REPLICA1_PID}"
echo "3. Observe replica 0 continues training"
echo "4. Restart replica 1 to see live recovery"
echo ""
echo "=== To stop all ==="
echo "kill ${LIGHTHOUSE_PID} ${REPLICA0_PID} ${REPLICA1_PID}"

# Wait for all processes
wait ${REPLICA0_PID} ${REPLICA1_PID}

# Cleanup
echo ""
echo "=== Training complete, cleaning up ==="
kill ${LIGHTHOUSE_PID} 2>/dev/null || true
echo "Done."
