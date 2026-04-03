# InternEvo × Greyhound Migration: Completed Status

**Date**: April 1, 2026
**Status**: ✅ Phase 1-3 Complete, Ready for Testing
**Target**: `/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo` (ResiHP branch)

## What Was Done

### Phase 1: L1 Detector Compilation ✅
- **Directory**: `/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/detector/`
- **Binary**: `detector/build/libncclprobe.so` (7.9 MB)
- **NCCL Version**: 2.20.5 (bundled with CUDA 12.8)
- **Compilation**: Used `-DNCCL_INCLUDE_DIR=/usr/local/cuda/targets/x86_64-linux/include -DNCCL_LIB_DIR=/usr/local/cuda/targets/x86_64-linux/lib`
- **Build Status**: ✅ SUCCESS

### Phase 2: L2 Control Plane Setup ✅
- **Wheel**: `detector/dist/control_plane-1.0-py3-none-any.whl` (26 KB)
- **Import Test**: ✅ Module loads successfully
- **Dependencies**: Uses Redis (localhost:6379) for inter-rank coordination

### Phase 3: L3 Framework Integration ✅
**File Modified**: `/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/internlm/core/trainer_builder.py`

**Changes Made**:
1. **New Imports** (lines 1-14):
   - `import signal` — for SIGUSR1 signal handling
   - `import redis` — for pause/resume coordination (with try/except fallback)

2. **Signal Handler** (lines 57-68):
   - `_CHECK_PAUSE` global flag
   - `_failslow_pause_handler()` — catches SIGUSR1 from L2 control plane
   - Logs pause event with rank info

3. **Pause/Resume Logic** (lines 299-338 in `_process_batch()`):
   - Checks `_CHECK_PAUSE` flag at iteration start
   - Connects to Redis on localhost:6379
   - Waits for recovery signal via key `failslow_pause_rank_<rank>`
   - Max pause duration: 5 minutes (timeout safety)
   - Fallback: 2-second sleep if Redis unavailable
   - Detailed logging of pause/resume events

**Backup**: Original saved to `trainer_builder.py.backup`

## How It Works (End-to-End)

Training Iteration:
├─ L1 (libncclprobe.so): Monitors NCCL operations via LD_PRELOAD
├─ L2 (control_plane): Detects anomalies via BOCD, sends SIGUSR1
├─ Signal Handler: Receives SIGUSR1 → sets _CHECK_PAUSE = True
└─ L3 (trainer_builder):
    ├─ Next iteration starts
    ├─ Checks _CHECK_PAUSE flag
    ├─ If true: Pause & wait for recovery signal via Redis
    ├─ Once unpaused: Resume training
    └─ Continue iterations

## Required Environment Variables (train.py or train_copy.py)

Before starting training, ensure these are set:

export LD_PRELOAD="/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/detector/build/libncclprobe.so"
export CONTROL_PLANE_WHL_PATH="/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/detector/dist/control_plane-1.0-py3-none-any.whl"
export NCCLPROBE_LOG_PATH="./logs"
export GLOBAL_CONTROLLER_LOG_PATH="./logs"
export LOCAL_CONTROLLER_LOG_PATH="./logs"

Or use train_copy.py which already has these configured.

## Next Steps

1. Run normal training (5-10 iterations) to verify no breakage
2. Check detector logs for NCCL interception
3. Manual signal test: kill -USR1 <pid> and verify pause behavior
4. Full fail-slow injection test when ready
