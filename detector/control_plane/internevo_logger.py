"""
InternEvo Parallel Topology Logger for Greyhound
=================================================
Equivalent of ``detector/control_plane/megatron_logger.py`` but uses
InternEvo's ``gpc`` (global parallel context) instead of Megatron's
``parallel_state`` module.

Call ``log_internevo_parallel_to_redis()`` once after
``initialize_distributed_env()`` completes.  It spawns a background
process that writes the TP/DP/PP group memberships — plus the micro-batch
and global-batch sizes — to Redis so that the Greyhound L2 control plane
(``global_analyzer.py``) can reconstruct the training topology.

Redis key schema (same as megatron_logger)::

    {rank}_tp  ->  "r0_r1_r2_r3"   (underscore-separated global ranks)
    {rank}_dp  ->  "r0_r4_r8_r12"
    {rank}_pp  ->  "r0_r1_r2_r3"
    micro_batch_size  ->  "2"
    global_batch_size ->  "256"
"""

import os
import time
from multiprocessing import Process

import redis
import redis.exceptions

try:
    from internlm.core.context import global_context as gpc
    from internlm.core.context.process_group_initializer import ParallelMode
    _HAS_GPC = True
except ImportError:
    _HAS_GPC = False


def _do_log(redis_addr, redis_port, my_rank,
            tp_ranks, dp_ranks, pp_ranks,
            micro_bsz, global_bsz):
    """Background worker: wait for Redis, then write all keys."""
    db = redis.StrictRedis(redis_addr, int(redis_port), db=0)

    # Wait until Redis is ready (up to 30 s)
    for _ in range(30):
        try:
            db.ping()
            break
        except redis.exceptions.ConnectionError:
            time.sleep(1)

    tp_str = '_'.join(str(r) for r in tp_ranks)
    dp_str = '_'.join(str(r) for r in dp_ranks)
    pp_str = '_'.join(str(r) for r in pp_ranks)

    db.set(f'{my_rank}_tp', tp_str)
    db.set(f'{my_rank}_dp', dp_str)
    db.set(f'{my_rank}_pp', pp_str)
    db.set('micro_batch_size', str(micro_bsz))
    db.set('global_batch_size', str(global_bsz))

    print(
        f'[InternEvoLogger] Rank {my_rank}: logged topology to Redis '
        f'tp={tp_str} dp={dp_str} pp={pp_str} '
        f'micro_bsz={micro_bsz} global_bsz={global_bsz}'
    )


def log_internevo_parallel_to_redis():
    """Log InternEvo parallel topology to Redis for the Greyhound control plane.

    Spawns a daemon background process so it does not block the caller.
    Safe to call on every rank; each rank writes its own topology keys.

    Requires ``gpc`` to be fully initialised (i.e., call after
    ``initialize_distributed_env()``).
    """
    if not _HAS_GPC:
        print('[InternEvoLogger] gpc not available, skipping topology logging.')
        return

    redis_addr = os.getenv('MASTER_ADDR', 'localhost')
    redis_port = os.getenv('REDIS_PORT', '6379')

    my_rank = gpc.get_global_rank()

    # Gather global ranks in each parallel group for *this* rank
    try:
        tp_ranks = list(gpc.get_ranks_in_group(ParallelMode.TENSOR))
    except Exception:
        tp_ranks = [my_rank]
    try:
        dp_ranks = list(gpc.get_ranks_in_group(ParallelMode.DATA))
    except Exception:
        dp_ranks = [my_rank]
    try:
        pp_ranks = list(gpc.get_ranks_in_group(ParallelMode.PIPELINE))
    except Exception:
        pp_ranks = [my_rank]

    # Compute batch sizes
    try:
        micro_bsz  = gpc.config.data.micro_bsz
        micro_num  = gpc.config.data.micro_num
        num_dp     = gpc.get_world_size(ParallelMode.DATA)
        global_bsz = micro_bsz * micro_num * num_dp
    except Exception:
        micro_bsz, global_bsz = 2, 256

    proc = Process(
        target=_do_log,
        args=(redis_addr, redis_port, my_rank,
              tp_ranks, dp_ranks, pp_ranks,
              micro_bsz, global_bsz),
        daemon=True,
    )
    proc.start()
