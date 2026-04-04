"""
Greyhound Fail-Slow Injector for InternEvo
==========================================
Ported from Greyhound/Megatron-LM/megatron/injection.py, adapted to use
InternEvo's ``gpc`` parallel context instead of Megatron's dist utilities.

Injection trace format (one event per line):
  Compute : <start_sec>;<end_sec>;<[global_rank_list]>;<delay_seconds>
  Comms   : comm;<start_sec>;<end_sec>;<[src_rank, dst_rank]>

Example injection.trace::

    30;120;[0,1];0.1
    comm;200;400;[0,4]

Usage (in train.py after distributed init)::

    from internlm.core.injection import FailSlowInjector
    import redis, os
    redis_cli = redis.StrictRedis(host=os.environ.get('MASTER_ADDR','localhost'))
    injector = FailSlowInjector('injection.trace', redis_cli)
    # then in the training loop each iteration:
    injector.check_computation(iteration)
    injector.check_communication(iteration, rank, device_id)
"""

import os
import socket
import subprocess
import sys
import time

import torch.distributed as dist

try:
    from internlm.core.context import global_context as gpc
    from internlm.core.context.process_group_initializer import ParallelMode
    _HAS_GPC = True
except ImportError:
    _HAS_GPC = False

# dp_planner lives under detector/injection/dp_planner.py; add that dir to path
_INJECTION_DIR = os.path.join(
    os.path.dirname(__file__),           # internlm/core/
    '..', '..', 'detector', 'injection'
)
_SINGLE_COMM_PATH = os.path.abspath(
    os.path.join(_INJECTION_DIR, 'single_comm.py')
)

import sys as _sys
_INJECTION_DIR_ABS = os.path.abspath(_INJECTION_DIR)
if _INJECTION_DIR_ABS not in _sys.path:
    _sys.path.insert(0, _INJECTION_DIR_ABS)

try:
    from dp_planner import PerformanceMetric, get_time_array, solve_dp
    _HAS_DP_PLANNER = True
except ImportError:
    _HAS_DP_PLANNER = False


class FailSlowInjector:
    """Wall-clock-time-scheduled fail-slow injector.

    Reads an *injection.trace* file and fires compute / communication
    fail-slow events at the specified wall-clock offsets relative to
    injector construction time.

    Compute events set ``delay_time_{rank}`` in Redis; the ``FailSlowHook``
    in :mod:`internlm.core.trainer_builder` picks these up each iteration
    and sleeps accordingly.

    Communication events launch ``single_comm.py`` sub-processes that
    flood a pair of ranks with 200 MB tensors, saturating their NIC.

    Args:
        injection_conf (str): Path to the injection trace file.
        redis_cli: A ``redis.StrictRedis`` client connected to the training
            cluster's Redis instance.
    """

    def __init__(self, injection_conf: str, redis_cli):
        self.init_time = time.time()
        self.comp_injection_info = []   # [start, end, [global_ranks], delay]
        self.comm_injection_info = []   # [start, end, [src_rank, dst_rank]]

        with open(injection_conf, 'r') as f:
            content = f.read().split('\n')

        for line in content:
            line = line.strip()
            if len(line) <= 1:
                continue
            if line.startswith('comm'):
                _, start, end, pair = line.split(';')
                self.comm_injection_info.append(
                    [float(start), float(end), eval(pair)]
                )
            else:
                start, end, global_ranks, delay_time = line.split(';')
                self.comp_injection_info.append(
                    [float(start), float(end), eval(global_ranks), float(delay_time)]
                )

        print(
            f'[FailSlowInjector] Loaded {len(self.comp_injection_info)} compute '
            f'and {len(self.comm_injection_info)} comm events from {injection_conf}',
            file=sys.stderr,
        )
        print(f'  Compute: {self.comp_injection_info}', file=sys.stderr)
        print(f'  Comms  : {self.comm_injection_info}', file=sys.stderr)

        self.comp_line_no = 0
        self.comm_line_no = 0
        self.version = 1          # monotonically increasing dp_version
        self.in_slow = False      # True while a compute slow is active
        self.iter_slow_start = -1 # iteration at which slow was injected
        self.port_version = 0     # offset added to base port for comm events
        self.redis_cli = redis_cli

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_parallel_sizes(self):
        """Return (num_tps, num_dps, num_pps) from Redis topology keys."""
        dp_data = self.redis_cli.get('0_dp')
        tp_data = self.redis_cli.get('0_tp')
        pp_data = self.redis_cli.get('0_pp')
        if dp_data is not None:
            num_dps = len(dp_data.decode().split('_'))
            num_tps = len(tp_data.decode().split('_')) if tp_data else 1
            num_pps = len(pp_data.decode().split('_')) if pp_data else 1
        else:
            if _HAS_GPC:
                num_dps = gpc.get_world_size(ParallelMode.DATA)
                num_tps = gpc.get_world_size(ParallelMode.TENSOR)
                num_pps = gpc.get_world_size(ParallelMode.PIPELINE)
            else:
                world = dist.get_world_size()
                num_dps = num_tps = num_pps = world
        return num_tps, num_dps, num_pps

    def _get_batch_sizes(self):
        """Return (micro_bsz, global_bsz) from Redis, with sensible defaults."""
        micro_bsz = self.redis_cli.get('micro_batch_size')
        global_bsz = self.redis_cli.get('global_batch_size')
        if micro_bsz is not None and global_bsz is not None:
            return int(micro_bsz.decode()), int(global_bsz.decode())
        if _HAS_GPC:
            try:
                mb = gpc.config.data.micro_bsz
                mn = gpc.config.data.micro_num
                _, num_dps, _ = self._get_parallel_sizes()
                return mb, mb * mn * num_dps
            except Exception:
                pass
        return 2, 256

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_computation(self, iteration: int):
        """Check and apply scheduled compute fail-slow events.

        Call once per iteration on every rank.  Only the affected ranks
        will have their ``delay_time`` Redis key changed.

        Args:
            iteration: Current global training iteration / batch count.
        """
        if self.comp_line_no >= len(self.comp_injection_info):
            return

        elapsed = time.time() - self.init_time
        start_t, end_t, global_ranks, delay_time = \
            self.comp_injection_info[self.comp_line_no]

        print(
            f'[FailSlowInjector] COMP check: elapsed={elapsed:.1f}s '
            f'iter={iteration} event=[{start_t},{end_t}] '
            f'ranks={global_ranks} delay={delay_time}s '
            f'in_slow={self.in_slow} iter_slow_start={self.iter_slow_start}',
            file=sys.stderr,
        )

        # Phase 1: inject
        if elapsed >= start_t and not self.in_slow:
            print(
                f'[FailSlowInjector] >>> INJECT COMP slow: ranks={global_ranks}, '
                f'delay={delay_time}s',
                file=sys.stderr,
            )
            for grank in global_ranks:
                self.redis_cli.set(f'delay_time_{grank}', delay_time)
            self.in_slow = True
            self.iter_slow_start = iteration
            return

        # Phase 2: DP rebalance (5 iters after injection)
        if (elapsed >= start_t and self.in_slow
                and iteration - self.iter_slow_start == 5
                and _HAS_DP_PLANNER):
            num_tps, num_dps, num_pps = self._get_parallel_sizes()
            micro_bsz, global_bsz = self._get_batch_sizes()

            locked_dp_ranks = []
            for grank in global_ranks:
                in_stage_rank = grank % (num_dps * num_tps)
                locked_dp_ranks.append(in_stage_rank // num_tps)

            print(
                f'[FailSlowInjector] DP rebalance: '
                f'TP/DP/PP={num_tps}/{num_dps}/{num_pps} '
                f'locked_dp_ranks={locked_dp_ranks} '
                f'micro_bsz={micro_bsz} global_bsz={global_bsz}',
                file=sys.stderr,
            )
            compute_time = {
                i: PerformanceMetric(65, 65, 65, 0.01) for i in range(num_dps)
            }
            for lr in locked_dp_ranks:
                compute_time[lr] = PerformanceMetric(515, 515, 515, 0.01)

            time_array = get_time_array(self.redis_cli, compute_time)
            print(f'[FailSlowInjector] Iter times: {time_array}', file=sys.stderr)
            dp_ret = solve_dp(time_array / 1000.0, micro_bsz, global_bsz)
            print(
                f'[FailSlowInjector] DP allocation: {dp_ret} version={self.version}',
                file=sys.stderr,
            )
            self.redis_cli.set('batch_distribution', str(dp_ret))
            self.redis_cli.set('dp_version', self.version)
            return

        # Phase 3: end injection
        if elapsed >= end_t and self.in_slow:
            print(
                f'[FailSlowInjector] <<< END COMP slow: ranks={global_ranks}',
                file=sys.stderr,
            )
            for grank in global_ranks:
                self.redis_cli.set(f'delay_time_{grank}', 0)

            if _HAS_DP_PLANNER:
                _, num_dps, _ = self._get_parallel_sizes()
                micro_bsz, global_bsz = self._get_batch_sizes()
                fair_dp = [global_bsz // (micro_bsz * num_dps)] * num_dps
                print(f'[FailSlowInjector] Restoring fair DP: {fair_dp}', file=sys.stderr)
                self.redis_cli.set('batch_distribution', str(fair_dp))
                self.redis_cli.set('dp_version', self.version + 1)

            self.version += 2
            self.in_slow = False
            self.comp_line_no += 1

    def check_communication(self, iteration: int, rank: int, device_id: int):
        """Check and trigger scheduled communication fail-slow events.

        Launches ``single_comm.py`` sub-processes to saturate the NIC between
        a pair of ranks.  Only the two involved ranks spawn a process.

        Args:
            iteration:  Current global training iteration / batch count.
            rank:       This process's global rank.
            device_id:  Local CUDA device index (for the comm worker).
        """
        if self.comm_line_no >= len(self.comm_injection_info):
            return

        elapsed = time.time() - self.init_time
        slow_start, duration, slow_pair = self.comm_injection_info[self.comm_line_no]

        print(
            f'[FailSlowInjector] COMM check: elapsed={elapsed:.1f}s '
            f'iter={iteration} rank={rank} event=[{slow_start},{slow_start+duration}] '
            f'pair={slow_pair}',
            file=sys.stderr,
        )

        if elapsed < slow_start:
            return

        print(
            f'[FailSlowInjector] >>> INJECT COMM slow: pair={slow_pair} '
            f'duration={duration}s',
            file=sys.stderr,
        )
        self.comm_line_no += 1
        self.port_version += 1

        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        base_port = 9969 + self.port_version

        cmd_base = (
            f'python {_SINGLE_COMM_PATH} '
            f'--tensor-size 200 --duration {int(duration)} --device {device_id}'
        )
        env_base = {
            **os.environ,
            'MASTER_ADDR': str(ip),
            'MASTER_PORT': str(base_port),
            'WORLD_SIZE': '2',
        }

        if rank == slow_pair[0]:
            env = {**env_base, 'RANK': '0'}
            print(f'[FailSlowInjector] Sender rank={rank}: {cmd_base}', file=sys.stderr)
            subprocess.Popen(cmd_base, shell=True, stdout=sys.stderr,
                             stderr=sys.stderr, env=env)
        elif rank == slow_pair[1]:
            env = {**env_base, 'RANK': '1'}
            print(f'[FailSlowInjector] Recver rank={rank}: {cmd_base}', file=sys.stderr)
            subprocess.Popen(cmd_base, shell=True, stdout=sys.stderr,
                             stderr=sys.stderr, env=env)
