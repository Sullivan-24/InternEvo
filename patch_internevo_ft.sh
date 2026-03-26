#!/bin/bash
# Patch InternEvo for torchft integration
set -e
BASE="/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo"

# 1. Patch trainer_builder.py - add FT import and dispatch in fit()
TB="$BASE/internlm/core/trainer_builder.py"
if ! grep -q 'ft_fit' "$TB"; then
    # Add import after the last import line
    sed -i '/^from internlm.utils.writer import Writer$/a\
\
try:\
    from internlm.ft.training import ft_fit\
    _FT_AVAILABLE = True\
except ImportError:\
    _FT_AVAILABLE = False' "$TB"

    # Replace fit() with FT-aware version that delegates to _original_fit
    python3 -c "
import re
with open('$TB') as f:
    content = f.read()

# Find 'def fit(self):' and rename it, add FT dispatch
old = '''    def fit(self):
        \"\"\"
        Run InternEvo training loop.
        \"\"\"'''

new = '''    def fit(self):
        \"\"\"
        Run InternEvo training loop (with optional fault tolerance).
        \"\"\"
        ft_cfg = gpc.config.get('ft', None)
        if _FT_AVAILABLE and ft_cfg and ft_cfg.get('enabled', False):
            ft_fit(self)
            return
        self._original_fit()

    def _original_fit(self):
        \"\"\"
        Original InternEvo training loop.
        \"\"\"'''

content = content.replace(old, new)
with open('$TB', 'w') as f:
    f.write(content)
"
    echo "Patched trainer_builder.py"
else
    echo "trainer_builder.py already patched"
fi

# 2. Patch launch.py - add init_fault_tolerance call
LP="$BASE/internlm/initialize/launch.py"
if ! grep -q 'init_fault_tolerance' "$LP"; then
    python3 -c "
with open('$LP') as f:
    content = f.read()

# Add init_fault_tolerance function before initialize_distributed_env
ft_func = '''
def init_fault_tolerance():
    \"\"\"Initialize fault tolerance if configured.\"\"\"
    ft_config = gpc.config.get('ft', None)
    if ft_config is None or not ft_config.get('enabled', False):
        return
    try:
        from internlm.ft.manager import FTManager
        from internlm.core.context.process_group_initializer import ParallelMode
        ft_mgr = FTManager(ft_config)
        ft_mgr.initialize(
            rank=gpc.get_local_rank(ParallelMode.DATA),
            world_size=gpc.get_world_size(ParallelMode.DATA),
        )
        if gpc.is_rank_for_log():
            logger.info('Fault tolerance initialized successfully')
    except ImportError:
        if gpc.is_rank_for_log():
            logger.warning('torchft not installed, fault tolerance disabled')
    except Exception as e:
        if gpc.is_rank_for_log():
            logger.error(f'Fault tolerance init failed: {e}')


'''

# Insert before initialize_distributed_env
content = content.replace(
    '@llm_timeout(func_name=\"initialize_distributed_env\")',
    ft_func + '@llm_timeout(func_name=\"initialize_distributed_env\")'
)

# Add init_fault_tolerance() call after args_sanity_check()
content = content.replace(
    '    if args_check:\n        args_sanity_check()',
    '    if args_check:\n        args_sanity_check()\n\n    init_fault_tolerance()'
)

with open('$LP', 'w') as f:
    f.write(content)
"
    echo "Patched launch.py"
else
    echo "launch.py already patched"
fi

# 3. Create demo FT config
mkdir -p "$BASE/configs"
cat > "$BASE/configs/ft_demo.py" << 'PYEOF'
# Fault tolerance demo configuration for InternEvo + torchft
# Usage: torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
#        Then launch training with this config

ft = dict(
    enabled=True,
    lighthouse_addr="",  # Set via TORCHFT_LIGHTHOUSE env var or here
    min_replicas=1,
    replica_id="",       # Set via TORCHFT_REPLICA_ID env var or here
    heartbeat_interval_ms=100,
    timeout_sec=60,
    quorum_timeout_sec=60,
    connect_timeout_sec=60,
    use_async_quorum=True,
    init_sync=True,
    checkpoint_transport="pg",  # "pg" for Gloo-based PGTransport, "http" for HTTP
)
PYEOF
echo "Created configs/ft_demo.py"

echo "All patches applied successfully!"
