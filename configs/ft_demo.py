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
