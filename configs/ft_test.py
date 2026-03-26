# Minimal InternEvo + torchft integration test config
# DP=2 (2 replica groups), PP=1, TP=1 on 2 GPUs per group
import os

JOB_NAME = "ft_test"
DO_ALERT = False

SEQ_LEN = 512
HIDDEN_SIZE = 512
NUM_ATTENTION_HEAD = 8
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 8 / 3
NUM_LAYER = 4
VOCAB_SIZE = 32000

SAVE_CKPT_FOLDER = "local:llm_ckpts"
LOAD_CKPT_FOLDER = None
CHECKPOINT_EVERY = 100

ckpt = dict(
    enable_save_ckpt=False,
    save_ckpt_folder=SAVE_CKPT_FOLDER,
    load_ckpt_info=dict(path=LOAD_CKPT_FOLDER, content=("model",), ckpt_type="internevo"),
    auto_resume=False,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=False,
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),
)

TRAIN_FOLDER = None
VALID_FOLDER = None
data = dict(
    seq_len=SEQ_LEN,
    micro_num=4,
    micro_bsz=2,
    valid_micro_num=4,
    valid_every=0,
    pack_sample_into_one=False,
    total_steps=50,
    skip_batches="",
    rampup_batch_size="",
    min_length=50,
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
)

grad_scaler = dict(
    fp16=dict(initial_scale=2**16, min_scale=1, growth_interval=1000),
    growth_factor=2,
    backoff_factor=0.5,
    max_scale=2**24,
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    overlap_sync_grad=False,
    overlap_sync_param=False,
    reduce_bucket_size=512 * 1024 * 1024,
    clip_grad_norm=1.0,
)

loss = dict(label_smoothing=0)

adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

monitor = dict(
    alert=dict(
        enable_feishu_alert=False,
        feishu_alert_address=None,
        light_monitor_address=None,
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(queue_max_length=10),
)

use_fp32_norm = False

parallel = dict(
    zero1=dict(size=1),
    tensor=dict(size=1, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=False),
    weight=dict(size=1, overlap=False),
)

model_type = "INTERNLM2_PUBLIC"
model = dict(
    num_chunks=1,
    checkpoint=0,
    dtype="torch.bfloat16",
    embed_split_hidden=True,
    num_layers=NUM_LAYER,
    hidden_size=HIDDEN_SIZE,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    num_attention_heads=NUM_ATTENTION_HEAD,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    mlp_ratio=MLP_RATIO,
    norm_type="rmsnorm",
    qk_interleaved=True,
    apply_post_layer_norm=False,
    no_bias=True,
    layer_norm_epsilon=1e-5,
    rope_base=1000000,
)

# Fault tolerance config
ft = dict(
    enabled=True,
    lighthouse_addr="",  # Set via TORCHFT_LIGHTHOUSE env var
    min_replicas=1,
    replica_id="",       # Set via env var
    heartbeat_interval_ms=100,
    timeout_sec=60,
    quorum_timeout_sec=60,
    connect_timeout_sec=60,
    use_async_quorum=True,
    init_sync=True,
    checkpoint_transport="pg",
)
