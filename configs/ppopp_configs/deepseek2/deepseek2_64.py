from configs.ppopp_configs.base import *

if "JOB_NAME" in os.environ:
    JOB_NAME = os.environ["JOB_NAME"]
else:
    JOB_NAME = os.path.basename(__file__).split(".py")[0]

model_type = "DEEPSEEK2_MoE"

HIDDEN_SIZE = 2048
NUM_ATTENTION_HEAD = 16
NUM_KV_ATTENTION_HEAD = 16

NUM_EXPERTS = 32
NUM_SELECTED_EXPERTS = 6
NUM_SHARED_EXPERTS = 2
MLP_RATIO = 5.34375
MULTIPLE_OF = 1
NUM_LAYER = 32
MoE_TYPE = "Dropless"
MoE_FFN_DIM = 1408
EP_SIZE = 2
EWP_SIZE = 1
# VOCAB_SIZE = 102400
VOCAB_SIZE = 128*1024
PP_SIZE = 8
TP_SIZE = 1
MICRO_NUM = PP_SIZE * 4
FIRST_K_DENSE_REPLACE = 3
PP_MODE = "1f1b"
CHUNK_NUM = 1

if SCHEDULE == 0:
    num_microbatches, pp_size, stage_placement, scheduler_type, \
    split_backward, unified_scheduler, comm_graph, first_stage, \
    last_stage, Devices_containing_last_stage, recomp_stages = generate_()

PP_MODE, CHUNK_NUM = set_pp_mode(JOB_NAME=JOB_NAME, pp_size=PP_SIZE, layer_num=NUM_LAYER, chunk_num=CHUNK_NUM, seq_len=SEQ_LEN, adap_partition=ALPA)

VOCAB_FILE = "/mnt/inspurfs/share_data/llm_data/tokenizers/deepseek/"
# ali
VOCAB_FILE = "/cpfs01/shared/alillm2/user/jiaopenglong/tokenizers/deepseek/"

# Dataset path
TRAIN_FOLDER: str = None
VALID_FOLDER: str = None

GRADIENT_ACCUMULATION = MICRO_NUM

ckpt = dict(
    enable_save_ckpt=False,
    save_ckpt_folder="boto3:s3://checkpoints_ssd_02.10.135.7.249/{JOB_NAME}/",
    checkpoint_every=1000,
    oss_snapshot_freq=1000 // 4,
    auto_resume=False,
    # load_ckpt_info=dict(
    #     path=None,
    #     content=["model", "sampler", "optimizer"],
    #     ckpt_type="internevo",
    # ),
    async_upload=True,
    async_upload_tmp_folder=f"/dev/shm/internlm_tmp_ckpt_{JOB_NAME}/",
    stop_file_path=f"llm_alter/{JOB_NAME}.log",
    evaluator_cfg=dict(
        type="internlm.eval.evaluator.OpenCompassEvaluator",
        evaluation_step=-6,
        evaluate_last=True,
    ),
)

SAVE_CKPT_FOLDER = "local:llm_ckpts"
CHECKPOINT_EVERY = 50
ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    # load_ckpt_folder= dict(path=MODEL_ONLY_FOLDER, content=["model"], ckpt_type="normal"),
    # load_ckpt_folder="local:llm_ckpts/",
    # 'load_ckpt_info' setting guide:
    # 1. the 'path' indicate ckpt path,
    # 2. the 'content‘ means what states will be loaded, support: "model", "sampler", "optimizer", "scheduler", "all"
    # 3. the ’ckpt_type‘ means the type of checkpoint to be loaded, support: "internevo", "hf", or other custom-defined
    # load function such as "llama"
    # load_ckpt_info=dict(path=MODEL_ONLY_FOLDER, content=("model",), ckpt_type="internevo"),
    # 'auto_resume' is designed to automatically load the latest checkpoint from 'save_ckpt_folder' when encountering
    # training interruptions/hangs caused by hardware failures, using a scheduling system (such as k8s/slurm)
    # with an automatic restart mechanism upon training reboot.
    # Please be aware that if `auto_resume` is not set (its default value is True), it will not load the checkpoint
    # path specified in `load_ckpt_info` by default.
    # If you want to initialize your model weights from another model, you must set `auto_resume` to False.
    # If you want to train from scratch, please set `auto_resume` to False and 'load_ckpt_info' to None.
    auto_resume=True,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
)

# Whether to enble `spawn` mode for pytorch multiprocessing.
# If set to False, will use `fork` mode during training.
MP_SPAWN = False

use_fp32_norm = True
parallel_check_freq = 100
data_check_counts = 2
compute_subset_loss = False

data = dict(
    type="tokenized",
    vocab_file=VOCAB_FILE,
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    micro_num=MICRO_NUM,
    valid_micro_num=1,
    gradient_accumulation=GRADIENT_ACCUMULATION,
    micro_bsz=1,
    seq_len=SEQ_LEN,
    max_length_per_sample=4096,
    min_length=50,
    valid_packed_length=4096,
    valid_pack_mode=None,
    valid_drop_last=False,
    pack_sample_into_one=False,
    total_steps=20,
    valid_every=0,
    num_worker=4,
    rampup_batch_size="",
    text_field="content",
    val_text_field="content",
    dataset_weights={"code": 1.0, "en": 2.0},
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
    skip_batches="",
)

grad_scaler = dict(
    fp16=dict(
        initial_scale=2**14,
        min_scale=1,
        growth_interval=1000,
    ),
    growth_factor=2,
    backoff_factor=0.5,
    max_scale=2**24,
    hysteresis=2,
)

loss = dict(
    moe_z_loss_coeff=1e-3,
    label_smoothing=0.0
)

adam = dict(
    lr=4e-5,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,
    warmup_ratio=0.025,
    eta_min=4e-6,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)


model = dict(
    checkpoint=0,
    num_chunks=1,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_attention_heads=NUM_ATTENTION_HEAD,
    num_layers=NUM_LAYER,
    mlp_ratio=MLP_RATIO,
    multiple_of=MULTIPLE_OF,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    attention_type="GQA",
    no_bias=True,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-6,
    qk_interleaved=False,
    # rope settings
    rope_base=10000,
    # moe settings
    num_experts=NUM_EXPERTS,
    top_k=NUM_SELECTED_EXPERTS,
    num_shared_experts=NUM_SHARED_EXPERTS,
    moe_type=MoE_TYPE,
    residual_type="deepseek",
    first_k_dense_replace=FIRST_K_DENSE_REPLACE,
    moe_layer_freq=1,
    moe_intermediate_size=MoE_FFN_DIM,
    moe_layer_kwargs=dict(),
    q_lora_rank=1536,
    kv_lora_rank=512,
    v_head_dim=128,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=OVERLAP_SYNC_GRAD,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
    # whether use new optm
    use_split_tensor_optim=False,
    # when use split tensor optm
    # Perform all gather with a set of parameters of all_gather_size
    all_gather_size=512 * 1024 * 1024,
)

parallel = dict(
    zero1=dict(size=-1, fsdp=False),
    tensor=dict(size=TP_SIZE, mode="fsp"),
    pipeline=dict(size=PP_SIZE, interleaved_overlap=True, mode=PP_MODE),
    weight=dict(size=1, overlap=True),
    expert=dict(size=EP_SIZE),
    expert_weight=dict(size=EWP_SIZE, overlap=True),
    expert_zero1=dict(size=-1),
)