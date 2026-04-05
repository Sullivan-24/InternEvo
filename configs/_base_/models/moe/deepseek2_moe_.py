# Copyright (c) InternLM. All rights reserved.
tokenizer_type = "internlm.data.tokenizers.LLaMAHFTokenizer"

# VOCAB_FILE = "/mnt/inspurfs/share_data/llm_data/tokenizers/deepseek/"
# ali
# VOCAB_FILE = "/cpfs01/shared/alillm2/user/jiaopenglong/tokenizers/deepseek/"
VOCAB_SIZE = 102400


HIDDEN_SIZE = 2048
NUM_ATTENTION_HEAD = 16
NUM_KV_ATTENTION_HEAD = 16

NUM_EXPERTS = 64
NUM_SELECTED_EXPERTS = 6
NUM_SHARED_EXPERTS = 2
MLP_RATIO = 5.34375
MULTIPLE_OF = 1
NUM_LAYER = 27
MoE_TYPE = "Dropless"
MoE_TYPE = "MegaBlocks"
MoE_TYPE = "GShard"
MoE_FFN_DIM = 1408
EP_SIZE = 4
EWP_SIZE = 1

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
    first_k_dense_replace=1,
    moe_layer_freq=1,
    moe_intermediate_size=MoE_FFN_DIM,
    moe_layer_kwargs=dict(),
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True,
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
    tensor=dict(size=1, mode="fsp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=True),
    expert=dict(size=EP_SIZE),
    expert_weight=dict(size=EWP_SIZE, overlap=True),
    expert_zero1=dict(size=-1),
)
