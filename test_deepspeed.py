import os
import torch
import logging
from profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from internlm.model.modeling_llama import Llama2
from internlm.initialize import initialize_distributed_env
from internlm.core.context import Config
from internlm.core.context import global_context as gpc
from tools.load_internlm2_model import initialize_internlm_model, get_tp_world_size, use_torchrun_starter, initialize_model_and_parallel_communicator

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

# 基于 configs/7B_llama2.py 的配置
JOB_NAME = "7b_llama2_train"
model_type = "LLAMA2"
DO_ALERT = False

VOCAB_SIZE = 32000
SEQ_LEN = 12*1024
HIDDEN_SIZE = 5120
NUM_ATTENTION_HEAD = 40
NUM_KV_ATTENTION_HEAD = 40
MLP_RATIO = 2.7
NUM_LAYER = 40

MODEL_ONLY_FOLDER = "local:llm_ckpts/xxxx"
# Ckpt folder format:
# fs: 'local:/mnt/nfs/XXX'
SAVE_CKPT_FOLDER = "local:llm_ckpts"
LOAD_CKPT_FOLDER = "local:llm_ckpts/49"
CHECKPOINT_EVERY = 50
# 创建完整配置字典（包含必要的 parallel 配置）
config_dict = dict(
    JOB_NAME="flops_profiling",
    model_type="LLAMA2",
    ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    auto_resume=False,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
    ),
    model = dict(
    checkpoint=False,
    num_chunks=1,
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    no_bias=True,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    use_flash_attn=True,
    # Whether the odd and even columns of the query and key in the model are normally interleaved.
    # If it's True, the model's odd and even columns are normally ordered; if it's False,
    # it means that the model has prematurely concatenated all odd columns and even columns in front
    # and back, in order to improve the RoPE's computational efficiency.
    # Example:
    # qk_interleaved = True: q[-1] = [q1,q2,q3,q4,q5,q6,...], k[-1] = [k1,k2,k3,k4,k5,k6,...]
    # qk_interleaved = False: q[-1] = [q1,q3,q5,...,q2,q4,q6,...], k[-1] = [k1,k3,k5,...,k2,k4,k6,...]
    qk_interleaved=False,
    mlp_layer_fusion=True,
    enable_qkv_fusion=True,
    ),
    parallel=dict(
        zero1=dict(size=-1),
        tensor=dict(size=1, mode="mtp"),  # 单GPU，不使用张量并行
        pipeline=dict(size=1, interleaved_overlap=False),
        weight=dict(size=1, overlap=False),
    ),
    data=dict(
        seq_len=SEQ_LEN,
        micro_num=1,
        micro_bsz=1,
        pack_sample_into_one=True,
        min_length=0,
        total_steps=1,
        valid_micro_num=1,
        valid_every=0,
        use_packed_dataset=True,
        rampup_batch_size="",
    ),
    grad_scaler=dict(
        fp16=dict(
            initial_scale=2**16,
            min_scale=1,
            growth_interval=1000,
        ),
        growth_factor=2,
        backoff_factor=0.5,
        max_scale=2**24,
        hysteresis=2,
    ),
    hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
),
    loss = dict(
    label_smoothing=0,
    op_type="flash_vocab_parallel",
),
    adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
),
    lr_scheduler = dict(
    total_steps=1,
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
),
beta2_scheduler = dict(
    init_beta2=0.95,
    c=0,
    cur_iter=-1,
),
use_fp32_norm = False,
cudnn_deterministic = False,
cudnn_benchmark = False,
)

def llama2_input_constructor(batch_size, seq_len, vocab_size, device):
    """构造 Llama2 模型的输入"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # 添加必要的 attention mask
    cu_seqlens = torch.arange(
        0, (batch_size ) * seq_len + 1, step=seq_len, dtype=torch.int32, device=device
    )
    print(f"input_ids shape: {input_ids}, cu_seqlens shape: {cu_seqlens}")
    return dict(input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=seq_len, cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len)

def initialize_internlm_model(
    config: dict,
    del_model_prefix: bool = False,
    param_dtype: torch.dtype = torch.bfloat16,
    training: bool = False,
    seed: int = 1024,
) -> torch.nn.Module:
    """Initialize internlm model.

    Args:
        model_type (str): The types of models supported by internlm framework, such as "INTERNLM".
        ckpt_dir (str): Directory where model checkpoints are stored. Its format needs to be like this:
            (a) local path, such as: "local:{your local path}";
            (b) boto3 path, such as: "boto3:s3://{bucket name}.{ip}/{your ceph path}".
        model_config (Optional[Union[Dict, str]], optional): Configuration of models. Defaults to None.
        del_model_prefix (bool, optional):  Whether to remove the "model." string in the key in state_dict.
            Defaults to False.
        param_dtype (torch.dtype, optional): The dtype of the model at inference time. This value can be a string.
            Use "torch.tf32" when you want to use tf32 to do the inference. Defaults to torch.bfloat16.
        training (bool, optional): model.train() or model.eval(). Defaults to False.
        seed (int, optional): Defaults to 1024.
    """
    
    model_config = config.model

    if gpc.is_rank_for_log():
        logger.info(f"tp world size: {get_tp_world_size()}.")

    if isinstance(param_dtype, str):
        try:
            param_dtype = eval(param_dtype)  # pylint: disable=W0123
        finally:
            pass
    # if param_dtype == "torch.tf32":
    #     param_dtype = torch.float32
    #     torch.backends.cudnn.allow_tf32 = True
    #     torch.backends.cuda.matmul.allow_tf32 = True

    # if not isinstance(param_dtype, torch.dtype):
    #     raise ValueError("Parameter ``param_dtype`` is not right.")

    # try:
    #     init_storage_manager(False, None, None)
    # except AssertionError:
    #     pass
    # except Exception as e:
    #     raise e

    # model_config["dtype"] = param_dtype
    model_config["parallel_output"] = False
    # FIXME: fix it.
    if gpc.is_rank_for_log():
        logger.info(f"model_config: {model_config}.")
    # import pdb; pdb.set_trace()
    initialize_distributed_env(
        config=config,
        launcher="torch" if use_torchrun_starter() else "slurm",
        seed=seed,
        master_port=29500,
        args_check=True,
    )
    # Directly get the origin model without NativeAMP wrapper.
    model, _ = initialize_model_and_parallel_communicator()
    model = model.model

    # state_dict = merge_pp_within_tp(ckpt_dir, del_model_prefix=del_model_prefix)

    # load_info = model.load_state_dict(state_dict, strict=False)
    # logger.info(f"Rank:{gpc.get_local_rank(ParallelMode.TENSOR)}. Load info: {load_info}.")

    model.to(param_dtype)
    if training:
        model.train()
    else:
        model.eval()
    model.cuda()
    torch.distributed.barrier()
    return model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    # 转换为百万 (M) 单位
    print(f"Total Parameters (M): {total_params/1e6:.2f} M")
    
    return total_params, trainable_params

if __name__ == "__main__":
    # 设置环境变量用于单GPU运行
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    
    # 初始化分布式环境
    config = Config(config_dict)
    model = initialize_internlm_model(config)
    # initialize_distributed_env(config=config, launcher="torch", master_port=29500, args_check=False)
    
    # 使用 GPU 0
    with get_accelerator().device(0):
        print("初始化 Llama2 模型...")
        model = model.to(get_accelerator().device_name(0))

        # 测试参数
        batch_size = 1
        seq_len = 2*1024  # 使用较短的序列长度进行测试

        print(f"\n开始 FLOPS 性能分析...")
        print(f"配置: batch_size={batch_size}, seq_len={seq_len}")
        print(f"模型: {NUM_LAYER} 层, hidden_size={HIDDEN_SIZE}, num_heads={NUM_ATTENTION_HEAD}\n")

        # 执行 FLOPS 分析
        flops, macs, params = get_model_profile(
            model,
            kwargs=llama2_input_constructor(
                batch_size, 
                seq_len, 
                VOCAB_SIZE, 
                get_accelerator().device_name(0)
            ),
            print_profile=True,
            detailed=True,
            module_depth=-1,  # 打印所有层级
            top_modules=3,    # 显示前3个最耗资源的模块
            warm_up=1,
            as_string=True,
            output_file=None,  # 可以指定输出文件路径
        )

        print("\n" + "="*80)
        print("总结:")
        print(f"  总 FLOPS: {flops}")
        print(f"  总 MACs: {macs}")
        print(f"  总参数量: {params}")
        print("="*80)
        
        count_parameters(model)