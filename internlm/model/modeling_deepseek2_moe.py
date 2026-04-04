# Copyright (c) InternLM. All rights reserved.
import math
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.initialize.initialize_tensor import (
    normal_,
    scaled_init_method_normal,
    scaled_init_method_uniform,
    uniform_,
)
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import new_linear
from internlm.model.modules.mha import GQA
from internlm.model.modules.mlp import new_feed_forward
from internlm.model.modules.norm import new_layer_norm
from internlm.model.moe import MoE
from internlm.model.ops.attention import SelfAttention
from internlm.model.utils import (
    convert_attn_args_to_kwargs,
    convert_attn_kwargs_to_args,
)
from internlm.solver.activation_checkpoint import activation_checkpoint
from internlm.utils.logger import get_logger

HAS_DLBLAS = False
if torch.cuda.is_available():
    try:
        import dlblas  # pyright: ignore

        HAS_DLBLAS = True
    except ImportError:
        pass


logger = get_logger(__file__)


def _convert_cu_seqlens_for_qksplited(kwargs: Dict):
    cu_seqlens = kwargs.pop("cu_seqlens", None)
    max_seqlen = kwargs.pop("max_seqlen", None)

    if cu_seqlens is not None:
        kwargs["cu_seqlens_q"] = cu_seqlens
        kwargs["cu_seqlens_k"] = cu_seqlens
        kwargs["max_seqlen_q"] = max_seqlen
        kwargs["max_seqlen_k"] = max_seqlen

    return kwargs


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV2RotaryEmbedding(nn.Module):
    # pylint: disable=missing-class-docstring
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):
    # pylint: disable=missing-class-docstring
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        sequence_process_group (torch.distributed.ProcessGroup): The process group for attention calculation.
        bias (boolean): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                        output projection. True by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        use_flash_attn (boolean): Whether to use flash attention or not.If False, vanilla attention module will be used.
                                    False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        use_flash_attn (bool): Whether to use flash-attn. True by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        tp_mode (str): The string value of tensor parallel mode, should be in ["mtp", "msp", "fsp", "isp"],
                       "mtp" by default.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        v_head_dim: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rope_base: int = 10000,
        rope_scaling_factor: float = 1.0,  # pylint: disable=unused-argument
        rotary_emb_scale_base: int = 0,  # pylint: disable=unused-argument
        norm_type: str = "rmsnorm",
        qk_interleaved: Optional[bool] = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_position_embeddings: int = 2048,
        rope_scaling_kwargs: Dict = None,
        fused_rope: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert self.embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"

        self.head_dim = self.embed_dim // self.num_heads

        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = min(self.head_dim, qk_rope_head_dim)
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.interleaved = qk_interleaved
        self.dtype = dtype

        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.kv_a_layernorm = new_layer_norm(norm_type, self.kv_lora_rank, eps=1e-6)

        self.yarn_embed = DeepseekV2YarnRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
            **rope_scaling_kwargs,
        )

        # TODO: get the column/row info and the dim
        if self.q_lora_rank is None:
            self.q_proj = new_linear(
                "wqkv",
                embed_dim,
                self.num_heads * self.q_head_dim,
                bias=bias,
                **factory_kwargs,
            )
        else:
            self.q_a_layernorm = new_layer_norm(norm_type, self.q_lora_rank, eps=1e-6)
            self.q_a_proj = new_linear(
                "wqkv",
                embed_dim,
                self.q_lora_rank,
                bias=bias,
                **factory_kwargs,
            )
            self.q_b_proj = new_linear(
                "wqkv",
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=bias,
                **factory_kwargs,
            )
        self.kv_a_proj_with_mqa = new_linear(
            "wqkv",
            embed_dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
            **factory_kwargs,
        )
        self.kv_b_proj = new_linear(
            "wqkv",
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=bias,
            **factory_kwargs,
        )
        softmax_scale = self.q_head_dim ** (-0.5)

        # copied from deepseekv2
        if rope_scaling_kwargs is not None:
            mscale_all_dim = rope_scaling_kwargs.get("mscale_all_dim", 0)
            scaling_factor = rope_scaling_kwargs["scaling_factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                softmax_scale = softmax_scale * mscale * mscale

        self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)

        self.inner_cross_attn_causal = causal
        self.inner_cross_attn_softmax_scale = softmax_scale
        self.inner_cross_attn_dropout = dropout

        # output projection always have the bias (for now)
        self.wo = new_linear(
            "wo",
            self.num_heads * self.v_head_dim,
            embed_dim,
            bias=bias,
            **factory_kwargs,
        )
        self.fused_rope = HAS_DLBLAS and fused_rope
        if self.fused_rope:
            logger.info("Using fused_rope!")
        else:
            logger.info(f"Not using fused_rope! HAS_DLBLAS:{HAS_DLBLAS}")

    def register_checkpoint_compatibility_hooks(
        self, pre_load_hook: Optional[Callable] = None, pre_save_hook: Optional[Callable] = None
    ):
        # Here we explicitly expose the checkpoint compatibility interface of the module,
        # hoping that model developers will make good use of it when adapting.
        # Is this interface already meeting all reasonable requirements?
        self._register_load_state_dict_pre_hook(pre_load_hook, with_module=True)
        self._register_state_dict_hook(pre_save_hook)

    def forward(self, x, inference_params=None, **kwargs):
        if inference_params is None:
            return self._training(x=x, **kwargs)
        else:
            return self._inference(x=x, inference_params=inference_params, **kwargs)

    def _training(self, x, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
        """
        bsz, q_len, _ = x.size()

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )

        # rotary embedding
        cos, sin = self.yarn_embed(q, seq_len=q_len)

        if self.fused_rope:
            position_ids = kwargs.pop("indexes", torch.arange(0, q_len)).to(q.device).unsqueeze(0)
            q, kv = dlblas.partial_rotary_emb(
                q.contiguous(),
                k_pe.contiguous(),
                kv.contiguous(),
                cos[position_ids].contiguous(),
                sin[position_ids].contiguous(),
            )
        else:
            q = q.transpose(1, 2)
            k_pe = k_pe.transpose(1, 2)
            kv = kv.transpose(1, 2)

            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            position_ids = kwargs.pop("indexes", torch.arange(0, q_len)).to(q.device).unsqueeze(0)
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

            q = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            q[:, :, :, : self.qk_nope_head_dim] = q_nope
            q[:, :, :, self.qk_nope_head_dim :] = q_pe

            k = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            k[:, :, :, : self.qk_nope_head_dim] = k_nope
            k[:, :, :, self.qk_nope_head_dim :] = k_pe

            if self.q_head_dim != self.v_head_dim:
                v = F.pad(v, [0, self.q_head_dim - self.v_head_dim])

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)

        # self attention
        kwargs = _convert_cu_seqlens_for_qksplited(kwargs)

        context = self.inner_attn(q, kv, **kwargs)

        if self.q_head_dim != self.v_head_dim:
            context = context[:, :, :, : self.v_head_dim]

        return self.wo(rearrange(context, "b s h d -> b s (h d)"))

    def _inference(self, x, inference_params, **kwargs):  # pylint: disable=W0613
        raise RuntimeError("Not support this right now")


class DeepSeek2MoEDecoder(nn.Module):
    """
    DeepSeek2 MoE Decoder layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_idx (int): The index of current layer. 0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        multiple_of (int): The value to make SwiGLU hidden layer size multiple of large power of 2.
        num_experts (int): The number of experts. <=1 means dense, >1 means MoE. 1 by default.
        top_k (int): The number of selected experts, should smaller than num_experts.
        num_shared_experts (int): The number of shard experts, alwarys placed in every device.
        residual_type (int): determine how to aggregation the outputs of selected experts and shared experts.
        moe_layer_kwargs (int): cumstom kwargs used in different moe implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        v_head_dim: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        attention_type: str = "GQA",
        num_kv_attention_heads: int = 8,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-6,
        checkpoint: bool = False,
        layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        apply_post_layer_norm: bool = False,
        fused_dropout_add_ln: bool = True,
        no_bias: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        init_type: str = "normal",
        rope_base: int = 10000,
        rope_scaling_factor: float = 1.0,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        fused_rope: bool = False,
        # moe-specific
        num_experts: int = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        residual_type: str = "deepseek",
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        moe_intermediate_size: int = 0,
        moe_layer_kwargs: dict = None,
        rope_scaling_kwargs: dict = None,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx
        self.prenorm = not apply_post_layer_norm
        assert not fused_dropout_add_ln, "dropout_add_layer_norm can not be used here"
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.attn_wqkv_init_std = attn_wqkv_init_std
        self.attn_other_init_std = attn_other_init_std
        self.ffn_uplayer_init_std = ffn_uplayer_init_std
        self.ffn_other_init_std = ffn_other_init_std

        self.max_position_embeddings = max_position_embeddings
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope

        if attention_type == "GQA":
            self.attention = GQA(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_kv_attention_heads,
                dropout=attn_drop_rate,
                max_position_embeddings=max_position_embeddings,
                causal=True,
                layer_idx=layer_idx,
                use_dynamic_ntk_rope=use_dynamic_ntk_rope,
                rotary_emb_scale_base=0,
                device=device,
                dtype=dtype,
                qk_interleaved=qk_interleaved,
                bias=not no_bias,
                rope_base=rope_base,
                enable_qkv_fusion=True,
            )
        elif attention_type == "MLA":
            self.attention = MLA(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                v_head_dim=v_head_dim,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                dropout=attn_drop_rate,
                causal=True,
                layer_idx=layer_idx,
                use_dynamic_ntk_rope=use_dynamic_ntk_rope,
                rope_base=rope_base,
                rope_scaling_factor=rope_scaling_factor,
                rotary_emb_scale_base=0,
                norm_type=norm_type,
                qk_interleaved=qk_interleaved,
                device=device,
                dtype=dtype,
                bias=not no_bias,
                fused_rope=fused_rope,
                max_position_embeddings=max_position_embeddings,
                rope_scaling_kwargs=rope_scaling_kwargs,
            )
        else:
            # TODO: to support MLA
            raise NotImplementedError(f"Attention type {attention_type} is not supported yet.")

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.attention_norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
        self.ffn_norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

        self.num_experts = num_experts
        if num_experts <= 1 or layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:  # dense, not MoE
            self.feed_forward = new_feed_forward(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                # TODO: to support more activation functions
                activation_type="swiglu" if use_swiglu else "swiglu",
            )
            self.num_experts = 1
        else:
            if moe_intermediate_size > 0:
                hidden_features = moe_intermediate_size
            else:
                hidden_features = int(hidden_size * mlp_ratio)
            self.feed_forward = MoE(
                hidden_size,
                hidden_features,
                out_features=hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                num_shared_experts=num_shared_experts,
                residual_type=residual_type,
                moe_layer_kwargs=moe_layer_kwargs,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                # TODO: to support more activation functions
                activation_type="swiglu" if use_swiglu else "swiglu",
            )

        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False

        if init_type == "normal":
            self.init_func = normal_
            self.scaled_init_func = scaled_init_method_normal
        else:
            self.init_func = uniform_
            self.scaled_init_func = scaled_init_method_uniform

        self.reset_parameters()  # TODO: check this should be changed when moe is added

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.attention.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "wq" in name or "wk" in name or "wv" in name:
                    self.init_func(std=self.attn_wqkv_init_std)(param.data)
                elif self.use_scaled_init:  # wo
                    self.scaled_init_func(sigma=self.attn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                else:
                    self.init_func(std=self.attn_other_init_std)(param.data)

            for name, param in self.feed_forward.named_parameters():
                if "gate" in name:
                    normal_(std=0.01)(param.data)
                elif self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        # candidate: w1, w3, fused_w1_w3
                        self.init_func(
                            std=self.ffn_uplayer_init_std if "w1" in name or "w3" in name else self.ffn_other_init_std
                        )(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        self.init_func(std=self.ffn_uplayer_init_std if "fc1" in name else self.ffn_other_init_std)(
                            param.data
                        )

    def forward(self, hidden_states, residual=None, **kwargs):
        if self.checkpoint and self.training:
            # NOTICE: activation_checkpiont do not support kwargs when use_reentrant = True.
            args = convert_attn_kwargs_to_args(kwargs)
            return activation_checkpoint(self._forward, False, hidden_states, residual, *args)
        else:
            return self._forward(hidden_states, residual, **kwargs)

    def _forward(self, hidden_states, residual, *args, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """
        final_hidden_states = None
        if self.prenorm:

            def _dropout_and_norm_attn(_residual, _hidden_states):
                _dropped = self.dropout1(_hidden_states)
                _residual = (_dropped + _residual) if _residual is not None else _dropped
                _hidden_states = self.attention_norm(_residual.to(dtype=self.attention_norm.weight.dtype))

                return _residual, _hidden_states

            if self.dropout_selective_checkpoint:
                residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, residual, hidden_states)
            else:
                residual, hidden_states = _dropout_and_norm_attn(residual, hidden_states)

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            attn_kwargs = convert_attn_args_to_kwargs(args, kwargs)
            hidden_states = self.attention(hidden_states, **attn_kwargs)

            if not isinstance(self.feed_forward, nn.Identity):
                if not self.fused_dropout_add_ln:

                    def _dropout_and_norm_ffn(_residual, _hidden_states):
                        _dropped = self.dropout2(_hidden_states)
                        _residual = (_dropped + _residual) if _residual is not None else _dropped
                        _hidden_states = self.ffn_norm(_residual.to(self.ffn_norm.weight.dtype))

                        return _residual, _hidden_states

                    if self.dropout_selective_checkpoint:
                        residual, hidden_states = activation_checkpoint(
                            _dropout_and_norm_ffn, False, residual, hidden_states
                        )
                    else:
                        residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                if self.num_experts <= 1:
                    hidden_states = self.feed_forward(hidden_states)
                    moe_loss = None
                    moe_z_loss = None
                else:
                    hidden_states, moe_loss, moe_z_loss = self.feed_forward(hidden_states)

            final_hidden_states = hidden_states + residual

        else:
            raise NotImplementedError("Post-norm is not supported yet.")

        return final_hidden_states, moe_loss, moe_z_loss


class DeepSeek2MoE(nn.Module):
    """
    DeepSeek2 MoE model.

    Args:
        num_layers (int): The number of layer. 12 by default.
        hidden_size (int): The size of hidden state. 768 by default.
        num_attention_heads (int): The number of attention head. 12 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        checkpoint_fraction (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 1.0 by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-6 by default.
        first (bool): Whether input embedding layer or not. False by default.
        last (bool): Whether output embedding layer or not. False by default.
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    True by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
        use_flash_attn (bool): Whether to use flash-attn. True by default.
        embedding_init_std (float): std used to init embedding weight. 0.02 by default,
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        out_head_init_std (float): std used to init output lmhead weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        extra_pred_tokens (int): The number of extra output head for multi-token-prediction. 0 by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        multiple_of (int): The value to make SwiGLU hidden layer size multiple of large power of 2.
        moe_type (str): determine which moe impl will be used, default is GShardMoE
        num_experts (int): The number of experts. <=1 means dense, >1 means MoE. 1 by default.
        top_k (int): The number of selected experts, should smaller than num_experts.
        num_shared_experts (int): The number of shard experts, alwarys placed in every device.
        residual_type (int): determine how to aggregation the outputs of selected experts and shared experts.
        moe_layer_kwargs (int): cumstom kwargs used in different moe implementation.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        v_head_dim: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        attention_type: str = "GQA",
        num_kv_attention_heads: int = 8,
        vocab_size: int = 50304,
        mlp_ratio: float = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        checkpoint: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        device: Optional[torch.device] = None,
        apply_post_layer_norm=False,
        no_bias=False,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        embedding_init_std: float = 0.02,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        out_head_init_std: float = 0.02,
        init_type: str = "normal",
        extra_pred_tokens: int = 0,
        rope_base: int = 10000,
        rope_scaling_factor: float = 1.0,
        norm_head: bool = False,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        fused_rope: bool = False,
        # moe-specific
        moe_type: str = None,  # pylint: disable=W0613
        num_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 0,
        residual_type: str = "deepseek",
        first_k_dense_replace: int = 8,
        moe_layer_freq: int = 1,
        moe_intermediate_size: int = 0,
        moe_layer_kwargs: dict = None,
        rope_scaling_kwargs: dict = None,
    ):
        super().__init__()

        checkpoint_layer_num = int(num_layers * checkpoint)
        self.embed_grad_scale = embed_grad_scale
        self.parallel_output = parallel_output

        if first:
            self.tok_embeddings = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)

            for _, param in self.tok_embeddings.named_parameters():
                if init_type == "normal":
                    normal_(std=embedding_init_std)(param)
                else:
                    uniform_(std=embedding_init_std)(param)

        self.layers = nn.ModuleList(
            [
                DeepSeek2MoEDecoder(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    q_lora_rank=q_lora_rank,
                    kv_lora_rank=kv_lora_rank,
                    v_head_dim=v_head_dim,
                    qk_nope_head_dim=qk_nope_head_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    attention_type=attention_type,
                    num_kv_attention_heads=num_kv_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    max_position_embeddings=max_position_embeddings,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    use_dynamic_ntk_rope=use_dynamic_ntk_rope,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    apply_post_layer_norm=apply_post_layer_norm,
                    fused_dropout_add_ln=False,
                    no_bias=no_bias,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    qk_interleaved=qk_interleaved,
                    attn_wqkv_init_std=attn_wqkv_init_std,
                    attn_other_init_std=attn_other_init_std,
                    ffn_uplayer_init_std=ffn_uplayer_init_std,
                    ffn_other_init_std=ffn_other_init_std,
                    init_type=init_type,
                    rope_base=rope_base,
                    rope_scaling_factor=rope_scaling_factor,
                    mlp_layer_fusion=mlp_layer_fusion,
                    multiple_of=multiple_of,
                    fused_rope=fused_rope,
                    # moe-specific
                    num_experts=num_experts,
                    top_k=top_k,
                    num_shared_experts=num_shared_experts,
                    residual_type=residual_type,
                    first_k_dense_replace=first_k_dense_replace,
                    moe_layer_freq=moe_layer_freq,
                    moe_intermediate_size=moe_intermediate_size,
                    moe_layer_kwargs=moe_layer_kwargs,
                    rope_scaling_kwargs=rope_scaling_kwargs,
                )
                for lid in range(num_layers)
            ]
        )

        if last:
            if not apply_post_layer_norm:
                self.norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

            self.output = new_linear(
                name="output",
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
                is_reward=is_reward,
                weight_scale=embed_grad_scale,
                norm_head=norm_head,
            )
            for _, param in self.output.named_parameters():
                if init_type == "normal":
                    normal_(std=out_head_init_std)(param)
                else:
                    uniform_(std=out_head_init_std)(param)

            if extra_pred_tokens > 0:
                self.extra_pred_tokens = extra_pred_tokens
                assert not is_reward, "extra_pred_tokens > 0 means using multi token prediction, not implement for RLHF"
                self.extra_outputs = nn.ModuleList(
                    [
                        new_linear(
                            name="output",
                            in_features=hidden_size,
                            out_features=vocab_size,
                            bias=False,
                            device=device,
                            dtype=dtype,
                            is_reward=is_reward,
                            weight_scale=embed_grad_scale,
                            norm_head=norm_head,
                        )
                        for _ in range(self.extra_pred_tokens)
                    ]
                )
                for _, param in self.extra_outputs.named_parameters():
                    if init_type == "normal":
                        normal_(std=out_head_init_std)(param)
                    else:
                        uniform_(std=out_head_init_std)(param)

    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "tok_embeddings") and input_ids is not None:
            hidden_states = self.tok_embeddings(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        moe_losses = []
        moe_z_losses = []
        for _, block in enumerate(self.layers):
            hidden_states, moe_loss, moe_z_loss = block(hidden_states, residual=None, **kwargs)
            if moe_loss is not None:
                moe_losses.append(moe_loss)
            if moe_z_loss is not None:
                moe_z_losses.append(moe_z_loss)

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.float())
        if hasattr(self, "extra_pred_tokens") and self.extra_pred_tokens > 0:
            extra_hidden_states_list = [self.extra_outputs[i](hidden_states) for i in range(self.extra_pred_tokens)]
        else:
            extra_hidden_states_list = None
        if hasattr(self, "output"):
            hidden_states = self.output(hidden_states)

        if extra_hidden_states_list is not None:
            return (hidden_states, moe_losses, moe_z_losses, extra_hidden_states_list)

        return hidden_states, moe_losses, moe_z_losses
