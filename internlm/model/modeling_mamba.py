# Copyright (c) InternLM. All rights reserved.

import math
from functools import partial
from typing import Optional

import torch
from mamba_ssm.modules.mamba_simple import Mamba
from torch import nn
from internlm.initialize.initialize_tensor import (
    normal_,
    uniform_,
)
# isort: off
from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.initialize.initialize_tensor import normal_
from internlm.core.context.parallel_context import IS_REPLICA_ZERO_PARALLEL
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.mha import GQA
from internlm.solver.activation_checkpoint import activation_checkpoint
from internlm.utils.logger import get_logger
from internlm.model.modules.norm import new_layer_norm
from internlm.core.naive_amp import set_output_attr_to_module
from internlm.model.modules.linear import new_linear
from internlm.model.modeling_llama import Llama2Decoder as TransformerLayer

logger = get_logger(__file__)

class MambaLayer(nn.Module):
    """
    Mamba layer. Mamba layer only use mamba module as mixer, no mlp part.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. False by default.
        layer_idx (int): The index of current layer. 0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-5,
        checkpoint: bool = False,
        layer_idx: int = 0,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        ssm_cfg=None,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx

        self.mixer = Mamba(
            d_model=hidden_size, layer_idx=layer_idx, dtype=dtype, **(ssm_cfg if ssm_cfg is not None else {})
        )
        self.dropout = nn.Dropout(drop_rate)
        self.norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False

        for param in self.mixer.parameters():
            setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

    def forward(self, hidden_states, cu_seqlens=None, inference_params=None, **kwargs):
        if self.checkpoint and self.training:
            return activation_checkpoint(self._forward, False, hidden_states, cu_seqlens, inference_params)
        else:
            return self._forward(hidden_states, cu_seqlens, inference_params)

    def _forward(self, hidden_states=None, cu_seqlens=None, inference_params=None):
        def _dropout_and_norm_attn(_hidden_states):
            _dropped = self.dropout(_hidden_states)
            _residual = _dropped
            if isinstance(self.norm, nn.LayerNorm):
                _hidden_states = self.norm(_residual)
            else:
                _hidden_states = self.norm(_residual.to(torch.float32))
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_attn(hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, inference_params)
        return hidden_states + residual

class InternevoNemotronH(nn.Module):
    """
    Mamba model.

    Args:
        num_layers (int): The number of layer. 12 by default.
        hidden_size (int): The size of hidden state. 768 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. False by default.
        checkpoint_fraction (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 1.0 by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        first (bool): Whether input embedding layer or not. True by default.
        last (bool): Whether output layer with norm and head or not. True by default.
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    False by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        is_reward (bool): Whether to use reward model. False by default. False by default.
        dropout_selective_checkpoint (bool): It can only be enabled when checkpoint is disabled. True by default.
        use_scaled_init (bool): Whether to use scaled init. True by default.
        ssm_cfg (dict): Configuration for mamba module. None by default.
    """

    def __init__(
        self,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        hidden_size: int = 768,
        vocab_size: int = 50304,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        checkpoint: bool = False,
        checkpoint_fraction: float = 1.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = True,
        last: bool = True,
        no_bias: bool = True,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        device: Optional[torch.device] = None,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        mlp_ratio: float = 5.25,
        apply_post_layer_norm=False,
        num_kv_attention_heads=8,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        ssm_cfg=None,
        out_head_init_std: float = 0.02,
        init_type: str = "normal",
        qk_interleaved=False,
        mlp_layer_fusion=False,
        enable_qkv_fusion=False,
        embedding_init_std: float = 0.02,
        attn_drop_rate: float = 0,
        use_swiglu: bool = True,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        rope_base: int = 10000,
        multiple_of: int = 256,
    ):
        super().__init__()
        if checkpoint_fraction <= 0:
            checkpoint = False
        if not checkpoint:
            checkpoint_fraction = 0
        
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


        self.layers = []
        for lid in range(num_layers):
            current_layer_idx = lid + start_layer_idx
            if (current_layer_idx + 1) % gpc.config.ATTN_FREQ == 0: # transformer layers
                self.layers.append(
                    TransformerLayer(
                        hidden_size=hidden_size,
                        num_attention_heads=num_attention_heads,
                        num_kv_attention_heads=num_kv_attention_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop_rate=attn_drop_rate,
                        drop_rate=drop_rate,
                        dtype=dtype,
                        layer_norm_epsilon=layer_norm_epsilon,
                        checkpoint=lid < checkpoint_layer_num,
                        layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
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
                        mlp_layer_fusion=mlp_layer_fusion,
                        multiple_of=multiple_of,
                        enable_qkv_fusion=enable_qkv_fusion,
                    )
                )
            else:
                self.layers.append(
                    MambaLayer(
                        hidden_size=hidden_size,
                        drop_rate=drop_rate,
                        dtype=dtype,
                        layer_norm_epsilon=layer_norm_epsilon,
                        checkpoint=lid < checkpoint_layer_num,
                        layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                        residual_in_fp32=residual_in_fp32,
                        norm_type=norm_type,
                        dropout_selective_checkpoint=dropout_selective_checkpoint,
                        use_scaled_init=use_scaled_init,
                        ssm_cfg=ssm_cfg,
                    )
                )
        self.layers = nn.ModuleList(self.layers)
        
        if last:
            if not apply_post_layer_norm:
                self.norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

            self.head = new_linear(
                name="output",
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
                is_reward=is_reward,
                weight_scale=embed_grad_scale,
            )
            set_output_attr_to_module(self.head)
            for _, param in self.head.named_parameters():
                if init_type == "normal":
                    normal_(std=out_head_init_std)(param)
                else:
                    uniform_(std=out_head_init_std)(param)
    
    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        if hasattr(self, "tok_embeddings") and input_ids is not None:
            hidden_states = self.tok_embeddings(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        for _, block in enumerate(self.layers):
            hidden_states = block(
                hidden_states,
                **kwargs,
            )

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.float())

        if hasattr(self, "head"):
            hidden_states = self.head(hidden_states)

        return hidden_states