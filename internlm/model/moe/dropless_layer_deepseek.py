"""
The file has been adapted from the following files:
https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py
 Git commit hash: f3943cf9109226ed3ecf2d5dbb639a11cd925555
 We retain the following license from the original files:
"""
import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.modules.mlp import new_feed_forward
from internlm.utils.logger import get_logger

from .base_layer import BaseMoELayer
from .monitor import MoEMonitor
from .utils import all_to_all, gather_along_first_dim_expert_parallel

internlm_accelerator = get_accelerator()

try:
    # To enable gemm permute optimizations on GPU:
    #   python3 -m pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@v1.1.3
    import grouped_gemm

    GEMM_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    # Fail silently so we don't spam logs unnecessarily if user isn't using gemm
    GEMM_INSTALLED = False
    pass

# global llm logger
logger = get_logger(__file__)

internlm_accelerator = get_accelerator()

uniform_map: Dict[torch.device, Callable] = {}


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device), high=torch.tensor(1.0 + epsilon, device=device)
        ).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def custom_argsort(x, stable=True):
    if internlm_accelerator.get_accelerator_backend() in [
        AcceleratorType.NPU,
        AcceleratorType.DIPU,
        AcceleratorType.DITORCH,
    ]:
        sorted_indices = torch.sort(x.to(torch.float), stable=stable)[1]
        return sorted_indices
    else:
        return torch.argsort(x, stable=stable)


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        topk: int = 1,
        noisy_gate_policy: Optional[str] = None,
        gate_monitor: Optional[MoEMonitor] = None,
    ) -> None:
        super().__init__()

        # Deepspeed's mechisms, alway use fp32
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.k = topk

        self.noisy_gate_policy = noisy_gate_policy
        self.gate_monitor = gate_monitor

    def forward(self, inputs: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        # input jittering
        if self.noisy_gate_policy == "Jitter" and self.training:
            inputs = multiplicative_jitter(inputs, device=inputs.device)
        logits = self.wg(inputs)
        if self.gate_monitor is not None and not gpc.is_forward_done:
            self.gate_monitor.pre_gate_profiling(logits)
        if self.training and gpc.config.loss.moe_z_loss_coeff > 0:
            z_loss = torch.logsumexp(logits, dim=1).square().mean()
        else:
            z_loss = None
        gates = F.softmax(logits, dim=1)

        return gates, z_loss


def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


class DeepSeekDroplessMoELayer(BaseMoELayer):
    """MoELayer module which implements MixtureOfExperts as described in Gshard_."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int,
        ep_group: Optional[torch.distributed.ProcessGroup],
        ep_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        activation_type: str = "swiglu",
        drop_and_pad: bool = False,
        drop_policy="probs",
        capacity_factor: float = None,
        noisy_gate_policy: str = None,
        moe_grouped_mlp: bool = True,
        enable_fused_permute: bool = True,
        deterministic_mode: bool = False,
        normalize_expert_weights: bool = True,
        seq_aux: bool = True,
        aux_loss_alpha: float = 0.0001,
        moe_loss_type: str = "default",
        routed_scaling_factor: float = 1.0,
        shared_expert=None,
        moe_monitor_cfg: Optional[Dict] = None,
        **_ignored,
    ) -> None:
        assert noisy_gate_policy is None or noisy_gate_policy in ["None", "Jitter", "RSample"], (
            "Unsupported noisy_gate_policy: " + noisy_gate_policy
        )
        assert (
            num_experts % ep_size == 0
        ), f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"

        if moe_monitor_cfg is not None:
            self.expert_monitor = MoEMonitor(
                ep_size=ep_size,
                num_experts=num_experts,
                topk=top_k,
                moe_monitor_cfg=moe_monitor_cfg,
            )
        else:
            self.expert_monitor = None

        if moe_grouped_mlp:
            experts = new_feed_forward(
                in_features,
                hidden_features,
                out_features,
                bias=False,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                activation_type=activation_type,
                is_moe=True,
                use_grouped_mlp=True,
                num_groups=num_experts // ep_size,
                backend="gmm",
            )
        else:
            experts = torch.nn.ModuleList(
                [
                    new_feed_forward(
                        in_features,
                        hidden_features,
                        out_features,
                        bias=False,
                        device=device,
                        dtype=dtype,
                        mlp_layer_fusion=mlp_layer_fusion,
                        multiple_of=multiple_of,
                        activation_type=activation_type,
                        is_moe=True,
                    )
                    for _ in range(num_experts // ep_size)
                ]
            )
        super().__init__(
            TopKGate(
                in_features,
                num_experts,
                top_k,
                noisy_gate_policy,
                gate_monitor=self.expert_monitor,
            ),
            experts,
            ep_group,
            ep_size,
            num_experts // ep_size,
        )

        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        local_expert_indices_offset = gpc.get_local_rank(ParallelMode.EXPERT) * self.num_local_experts
        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        self.topk = top_k
        self.moe_grouped_mlp = moe_grouped_mlp
        self.enable_fused_permute = (
            GEMM_INSTALLED and enable_fused_permute and not drop_and_pad and capacity_factor is None
        )
        self.deterministic_mode = deterministic_mode
        self.normalize_expert_weights = normalize_expert_weights
        self.seq_aux = seq_aux
        self.aux_loss_alpha = aux_loss_alpha
        self.moe_loss_type = moe_loss_type
        self.routed_scaling_factor = routed_scaling_factor
        self.shared_expert = shared_expert

        self.input_splits = None
        self.output_splits = None
        # self.num_global_tokens_per_local_expert = None
        self.num_global_tokens_per_local_expert_cpu = None
        # track balance
        self.num_global_tokens_per_expert_global_expert = None
        # We need to keep track of the token num if we drop tokens without padding them.
        self.num_out_tokens = None
        self.hidden_shape = None
        # A cuda stream synchronization is needed due to no blocking sync between host and device
        self.device_sync_point = "no_sync"
        input_chunk_idxs = torch.arange(self.num_experts)  # * self.tp_size
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(-1, self.num_local_experts).T.ravel().tolist()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(self.num_local_experts, -1).T.ravel().tolist()
        # self.num_input_tokens = None

        self.drop_and_pad = drop_and_pad
        self.capacity_factor = capacity_factor
        self.drop_policy = drop_policy
        if self.drop_and_pad:
            assert self.capacity_factor is not None
        self.capacity = None

    def forward(self, *inputs: Tensor) -> Tensor:
        self.hidden_shape = inputs[0].shape

        d_model = inputs[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_inputs = inputs[0].reshape(-1, d_model)

        gates, self.z_loss = self.gate(reshaped_inputs)
        expert_weights, indices, tokens_per_expert_before_capacity = self.topk_softmax_with_capacity(gates)

        if self.moe_loss_type == "deepseek":
            self.l_aux = self.deepseek_aux_loss(gates, indices, self.hidden_shape[0], self.hidden_shape[1])
        else:
            self.l_aux = self.load_balancing_loss(tokens_per_expert_before_capacity, gates)
        (dispatched_input, tokens_per_expert) = self.token_permutation(
            reshaped_inputs, expert_weights, indices, tokens_per_expert_before_capacity
        )
        if self.moe_grouped_mlp:
            expert_output = self.experts(dispatched_input, batch_sizes=tokens_per_expert)
        else:
            expert_output = self.experts(dispatched_input, split_size_or_sections=tokens_per_expert, split_dim=0)
        output = self.token_unpermutation(expert_output, expert_weights)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output

    def topk_softmax_with_capacity(self, gates):
        expert_weights, indices = torch.topk(gates, self.topk, dim=1)
        if self.normalize_expert_weights:
            expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        # we compute num_local_tokens_per_expert here. If no drop and padding, num_local_tokens_per_expert should be
        # the final value, otherwise we recompute it in self.process(.)
        # histc(.) can be faster the bincount(.), but will cause non-deterministic behavior
        if self.deterministic_mode:
            num_local_tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)
        else:
            num_local_tokens_per_expert = torch.histc(indices, bins=self.num_experts, min=0, max=self.num_experts)

        expert_weights = expert_weights * self.routed_scaling_factor
        # without capacity
        if self.capacity_factor is None:
            # shape: [num_token, topk]
            return expert_weights, indices, num_local_tokens_per_expert

        # with capacity
        expert_capacity = get_capacity(
            num_tokens=gates.shape[0] * self.topk,
            num_experts=gates.shape[1],
            capacity_factor=self.capacity_factor,
        )
        # TopK selection, Maskout unused experts
        topk_masked_gates = torch.zeros_like(gates).scatter(1, indices, expert_weights)
        topk_mask = torch.zeros_like(gates).scatter(1, indices, 1)

        if self.drop_policy == "probs":
            capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(gates).scatter(0, capacity_indices, 1)
        elif self.drop_policy == "position":
            _, capacity_indices = torch.topk(topk_mask, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(gates).scatter(0, capacity_indices, 1)
            capacity_probs = torch.gather(topk_masked_gates, 0, capacity_indices)
        else:
            raise ValueError(f"Invalid drop_policy: {self.drop_policy}")

        if self.drop_and_pad:
            # shape: [num_expert, capacity]
            final_expert_weights, final_indices = (
                capacity_probs.T.contiguous(),
                capacity_indices.T.contiguous(),
            )
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_mask = torch.logical_and(topk_mask, capacity_mask)
            drop_mask = torch.logical_not(final_mask)
            exceed_mask = torch.gather(drop_mask, 1, indices)
            # shape: [num_token, topk]
            final_expert_weights = expert_weights * torch.logical_not(exceed_mask)
            final_indices = indices.clone().masked_fill_(exceed_mask, torch.iinfo(torch.long).max)

        tokens_per_expert_before_capacity = topk_mask.sum(dim=0)

        return final_expert_weights, final_indices, tokens_per_expert_before_capacity

    def preprocess(self, indices, expert_weight, tokens_per_expert_before_capacity) -> torch.Tensor:
        """
        Preprocess token indices for AlltoAll communication and token permutation. This method computes
        the number of tokens assigned to each expert based on the input indices.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            indices (torch.Tensor): Tensor of indices mapping tokens to experts.

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        if self.capacity_factor is not None:
            if self.deterministic_mode:
                num_local_tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)
            else:
                num_local_tokens_per_expert = torch.histc(indices, bins=self.num_experts, min=0, max=self.num_experts)
        else:
            num_local_tokens_per_expert = tokens_per_expert_before_capacity

        # num_local_tokens_per_expert: [num_experts]
        self.num_local_tokens_per_expert = num_local_tokens_per_expert

        if self.drop_and_pad:
            # expert_weights: [num_experts, capacity]
            self.capacity = expert_weight.size(1)
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long
            )
            self.num_global_tokens_per_local_expert_cpu = torch.full(
                (self.num_experts,), self.capacity, dtype=torch.long
            )
            return num_tokens_per_local_expert

        elif self.capacity_factor is not None:
            self.num_out_tokens = num_local_tokens_per_expert.sum().to(torch.device("cpu"), non_blocking=True)
            self.device_sync_point = "before_permutation_1"
        elif self.ep_size > 1 or self.num_local_experts > 1:
            # wait for input_splits and output_splits, or num_global_tokens_per_local_expert_cpu sync
            self.device_sync_point = "before_ep_alltoall"
        else:
            # wait for tokens_per_expert sync
            self.device_sync_point = "before_premute_finish"

        if self.ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(self.ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )

            num_global_tokens_per_expert = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert).reshape(
                self.ep_size, self.num_experts
            )

            self.num_global_tokens_per_expert = num_global_tokens_per_expert

            self.num_global_tokens_per_expert_global_expert = num_global_tokens_per_expert.sum(dim=0)
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices]

            self.output_splits = (
                num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
            )
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(self.num_experts)
            self.num_global_tokens_per_expert_global_expert = num_local_tokens_per_expert
            num_tokens_per_local_expert = num_local_tokens_per_expert

        if self.expert_monitor is not None and not gpc.is_forward_done:
            self.expert_monitor.expert_profiling(
                tokens_per_expert_before_capacity=tokens_per_expert_before_capacity
                if self.capacity_factor is not None
                else None,
                num_tokens_per_expert=num_global_tokens_per_expert.sum(axis=0)
                if self.ep_size > 1
                else num_tokens_per_local_expert,
            )

        if self.moe_grouped_mlp:
            num_tokens_per_local_expert = num_tokens_per_local_expert.to(torch.device("cpu"), non_blocking=True)

        if self.num_local_experts > 1 and self.ep_size > 1:
            # expert_ids_per_ep_rank = torch.remainder(
            #     torch.arange(self.num_experts, dtype=torch.int32, device=indices.device),
            #     self.num_local_experts,  # mpu.experts_per_rank(self.args),
            # )
            self.num_global_tokens_per_local_expert_cpu = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            ).to(torch.device("cpu"), non_blocking=True)

        return num_tokens_per_local_expert

    def permute_with_padded_tokens(self, tokens, indices):
        """Permute the tokens based on the indices, only used in padding mode.
        The input indices shape is [num_expert, capacity], it indicates which tokens were selected
        by each expert separately.
        Args:
            tokens (torch.Tensor): The input token tensor.
            indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the
                                    selected tokens for each expert.

        Returns:
            torch.Tensor: The permuted tensor.
            torch.Tensor: The sorted_indices corresponding permuted tensor.
        """
        permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))

        return permuted_tokens, indices

    def unpermute_with_padded_tokens(
        self,
        permuted_tokens: torch.Tensor,
        indices: torch.Tensor,
        expert_weights: torch.Tensor,
        restore_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Unpermutes a padded permuted tokens based on sorted indices and merges the tokens
        with their corresponding expert_weights.

        This function takes a tensor of permuted tokens and reorders them according to the provided indices.
        It also combines the tokens with their associated expert_weights.

        Parameters:
            permuted_tokens (torch.Tensor): A 2D tensor containing permuted tokens.
            indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected tokens for
                                    each expert.
            expert_weights (torch.Tensor): A tensor with the same shape as indices, containing weights
                                           corresponding to each token.
            restore_shape (torch.Size): The target shape for the unpermuted tokens tensor.

        Returns:
            torch.Tensor: A tensor of unpermuted tokens, merged with their expert_weights.

        """
        # Ensure permuted_tokens is 2D
        assert permuted_tokens.dim() == 2, f"Got {permuted_tokens.dim()}D."

        # Reshape and expand expert_weights and indices to match permuted_tokens
        expert_weights = expert_weights.view(-1).unsqueeze(-1)

        # Combine tokens with their expert_weights
        combined_output = expert_weights * permuted_tokens

        # Prepare a tensor of zeros with the desired output shape
        empty_tokens = torch.zeros(
            restore_shape,
            dtype=combined_output.dtype,
            device=combined_output.device,
        )

        # Scatter the combined tokens back to their original positions
        flatten_indices = indices.view(-1)
        unpermuted_tokens = empty_tokens.index_put_((flatten_indices,), combined_output, accumulate=True)

        return unpermuted_tokens

    def permute(self, tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
        """Permute the tokens based on the indices. Token with the same index will be grouped together.
        The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.
        Args:
            tokens (torch.Tensor): The input token tensor.
            indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens] or
                        [num_tokens, topk].
            num_out_tokens (int, optional): The effective output token count, when enabling the capacity factor, should
                        equal the number of tokens not dropped.  By default, set to None, meaning no tokens are dropped.
            padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity]
                        to denote selected tokens per expert. Defaults to False.

        Returns:
            torch.Tensor: The permuted tensor.
            torch.Tensor: The sorted_indices corresponding permuted tensor.
        """
        if padded_mode:
            return self.permute_with_padded_tokens(tokens, indices)

        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = custom_argsort(flatten_indices, stable=True)

        if num_out_tokens is not None:
            sorted_indices = sorted_indices[:num_out_tokens]
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)
        return permuted_tokens, sorted_indices

    def unpermute(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        expert_weights: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
    ):
        """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens
           with their corresponding expert_weights.

        Args:
            permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
            sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
            expert_weights (torch.Tensor, optional): The tensor of expert_weights corresponding to the permuted tokens.
                        If provided, the unpermuted tokens will be merged with their respective expert_weights.
            padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity]
                        to denote selected tokens per expert. Defaults to False.
            restore_shape (torch.Size, optional): The input shape before permutation, only used in padding mode.
                        Defaults to None.

        Returns:
            torch.Tensor: The unpermuted tokens, optionally merged with expert_weights.
        """
        if padded_mode:
            return self.unpermute_with_padded_tokens(
                permuted_tokens, sorted_indices, expert_weights, restore_shape=restore_shape
            )

        assert sorted_indices.numel() == permuted_tokens.size(0)
        if expert_weights is not None:
            # Unpermute and merge the tokens with their expert_weights
            num_unpermuted_tokens = expert_weights.numel()
            topk = expert_weights.size(1)
        else:
            # Unpermute the tokens without merge
            num_unpermuted_tokens = permuted_tokens.size(0)
            topk = 1

        unpermuted_tokens = torch.zeros(
            [num_unpermuted_tokens, permuted_tokens.shape[-1]],
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        unpermuted_tokens.index_put_((sorted_indices,), permuted_tokens, accumulate=False)
        unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
        if expert_weights is not None:
            unpermuted_tokens = unpermuted_tokens * expert_weights.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        return unpermuted_tokens

    def sort_chunks_by_idxs(self, inputs: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
        """Split and sort the input tensor based on the split_sizes and sorted indices."""
        try:
            inputs = torch.split(inputs, split_sizes.tolist(), dim=0)
            output = torch.cat([inputs[i] for i in sorted_idxs], dim=0)

        except Exception as e:
            import os

            rank_info = (
                f"ep_rank_{gpc.get_local_rank(ParallelMode.EXPERT)},global_rank_{gpc.get_global_rank()},"
                f"{os.environ['HOSTNAME'], os.environ['KUBERNETES_POD_NAME']}"
            )

            input_shapess = [inp.shape for inp in inputs] if isinstance(inputs, (tuple, list)) else inputs.shape
            logger.info(f"{rank_info}, inputs.shape: {input_shapess};")
            logger.info(
                f"{rank_info}, self.permutated_local_input_tokens_shape:{self.permutated_local_input_tokens_shape}"
            )
            logger.info(f"{rank_info}, split_sizes:{split_sizes}, sorted_idxs:{sorted_idxs}")
            logger.info(
                f"{rank_info}, self.num_global_tokens_per_expert_global_expert:"
                f"{self.num_global_tokens_per_expert_global_expert}"
            )
            logger.info(f"self.output_splits: {self.output_splits}")
            logger.info(f"self.input_splits: {self.input_splits}")
            logger.info(f"{rank_info}, self.num_global_tokens_per_expert: {self.num_global_tokens_per_expert}")
            logger.info(f"{rank_info}, self.num_local_tokens_per_expert:{self.num_local_tokens_per_expert}")
            logger.warning(e)

            raise RuntimeError()
        return output

    def token_permutation(
        self,
        reshaped_inputs: torch.Tensor,
        expert_weights: torch.Tensor,
        indices: torch.Tensor,
        tokens_per_expert_before_capacity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            expert_weights (torch.Tensor): expert_weights of tokens assigned to experts.
            indices (torch.Tensor): Indices of tokens assigned to experts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        assert expert_weights.dim() == 2, "Expected 2D tensor for expert weights"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        tokens_per_expert = self.preprocess(indices, expert_weights, tokens_per_expert_before_capacity)

        # Permutation 1: input to AlltoAll input
        self.hiddden_shape_before_permute = reshaped_inputs.shape
        if self.device_sync_point == "before_permutation_1":
            internlm_accelerator.current_stream().synchronize()
        if self.enable_fused_permute:
            permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = grouped_gemm.ops.permute(
                reshaped_inputs, indices.to(torch.int32), self.num_out_tokens
            )
        else:
            permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = self.permute(
                reshaped_inputs,
                indices,
                num_out_tokens=self.num_out_tokens,
                padded_mode=self.drop_and_pad,
            )

        if self.expert_monitor is not None and not gpc.is_forward_done:
            self.expert_monitor.expert_profiling_wait()

        # Perform expert parallel AlltoAll communication
        if self.device_sync_point == "before_ep_alltoall":
            internlm_accelerator.current_stream().synchronize()
        if self.shared_expert is not None:
            self.shared_expert.launch_w1w3()
        global_input_tokens, _ = all_to_all(
            permutated_local_input_tokens, self.output_splits, self.input_splits, gpc.get_group(ParallelMode.EXPERT)
        )
        self.permutated_local_input_tokens_shape = permutated_local_input_tokens.shape
        # Permutation 2: Sort alltoall output by local experts when num_local_experts > 1.
        if self.num_local_experts > 1 and self.ep_size > 1:
            global_input_tokens = self.sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert_cpu.ravel(),
                self.sort_input_by_local_experts,
            )
        if self.device_sync_point == "before_premute_finish":
            internlm_accelerator.current_stream().synchronize()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1 and self.ep_size > 1:
            hidden_states = self.sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                self.restore_output_by_local_experts,
            )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if self.shared_expert is not None:
            self.shared_expert.launch_w2()

        permutated_local_input_tokens, _ = all_to_all(
            hidden_states, self.input_splits, self.output_splits, gpc.get_group(ParallelMode.EXPERT)
        )

        # Unpermutation 1: AlltoAll output to output
        if self.enable_fused_permute:
            output = grouped_gemm.ops.unpermute(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                expert_weights.to(torch.float32),
            )
        else:
            output = self.unpermute(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                expert_weights=expert_weights,
                padded_mode=self.drop_and_pad,
                restore_shape=self.hiddden_shape_before_permute,
            )

        return output

    def load_balancing_loss(self, num_local_tokens_per_expert, gates):
        """Calculate the load balancing loss contribution."""
        assert len(gates.size()) == 2
        tokens, num_experts = gates.size()
        assert num_experts == self.num_experts
        assert len(num_local_tokens_per_expert.size()) == 1
        (num_experts,) = num_local_tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.topk)
        return scale * torch.dot(num_local_tokens_per_expert.to(gates.dtype), gates.mean(dim=0))

    def deepseek_aux_loss(self, scores, topk_idx, bsz, seq_len):
        scores_for_aux = scores
        aux_topk = self.topk
        # always compute aux loss based on the naive greedy topk method
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if self.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.num_experts, device=scores.device)
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * aux_topk, device=scores.device),
            ).div_(seq_len * aux_topk / self.num_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.aux_loss_alpha
        else:
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.num_experts)
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.num_experts
            aux_loss = (Pi * fi).sum() * self.aux_loss_alpha
        return aux_loss
