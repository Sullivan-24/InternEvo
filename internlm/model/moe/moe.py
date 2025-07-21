import torch
import torch.nn.functional as F

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import set_fp32_attr_to_module
from internlm.model.modules.mlp import new_feed_forward
from internlm.model.moe.dropless_layer import DroplessMoELayer
from internlm.model.moe.dropless_layer_deepseek import DeepSeekDroplessMoELayer
from internlm.model.moe.gshard_layer import GShardMoELayer
from internlm.model.moe.megablocks.megablock_dmoe import MegaBlockdMoE
from internlm.model.moe.megablocks.megablock_moe import MegaBlockMoE
from internlm.utils.logger import get_logger

# global llm logger
logger = get_logger(__file__)


def new_moe_layer(moe_type: str, **kwargs):
    if moe_type == "GShard":
        return GShardMoELayer(**kwargs)
    elif moe_type == "Dropless":
        return DroplessMoELayer(**kwargs)
    elif moe_type == "MegaBlock":
        return MegaBlockMoE(**kwargs)
    elif moe_type == "MegaBlock-Dropless":
        return MegaBlockdMoE(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {moe_type}")

def new_moe_layer(moe_type: str, **kwargs):
    if moe_type == "GShard":
        return GShardMoELayer(**kwargs)
    elif moe_type == "Dropless":
        if gpc.config.model_type == "DEEPSEEK2_MoE":
            return DeepSeekDroplessMoELayer(**kwargs)
        else:
            return DroplessMoELayer(**kwargs)
    elif moe_type == "MegaBlock":
        return MegaBlockMoE(**kwargs)
    elif moe_type == "MegaBlock-Dropless":
        return MegaBlockdMoE(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {moe_type}")

class MoEBase(torch.nn.Module):
    """Initialize an MoE base layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
                                            or 'None'.
        using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
                                        infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        moe_use_residual (bool, optional): default=False, make this MoE layer a Residual MoE
                                          (https://arxiv.org/abs/2201.05596) layer.
        residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        moe_layer_kwargs=None,
        device=None,
        dtype=None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        activation_type: str = "swiglu",
    ):

        super().__init__()

        if moe_layer_kwargs is None:
            moe_layer_kwargs = dict()

        ep_group = gpc.get_group(ParallelMode.EXPERT)
        ep_size = gpc.get_world_size(ParallelMode.EXPERT)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.moe_layer = new_moe_layer(
            moe_type=gpc.config.model.moe_type,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            ep_group=ep_group,
            ep_size=ep_size,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            activation_type=activation_type,
            **moe_layer_kwargs,
        )
        set_fp32_attr_to_module(self.moe_layer.gate)

        # residual network, see https://arxiv.org/pdf/2201.05596.pdf, seems useful for convergence
        self.num_shared_experts = num_shared_experts
        if self.num_shared_experts > 0:
            self.residual_mlp = new_feed_forward(
                in_features=in_features,
                hidden_features=int(hidden_features * num_shared_experts),
                out_features=out_features,
                bias=False,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                activation_type=activation_type,
            )
            # coefficient is used for weighted sum of the output of expert and residual mlp
            self.coefficient = torch.nn.Linear(in_features, 2)


# class MoE(MoEBase):
#     """Initialize an MoE layer.

#     Arguments:
#         hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
#         num_experts (int, optional): default=1, the total number of experts per layer.
#         ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
#         k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
#         capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
#         eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
#         min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
#         noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
#                                             or 'None'.
#         using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
#         drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
#                                         infinite capacity).
#         use_rts (bool, optional): default=True, whether to use Random Token Selection.
#         moe_use_residual (bool, optional): default=False, make this MoE layer a Residual MoE
#                                           (https://arxiv.org/abs/2201.05596) layer.
#         residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
#     """

#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: int,
#         out_features: int,
#         num_experts: int = 1,
#         top_k: int = 1,
#         num_shared_experts: int = 0,
#         moe_layer_kwargs=None,
#         device=None,
#         dtype=None,
#         mlp_layer_fusion: bool = False,
#         multiple_of: int = 256,
#         activation_type: str = "swiglu",
#     ):
#         super().__init__(
#             in_features=in_features,
#             hidden_features=hidden_features,
#             out_features=out_features,
#             num_experts=num_experts,
#             top_k=top_k,
#             num_shared_experts=num_shared_experts,
#             moe_layer_kwargs=moe_layer_kwargs,
#             device=device,
#             dtype=dtype,
#             mlp_layer_fusion=mlp_layer_fusion,
#             multiple_of=multiple_of,
#             activation_type=activation_type,
#         )

#         if self.num_shared_experts > 0:
#             self.coefficient = torch.nn.Linear(in_features, 2)

#     def forward(self, hidden_states, used_token=None):
#         """MoE forward

#         Arguments:
#             hidden_states (Tensor): input to the layer
#             used_token (Tensor, optional): default: None, mask only used tokens

#         Returns:
#             A tuple including output, gate loss, and expert count.

#             * output (Tensor): output of the model

#             * l_aux (Tensor): gate loss value

#             * exp_counts (int): expert count
#         """
#         output = self.moe_layer(hidden_states, used_token)
#         if self.num_shared_experts > 0:
#             # Residual MoE
#             output_mlp = self.residual_mlp(hidden_states)
#             if isinstance(output_mlp, tuple):
#                 output_mlp = output_mlp[0]  # Ignore the bias term for now
#             coef = self.coefficient(hidden_states)
#             coef = torch.nn.functional.softmax(coef, dim=-1)
#             output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
#         return output, self.moe_layer.l_aux, self.moe_layer.exp_counts

class MoE(torch.nn.Module):
    """Initialize an MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
                                            or 'None'.
        using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
                                        infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        moe_use_residual (bool, optional): default=False, make this MoE layer a Residual MoE
                                          (https://arxiv.org/abs/2201.05596) layer.
        residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        residual_type: str = "deepspeed",
        moe_layer_kwargs=None,
        device=None,
        dtype=None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        activation_type: str = "swiglu",
    ):
        super().__init__()

        if moe_layer_kwargs is None:
            moe_layer_kwargs = dict()

        ep_group = gpc.get_group(ParallelMode.EXPERT)
        ep_size = gpc.get_world_size(ParallelMode.EXPERT)

        self.moe_layer = new_moe_layer(
            moe_type=gpc.config.model.moe_type,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            ep_group=ep_group,
            ep_size=ep_size,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            activation_type=activation_type,
            **moe_layer_kwargs,
        )
        set_fp32_attr_to_module(self.moe_layer.gate)

        # residual network, see https://arxiv.org/pdf/2201.05596.pdf, seems useful for convergence
        self.num_shared_experts = num_shared_experts
        self.residual_type = residual_type
        if self.num_shared_experts > 0:
            self.residual_mlp = new_feed_forward(
                in_features=in_features,
                hidden_features=int(hidden_features * num_shared_experts),
                out_features=out_features,
                bias=False,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                activation_type=activation_type,
            )

            if self.residual_type == "deepspeed":
                # residual network, see https://arxiv.org/pdf/2201.05596.pdf, seems useful for convergence
                # https://github.com/microsoft/DeepSpeed/blob/42a8eaa705ed30b4a656ac71bdb400772df2cb21/deepspeed/moe/layer.py#L129C13-L129C51
                # coefficient is used for weighted sum of the output of expert and residual mlp
                self.coefficient = torch.nn.Linear(in_features, 2)
            elif self.residual_type == "deepspeed_no_bias":
                # residual network, see https://arxiv.org/pdf/2201.05596.pdf, seems useful for convergence
                # https://github.com/microsoft/DeepSpeed/blob/42a8eaa705ed30b4a656ac71bdb400772df2cb21/deepspeed/moe/layer.py#L129C13-L129C51
                # coefficient is used for weighted sum of the output of expert and residual mlp
                self.coefficient = torch.nn.Linear(in_features, 2, bias=False)
            elif self.residual_type == "qwen":
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
                self.coefficient = torch.nn.Linear(in_features, 1, bias=False)
            elif self.residual_type == "deepseek":
                # https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/521d2bc4fb69a3f3ae565310fcc3b65f97af2580/modeling_deepseek.py#L392
                # no nothing, add directly
                pass
            else:
                raise NotImplementedError(f"Residual type {residual_type} is not supported")

    def forward(self, hidden_states, used_token=None):
        """MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * z_loss (Tensor): gate z loss
        """
        output = self.moe_layer(hidden_states, used_token)
        if self.num_shared_experts > 0:
            # Residual MoE
            output_mlp = self.residual_mlp(hidden_states)
            if isinstance(output_mlp, tuple):
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            if self.residual_type in ["deepspeed", "deepspeed_no_bias"]:
                coef = self.coefficient(hidden_states)
                coef = torch.nn.functional.softmax(coef, dim=-1)
                output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
            elif self.residual_type == "qwen":
                coef = torch.sigmoid(self.coefficient(hidden_states))
                output = output + output_mlp * coef
            elif self.residual_type == "deepseek":
                output = output + output_mlp
            else:
                raise NotImplementedError(f"Residual type {self.residual_type} is not supported")

        return output, self.moe_layer.l_aux.to(torch.float32), self.moe_layer.z_loss.to(torch.float32)


class Qwen2MoE(MoEBase):
    """Initialize an Qwen2MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
                                            or 'None'.
        using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
                                        infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        moe_use_residual (bool, optional): default=False, make this MoE layer a Residual MoE
                                          (https://arxiv.org/abs/2201.05596) layer.
        residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        moe_layer_kwargs=None,
        device=None,
        dtype=None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        activation_type: str = "swiglu",
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            moe_layer_kwargs=moe_layer_kwargs,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            activation_type=activation_type,
        )

        if self.num_shared_experts > 0:
            self.coefficient = torch.nn.Linear(in_features, 1, bias=False)

    def forward(self, hidden_states, used_token=None):
        """Qwen2MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.moe_layer(hidden_states, used_token)
        if self.num_shared_experts > 0:
            # Residual MoE
            output_mlp = self.residual_mlp(hidden_states)
            if isinstance(output_mlp, tuple):
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            output_mlp = F.sigmoid(coef) * output_mlp
            output = output + output_mlp
        return output, self.moe_layer.l_aux, self.moe_layer.exp_counts
