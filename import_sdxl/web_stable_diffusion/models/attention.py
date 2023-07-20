import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from .attention_processor import Attention
from typing import Any, Dict, Optional
from .embeddings import CombinedTimestepLabelEmbeddings

def get_activation(act_fn):
    if act_fn in ["swish", "silu"]:
        return nn.SiLU()
    elif act_fn == "mish":
        return nn.Mish()
    elif act_fn == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class AdaGroupNorm(nn.Module):
    """
    GroupNorm layer modified to incorporate timestep embeddings.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x, emb):
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep, class_labels, hidden_dtype=None):
        emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)



# class AttentionBlock(nn.Module):
#     """
#     An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
#     to the N-d case.
#     https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
#     Uses three q, k, v linear layers to compute attention.

#     Parameters:
#         channels (`int`): The number of channels in the input and output.
#         num_head_channels (`int`, *optional*):
#             The number of channels in each head. If None, then `num_heads` = 1.
#         norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for group norm.
#         rescale_output_factor (`float`, *optional*, defaults to 1.0): The factor to rescale the output by.
#         eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
#     """

#     def __init__(
#         self,
#         channels: int,
#         num_head_channels: Optional[int] = None,
#         norm_num_groups: int = 32,
#         rescale_output_factor: float = 1.0,
#         eps: float = 1e-5,
#     ):
#         super().__init__()
#         self.channels = channels

#         self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
#         self.num_head_size = num_head_channels
#         self.group_norm = nn.GroupNorm(
#             num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)

#         # define q,k,v as linear layers
#         self.query = nn.Linear(channels, channels)
#         self.key = nn.Linear(channels, channels)
#         self.value = nn.Linear(channels, channels)

#         self.rescale_output_factor = rescale_output_factor
#         self.proj_attn = nn.Linear(channels, channels, 1)

#         self._use_memory_efficient_attention_xformers = False

#     def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
#         if not is_xformers_available():
#             raise ModuleNotFoundError(
#                 "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
#                 " xformers",
#                 name="xformers",
#             )
#         elif not torch.cuda.is_available():
#             raise ValueError(
#                 "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
#                 " available for GPU "
#             )
#         else:
#             try:
#                 # Make sure we can run the memory efficient attention
#                 _ = xformers.ops.memory_efficient_attention(
#                     torch.randn((1, 2, 40), device="cuda"),
#                     torch.randn((1, 2, 40), device="cuda"),
#                     torch.randn((1, 2, 40), device="cuda"),
#                 )
#             except Exception as e:
#                 raise e
#             self._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

#     def reshape_heads_to_batch_dim(self, tensor):
#         batch_size, seq_len, dim = tensor.shape
#         head_size = self.num_heads
#         tensor = tensor.reshape(batch_size, seq_len,
#                                 head_size, dim // head_size)
#         tensor = tensor.permute(0, 2, 1, 3).reshape(
#             batch_size * head_size, seq_len, dim // head_size)
#         return tensor

#     def reshape_batch_dim_to_heads(self, tensor):
#         batch_size, seq_len, dim = tensor.shape
#         head_size = self.num_heads
#         tensor = tensor.reshape(batch_size // head_size,
#                                 head_size, seq_len, dim)
#         tensor = tensor.permute(0, 2, 1, 3).reshape(
#             batch_size // head_size, seq_len, dim * head_size)
#         return tensor

#     def forward(self, hidden_states):
#         residual = hidden_states
#         batch, channel, height, width = hidden_states.shape

#         # norm

#         hidden_states = self.group_norm(hidden_states)

#         hidden_states = hidden_states.view(
#             batch, channel, height * width).transpose(1, 2)
#         # proj to q, k, v
#         query_proj = self.query(hidden_states)
#         key_proj = self.key(hidden_states)
#         value_proj = self.value(hidden_states)
#         scale = 1 / math.sqrt(self.channels / self.num_heads)

#         query_proj = self.reshape_heads_to_batch_dim(query_proj)
#         key_proj = self.reshape_heads_to_batch_dim(key_proj)
#         value_proj = self.reshape_heads_to_batch_dim(value_proj)

#         if self._use_memory_efficient_attention_xformers:
#             # Memory efficient attention
#             hidden_states = xformers.ops.memory_efficient_attention(
#                 query_proj, key_proj, value_proj, attn_bias=None)
#             hidden_states = hidden_states.to(query_proj.dtype)
#         else:
#             attention_scores = torch.matmul(query_proj, key_proj.transpose(-1, -2))*scale
#             attention_probs = torch.softmax(
#                 attention_scores.float(), dim=-1).type(attention_scores.dtype)
#             hidden_states = torch.matmul(attention_probs, value_proj)

#         # reshape hidden_states
#         hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
#         # compute next hidden_states
#         hidden_states = self.proj_attn(hidden_states)
#         hidden_states = hidden_states.transpose(
#             -1, -2).reshape(batch, channel, height, width)

#         # res connect and rescale
#         hidden_states = (hidden_states + residual) / self.rescale_output_factor
#         return hidden_states


# class BasicTransformerBlock(nn.Module):
#     r"""
#     A basic Transformer block.

#     Parameters:
#         dim (:obj:`int`): The number of channels in the input and output.
#         n_heads (:obj:`int`): The number of heads to use for multi-head attention.
#         d_head (:obj:`int`): The number of channels in each head.
#         dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
#         context_dim (:obj:`int`, *optional*): The size of the context vector for cross attention.
#         gated_ff (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use a gated feed-forward network.
#         checkpoint (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use checkpointing.
#     """

#     def __init__(
#         self,
#         dim: int,
#         n_heads: int,
#         d_head: int,
#         dropout=0.0,
#         context_dim: Optional[int] = None,
#         gated_ff: bool = True,
#         checkpoint: bool = True,
#     ):
#         super().__init__()
#         self.attn1 = CrossAttention(
#             query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
#         )  # is a self-attention
#         self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
#         self.attn2 = CrossAttention(
#             query_dim=dim,
#             context_dim=context_dim,
#             heads=n_heads,
#             dim_head=d_head,
#             dropout=dropout,
#         )  # is self-attn if context is none
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#         self.checkpoint = checkpoint

#     def forward(self, hidden_states, context=None):
#         hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
#         hidden_states = (
#             self.attn2(self.norm2(hidden_states), context=context) + hidden_states
#         )
#         hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
#         return hidden_states


# class SpatialTransformer(nn.Module):
#     """
#     Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
#     standard transformer action. Finally, reshape to image.

#     Parameters:
#         in_channels (:obj:`int`): The number of channels in the input and output.
#         n_heads (:obj:`int`): The number of heads to use for multi-head attention.
#         d_head (:obj:`int`): The number of channels in each head.
#         depth (:obj:`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
#         dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
#         context_dim (:obj:`int`, *optional*): The number of context dimensions to use.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         n_heads: int,
#         d_head: int,
#         depth: int = 1,
#         dropout: float = 0.0,
#         num_groups: int = 32,
#         context_dim: Optional[int] = None,
#         use_linear_projection=False,
#     ):
#         super().__init__()
#         self.n_heads = n_heads
#         self.d_head = d_head
#         self.in_channels = in_channels
#         # We always use 1x1 conv2d instead of linear regardless of the value of
#         # use_linear_projection that the model uses.
#         self.use_linear_projection = False # use_linear_projection

#         inner_dim = n_heads * d_head
#         self.norm = torch.nn.GroupNorm(
#             num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
#         )

#         if self.use_linear_projection:
#             self.proj_in = nn.Linear(in_channels, inner_dim)
#         else:
#             self.proj_in = nn.Conv2d(
#                 in_channels, inner_dim, kernel_size=1, stride=1, padding=0
#             )


#         self.transformer_blocks = nn.ModuleList(
#             [
#                 BasicTransformerBlock(
#                     inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
#                 )
#                 for d in range(depth)
#             ]
#         )

#         if self.use_linear_projection:
#             self.proj_out = nn.Linear(in_channels, inner_dim)
#         else:
#             self.proj_out = nn.Conv2d(
#                 inner_dim, in_channels, kernel_size=1, stride=1, padding=0
#             )

#     def forward(self, hidden_states, context=None):
#         # note: if no context is given, cross-attention defaults to self-attention
#         batch, channel, height, width = hidden_states.shape
#         residual = hidden_states
#         hidden_states = self.norm(hidden_states)

#         if not self.use_linear_projection:
#             hidden_states = self.proj_in(hidden_states)
#             inner_dim = hidden_states.shape[1]
#             hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
#         else:
#             inner_dim = hidden_states.shape[1]
#             hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
#             hidden_states = self.proj_in(hidden_states)

#         for block in self.transformer_blocks:
#             hidden_states = block(hidden_states, context=context)

#         if not self.use_linear_projection:
#             hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2)
#             hidden_states = self.proj_out(hidden_states)
#         else:
#             hidden_states = self.proj_out(hidden_states)
#             hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2)

#         return hidden_states + residual


# class CrossAttention(nn.Module):
#     r"""
#     A cross attention layer.

#     Parameters:
#         query_dim (:obj:`int`): The number of channels in the query.
#         context_dim (:obj:`int`, *optional*):
#             The number of channels in the context. If not given, defaults to `query_dim`.
#         heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
#         dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
#         dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
#     """

#     def __init__(
#         self,
#         query_dim: int,
#         context_dim: Optional[int] = None,
#         heads: int = 8,
#         dim_head: int = 64,
#         dropout: int = 0.0,
#     ):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = context_dim if context_dim is not None else query_dim

#         self.scale = dim_head**-0.5
#         self.heads = heads
#         # for slice_size > 0 the attention score computation
#         # is split across the batch axis to save memory
#         # You can set slice_size with `set_attention_slice`
#         self._slice_size = None

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
#         )

#     def reshape_heads_to_batch_dim(self, tensor):
#         batch_size, seq_len, dim = tensor.shape
#         head_size = self.heads
#         tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
#         tensor = tensor.permute(0, 2, 1, 3).reshape(
#             batch_size * head_size, seq_len, dim // head_size
#         )
#         return tensor

#     def reshape_batch_dim_to_heads(self, tensor):
#         batch_size, seq_len, dim = tensor.shape
#         head_size = self.heads
#         tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
#         tensor = tensor.permute(0, 2, 1, 3).reshape(
#             batch_size // head_size, seq_len, dim * head_size
#         )
#         return tensor

#     def forward(self, hidden_states, context=None, mask=None):
#         batch_size, sequence_length, _ = hidden_states.shape

#         query = self.to_q(hidden_states)
#         context = context if context is not None else hidden_states
#         key = self.to_k(context)
#         value = self.to_v(context)

#         dim = query.shape[-1]

#         query = self.reshape_heads_to_batch_dim(query)
#         key = self.reshape_heads_to_batch_dim(key)
#         value = self.reshape_heads_to_batch_dim(value)

#         # TODO(PVP) - mask is currently never used. Remember to re-implement when used

#         # attention, what we cannot get enough of
#         if self._slice_size is None or query.shape[0] // self._slice_size == 1:
#             hidden_states = self._attention(query, key, value)
#         else:
#             hidden_states = self._sliced_attention(
#                 query, key, value, sequence_length, dim
#             )

#         return self.to_out(hidden_states)

#     def _attention(self, query, key, value):
#         # TODO: use baddbmm for better performance
#         attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
#         attention_probs = attention_scores.softmax(dim=-1)
#         # compute attention output
#         hidden_states = torch.matmul(attention_probs, value)
#         # reshape hidden_states
#         hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
#         return hidden_states


# class FeedForward(nn.Module):
#     r"""
#     A feed-forward layer.

#     Parameters:
#         dim (:obj:`int`): The number of channels in the input.
#         dim_out (:obj:`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
#         mult (:obj:`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
#         glu (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use GLU activation.
#         dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
#     """

#     def __init__(
#         self,
#         dim: int,
#         dim_out: Optional[int] = None,
#         mult: int = 4,
#         glu: bool = False,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         inner_dim = int(dim * mult)
#         dim_out = dim_out if dim_out is not None else dim
#         project_in = GEGLU(dim, inner_dim)

#         self.net = nn.Sequential(
#             project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
#         )

#     def forward(self, hidden_states):
#         return self.net(hidden_states)


# class GEGLU(nn.Module):
#     r"""
#     A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

#     Parameters:
#         dim_in (:obj:`int`): The number of channels in the input.
#         dim_out (:obj:`int`): The number of channels in the output.
#     """

#     def __init__(self, dim_in: int, dim_out: int):
#         super().__init__()
#         self.proj1 = nn.Linear(dim_in, dim_out)
#         self.proj2 = nn.Linear(dim_in, dim_out)

#     def forward(self, hidden_states):
#         return self.proj1(hidden_states) * F.gelu(self.proj2(hidden_states))
