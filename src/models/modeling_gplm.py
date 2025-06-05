#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 10:01
@project: LucaOne
@file: modeling_gplm
@desc: LucaOne Model Detail
'''
import math
from typing import Dict, Optional, Sequence, Tuple, List, Union
import uuid
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def symmetrize(x):
    return x + x.transpose(-1, -2)


def apc(x):
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


class LucaGPLM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


try:
    # Optimized LayerNorm
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
    class LucaGPLM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError as e:
    print("import apex err:", e)
    from torch.nn import LayerNorm as LucaGPLM1bLayerNorm


class LucaGPLMTransformerLayer(nn.Module):
    """LucaGPLM Transformer layer block."""

    def __init__(
            self,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            add_bias_kv=True,
            use_lucagplm1b_layer_norm=False,
            use_rotary_embeddings: bool = False,
    ):
        '''
        Tramsformer-Encoder 层
        :param embed_dim: token embedding dim
        :param ffn_embed_dim: fully connected layer dim
        :param attention_heads: heads num
        :param add_bias_kv: key-value layer add bias
        :param use_lucagplm1b_layer_norm:  whether to use lucagplm 1b layer norm
        :param use_rotary_embeddings: whether to use rotary embedding
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv, use_lucagplm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_lucagplm1b_layer_norm):
        LucaGPLMLayerNorm = LucaGPLM1bLayerNorm if use_lucagplm1b_layer_norm else LucaGPLM1LayerNorm

        # pre layer norm
        self.pre_layer_norm = LucaGPLMLayerNorm(self.embed_dim)

        self.self_attn = LucaGPLMMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        # post layer norm
        self.post_layer_norm = LucaGPLMLayerNorm(self.embed_dim)

        # dimension increase by the fully connected layer
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)

        # dimension reduction by the fully connected layer
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

    def forward(
            self,
            x,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            need_head_weights=False
    ):
        residual = x
        x = self.pre_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.post_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn


class AxialTransformerLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            max_tokens_per_msa: int = 2**14,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        row_self_attention = RowSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        column_self_attention = ColumnSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
            activation_dropout=activation_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        self.row_self_attention = self.build_residual(row_self_attention)
        self.column_self_attention = self.build_residual(column_self_attention)
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module):
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
            self.dropout_prob,
        )

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_head_weights: bool = False,
    ):
        x, row_attn = self.row_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        x, column_attn = self.column_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        x = self.feed_forward_layer(x)
        if need_head_weights:
            return x, column_attn, row_attn
        else:
            return x


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, learned=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self, x):
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def make_positions(self, x):
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb


class RobertaLMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = LucaGPLM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ContactPredictionHead(nn.Module):
    def __init__(
            self,
            in_features: int,
            prepend_bos: bool,
            append_eos: bool,
            bias=True,
            eos_idx: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))


class NormalizedResidualBlock(nn.Module):
    def __init__(
            self,
            layer: nn.Module,
            embedding_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.layer = layer
        self.dropout_module = nn.Dropout(
            dropout,
        )
        self.layer_norm = LucaGPLM1bLayerNorm(self.embedding_dim)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x


class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            ffn_embedding_dim: int,
            activation_dropout: float = 0.1,
            max_tokens_per_msa: int = 2**14,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens_per_msa = max_tokens_per_msa
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout,
        )
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class RowSelfAttention(nn.Module):
    """Compute self-attention over rows of a 2D input."""

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            max_tokens_per_msa: int = 2 ** 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attn_shape = "hnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(
            self,
            x,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                x[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, start : start + max_rows]
                if self_attn_padding_mask is not None
                else None,
            )
            attns += attn_weights
        attn_probs = attns.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(x[start : start + max_rows], attn_probs)
            outputs.append(output)

        output = torch.cat(outputs, 0)
        return output, attn_probs

    def compute_attention_weights(
            self,
            x,
            scaling: float,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        if self_attn_padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(4).to(q)

        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        if self_attn_mask is not None:
            raise NotImplementedError
            # Mask Size: [B x R x C], Weights Size: [H x B x C x C]

        if self_attn_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2),
                -10000,
            )

        return attn_weights

    def compute_attention_update(
            self,
            x,
            attn_probs,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
            self,
            x,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if (num_rows * num_cols > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)
        else:
            scaling = self.align_scaling(x)
            attn_weights = self.compute_attention_weights(
                x, scaling, self_attn_mask, self_attn_padding_mask
            )
            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            output = self.compute_attention_update(x, attn_probs)
            return output, attn_probs


class ColumnSelfAttention(nn.Module):
    """Compute self-attention over columns of a 2D input."""

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            max_tokens_per_msa: int = 2 ** 16,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def _batched_forward(
            self,
            x,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        outputs = []
        attns = []
        for start in range(0, num_cols, max_cols):
            output, attn = self(
                x[:, start : start + max_cols],
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, :, start : start + max_cols]
                if self_attn_padding_mask is not None
                else None,
            )
            outputs.append(output)
            attns.append(attn)
        output = torch.cat(outputs, 1)
        attns = torch.cat(attns, 1)
        return output, attns

    def compute_attention_update(
            self,
            x,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if num_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with padding
            attn_probs = torch.ones(
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device=x.device,
                dtype=x.dtype,
            )
            output = self.out_proj(self.v_proj(x))
        else:
            q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            q *= self.scaling

            attn_weights = torch.einsum("icnhd,jcnhd->hcnij", q, k)

            if self_attn_mask is not None:
                raise NotImplementedError
            if self_attn_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    self_attn_padding_mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(3),
                    -10000,
                )

            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            context = torch.einsum("hcnij,jcnhd->icnhd", attn_probs, v)
            context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
            output = self.out_proj(context)
        return output, attn_probs

    def forward(
            self,
            x,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        # if False and num_rows * num_cols > 2 ** 14 and not torch.is_grad_enabled():
        if (num_rows * num_cols) > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(
                x,
                self_attn_mask,
                self_attn_padding_mask,
            )
        else:
            return self.compute_attention_update(x, self_attn_mask, self_attn_padding_mask)


def utils_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
            self,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
            key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
            self,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
            key: str,
            value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


@with_incremental_state
class LucaGPLMMultiheadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            self_attention: bool = False,
            encoder_decoder_attention: bool = False,
            use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=nn.init.calculate_gain("relu"))

        nn.init.xavier_uniform_(self.out_proj.weight, gain=nn.init.calculate_gain("relu"))
        # nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
                not self.rot_emb
                and self.enable_torch_version
                and not self.onnx_trace
                and incremental_state is None
                and not static_kv
                # A workaround for quantization to work. Otherwise JIT compilation
                # treats bias in linear module as method.
                and not torch.jit.is_scripting()
                and not need_head_weights
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = LucaGPLMMultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = LucaGPLMMultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).type_as(attn).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(
                            0
                    ):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, *_, **__):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )




