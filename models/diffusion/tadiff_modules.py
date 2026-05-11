import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import copy
from functools import partial, cached_property
from einops import einsum, rearrange, repeat
import abc

def size_to_mask(
    max_size,
    sizes,
):
    mask = torch.arange(max_size, device=sizes.device)
    return torch.lt(mask, sizes.unsqueeze(-1))

class DiffusionStepEmbedder(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 frequency_embedding_size: int=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class MultiInSizeLinear(nn.Module):
    def __init__(
        self,
        in_features_ls,
        out_features,
        bias = True,
        dtype = None,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                (len(in_features_ls), out_features, max(in_features_ls)), dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(in_features_ls), out_features), dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(in_features_ls), torch.as_tensor(in_features_ls)),
                "num_feats max_feat -> num_feats 1 max_feat",
            ),
            persistent=False,
        )
        self.register_buffer(
            "in_features_buffer",
            torch.tensor(in_features_ls),
            persistent=False,
        )

    def reset_parameters(self):
        for idx, feat_size in enumerate(self.in_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :, :feat_size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, :, feat_size:])
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[idx, :, :feat_size]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx], -bound, bound)

    def forward(
        self,
        x,
        in_feat_size,
    ):
        out = 0
        for idx, feat_size in enumerate(self.in_features_ls):
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                torch.eq(in_feat_size, feat_size).unsqueeze(-1)
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out

    def extra_repr(self):
        return (
            f"in_features_ls={self.in_features_ls}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )

class MultiOutSizeLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features_ls,
        dim = 1,
        bias = True,
        dtype = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_ls = out_features_ls
        self.dim = dim

        self.weight = nn.Parameter(
            torch.empty(
                (len(out_features_ls), max(out_features_ls), in_features), dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(out_features_ls), max(out_features_ls)), dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(out_features_ls), torch.as_tensor(out_features_ls)),
                "num_feats max_feat -> num_feats max_feat 1",
            ),
            persistent=False,
        )
        self.register_buffer(
            "out_features_buffer",
            torch.tensor(out_features_ls),
            persistent=False,
        )

    def reset_parameters(self):
        for idx, feat_size in enumerate(self.out_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :feat_size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, feat_size:])
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[idx, :feat_size]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx, :feat_size], -bound, bound)
                nn.init.zeros_(self.bias[idx, feat_size:])

    def forward(
        self,
        x,
        out_feat_size,
    ):
        out = 0
        for idx, feat_size in enumerate(self.out_features_ls):
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                torch.eq(out_feat_size, feat_size // self.dim).unsqueeze(-1)
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features_ls={self.out_features_ls}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_groups,
        bias = True,
        norm_layer = nn.LayerNorm,
        softmax_scale = None,
        attn_dropout_p = 0.0,
        var_attn_bias = None,
        time_attn_bias = None,
        var_qk_proj = None,
        time_qk_proj = None,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.heads_per_group = num_heads // num_groups
        self.var_attn_bias = var_attn_bias() if var_attn_bias is not None else None
        self.time_attn_bias = time_attn_bias() if time_attn_bias is not None else None
        self.var_qk_proj = var_qk_proj() if var_qk_proj is not None else None
        self.time_qk_proj = time_qk_proj() if time_qk_proj is not None else None

        self.softmax_scale = softmax_scale or 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.q_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.attn_dropout_p = attn_dropout_p
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _get_var_id(
        self,
        query,
        key,
        query_var_id,
        kv_var_id,
    ):
        if self.var_attn_bias is not None or self.var_qk_proj is not None:
            if query_var_id is None:
                query_var_id = repeat(
                    torch.zeros((), device=query.device, dtype=torch.long),
                    f" -> {' '.join(map(str, query.shape[:-4]))} 1 1 {query.shape[-2]}",
                )
            else:
                query_var_id = rearrange(query_var_id, "... q_len -> ... 1 1 q_len")

            if kv_var_id is None:
                kv_var_id = repeat(
                    torch.zeros((), device=key.device, dtype=torch.long),
                    f" -> {' '.join(map(str, key.shape[:-4]))} 1 1 {key.shape[-2]}",
                )
            else:
                kv_var_id = rearrange(kv_var_id, "... kv_len -> ... 1 1 kv_len")

        return query_var_id, kv_var_id

    def _get_time_id(
        self,
        query,
        key,
        query_time_id,
        kv_time_id,
    ):
        if self.time_attn_bias is not None or self.time_qk_proj is not None:
            if query_time_id is None:
                query_time_id = repeat(
                    torch.arange(
                        query.shape[-2], device=query.device, dtype=torch.long
                    ),
                    f"q_len -> {' '.join(map(str, query.shape[:-4]))} 1 1 q_len",
                )
            else:
                query_time_id = rearrange(query_time_id, "... q_len -> ... 1 1 q_len")

            if kv_time_id is None:
                kv_time_id = repeat(
                    torch.arange(key.shape[-2], device=key.device, dtype=torch.long),
                    f"kv_len -> {' '.join(map(str, key.shape[:-4]))} 1 1 kv_len",
                )
            else:
                kv_time_id = rearrange(kv_time_id, "... kv_len-> ... 1 1 kv_len")

        return query_time_id, kv_time_id

    def _update_attn_mask(
        self,
        attn_mask,
        query,
        key,
        query_var_id = None,
        kv_var_id = None,
        query_time_id = None,
        kv_time_id = None,
    ):
        if attn_mask is not None:
            attn_mask = rearrange(
                attn_mask,
                "... q_len kv_len -> ... 1 1 q_len kv_len",
            )

        attn_bias = 0
        if self.var_attn_bias is not None:
            attn_bias = attn_bias + self.var_attn_bias(
                query,
                key,
                query_id=query_var_id,
                kv_id=kv_var_id,
            )

        if self.time_attn_bias is not None:
            attn_bias = attn_bias + self.time_attn_bias(
                query,
                key,
                query_id=query_time_id,
                kv_id=kv_time_id,
            )

        attn_mask = (
            attn_mask
            if isinstance(attn_bias, int)
            else (
                attn_bias
                if attn_mask is None
                else attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
            )
        )
        return attn_mask

    def _qk_proj(
        self,
        query,
        key,
        query_var_id,
        kv_var_id,
        query_time_id,
        kv_time_id
    ):
        if self.var_qk_proj is not None:
            query, key = self.var_qk_proj(
                query, key, query_id=query_var_id, kv_id=kv_var_id
            )

        if self.time_qk_proj is not None:
            query, key = self.time_qk_proj(
                query, key, query_id=query_time_id, kv_id=kv_time_id
            )

        return query, key

    def forward(
        self,
        query,
        key,
        value,
        attn_mask = None,
        query_var_id = None,
        kv_var_id = None,
        query_time_id = None,
        kv_time_id = None,
    ):
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self.q_norm(
            rearrange(
                query,
                "... q_len (group hpg dim) -> ... group hpg q_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )
        key = self.k_norm(
            repeat(
                key,
                "... kv_len (group dim) -> ... group hpg kv_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )
        value = repeat(
            value,
            "... kv_len (group dim) -> ... group hpg kv_len dim",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )

        query_var_id, kv_var_id = self._get_var_id(query, key, query_var_id, kv_var_id)
        query_time_id, kv_time_id = self._get_time_id(
            query,
            key,
            query_time_id,
            kv_time_id,
        )

        attn_mask = self._update_attn_mask(
            attn_mask,
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        query, key = self._qk_proj(
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            scale=self.softmax_scale,
        )
        out = rearrange(out, "... group hpg q_len dim -> ... q_len (group hpg dim)")
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim = None,
        out_dim = None,
        activation = F.gelu,
        bias = True,
        ffn_dropout_p = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * in_dim
        out_dim = out_dim or in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias
        self.ffn_dropout_p = ffn_dropout_p

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(ffn_dropout_p)
        self.dropout2 = nn.Dropout(ffn_dropout_p)
        self.activation = activation

    def forward(
        self,
        x,
        centroid = None,
    ):
        x = self._in_proj(x)
        return self.dropout2(self.fc2(self.dropout1(x)))

    def _in_proj(self, x):
        return self.activation(self.fc1(x))

class GatedLinearUnitFeedForward(FeedForward):
    def __init__(
        self,
        in_dim,
        hidden_dim = None,
        out_dim = None,
        activation = F.silu,
        bias = True,
        ffn_dropout_p = 0.0,
    ):
        super().__init__(
            in_dim,
            hidden_dim=hidden_dim or self.adjust_hidden_dim(4 * in_dim),
            out_dim=out_dim,
            activation=activation,
            bias=bias,
            ffn_dropout_p=ffn_dropout_p,
        )
        self.fc_gate = nn.Linear(self.in_dim, self.hidden_dim, bias=self.bias)

    @staticmethod
    def adjust_hidden_dim(dim):
        return (int(dim * 2 / 3) + 7) // 8 * 8

    def _in_proj(self, x):
        return self.activation(self.fc_gate(x)) * self.fc1(x)
    
class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        self_attn: GroupedQueryAttention,
        ffn: FeedForward,
        norm1,
        norm2,
        post_attn_dropout_p = 0.0,
        pre_norm = True,
        diff_step_proj = None,
        adaln_attn = None,
        adaln_ffn = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout_p = post_attn_dropout_p

        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1 or nn.Identity()
        self.norm2 = norm2 or nn.Identity()
        self.dropout = nn.Dropout(post_attn_dropout_p)
        self.diff_step_proj = diff_step_proj or nn.Identity()
        self.adaln_attn = adaln_attn or self._dummy_adaln
        self.adaln_ffn = adaln_ffn or self._dummy_adaln

    def forward(
        self,
        x,
        attn_mask = None,
        sample_id = None,
        var_id = None,
        time_id = None,
        centroid = None,
        diff_step = None,
        prediction_mask = None,
        time_emb = None,
    ):
        diff_step = self.diff_step_proj(diff_step)
        sa_func = partial(
            self._sa_block,
            attn_mask=attn_mask, 
            var_id=var_id, 
            time_id=time_id
        )
        ffn_func = partial(
            self.ffn,
            centroid=centroid,
        )

        if self.pre_norm:
            x = x + self.adaln_attn(
                self.norm1(x),
                c=diff_step,
                mask=prediction_mask,
                func=sa_func
            )
            x = x + self.adaln_ffn(
                self.norm2(x),
                c=diff_step,
                mask=prediction_mask,
                func=ffn_func
            )
        else:
            x = self.norm1(
                x + self.adaln_attn(
                    x,
                    c=diff_step,
                    mask=prediction_mask,
                    func=sa_func,
                )
            )
            x = self.norm2(
                x + self.adaln_ffn(
                    x,
                    c=diff_step,
                    mask=prediction_mask,
                    func=ffn_func
                )
            )
        return x

    def _sa_block(
        self,
        x,
        attn_mask = None,
        var_id = None,
        time_id = None,
    ):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            query_var_id=var_id,
            kv_var_id=var_id,
            query_time_id=time_id,
            kv_time_id=time_id,
        )
        return self.dropout(x)
    
    @staticmethod
    def _dummy_adaln(
        x,
        c,
        mask = None,
        func = None,
    ):
        if c is None:
            return x
        mask = AdaLN.get_mask(mask)
        func = AdaLN.get_func(func)
        return func(x + c * mask)
    
class AdaLN(nn.Module):
    n_params = 2
    
    def __init__(
        self,
        in_dim,
        out_dim = None,
        zero_init = True,
        gate = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim
        self.gate = gate

        if self.gate:
            self.n_params += 1

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.in_dim, 
                self.n_params*self.out_dim)
        )
        if zero_init:
            self._init_params()

    def _init_params(self):
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(
        self, 
        x,
        c,
        mask = None,
        func = None,
    ):
        func = self.get_func(func)
        mask = self.get_mask(mask)
        params = self.mlp(c) * mask

        if self.gate:
            scale, shift, gate = params.chunk(self.n_params, dim=-1)
            gate = gate + 1 * ~mask
        else:
            scale, shift = params.chunk(self.n_params, dim=-1)
            gate = 1
        x = self.rescale(x, scale, shift)
        x = gate * func(x)
        return x
    
    @staticmethod
    def get_mask(mask):
        return True if mask is None else mask.unsqueeze(-1)
    
    @staticmethod
    def get_func(func):
        return func or nn.Identity()

    @staticmethod
    def rescale(
        x,
        scale,
        shift,
    ):     
        return x * (1 + scale) + shift

class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps = 1e-5,
        weight = True,
        dtype = None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.mean_dim = tuple(range(-len(normalized_shape), 0))

        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        output = x * torch.rsqrt(
            x.pow(2).mean(dim=self.mean_dim, keepdim=True) + self.eps
        )
        if self.weight is not None:
            return output * self.weight
        return output

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"weight={self.weight is not None}"
        )

class AttentionBias(nn.Module, abc.ABC):
    def __init__(
        self,
        dim,
        num_heads,
        num_groups,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.head_dim = dim // num_heads

    @abc.abstractmethod
    def forward(
        self,
        query,
        key,
        query_id,
        kv_id,
    ): ...

class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim, num_heads, num_groups):
        super().__init__(dim, num_heads, num_groups)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

    def forward(
        self,
        query,
        key,
        query_id,
        kv_id,
    ):
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = rearrange(
            ~ind * weight[:1] + ind * weight[1:],
            "... 1 (group hpg) q_len kv_len -> ... group hpg q_len kv_len",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )
        return bias

class Projection(nn.Module, abc.ABC):
    def __init__(self, proj_width, num_heads, num_groups, **kwargs):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups

    @abc.abstractmethod
    def forward(
        self,
        x,
        seq_id,
    ): ...

class RotaryProjection(Projection):
    def __init__(
        self,
        *,
        proj_width,
        num_heads,
        num_groups,
        max_len = 512,
        base = 10000,
    ):
        super().__init__(proj_width, num_heads, num_groups)
        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x):
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)

    def forward(
        self,
        x,
        seq_id,
    ):
        self._init_freq(max_len=seq_id.max() + 1)
        rot_cos = self.cos[seq_id.long()]
        rot_sin = self.sin[seq_id.long()]
        return rot_cos * x + rot_sin * self._rotate(x)

class QueryKeyProjection(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_groups,
        proj_layer,
        kwargs = None,
        key_proj_layer = None,
        key_kwargs = None,
        partial_factor = None,
    ):
        super().__init__()
        if partial_factor is not None:
            assert (
                0.0 <= partial_factor[0] < partial_factor[1] <= 1.0
            ), f"got {partial_factor[0]}, {partial_factor[1]}"
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            num_groups=num_groups,
            **(kwargs or {}),
        )
        if key_proj_layer is None:
            self.key_proj = self.query_proj
        else:
            self.key_proj = key_proj_layer(
                proj_width=self.proj_width,
                num_heads=num_heads,
                num_groups=num_groups,
                **(key_kwargs or {}),
            )

    @cached_property
    def proj_width(self):
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self):
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(
        self,
        query,
        key,
        query_id,
        kv_id,
    ):
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_id=query_id)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id)
            key = self.key_proj(key, seq_id=kv_id)
        return query, key

class DiffusionStepProjector(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim = None,
        activation = F.silu,
        bias = True,
        dropout_p = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim
        self.bias = bias
        self.dropout_p = dropout_p

        self.activation = activation
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(
        self,
        x,
    ):
        x = self.dropout1(self.activation(x))
        x = self.dropout2(self.fc(x))
        return x