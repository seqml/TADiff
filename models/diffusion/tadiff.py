import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import copy
from functools import partial
from models.diffusion.tadiff_modules import DiffusionStepEmbedder, MultiInSizeLinear, MultiOutSizeLinear, GroupedQueryAttention, FeedForward, GatedLinearUnitFeedForward
from models.diffusion.tadiff_modules import DiffusionTransformerBlock, RMSNorm, AdaLN, BinaryAttentionBias, QueryKeyProjection, RotaryProjection, DiffusionStepProjector
class DiffusionTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers,
        num_heads = None,
        num_groups = None,
        pre_norm = True,
        attn_dropout_p = 0.0,
        dropout_p = 0.0,
        norm_layer = nn.LayerNorm,
        activation = F.silu,
        use_glu = True,
        use_qk_norm = True,
        var_attn_bias_layer = None,
        time_attn_bias_layer = None,
        var_qk_proj_layer = None,
        time_qk_proj_layer = None,
        shared_var_attn_bias = False,
        shared_time_attn_bias = False,
        shared_var_qk_proj = False,
        shared_time_qk_proj = False,
        d_ff = None,
        diff_step_proj = None,
        adaln = None,
    ):
        super().__init__()
        num_heads = num_heads or d_model // 64
        num_groups = num_groups or num_heads

        var_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            var_attn_bias_layer,
            shared_var_attn_bias,
        )
        time_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            time_attn_bias_layer,
            shared_time_attn_bias,
        )
        var_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, var_qk_proj_layer, shared_var_qk_proj
        )
        time_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, time_qk_proj_layer, shared_time_qk_proj
        )

        get_self_attn = partial(
            GroupedQueryAttention,
            dim=d_model,
            num_heads=num_heads,
            num_groups=num_groups,
            bias=False,
            norm_layer=norm_layer if use_qk_norm else None,
            softmax_scale=None,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )
        get_ffn = partial(
            GatedLinearUnitFeedForward if use_glu else FeedForward,
            in_dim=d_model,
            hidden_dim=d_ff,
            out_dim=None,
            activation=activation,
            bias=False,
            ffn_dropout_p=dropout_p,
        )
        get_encoder_layer_norm = partial(norm_layer, d_model)
        get_diff_step_proj = partial(
            self.get_cond_projector,
            d_model,
            diff_step_proj,
        )
        get_adaln = partial(
            self.get_cond_projector,
            d_model, 
            adaln,
        )

        self.layers = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    self_attn=get_self_attn(),
                    ffn=get_ffn(),
                    norm1=get_encoder_layer_norm(),
                    norm2=get_encoder_layer_norm(),
                    pre_norm=pre_norm,
                    post_attn_dropout_p=dropout_p,
                    diff_step_proj=get_diff_step_proj(),
                    adaln_attn=get_adaln(),
                    adaln_ffn=get_adaln(),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = norm_layer(d_model)

    @staticmethod
    def get_layer(
        dim,
        num_heads,
        num_groups,
        layer,
        shared_layer,
    ):
        if layer is None:
            return None
        if shared_layer:
            module = layer(dim=dim, num_heads=num_heads, num_groups=num_groups)
            return lambda: module
        return partial(layer, dim=dim, num_heads=num_heads, num_groups=num_groups)
    
    @staticmethod
    def get_cond_projector(
        d_model,
        projector = None,
    ):
        if projector is None:
            return None
        return projector(in_dim=d_model, out_dim=d_model)


    def forward(
        self,
        x,
        attn_mask = None,
        sample_id = None,
        var_id = None,
        time_id = None,
        diff_step = None,
        prediction_mask = None,
        time_emb = None,
    ):
        for layer in self.layers:
            x = layer(
                x, 
                attn_mask,
                sample_id=sample_id,
                var_id=var_id, 
                time_id=time_id, 
                diff_step=diff_step,
                prediction_mask=prediction_mask,
                time_emb=time_emb,
            )
        return self.norm(x)

class TADiff(nn.Module):
    def __init__(self, config, inputdim=1):
        super().__init__()
        self.d_model = config["channels"]
        self.num_layers = config["num_layers"]
        self.patch_size = config["patch_size"]
        self.max_seq_len = config["max_seq_len"]
        self.learn_sigma = config["learn_sigma"]
        self.diff_step_proj = config["diff_step_proj"]
        self.adaln = config["adaln"]
        print(self.patch_size)
        self.diff_step_embedder = DiffusionStepEmbedder(embedding_dim=self.d_model)
        self.mask_encoding = nn.Embedding(num_embeddings=2, embedding_dim=self.d_model)
        self.in_proj = MultiInSizeLinear(
            in_features_ls=[self.patch_size],
            out_features=self.d_model,
        )
        self.encoder = DiffusionTransformerEncoder(
            self.d_model,
            self.num_layers,
            num_heads=config["num_heads"],
            pre_norm=True,
            attn_dropout_p=config["attn_dropout_p"],
            dropout_p=config["dropout_p"],
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=self.max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
            diff_step_proj=partial(DiffusionStepProjector) if self.diff_step_proj else None,
            adaln=partial(AdaLN) if self.adaln else None,
        )
        self.n_chunk = 2 if self.learn_sigma else 1
        self.out_proj = MultiOutSizeLinear(
            in_features=self.d_model,
            out_features_ls=tuple(ps*self.n_chunk for ps in [self.patch_size]),
            dim=self.n_chunk,
        )
        if self.adaln:
            self.out_norm = nn.LayerNorm(self.d_model)
            self.adaln_out = AdaLN(self.d_model, self.d_model, gate=False)
    
    def ts_patchify(self, x):
        x = x.unfold(dimension=3, size=self.patch_size, step=self.patch_size)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, Nl, Pl)
        return x
    
    def mask_patchify(self, mask):
        mask = mask.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        mask, _ = torch.max(mask, dim=3)
        mask = mask[:,0,:]
        return mask

    def forward(self, x_raw, prediction_mask, tp, attr_emb_raw, diffusion_step):
        x = self.ts_patchify(x_raw)
        B, N, P = x.shape
        prediction_mask = self.mask_patchify(prediction_mask)
        tp = torch.arange(0, N)[None].expand(B, N).to(device=x.device)
        diffusion_step = diffusion_step[:,None].expand(B, N)

        patch_size_list = torch.zeros_like(diffusion_step) + self.patch_size
        reprs = self.in_proj(x, patch_size_list)
        if attr_emb_raw is None:
            attr_emb = torch.zeros_like(reprs)
        else:
            attr_emb = attr_emb_raw[:,0,0,:][:,None,:]
        mask_emb = self.mask_encoding(prediction_mask.long())
        masked_reprs = reprs + mask_emb
        diff_step_reprs = self.diff_step_embedder(diffusion_step)
        diff_step_reprs = diff_step_reprs * prediction_mask.unsqueeze(-1)
        condition_reprs = diff_step_reprs + attr_emb
        out_reprs = self.encoder(
            masked_reprs,
            attn_mask=None,
            time_id=tp,
            diff_step=condition_reprs,
            prediction_mask=prediction_mask,
            time_emb=None,
        )

        if self.adaln:
            out_reprs = self.adaln_out(
                self.out_norm(out_reprs),
                c=diff_step_reprs,
                mask=prediction_mask,
            )

        outputs = self.out_process(
            reprs=out_reprs,
            patch_size=patch_size_list
        )
        return {
            "pred_noise": outputs,
            "patch_embedding": reprs,
            "last_hiddenstate": out_reprs,
            "loss_dict": {}
        }
    
    def out_process(
            self, 
            reprs, 
            patch_size,
        ):
        outputs = self.out_proj(reprs, patch_size)
        if self.n_chunk > 1:
            seq, var = outputs.chunk(self.n_chunk, dim=-1)
            outputs = torch.cat([seq, var], dim=1)
        else:
            outputs = outputs
        outputs = outputs.reshape((outputs.shape[0], 1, -1))
        return outputs