# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.layers import DropPath
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import \
        flash_attn_varlen_qkvpacked_func
    has_flash_attn = True
except:
    print('FlashAttention2 is not installed.')
    has_flash_attn = False

import math

logger = logging.get_logger(__name__)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                          device=qkv.device)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                             indices, batch_size, seqlen),
                                   'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm')
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning('discovered apex but it failed to load, falling back to InternRMSNorm')
    pass


NORM2FN = {
    'rms_norm': InternRMSNorm,
    'layer_norm': nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, k  

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        q, k, v = qkv.unbind(2)
        if self.qk_normalization:
            
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs, None, k

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x, attn ,k = self._naive_attn(hidden_states) if not self.use_flash_attn else self._flash_attn(hidden_states)
        return {'hidden_states':x,
                'attn_scores':attn,
                'k_states':k}


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(
            self,
            hidden_states: torch.Tensor,
            tome_r=0,
            method="average"
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        attn_outputs = self.attn(self.norm1(hidden_states).to(hidden_states.dtype))
        #print("attn_out",attn_outputs)
        hidden_states = hidden_states + self.drop_path1(attn_outputs['hidden_states'] * self.ls1)


        B, L ,D = hidden_states.shape
        #print("bld" ,B,L,D,attn_outputs['k_states'].shape,hidden_states.shape)
        merged_idx = None
    
        # 直接合并（精简版 ToMe）
        if tome_r > 0:
            #for b in range(B):
            metric = attn_outputs['k_states'].mean(2)  # [b,seq_len, head_dim]
            # 这里假设 batch=1，如果有多 batch 需要 reshape
            # 跳过cls
            cls_token = hidden_states[:, :1, :]
            merged_tokens, merged_idx = bipartite_soft_fusion(
                metric[:,1:,:],  # [1, T, C]
                hidden_states[:,1:,:],     # [1, T, C]
                tome_r,
                method=method
            )
            # 拼回 CLS token
            hidden_states = torch.cat([cls_token, merged_tokens], dim=1)
            merged_idx = merged_idx + 1  # 让索引对齐到原始 hidden_states 的位置
            merged_idx = torch.cat([torch.zeros((B,1),device=merged_idx.device), merged_idx], dim=1)
            #print(hidden_states.shape,merged_idx.shape,merged_idx)
            # hidden_states = hidden_states.squeeze(0)


            # merged_idx 是 [B, kept_T]，但这里 batch=1 时 squeeze 了
            # 如果 batch=1，可以先还原成一维索引
            
            # merged_idx_flat = merged_idx


        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2)

        #return hidden_states
    
        return {'hidden_states':hidden_states,
                'attn_scores':attn_outputs['attn_scores'],
                'k_states':attn_outputs['k_states'],
                'merged_idx': merged_idx
                }
    
def bipartite_soft_fusion(metric: torch.Tensor, x: torch.Tensor, r: int, mode="mean", method="average"):
    """
    简化版 ToMe bipartite soft matching，只做 token 合并。
    metric: [B, T, C] 特征向量（通常是 K）
    x:      [B, T, C] 要合并的 token 特征
    r:      要合并的 token 对数（最多 50%）
    返回:
        merged_x: 合并后的 token 特征
        merged_idx: 合并后 token 在合并前的绝对位置索引
    """
    B, T, C = x.shape
    r = min(r, T // 2)
    if r <= 0:
        return x, torch.arange(T, device=x.device)[None, :].expand(B, -1)

    # 本层的原始顺序索引
    idx = torch.arange(T, device=x.device)[None, :].expand(B, -1)

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        #print("a b s",a.shape,b.shape,scores.shape)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]   # 未合并的 A 集
        src_idx = edge_idx[..., :r, :]   # 要合并的 A 集
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # 对应的 B 集位置

    # 同步位置索引
    idx_a, idx_b = idx[..., ::2], idx[..., 1::2]
    unm_pos = idx_a.gather(dim=-1, index=unm_idx.squeeze(-1))  # 未合并 A 集的原位置
    dst_pos = idx_b  # B 集 token 的原位置（合并后继承）

    # 特征合并
    src, dst = x[..., ::2, :], x[..., 1::2, :]
    #print("unm",unm_idx.shape,src_idx.shape,dst_idx.shape)
    #print(src.shape,dst.shape)
    unm = src.gather(dim=-2, index=unm_idx.expand(B, math.ceil(T / 2) - r, C))
    
    if method == "pruned":
        # 直接丢掉 src，不做合并
        pass
    elif method == "average":
        src = src.gather(dim=-2, index=src_idx.expand(B, r, C))
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce="mean")
    elif method == "mlerp":
        src = src.gather(dim=-2, index=src_idx.expand(B, r, C))
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce="mean")
        # 范数修正
        src_norm = src.norm(dim=-1, keepdim=True)
        dst_norm = dst.norm(dim=-1, keepdim=True)
        max_norm = torch.max(dst_norm, src_norm)  # 这里简化成两者 max
        dst = dst / (dst.norm(dim=-1, keepdim=True) + 1e-6) * max_norm

    merged_idx = torch.cat([unm_pos, dst_pos], dim=1)
    return torch.cat([unm, dst], dim=1), merged_idx


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


class InternVisionEncoder_Sparse(InternVisionEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)

        self.update_attention_layer=False

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds



        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # if self.config.DART_config is not None and self.config.DART_config['vit_Sparse'] and self.config.DART_config['vit_attn_scores_choose']\
        #     and not self.update_attention_layer:
        #     k = self.config.DART_config['vit_pruned_layer'] - 1
        #     self.layers[k].attn.use_flash_attn = False
        #     self.update_attention_layer=True

        #--------------------BEGIN------------------------------------
        hidden_states_pkg = {'hidden_states':hidden_states, # [batch_size,seq_len, embed_dim]
                            'k_states':None,                # [batch_size,seq_len, num_heads, head_dim]  TODO:优化显存占用
                            'attn_scores':None}                 # [batch_size, nheads,seqlen,seqlen] TODO: 优化显存占用
        #frame_counts = torch.zeros(1, device=device)
        hidden_states_prev = None
        #print("hid state",hidden_states.shape)

        target_keep_ratio = 1.0 - self.config.DART_config['vit_reduction_ratio']  # 0.2
        current_keep_ratio = 1.0  # 初始保留 100%
        min_layer_keep = 0.8  # 每层最多剪去20%
        total_prune_layer = math.ceil(math.log(target_keep_ratio) / math.log(min_layer_keep))  # 需要进行剪枝的层数
        
        for idx, blk in enumerate(self.layers):
            #print("hid state",hidden_states.shape)
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states_pkg = torch.utils.checkpoint.checkpoint(
                    blk,
                    hidden_states
                )
                hidden_states = hidden_states_pkg['hidden_states']
            else:
                DART_config = self.config.DART_config
                if DART_config is not None and DART_config['vit_Sparse']:
                    K = DART_config['vit_pruned_layer']
                    seq_len = hidden_states_pkg['hidden_states'].shape[1]  # (B, N, C) 取 N

                    # if K - 1 > 0 and idx == K - 1 and DART_config['vit_diff_choose'] and seq_len > 1:
                    #     hidden_states_prev = hidden_states_pkg['hidden_states']  # (B, N, C)

                    if idx >= K and seq_len > 1 and\
                        current_keep_ratio > target_keep_ratio:
                        # 如果还没达到目标比例
                        # 计算本层需要的保留比例
                        # 每层最多剪 50% → 最少保留 50%
                        #desired_keep_ratio = max(target_keep_ratio / current_keep_ratio, 0.5)
                        desired_keep_ratio = max(target_keep_ratio / current_keep_ratio, 0.8)

                        # 本层要保留的 token 数
                        keep_tokens = int(seq_len * desired_keep_ratio)
                        # 本层要合并的 token 对数
                        tome_r = seq_len - keep_tokens
                        tome_r = (tome_r // 2) * 2  # 取下限
                        if desired_keep_ratio == target_keep_ratio / current_keep_ratio:
                            max_square_root = int(math.sqrt(seq_len-1-tome_r))  # 向下取整平方根
                            tome_r = max_square_root ** 2  # 得到平方数

                            TOKEN_TOPK = 0
                            for r in range(max_square_root, 0, -1):
                                candidate = r ** 2
                                if candidate % 2 == 0:
                                    TOKEN_TOPK = int(candidate)
                                    break
                            tome_r = seq_len-1 -TOKEN_TOPK

                        if idx - K  < total_prune_layer/2:
                            method = "pruned"
                        else:
                            method = "average"
                        hidden_states_pkg = blk(hidden_states,tome_r=tome_r,method=method)
                        hidden_states = hidden_states_pkg['hidden_states']

                        # 更新当前保留比例
                        current_keep_ratio *= desired_keep_ratio
                        

                
                    else:
                        hidden_states_pkg = blk(hidden_states)
                        hidden_states = hidden_states_pkg['hidden_states']
                else:
                    hidden_states_pkg = blk(hidden_states)
                    hidden_states = hidden_states_pkg['hidden_states']


            #hidden_states = hidden_states_pkg['hidden_states']

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


                    
                    

    
    def get_retained_image_token(self, config, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        # any_state [seq_len, num_heads, head_dim]
        DART_config = config.DART_config
        #K = DART_config['vit_pruned_layer']
        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]

        pivot_image_token = DART_config['pivot_image_token']
        # pivot_text_token = DART_config['pivot_text_token']

        reduction_ratio = DART_config['vit_reduction_ratio']
        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio) / (pivot_image_token)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     TOKEN_TOPK = TOKEN_TOPK_down
        # else:
        #     TOKEN_TOPK = TOKEN_TOPK_up

        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio) / (pivot_image_token)

        # # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        # import math
        # max_square_root = int(math.sqrt(pivot_image_token*TOKEN_TOPK_RAW))  # 向下取整平方根
        # TOKEN_TOPK = max_square_root ** 2  # 得到平方数

        # 原始计算
        TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio) / pivot_image_token

        # 最大可能的平方根
        max_square_root = int(math.sqrt(pivot_image_token * (TOKEN_TOPK_RAW+1)))

        # 从 max_square_root 往下找，直到找到既是平方数又是 pivot_image_token 的倍数
        TOKEN_TOPK = 0
        for r in range(max_square_root, 0, -1):
            candidate = r ** 2
            if candidate % pivot_image_token == 0:
                TOKEN_TOPK = int(candidate /pivot_image_token - 1)
                break
        #print("topk",TOKEN_TOPK)


        # 向下取
        # TOKEN_TOPK = TOKEN_TOPK_down - 1
        device = last_layer_state.device

        device = last_layer_state.device

        any_states = any_states.reshape(any_states.shape[0], -1) # [seq_len, embed_dim]

        k_states_image_token = any_states[image_token_start_index:image_token_start_index + image_token_length, :] # [valid_seq_len, hidden_dim]
        #k_states_query_token = any_states[image_token_start_index + image_token_length:, :]

        k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1) # [valid_seq_len]
        #k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1) # [valid_seq_len]

        image_indices = (k_states_image_token_L1_norm.topk(pivot_image_token).indices + image_token_start_index).tolist() # pivot indices (list)
        #query_indices = (k_states_query_token_L1_norm.topk(pivot_text_token).indices + image_token_start_index + image_token_length).tolist() # pivot indices (list)
        #indices_set = set(image_indices + query_indices) # merge 2 lists
        indices_set = set(image_indices)

        valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(image_indices)

        valid_indices_list = list(valid_indices)
        for item in list(indices_set):
            valid_vectors = last_layer_state[valid_indices_list, :] # last_layer_state中待处理image token的对应向量 [valid_seq_len - num_pivot_tokens, hidden_dim]
            cos_sim = -torch.nn.functional.cosine_similarity(last_layer_state[item, :], valid_vectors, dim=-1) # 计算余弦相似度 [valid_seq_len - num_pivot_tokens]
            #print("cossim",cos_sim.shape)
            top_k_indices = cos_sim.topk(TOKEN_TOPK).indices

            top_k_real_indices = [valid_indices_list[i] for i in top_k_indices] # 待保留的image token的index
            indices_set.update(top_k_real_indices)

            valid_indices.difference_update(top_k_real_indices)
            valid_indices_list = list(valid_indices) 

            retained_image_tokens_index = torch.tensor(list(indices_set), device=device)

        return retained_image_tokens_index

    def get_retained_image_token_random(self, config, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        reduction_ratio = DART_config['vit_reduction_ratio']

        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]
        device = last_layer_state.device

        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     retained_count = TOKEN_TOPK_down
        # else:
        #     retained_count = TOKEN_TOPK_up

        # 计算原始值
        TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)

        # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        
        max_square_root = int(math.sqrt(TOKEN_TOPK_RAW))  # 向下取整平方根
        retained_count = max_square_root ** 2  # 得到平方数

        # 向下取
        # retained_count = TOKEN_TOPK_down
        # 确保至少保留一个token
        retained_count = max(retained_count, 1)
        
        # 生成所有图像token的索引并随机选择
        all_indices = torch.arange(image_token_start_index, image_token_start_index + image_token_length, device=device)
        retained_indices = all_indices[torch.randperm(all_indices.size(0))[:retained_count]]
        
        return retained_indices

    def get_retained_image_token_attn_scores(self, config, last_layer_state: torch.Tensor, any_states: torch.Tensor, attn_scores: torch.Tensor) -> torch.Tensor:
        # any_state [seq_len, num_heads, head_dim]
        DART_config = config.DART_config
        # K = DART_config['K']
        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]

        reduction_ratio = DART_config['vit_reduction_ratio']
        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     TOKEN_TOPK = TOKEN_TOPK_down
        # else:
        #     TOKEN_TOPK = TOKEN_TOPK_up

        # 计算原始值
        TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)

        # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        
        max_square_root = int(math.sqrt(TOKEN_TOPK_RAW))  # 向下取整平方根
        TOKEN_TOPK = max_square_root ** 2  # 得到平方数
        device = last_layer_state.device

        #attn_scores.squeeze(0) # [nheads,seqlen,seqlen]
        attn_scores = attn_scores.sum(dim=-2) # 沿着query维度求和
        attn_scores = attn_scores.mean(dim=0) # 对不同的注意力头求平均
        top_k_indices = attn_scores.topk(TOKEN_TOPK).indices
        top_k_real_indices = top_k_indices
        retained_image_tokens_index = torch.tensor(top_k_real_indices, device=device)
        return retained_image_tokens_index

    def get_retained_image_token_diff(self,config,hidden_states_prev,hidden_states_cur,last_layer_state):
        DART_config = config.DART_config
        #K = DART_config['K']
        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]

        reduction_ratio = DART_config['vit_reduction_ratio']
        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     TOKEN_TOPK = TOKEN_TOPK_down
        # else:
        #     TOKEN_TOPK = TOKEN_TOPK_up

        # 计算原始值
        TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)
        

        # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        
        max_square_root = int(math.sqrt(TOKEN_TOPK_RAW))  # 向下取整平方根
        TOKEN_TOPK = max_square_root ** 2  # 得到平方数

        #print("topk",TOKEN_TOPK_RAW,TOKEN_TOPK)
        # # 向下取
        # TOKEN_TOPK = TOKEN_TOPK_down - 1
        device = last_layer_state.device

        diff = hidden_states_cur - hidden_states_prev # [seqlen,embed_dim]
        diff_norm = torch.norm(diff,dim=-1) # [seqlen]
        top_k_indices = diff_norm.topk(TOKEN_TOPK).indices
        top_k_real_indices = top_k_indices
        retained_image_tokens_index = torch.tensor(top_k_real_indices, device=device)
        return retained_image_tokens_index

    def get_retained_image_token_pivot_sim(self, config, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        DART_config = config.DART_config
        #K = DART_config['K']
        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]

        reduction_ratio = DART_config['vit_reduction_ratio']
        # # 计算原始值
        # TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)
        # # 向下取4的倍数
        # TOKEN_TOPK_down = int(TOKEN_TOPK_RAW) // 4 * 4
        # # 向上取4的倍数
        # TOKEN_TOPK_up = (int(TOKEN_TOPK_RAW) + 3) // 4 * 4
        # # 选择与原始值更接近的结果
        # if abs(TOKEN_TOPK_RAW - TOKEN_TOPK_down) <= abs(TOKEN_TOPK_RAW - TOKEN_TOPK_up):
        #     TOKEN_TOPK = TOKEN_TOPK_down
        # else:
        #     TOKEN_TOPK = TOKEN_TOPK_up

        # 计算原始值
        TOKEN_TOPK_RAW = image_token_length * (1 - reduction_ratio)

        # 找到比 TOKEN_TOPK_RAW 小的最大平方数
        
        max_square_root = int(math.sqrt(TOKEN_TOPK_RAW))  # 向下取整平方根
        TOKEN_TOPK = max_square_root ** 2  # 得到平方数
        
        device = last_layer_state.device
        pivot_token = last_layer_state.mean(dim=0) # 求出平均token [embed_dim]
        cos_sim = -torch.nn.functional.cosine_similarity(pivot_token, last_layer_state, dim=-1) # 计算余弦相似度
        top_k_indices = cos_sim.topk(TOKEN_TOPK).indices
        top_k_real_indices = top_k_indices
        retained_image_tokens_index = torch.tensor(top_k_real_indices, device=device)
        return retained_image_tokens_index


class InternVisionModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    config_class = InternVisionConfig
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class InternVisionModel_Sparse(InternVisionModel):
    def __init__(self, config: InternVisionConfig):
        super().__init__(config)

        self.encoder = InternVisionEncoder_Sparse(config)