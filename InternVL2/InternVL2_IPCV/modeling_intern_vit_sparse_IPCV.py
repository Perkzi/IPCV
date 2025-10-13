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
            sparse_vit_saved=None
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        if sparse_vit_saved is not None:
            saved = sparse_vit_saved  
            B, L, D = saved["removed_states"].size(0), saved["orig_seq_len"], saved["removed_states"].size(2)
            device = hidden_states.device  

            # global new_kept / orig_kept
            K = saved["orig_kept_states"].shape[1]
            flat_new_kept = hidden_states.reshape(B*K, D)  
            flat_orig_kept = saved["orig_kept_states"].reshape(B*K, D)

            full_states_patch = []
            for b in range(B):
                rem_to_kept_idx = saved["rem_to_kept_idx"][b]  # [R, topk],  Global Index
                removed_indexs_per_batch = saved["removed_indices"][b]
                unique_idx = saved["unique_idx"][b]
                inv_idx = saved["inv_idx"][b]

                # compute delta (Global Index)
                delta_unique = flat_new_kept[unique_idx] - flat_orig_kept[unique_idx]  # [U, D]

                inv_idx = inv_idx.view(rem_to_kept_idx.shape)  # [R, topk]
                avg_delta_removed = delta_unique[inv_idx].mean(dim=1)  # [R, D]

                # prepare full_states
                full_states = torch.zeros(L, D, device=device, dtype=flat_new_kept.dtype)
                
                full_states[saved["keep_indexs"][b]] = hidden_states[b]  
                full_states[removed_indexs_per_batch] = saved["removed_states"][b] + avg_delta_removed

                full_states_patch.append(full_states)

            full_states_patch = torch.stack(full_states_patch, dim=0)
            hidden_states = full_states_patch  

        attn_outputs = self.attn(self.norm1(hidden_states).to(hidden_states.dtype))
        #print("attn_out",attn_outputs)
        hidden_states = hidden_states + self.drop_path1(attn_outputs['hidden_states'] * self.ls1)

        if sparse_vit_saved is not None:
            # prune again
            pruned = []
            for b in range(B):
                pruned.append(hidden_states[b, saved["keep_indexs"][b], :])  # (K, C)
            hidden_states = torch.stack(pruned, dim=0)  # (B, K, C)

        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2)

        #return hidden_states
    
        return {'hidden_states':hidden_states,
                'attn_scores':attn_outputs['attn_scores'],
                'k_states':attn_outputs['k_states']}


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


class IPCV_ViT(InternVisionEncoder):
    """
    IPCV implementation on ViT
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

        #--------------------BEGIN------------------------------------
        hidden_states_pkg = {'hidden_states':hidden_states, # [batch_size,seq_len, embed_dim]
                            'k_states':None,                # [batch_size,seq_len, num_heads, head_dim]  
                            'attn_scores':None}                 # [batch_size, nheads,seqlen,seqlen] 
        #frame_counts = torch.zeros(1, device=device)
        hidden_states_prev = None
        #print("hid state",hidden_states.shape,len(self.layers))

        for idx, blk in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states_pkg = torch.utils.checkpoint.checkpoint(
                    blk,
                    hidden_states
                )
                hidden_states = hidden_states_pkg['hidden_states']
            else:
                Sparse_config = self.config.Sparse_config
                if Sparse_config is not None and Sparse_config['vit_Sparse']:
                    K = Sparse_config['vit_pruned_layer']
                    seq_len = hidden_states_pkg['hidden_states'].shape[1]  # (B, N, C) 

                    if K - 1 > 0 and idx == K - 1 and seq_len > 1:
                        hidden_states_prev = hidden_states_pkg['hidden_states']  # (B, N, C)

                    if idx == K and seq_len > 1:
                        device = hidden_states_pkg['hidden_states'].device
                        last_layer_state = hidden_states_pkg['hidden_states'].detach().clone()  # (B, N, C)
                        k_states = hidden_states_pkg['k_states']   # (B, heads, N, C) 
                        attn_scores = hidden_states_pkg['attn_scores']  # (B, heads, N, N) 
                        #print("k_states",k_states.shape)

                        orig_seq_len = seq_len

                        B = last_layer_state.shape[0]
                        keep_indexs_per_batch = []
                        removed_indexs_per_batch = []

                        for b in range(B):
            
                            keep_idx = self.get_retained_image_token_diff(
                                self.config,
                                hidden_states_pkg['hidden_states'][b,1:],
                                hidden_states_prev[b,1:],
                                last_layer_state[b,1:]
                            )
                            
                            keep_idx = keep_idx.sort().values
                           
                            # index +1 for cls
                            keep_idx = keep_idx + 1

                            # add cls token idx
                            keep_idx = torch.cat([
                                torch.tensor([0], device=keep_idx.device, dtype=keep_idx.dtype),
                                keep_idx
                            ])

                            removed_mask = torch.ones(orig_seq_len, dtype=torch.bool, device=device)
                            removed_mask[keep_idx] = False
                            removed_idx = torch.nonzero(removed_mask, as_tuple=False).view(-1)

                           
                            keep_indexs_per_batch.append(keep_idx)
                            removed_indexs_per_batch.append(removed_idx)
                        keep_indexs_per_batch = torch.stack(keep_indexs_per_batch, dim=0)
                        removed_indexs_per_batch = torch.stack(removed_indexs_per_batch, dim=0)

                        
                        orig_states = hidden_states_pkg['hidden_states'].detach().clone()

                        # prune for each sample
                        orig_kept_states = []
                        removed_states = []
                        for b in range(B):
                            orig_kept_states.append(orig_states[b, keep_indexs_per_batch[b], :])
                            removed_states.append(orig_states[b, removed_indexs_per_batch[b], :])
                        orig_kept_states = torch.stack(orig_kept_states, dim=0)
                        removed_states = torch.stack(removed_states, dim=0)


                        with torch.no_grad():
                            B, K, D = orig_kept_states.shape
                            orig_kept_states_all = orig_kept_states.reshape(B*K, D)  # (B*K, D)

                            rem_to_kept_idx_patch = []
                            unique_idx_patch = []
                            inv_idx_patch = []

                            for b in range(B):
                                # compute the distance from  removed_states[b] to kept token 
                                dists = torch.cdist(
                                    removed_states[b].float(),
                                    orig_kept_states_all.float(),
                                    p=2.0
                                )
                                # Global kept index corresponding to topk minimum distance
                                Top_K = Sparse_config['Top_K']
                                _, rem_to_kept_idx_global = dists.topk(min(Top_K, orig_kept_states_all.shape[0]), largest=False, dim=1)

                                flat_idx = rem_to_kept_idx_global.view(-1)
                                unique_idx, inv_idx = torch.unique(flat_idx, return_inverse=True)

                                rem_to_kept_idx_patch.append(rem_to_kept_idx_global)
                                unique_idx_patch.append(unique_idx)
                                inv_idx_patch.append(inv_idx)

                            rem_to_kept_idx_patch = torch.stack(rem_to_kept_idx_patch, dim=0)


                        hidden_states_pkg['hidden_states'] = orig_kept_states
                        

                        hidden_states = hidden_states_pkg['hidden_states']


                        self._sparse_vit_saved = {
                            "orig_seq_len": orig_seq_len,

                            "keep_indexs": keep_indexs_per_batch,
                            "removed_indices": removed_indexs_per_batch,
                            "removed_states": removed_states,
                            "orig_kept_states": orig_kept_states,

                            "orig_kept_states_flat": orig_kept_states.reshape(-1, D),

                            
                            "rem_to_kept_idx":    rem_to_kept_idx_patch,  
                            "unique_idx": unique_idx_patch,
                            "inv_idx": inv_idx_patch,
                        }
                #print("hid state",hidden_states.shape)
                
                if hasattr(self, "_sparse_vit_saved") and idx < Sparse_config['vit_pruned_layer'] + Sparse_config['AS_layer']: # 3
                    hidden_states_pkg = blk(hidden_states, sparse_vit_saved=self._sparse_vit_saved)
                else:
                    hidden_states_pkg = blk(hidden_states)


                if idx == self.config.num_hidden_layers - 1 and hasattr(self, "_sparse_vit_saved"):
                    saved = self._sparse_vit_saved  
                    B, L, D = saved["removed_states"].size(0), saved["orig_seq_len"], saved["removed_states"].size(2)
                    device = hidden_states_pkg['hidden_states'].device  

                    # new_kept / orig_kept
                    K = saved["orig_kept_states"].shape[1]
                    flat_new_kept = hidden_states_pkg['hidden_states'].reshape(B*K, D) 
                    flat_orig_kept = saved["orig_kept_states"].reshape(B*K, D)

                    full_states_patch = []
                    for b in range(B):
                        rem_to_kept_idx = saved["rem_to_kept_idx"][b]  # [R, topk]
                        removed_indexs_per_batch = saved["removed_indices"][b]
                        unique_idx = saved["unique_idx"][b]
                        inv_idx = saved["inv_idx"][b]

                        # compute delta
                        delta_unique = flat_new_kept[unique_idx] - flat_orig_kept[unique_idx]  # [U, D]

                        inv_idx = inv_idx.view(rem_to_kept_idx.shape)  # [R, topk]
                        avg_delta_removed = delta_unique[inv_idx].mean(dim=1)  # [R, D]

                        # prepare full_states
                        full_states = torch.zeros(L, D, device=device, dtype=flat_new_kept.dtype)
                        
                        full_states[saved["keep_indexs"][b]] = hidden_states_pkg['hidden_states'][b]  
                        full_states[removed_indexs_per_batch] = saved["removed_states"][b] + avg_delta_removed

                        full_states_patch.append(full_states)

                    full_states_patch = torch.stack(full_states_patch, dim=0)
                    hidden_states_pkg['hidden_states'] = full_states_patch  


                    del self._sparse_vit_saved

                hidden_states = hidden_states_pkg['hidden_states']

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )

    # # pruning by feature-difference
    def get_retained_image_token_diff(self,config,hidden_states_prev,hidden_states_cur,last_layer_state):
        Sparse_config = config.Sparse_config
        
        image_token_start_index = 0
        image_token_length = last_layer_state.shape[0]

        reduction_ratio = Sparse_config['vit_reduction_ratio']
        
        TOKEN_TOPK = int(image_token_length * (1 - reduction_ratio))
        

        device = last_layer_state.device

        diff = hidden_states_cur - hidden_states_prev # [seqlen,embed_dim]
        diff_norm = torch.norm(diff,dim=-1) # [seqlen]
        top_k_indices = diff_norm.topk(TOKEN_TOPK).indices
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

        self.encoder = IPCV_ViT(config)