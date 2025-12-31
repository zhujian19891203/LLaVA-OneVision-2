# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLaVA-OneVision 2.0 model implementation."""

from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers import AutoModel
from .configuration_llava_onevision2 import LlavaOnevision2Config, LlavaOnevision2VisionConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.utils import replace_return_docstrings, is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava-Onevision-1.5 outputs, with hidden states and attentions.
    """
)
class LlavaOnevision2ModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava-Onevision-1.5 causal language model (or autoregressive) outputs.
    """
)
class LlavaOnevision2CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


# ---------------------------------------------------------------------------
# Vision Rotary Embedding
# ---------------------------------------------------------------------------

class VisionRotaryEmbedding(nn.Module):
    """
    3D (T,H,W) Rotary frequency constructor with 4:6:6 split.
    Supports both grid_thw-based and explicit position-based RoPE computation.
    """
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half

        # 4:6:6 split for T:H:W
        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer("inv_freq_t", 1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)), persistent=False)
        self.register_buffer("inv_freq_h", 1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)), persistent=False)
        self.register_buffer("inv_freq_w", 1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)), persistent=False)

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from grid_thw (Qwen2VL style).
        
        Args:
            grid_thw: [num_samples, 3] tensor with [t, h, w] for each sample
            
        Returns:
            freqs: [total_seq_len, half] tensor of position frequencies
        """
        device = grid_thw.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)
        
        all_freqs = []
        for sample_thw in grid_thw:
            t, h, w = sample_thw[0].item(), sample_thw[1].item(), sample_thw[2].item()
            
            # Compute frequency tables
            ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
            fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
            fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)
            
            # Build position indices for this sample
            t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
            h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            w_ids = torch.arange(w, device=device).repeat(h).repeat(t)
            
            # Concatenate frequencies: [seq_len, half]
            sample_freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
            all_freqs.append(sample_freqs)
        
        return torch.cat(all_freqs, dim=0)

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit patch positions.
        
        Args:
            patch_positions: [seq_len, 3] tensor with [t, h, w] positions for each patch
            
        Returns:
            freqs: [seq_len, half] tensor of position frequencies
        """
        device = patch_positions.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)
        
        t_pos = patch_positions[:, 0].float()
        h_pos = patch_positions[:, 1].float()
        w_pos = patch_positions[:, 2].float()
        
        ft = torch.outer(t_pos, inv_t)
        fh = torch.outer(h_pos, inv_h)
        fw = torch.outer(w_pos, inv_w)
        
        return torch.cat([ft, fh, fw], dim=-1)

    def forward_with_thw(self, t: int, h: int, w: int, device=None) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit t, h, w dimensions.
        
        Args:
            t: Number of temporal frames
            h: Number of height patches
            w: Number of width patches
            device: Target device
            
        Returns:
            freqs: [t*h*w, half] tensor of position frequencies
        """
        if device is None:
            device = self.inv_freq_t.device

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class LlavaViTEmbeddings(nn.Module):
    """
    Patch embedding layer that converts images to patch embeddings.
    Supports both 4D (B, C, H, W) and 5D (B, C, T, H, W) inputs.
    """
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:

        target_dtype = self.patch_embedding.weight.dtype
        # Handle 4D (B, C, H, W) or 5D (B, C, T, H, W) inputs
        if pixel_values.dim() == 4:
             pixel_values = pixel_values.unsqueeze(2) # (B, C, 1, H, W)

        batch_size, channels, t_frames, height, width = pixel_values.shape

        # Merge time into batch for Conv2d
        x_2d = pixel_values.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, channels, height, width)

        # Patch Embed
        embeddings = self.patch_embedding(x_2d.to(dtype=target_dtype))  # (B*T, C, Hp, Wp)
        embeddings = embeddings.flatten(2).transpose(1, 2) # (B*T, L_frame, C)

        # Flatten all patches
        total_patches = t_frames * (height // self.patch_size) * (width // self.patch_size)
        embeddings = embeddings.reshape(batch_size, total_patches, self.embed_dim)

        return embeddings


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------

class LlavaOnevision2VisionPatchMerger(nn.Module):
    """
    Patch merger that merges spatial_merge_size x spatial_merge_size patches into one.
    Supports both packing format and standard batch format.
    """
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-05,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )
        self.spatial_merge_size = spatial_merge_size

    def forward(self, x: torch.Tensor, grid_thw: Optional[torch.Tensor] = None, 
                height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:
        """
        Merge patches with support for both packing and batch formats.
        
        Packing format:
            Input: [total_seq_len, hidden_size] with grid_thw defining sample boundaries
            Output: [total_merged_seq_len, dim]
            
        Batch format:
            Input: [B, N, C] where N = H * W
            Output: [B, N // (spatial_merge_size^2), dim]
        """
        merge_size = self.spatial_merge_size
        
        # ============================================================
        # 【PACKING FORMAT】: Input is [seq_len, C] with grid_thw
        # ============================================================
        if grid_thw is not None and x.dim() == 2:
            x = self.ln_q(x)
            
            all_merged = []
            start_idx = 0
            
            for sample_thw in grid_thw:
                t, h, w = sample_thw[0].item(), sample_thw[1].item(), sample_thw[2].item()
                seq_len = t * h * w
                sample_x = x[start_idx:start_idx + seq_len]  # [t*h*w, C]
                
                # Validate divisibility
                assert h % merge_size == 0 and w % merge_size == 0, \
                    f"Grid size ({h}, {w}) not divisible by merge_size {merge_size}"
                
                C = sample_x.shape[-1]
                new_h = h // merge_size
                new_w = w // merge_size
                
                # Reshape: [t*h*w, C] -> [t, h, w, C]
                sample_x = sample_x.view(t, h, w, C)
                
                # Merge 2x2 spatial patches
                # [t, h, w, C] -> [t, new_h, merge_size, new_w, merge_size, C]
                sample_x = sample_x.view(t, new_h, merge_size, new_w, merge_size, C)
                sample_x = sample_x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [t, new_h, new_w, merge_size, merge_size, C]
                sample_x = sample_x.view(t * new_h * new_w, merge_size * merge_size * C)  # [t*new_h*new_w, hidden_size]
                
                all_merged.append(sample_x)
                start_idx += seq_len
            
            merged_x = torch.cat(all_merged, dim=0)  # [total_merged_seq_len, hidden_size]
            return self.mlp(merged_x)
        
        # ============================================================
        # 【BATCH FORMAT】: Input is [B, N, C]
        # ============================================================
        B, N, C = x.size()

        # Infer H and W if not provided
        if height is None or width is None:
            H, W = self._infer_hw(N)
        else:
            H, W = height, width

        assert H * W == N, f"Height {H} * Width {W} != N {N}"

        # Validate divisibility by merge_size
        assert H % merge_size == 0 and W % merge_size == 0, \
            f"Grid size ({H}, {W}) not divisible by merge_size {merge_size}"

        # Apply LayerNorm
        x = self.ln_q(x)

        # Reshape to (B, H, W, C)
        x = x.view(B, H, W, C)

        # Merge 2x2 spatial patches
        new_H = H // merge_size
        new_W = W // merge_size
        x = x.view(B, new_H, merge_size, new_W, merge_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, new_H, new_W, merge_size, merge_size, C)
        x = x.view(B, new_H * new_W, merge_size * merge_size * C)  # (B, N', hidden_size)

        # Project to LLM dimension
        x = self.mlp(x)
        return x
    
    def _infer_hw(self, N: int) -> Tuple[int, int]:
        """Infer height and width from number of patches."""
        merge_size = self.spatial_merge_size
        sqrt_n = int(N ** 0.5)
        
        # Try to find factors closest to square
        for h in range(sqrt_n, 0, -1):
            if N % h == 0:
                w = N // h
                if h % merge_size == 0 and w % merge_size == 0:
                    return h, w
        
        # Fallback: try all factors
        for h in range(1, N + 1):
            if N % h == 0:
                w = N // h
                if h % merge_size == 0 and w % merge_size == 0:
                    return h, w
        
        raise ValueError(f"Cannot find valid H, W for N={N} with merge_size={merge_size}")


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)



def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # We need to broadcast freqs to match heads
    # (B, L, D) -> (B, 1, L, D)
    # Keep the same dtype as q, k to avoid memory doubling from float32 promotion
    cos = freqs.cos().unsqueeze(1).float()
    sin = freqs.sin().unsqueeze(1).float()

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class LlavaViTFlashAttention2(nn.Module):
    """
    Multi-headed attention with RoPE support using Flash Attention 2.
    """
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using Flash Attention 2.
        """
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash Attention requires (B, L, H, D) format
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim)

        # Apply RoPE if provided
        if rotary_pos_emb is not None:
            # Transpose for RoPE application: (B, L, H, D) -> (B, H, L, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            # NOTE: apply_rotary_pos_emb now ensures NO float32 cast happens
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)
            # Transpose back: (B, H, L, D) -> (B, L, H, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        # FIX: Removed the explicit float32 check and downcast.
        # We assume input is already correct (bf16/fp16) thanks to RoPE fix.

        # Flash Attention forward pass
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        # Reshape to (B, L, embed_dim)
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        # No extra casting here.
        attn_output = self.out_proj(attn_output)

        return attn_output, None


class LlavaViTEncoderLayer(nn.Module):
    """Vision encoder layer with pre-norm and Flash Attention 2."""
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = LlavaViTFlashAttention2(config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attn_weights) if output_attentions else (hidden_states,)
        return outputs

class LlavaViTEncoder(nn.Module):
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([LlavaViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # Gradient checkpointing support
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LlavaOnevision2PreTrainedModel(PreTrainedModel):
    config_class = LlavaOnevision2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _no_split_modules = ["LlavaViTEncoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, LlavaOnevision2VisionPretrainedModel):
            std_cls = float(module.config.hidden_size) ** -0.5
            # torch.nn.init.normal_(module.class_embedding, mean=0.0, std=std_cls)
            # torch.nn.init.normal_(module.class_pos_emb, mean=0.0, std=std_cls)

class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling with a learned probe (PMA-style).
    """
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        attn_output, _ = self.attention(probe, hidden_states, hidden_states)

        residual = attn_output
        attn_output = self.norm(attn_output)
        attn_output = residual + self.mlp(attn_output)

        return attn_output[:, 0]


def interpolate_frame_indices(frame_indices: torch.Tensor, total_frames: torch.Tensor, target_frames: int = 64) -> torch.Tensor:
    """
    Interpolate frame indices from the original video frame count to the target frame count.

    Args:
        frame_indices: [B, seq_len] Original frame indices
        total_frames: [B] Total number of frames for each video
        target_frames: Target number of frames (default: 64)

    Returns:
        interpolated_indices: [B, seq_len] Interpolated frame indices, range in [0, target_frames-1]
    """
    bs, seq_len = frame_indices.shape
    device = frame_indices.device

    # Convert total_frames to float for interpolation calculation
    total_frames_float = total_frames.float().view(bs, 1)  # [B, 1]
    frame_indices_float = frame_indices.float()  # [B, seq_len]

    # Interpolation formula: new_idx = (old_idx / (total_frames - 1)) * (target_frames - 1)
    total_frames_safe = torch.clamp(total_frames_float - 1, min=1.0)
    interpolated_indices = (frame_indices_float / total_frames_safe) * (target_frames - 1)

    # Round and convert to integer
    interpolated_indices = torch.round(interpolated_indices).long()

    # Ensure indices are within valid range
    interpolated_indices = torch.clamp(interpolated_indices, 0, target_frames - 1)

    return interpolated_indices


def compute_patch_positions_from_grid_thw(grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Compute patch positions from grid_thw for RoPE calculation.
    
    Args:
        grid_thw: [num_samples, 3] tensor with [t, h, w] for each sample
        
    Returns:
        patch_positions: [total_seq_len, 3] tensor with [t, h, w] position for each patch
    """
    device = grid_thw.device
    all_positions = []
    
    for sample_thw in grid_thw:
        t, h, w = sample_thw[0].item(), sample_thw[1].item(), sample_thw[2].item()
        
        # Build position indices
        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)
        
        positions = torch.stack([t_ids, h_ids, w_ids], dim=-1)  # [t*h*w, 3]
        all_positions.append(positions)
    
    return torch.cat(all_positions, dim=0)


def compute_patch_positions_with_interpolated_temporal(
    interpolated_indices: torch.Tensor,
    h_patches: int,
    w_patches: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute patch positions with interpolated temporal positions for RoPE.
    
    This function computes patch positions where the temporal positions are
    based on the interpolated frame indices.
    
    Args:
        interpolated_indices: [B, num_frames] Interpolated frame indices in 64-frame context
        h_patches: Number of patches in height dimension
        w_patches: Number of patches in width dimension
        device: Target device
    
    Returns:
        visible_indices: Tensor of shape (B, total_patches) with flattened patch indices
    """
    num_patches_per_frame = h_patches * w_patches
    B, T = interpolated_indices.shape
    visible_indices = []

    for b in range(B):
        indices_b = []
        for t in range(T):
            t_new = interpolated_indices[b, t].item()
            for p in range(num_patches_per_frame):
                idx = t_new * num_patches_per_frame + p
                indices_b.append(idx)
        visible_indices.append(indices_b)

    visible_indices = torch.tensor(visible_indices, device=device, dtype=torch.long)
    return visible_indices


# ---------------------------------------------------------------------------
# Vision Model
# ---------------------------------------------------------------------------

class LlavaOnevision2VisionPretrainedModel(LlavaOnevision2PreTrainedModel):
    """
    LLaVA-OneVision 2.0 Vision Model.
    
    Supports:
        - 4D input: [B, C, H, W] for images
        - 5D input: [B, C, T, H, W] for videos
        - visible_indices for sparse patch selection
    """
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        # Vision components
        self.embeddings = LlavaViTEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = LlavaViTEncoder(config)
        self.video_rope = VisionRotaryEmbedding(config)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        else:
            self.layernorm_post = None
            self.head = None

        self.merger = LlavaOnevision2VisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.post_init()

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=LlavaOnevision2VisionConfig)
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        visible_indices: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_merger: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Forward pass for vision model.
        
        Args:
            pixel_values: 4D [B, C, H, W] or 5D [B, C, T, H, W] tensor
            grid_thw: Optional grid sizes for each sample
            visible_indices: Optional indices for sparse patch selection (for video)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a ModelOutput instead of tuple
            skip_merger: If True, skip patch merger
        
        Returns:
            BaseModelOutputWithPooling with last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, 'output_attentions', False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, 'output_hidden_states', False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, 'use_return_dict', True)

        # Handle special case for video input
        if pixel_values.shape[0] == 8 and pixel_values.dim() == 4:
            # [8, C, H, W] -> [1, C, 8, H, W]
            pixel_values = pixel_values.unsqueeze(0).permute(0, 2, 1, 3, 4)

        # Determine video dimensions for RoPE
        if pixel_values.dim() == 5:
            t_frames = pixel_values.shape[2]
            height = pixel_values.shape[3]
            width = pixel_values.shape[4]
        else:
            t_frames = 1
            height = pixel_values.shape[2]
            width = pixel_values.shape[3]

        # 1. Embeddings
        hidden_states = self.embeddings(pixel_values)
        batch_size, total_patches, _ = hidden_states.shape

        # 2. Visible Indices Handling
        if visible_indices is None or (isinstance(visible_indices, list) and visible_indices[0] is None):
            if t_frames == 1:
                visible_indices = torch.arange(total_patches, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)
            else:
                # Compute interpolated frame indices for video
                frame_indices = torch.arange(t_frames).unsqueeze(0).to(pixel_values.device)
                total_frames_tensor = torch.tensor([t_frames]).to(pixel_values.device)
                interpolated_indices = interpolate_frame_indices(
                    frame_indices, total_frames_tensor, 64
                )
                visible_indices = compute_patch_positions_with_interpolated_temporal(
                    interpolated_indices, height // self.config.patch_size, width // self.config.patch_size, pixel_values.device
                )
        else:
            # Handle visible_indices as list
            if isinstance(visible_indices, list):
                if len(visible_indices) == 1:
                    visible_indices = visible_indices[0]
                    if not isinstance(visible_indices, torch.Tensor):
                        visible_indices = torch.tensor(visible_indices, device=pixel_values.device)
                    if visible_indices.dim() == 1:
                        visible_indices = visible_indices.unsqueeze(0)
                else:
                    visible_indices = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in visible_indices])
            elif not isinstance(visible_indices, torch.Tensor):
                visible_indices = torch.tensor(visible_indices, device=pixel_values.device)
            
            visible_indices = visible_indices.to(pixel_values.device)

        # Gather visible patches for images (t_frames == 1)
        if t_frames == 1:
            gather_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
            hidden_states = torch.gather(hidden_states, 1, gather_index)

        # 3. RoPE Construction
        freqs_full = self.video_rope.forward_with_thw(
            t=64 if t_frames > 1 else 1,
            h=height // self.config.patch_size,
            w=width // self.config.patch_size,
            device=pixel_values.device
        )
        freqs_visible = freqs_full[visible_indices]

        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)

        # 4. Pre-Norm & Encoder
        hidden_states = self.layernorm_pre(hidden_states)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states to use -2 layer
            return_dict=True,
        )

        # Use second-to-last layer output for better feature representation
        if encoder_outputs.hidden_states is not None and len(encoder_outputs.hidden_states) >= 2 and not skip_merger:
            sequence_output = encoder_outputs.hidden_states[-2]
        else:
            sequence_output = encoder_outputs[0]

        # Post-Norm
        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        # Skip merger for consistency check with original ViT
        if skip_merger:
            pooled_output = None
            if self.head is not None:
                pooled_output = self.head(sequence_output)
            
            if not return_dict:
                return (sequence_output, pooled_output) + (encoder_outputs.hidden_states if output_hidden_states else None,)
            return BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                attentions=encoder_outputs.attentions if output_attentions else None,
            )

        # Compute grid_thw for merger
        h_patches = height // self.config.patch_size
        w_patches = width // self.config.patch_size
        grid_thw = torch.tensor(
            [[t_frames, h_patches, w_patches]] * batch_size,
            dtype=torch.long,
            device=pixel_values.device
        )

        # Patch merger (batch format)
        merged_output = self.merger(sequence_output, grid_thw=None, height=h_patches, width=w_patches)

        if not return_dict:
            return (merged_output,) + (encoder_outputs.hidden_states if output_hidden_states else None,)

        return BaseModelOutputWithPooling(
            last_hidden_state=merged_output,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )


@auto_docstring
class LlavaOnevision2Model(LlavaOnevision2PreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: LlavaOnevision2Config
    _no_split_modules = ["LlavaViTEncoderLayer"]

    def __init__(self, config: LlavaOnevision2Config):
        super().__init__(config)
        self.visual = LlavaOnevision2VisionPretrainedModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, temporal, height, width)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Convert to correct dtype
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        
        # Forward through vision model (batch mode)
        vision_output = self.visual(pixel_values_videos, visible_indices=None)
        
        # Extract the actual tensor from BaseModelOutputWithPooling
        if hasattr(vision_output, 'last_hidden_state'):
            video_embeds = vision_output.last_hidden_state
        else:
            video_embeds = vision_output[0]  # Fallback for tuple output
        
        # Compute split sizes from video_grid_thw or from input shape
        if video_grid_thw is not None:
            split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        else:
            # Compute from input shape
            batch_size = pixel_values_videos.shape[0]
            split_sizes = [video_embeds.shape[1]] * batch_size
        
        # Split embeddings per video
        if len(split_sizes) > 1:
            video_embeds = torch.split(video_embeds.view(-1, video_embeds.shape[-1]), split_sizes)
        else:
            video_embeds = [video_embeds.view(-1, video_embeds.shape[-1])]
        
        return video_embeds

    def get_image_features(self, pixel_values, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values: Can be one of:
                - `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`
                - `List[torch.FloatTensor]` of variable-size images
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        # Handle list of images (variable sizes) - need to process one by one
        if isinstance(pixel_values, (list, tuple)):
            all_image_embeds = []
            for img in pixel_values:
                if img.dim() == 3:
                    # [C, H, W] -> [1, C, H, W]
                    img = img.unsqueeze(0)
                
                # Convert to correct dtype
                img = img.type(self.visual.dtype)
                
                # Forward through vision model (batch mode)
                vision_output = self.visual(img, visible_indices=None)
                
                # Extract the actual tensor from BaseModelOutputWithPooling
                if hasattr(vision_output, 'last_hidden_state'):
                    img_embeds = vision_output.last_hidden_state
                else:
                    img_embeds = vision_output[0]
                
                all_image_embeds.append(img_embeds.view(-1, img_embeds.shape[-1]))
            
            return all_image_embeds
        
        # Standard [B, C, H, W] format
        if pixel_values.dim() == 4:
            # Convert to correct dtype
            pixel_values = pixel_values.type(self.visual.dtype)
            
            # Forward through vision model (batch mode)
            vision_output = self.visual(pixel_values, visible_indices=None)
            
            # Extract the actual tensor from BaseModelOutputWithPooling
            if hasattr(vision_output, 'last_hidden_state'):
                image_embeds = vision_output.last_hidden_state
            else:
                image_embeds = vision_output[0]
            
            # Compute split sizes
            batch_size = pixel_values.shape[0]
            if image_grid_thw is not None:
                split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            else:
                split_sizes = [image_embeds.shape[1]] * batch_size
            
            # Split embeddings per image
            if len(split_sizes) > 1:
                image_embeds = torch.split(image_embeds.view(-1, image_embeds.shape[-1]), split_sizes)
            else:
                image_embeds = [image_embeds.view(-1, image_embeds.shape[-1])]
            
            return image_embeds
        else:
            raise ValueError(f"Unsupported pixel_values type/shape: {type(pixel_values)}, {pixel_values.shape if hasattr(pixel_values, 'shape') else 'N/A'}")

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaOnevision2ModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        image_embeds = None
        
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)

        if image_embeds is not None:
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Use simple 1D position_ids
        if position_ids is None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
            
            # Handle cache_position for generation
            if cache_position is not None and cache_position[0] != 0:
                position_ids = position_ids + cache_position[0]

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = LlavaOnevision2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output if return_dict else output.to_tuple()


@auto_docstring
class LlavaOnevision2ForConditionalGeneration(LlavaOnevision2PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaOnevision2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaOnevision2CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaOnevision2ForConditionalGeneration

        >>> model = LlavaOnevision2ForConditionalGeneration.from_pretrained("Deep-VLM/LLaVA-OneVision-1.5-8B-Instruct-hf", trust_remote_code=True)
        >>> processor = AutoProcessor.from_pretrained("Deep-VLM/LLaVA-OneVision-1.5-8B-Instruct-hf", trust_remote_code=True)

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return LlavaOnevision2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "LlavaOnevision2ForConditionalGeneration", 
    "LlavaOnevision2Model", 
    "LlavaOnevision2PreTrainedModel",
    "LlavaOnevision2VisionPretrainedModel",
    # Vision components
    "VisionRotaryEmbedding",
    "LlavaViTEmbeddings",
    "LlavaViTFlashAttention2",
    "LlavaViTEncoderLayer",
    "LlavaViTEncoder",
    "LlavaOnevision2VisionPatchMerger",
    "Siglip2MultiheadAttentionPoolingHead",
    # Utility functions
    "interpolate_frame_indices",
    "compute_patch_positions_from_grid_thw",
    "compute_patch_positions_with_interpolated_temporal",
]
