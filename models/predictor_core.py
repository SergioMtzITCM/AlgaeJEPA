import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import BaseConfig
from typing import Tuple, Optional

from models.vit_core import SelfAttention, LayerScale, SwiGLU, DropPath
from models.rope import apply_rotary_pos_embed_single


def build_2d_sincos_pos_embed(embed_dim: int,
                              grid_size: int) -> torch.Tensor:
    """
    Fixed (non-learnable) 2D sin-cos absolute position embedding, I-JEPA style.
 
    Returns: [grid_size * grid_size, embed_dim] in float32. Row-major order, i.e. the same order as
    the patch indices used everywhere else (index = row * grid_size + col). The first half of the
    channels encodes the row and the second half encodes the column.
    """
 
    if embed_dim % 4 != 0:
        raise ValueError(f"'embed_dim' must be divisible by 4 for a 2D sin-cos embedding (embed_dim = {embed_dim})")
 
    def _embed_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
        # pos: [N] -> [N, dim]  (sin | cos)
        omega = torch.arange(dim // 2, dtype = torch.float64) / (dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)
        angles = pos.reshape(-1, 1) * omega.reshape(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim = 1)
 
    coords = torch.arange(grid_size, dtype = torch.float64)
    rows = coords.reshape(-1, 1).expand(grid_size, grid_size).reshape(-1)
    cols = coords.reshape(1, -1).expand(grid_size, grid_size).reshape(-1)
 
    embed = torch.cat([_embed_1d(embed_dim // 2, rows), _embed_1d(embed_dim // 2, cols)], dim = 1)
 
    return embed.float()

class CrossAttention(nn.Module):
    def __init__(self,
                 config: BaseConfig) -> None:
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.is_causal = config.is_causal

        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias = config.qkv_bias)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias = config.qkv_bias)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias = config.qkv_bias)

        # QK-Norm
        if config.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps = config.layer_norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps = config.layer_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self,
                x_q: torch.Tensor,
                x_k: torch.Tensor,
                x_v: torch.Tensor,
                rope_embed_q: Tuple[torch.Tensor, torch.Tensor],
                rope_embed_k: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        # x_q: Tokens of Target Patches
        # x_k, x_v: Context Embeddings
        # rope_embed_q: Target RoPE (cos/sin)
        # rope_embed_k: Context RoPE (cos/sin)

        B, N_q, C = x_q.shape
        _, N_kv, _ = x_k.shape

        # Projections [B, N, Num_Heads, Head_Dim]
        q = self.q(x_q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x_k).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_v).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        cos_q, sin_q = rope_embed_q
        cos_k, sin_k = rope_embed_k

        q = apply_rotary_pos_embed_single(q, cos_q, sin_q) # Only rotate q (target RoPE)
        k = apply_rotary_pos_embed_single(k, cos_k, sin_k) # Only rotate k (context RoPE)

        # Attention: (Q @ K.T) * scale
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.attn_drop.p if self.training else 0.0,
            scale = self.scale
        )
        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PredictorLayer(nn.Module):
    def __init__(self,
                 config: BaseConfig,
                 drop_path_radio: float = 0.0) -> None:
        super().__init__()

        # Pre-Norm Architecture
        self.norm1 = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.self_attn = SelfAttention(config)
        self.ls1 = LayerScale(config.hidden_size, config.layerscale_value)
        self.drop_path1 = DropPath(drop_path_radio) if drop_path_radio > 0 else nn.Identity()

        # Cross Attention
        self.norm2 = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.cross_attn = CrossAttention(config)

        self.norm3 = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.mlp = SwiGLU(config) if config.use_swiglu else nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )

        self.ls2 = LayerScale(config.hidden_size, config.layerscale_value)
        self.drop_path2 = DropPath(drop_path_radio) if drop_path_radio > 0 else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                context_embeddings: torch.Tensor,
                context_rope_embed: Tuple[torch.Tensor, torch.Tensor],
                target_rope_embed: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        # x: Tokens of Target Patches
        # context_embeddings: Visible Context Embeddings 
        # context_rope_embed: RoPE of Context (cos/sin)
        # target_rope_embed: RoPE of Target (cos/sin)

        # Self Attention Block
        # x = x + drop_path(layerscale(attn(norm1(x))))
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, target_rope_embed)
        x = x + self.drop_path1(self.ls1(attn_out))

        # Cross Attention Block
        x_norm = self.norm2(x)
        x = x + self.cross_attn(
            x_norm,
            context_embeddings,
            context_embeddings,
            target_rope_embed,
            context_rope_embed
        )

        # MLP Block
        # x = x + drop_path(layerscale(mlp(norm2(x))))
        x = x + self.drop_path2(
            self.ls2(
                self.mlp(self.norm3(x))
            )
        )

        return x

class PredictorModel(nn.Module):
    def __init__(self,
                 config: BaseConfig) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"'num_layers' must be >= 1 (received {config.num_predictor_layers})")

        self.config = config

        # Target Patches Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Predictor Blocks
        # Stochastic Depth Decay Rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_prob, config.num_predictor_layers)]

        self.layers = nn.ModuleList([
            PredictorLayer(config, drop_path_radio = dpr[i])
            for i in range(config.num_predictor_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

        if config.use_abs_pos:
            grid_size = config.image_size // config.patch_size
            abs_pos_embed = build_2d_sincos_pos_embed(config.hidden_size, grid_size) # [N_patches, D]
            self.register_buffer("abs_pos_embed", abs_pos_embed, persistent = False)

        self._init_weights()

    def _init_weights(self) -> None:
        # Truncated Normal Initialization for Tokens and Linear Weights
        nn.init.trunc_normal_(self.mask_token, std = self.config.initializer_range)
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = self.config.initializer_range)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,
                context_embeddings: torch.Tensor,
                context_rope_embed: Tuple[torch.Tensor, torch.Tensor],
                target_rope_embed: Tuple[torch.Tensor, torch.Tensor],
                target_idx: Optional[torch.Tensor] = None) -> torch.Tensor:

        # target_idx: Indices of the target patches [B, N_tgt]. Only required when use_abs_pos = True.

        B, N_tgt = target_rope_embed[0].shape[0], target_rope_embed[0].shape[1]

        if context_embeddings.shape[1] != context_rope_embed[0].shape[-2]:
            raise ValueError(
                f"Context tokens ({context_embeddings.shape[1]}) do not match the context RoPE positions "
                f"({context_rope_embed[0].shape[-2]})."
            )

        # Expand the Mask Token
        mask_tokens = self.mask_token.expand(B, N_tgt, -1)

        # Absolute position of each target (optional)
        if self.use_abs_pos:
            if target_idx is None:
                raise ValueError("'target_idx' is required when the predictor is built with use_abs_pos = True")
            if tuple(target_idx.shape) != (B, N_tgt):
                raise ValueError(
                    f"'target_idx' must have shape {(B, N_tgt)} (received {tuple(target_idx.shape)})"
                )
            # abs_pos_embed: [N_patches, D] -> gather -> [B, N_tgt, D]
            target_pos = self.abs_pos_embed[target_idx]
            mask_tokens = mask_tokens + target_pos.to(mask_tokens.dtype)

        # Forward Pass Through Predictor Layers
        x = mask_tokens
        for layer in self.layers:
            x = layer(
                x,
                context_embeddings,
                context_rope_embed,
                target_rope_embed
            )

        # Final Norm
        x = self.norm(x)

        return x