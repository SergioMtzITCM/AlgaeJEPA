import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import ViTRoPE, apply_rotary_pos_embed

from configs.config import BaseConfig

from typing import Optional, Tuple

class ViTPatchEmbeddings(nn.Module):
    """Transform Image to Patch Sequence"""
    def __init__(self,
                 config: BaseConfig) -> None:
        super().__init__()
        
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size

        # Convolution with stride = patch_size is equal to divide into patches
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size = config.patch_size,
            stride = config.patch_size
        )

    def forward(self,
                x_input: torch.Tensor) -> torch.Tensor:
        # x_input: [B, C, H, W] -> [B, Hidden, Grid_H, Grid_W]
        x = self.projection(x_input)
        # Flat and transpose: [B, Hidden, N_patches] -> [B, N_patches, Hidden]
        x = x.flatten(2).transpose(1, 2)

        return x

class SelfAttention(nn.Module):
    def __init__(self,
                 config: BaseConfig) -> torch.Tensor:
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

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
                x_query: torch.Tensor,
                x_key: torch.Tensor,
                x_value: torch.Tensor,
                rope_embed: Tuple[torch.Tensor, torch.Tensor],
                output_attentions: bool = False) -> torch.Tensor:

        B, N, C = x_query.shape

        # Projections [B, N, Num_Heads, Head_Dim]
        q = self.q(x_query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x_key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        cos, sin = rope_embed
        q, k = apply_rotary_pos_embed(q, k, cos, sin)

        # Manual Attention Only for Visualization
        if output_attentions:

            # Attention: (Q @ K.T) * scale
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim = -1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x, attn

        # Optimized Attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.attn_drop.p if self.training else 0.0,
            scale = self.scale
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, None

class SwiGLU(nn.Module):
    """MLP Gated with SiLU activation"""
    def __init__(self,
                 config: BaseConfig) -> None:
        super().__init__()

        in_features = config.hidden_size
        hidden_features = config.intermediate_size

        self.w1 = nn.Linear(in_features, hidden_features) # Gate
        self.w2 = nn.Linear(in_features, hidden_features) # Up
        self.w3 = nn.Linear(hidden_features, in_features) # Down
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                x_input: torch.Tensor) -> torch.Tensor:
        # SwiGLU operation: (SiLU(xW1) * xW2) W3
        x1 = F.silu(self.w1(x_input))
        x2 = self.w2(x_input)
        hidden = x1 * x2
        out = self.w3(hidden)
        out = self.drop(out)

        return out

class LayerScale(nn.Module):
    def __init__(self,
                 dim: int,
                 init_values: float = 1e-5) -> None:
        super().__init__()

        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self,
                x_input: torch.Tensor) -> torch.Tensor:
        return x_input * self.gamma

def drop_path(
    x_input: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False
) -> torch.Tensor:

    """Stochastic Depth"""

    if drop_prob == 0.0 or not training:
        return x_input

    keep_prob = 1 - drop_prob
    shape = (x_input.shape[0],) + (1,) * (x_input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype = x_input.dtype, device = x_input.device)
    random_tensor.floor_() # Binarize
    output = x_input.div(keep_prob) * random_tensor

    return output

class DropPath(nn.Module):
    def __init__(self,
                 drop_prob: float = 0.0) -> None:
        super().__init__()

        self.drop_prob = drop_prob

    def forward(self,
                x_input: torch.Tensor) -> torch.Tensor:
        return drop_path(x_input, self.drop_prob, self.training)

class ViTLayer(nn.Module):
    """Vision Transformer Layer"""
    def __init__(self,
                 config: BaseConfig,
                 drop_path_radio: float = 0.0) -> None:
        super().__init__()

        # Pre-Norm Architecture
        self.norm1 = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.attn = SelfAttention(config)
        self.ls1 = LayerScale(config.hidden_size, config.layerscale_value)
        self.drop_path1 = DropPath(drop_path_radio) if drop_path_radio > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.mlp = SwiGLU(config) if config.use_swiglu else nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )

        self.ls2 = LayerScale(config.hidden_size, config.layerscale_value)
        self.drop_path2 = DropPath(drop_path_radio) if drop_path_radio > 0 else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                rope_embed: torch.Tensor,
                output_attentions: bool = False) -> torch.Tensor:

        # Attention Block
        # x = x + drop_path(layerscale(attn(norm1(x))))
        x_norm = self.norm1(x)

        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, rope_embed, output_attentions)

        x = x + self.drop_path1(self.ls1(attn_out))

        # MLP Block
        # x = x + drop_path(layerscale(mlp(norm2(x))))
        x = x + self.drop_path2(
            self.ls2(
                self.mlp(self.norm2(x))
            )
        )

        if output_attentions:
            return x, attn_weights

        return x, None

class ViTModel(nn.Module):
    def __init__(self,
                config: BaseConfig) -> None:
        super().__init__()

        self.config = config

        # Patch Embeddings
        self.patch_embed = ViTPatchEmbeddings(config)

        # RoPE Generator
        self.rope = ViTRoPE(config)

        # Embeddings Dropout
        self.pos_drop = nn.Dropout(config.hidden_dropout_prob)

        # Encoder Blocks
        # Stochastic Depth Decay Rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_prob, config.num_hidden_layers)]

        self.layers = nn.ModuleList([
            ViTLayer(config, drop_path_radio = dpr[i])
            for i in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

        # ---- Classifier ----
        self.num_classes = config.num_classes
        if self.num_classes is not None:
            self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        # --------------------

        self._init_weights()

    def _init_weights(self) -> None:
        # Truncated Normal Initialization for Linear Weights
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = self.config.initializer_range)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid_size(self) -> Tuple[int, int]:
        """Patch grid (rows, cols) of the encoder output. This ViT is square (single 'image_size')."""
        n_patch = self.config.image_size // self.config.patch_size
        return (n_patch, n_patch)

    def get_output_size(self) -> Tuple[int, int]:
        """(number of output tokens, hidden size). There is no CLS token: tokens = grid_h * grid_w."""
        grid_h, grid_w = self.get_grid_size()
        return (grid_h * grid_w, self.config.hidden_size)

    def forward(self,
                pixel_values: torch.Tensor,
                output_attentions: bool = False) -> torch.Tensor:
        """
        Args:
            pixel_values: Image Tensor [Batch, C, H, W]

        Returns:
            x: Transformer Output [Batch, N_patches, Hidden_Size]
        """

        # Patch Embeddings
        # x_patches: [B, N_patches, Dim]
        x_patches = self.patch_embed(pixel_values)
        x = self.pos_drop(x_patches)

        # Compute RoPE 2D
        rope_cos_sin = self.rope(pixel_values)

        all_attentions = () if output_attentions else None

        # Forward Pass Through Transformer Layers
        for layer in self.layers:
            # Each Layer receives embeddings and positional info (RoPE)
            x, attn_weights = layer(x, rope_cos_sin, output_attentions = output_attentions)
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        # Final Norm
        x = self.norm(x)

        # ---- Classification ----
        if self.num_classes is not None:
            x = x.mean(dim = 1) 
            x = self.classifier(x)
            
            if output_attentions:
                return x, all_attentions
            return x
        # ------------------------

        if output_attentions:
            return x, all_attentions

        return x

    def get_patch_embeddings(self,
                             pixel_values: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(pixel_values)

    def forward_embeddings(self,
                           patch_embeddings: torch.Tensor,
                           rope_cos_sin: torch.Tensor) -> torch.Tensor:

        x = self.pos_drop(patch_embeddings)

        # Forward Pass Through Transformer Layers
        for layer in self.layers:
            # Each Layer receives embeddings and positional info (RoPE)
            x, _ = layer(x, rope_cos_sin)

        # Final Norm
        x = self.norm(x)

        return x