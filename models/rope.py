from configs.config import BaseConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
import math

# --- Coordinate Utilities ---
def get_patches_center_coordinates(
    num_patches_h: int,
    num_patches_w: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:

    """Generates a normalized coordinate grid between [-1, 1]"""

    # Make 0.5 to N-0.5 ranges (patch centers)
    coords_h = torch.arange(0.5, num_patches_h, dtype = dtype, device = device)
    coords_w = torch.arange(0.5, num_patches_w, dtype = dtype, device = device)

    # Normalize to [0, 1]
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w

    # Make the grid (Meshgrid) [H, W, 2]
    # Stack to get (x, y) pairs in each point
    grid = torch.stack(torch.meshgrid(coords_h, coords_w, indexing = "ij"), dim = -1)

    # Flat: [H, W, 2] -> [H*W, 2]
    coords = grid.flatten(0, 1)

    # Scale from [0, 1] to [-1, 1]
    coords = 2.0 * coords - 1.0

    return coords

def augment_patches_center_coordinates(
    coords: torch.Tensor,
    shift: Optional[float] = None,
    jitter: Optional[float] = None,
    rescale: Optional[float] = None,
) -> torch.Tensor:

    """
    Apply data augment to positional coords
    """

    # Shift
    if shift is not None:
        shift_hw = torch.empty((1, 2), device = coords.device, dtype = coords.dtype)
        shift_hw = shift_hw.uniform_(-shift, shift)
        coords = coords + shift_hw

    # Jitter
    if jitter is not None:
        jitter_range = math.log(jitter)
        jitter_hw = torch.empty((1, 2), device = coords.device, dtype = coords.dtype)
        jitter_hw = jitter_hw.uniform_(-jitter_range, jitter_range).exp()
        coords = coords * jitter_hw

    # Global Rescale
    if rescale is not None:
        rescale_range = math.log(rescale)
        rescale_hw = torch.empty(1, device = coords.device, dtype = coords.dtype)
        rescale_hw = rescale_hw.uniform_(-rescale_range, rescale_range).exp()
        coords = coords * rescale_hw

    return coords

# --- Rotary Positional Embeddings (RoPE) 2D ---
class ViTRoPE(nn.Module):
    def __init__(self,
                 config: BaseConfig) -> torch.Tensor:
        super().__init__()

        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.base = config.rope_theta

        # Pre-compute inverse frequencies (Theta)
        # shape: [Head_Dim / 2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

        # Number of Patches
        num_patches_h = config.image_size // self.config.patch_size
        num_patches_w = config.image_size // self.config.patch_size

        # Compute Patch Coords
        # patch_coords: [Total_Patches, 2]
        patch_coords = get_patches_center_coordinates(
            num_patches_h, num_patches_w, dtype = torch.float32, device =
            torch.device("cpu")
        )

        # Compute Angles for RoPE
        # patch_coords: [Total_Patches, 2]
        # inv_freq: [Head_Dim / 2]
        # angles shape: [N_patches, 2, Head_Dim/2]
        angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
        # Flatten: [Total_Patches, Head_Dim]
        angles = angles.flatten(1, 2)

        cos_full = torch.cos(angles)
        sin_full = torch.sin(angles)

        self.register_buffer("cos_full", cos_full, persistent = False)
        self.register_buffer("sin_full", sin_full, persistent = False)

    def forward(self,
                 input_tensor: torch.Tensor,
                 patch_indices: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
         
         """
         Args:
             input_tensor: Reference tensor for device/dtype info [B, C, H, W]
             patch_indices: Indices of the patches that are being processed [Batch, N_subset]
             (e.g., only context patches or only target patches)
             If None, assumes all patches are in standard order
         """

         device = input_tensor.device
         dtype = input_tensor.dtype

         # Force float32 precision for coords
         with torch.autocast(device_type = device.type, enabled = False):

             cos = self.cos_full
             sin = self.sin_full
             
             if patch_indices is not None:
                 patch_indices = patch_indices.to(device)

                 # Gather (Select specific index per batch)
                 # patch_indices: [B, N_subset]
                 # cos_full: [Total, D] -> Expand to [B, Total, D] for gather
    
                 B = patch_indices.shape[0]
                 cos_expanded = cos.unsqueeze(0).expand(B, -1, -1)
                 sin_expanded = sin.unsqueeze(0).expand(B, -1, -1)
    
                 # Expand indices to [B, N_subset, D]
                 D = cos.shape[-1]
                 indices_expanded = patch_indices.unsqueeze(-1).expand(-1, -1, D)
    
                 # Final Gather
                 cos_subset = torch.gather(cos_expanded, 1, indices_expanded)
                 sin_subset = torch.gather(sin_expanded, 1, indices_expanded)
    
                 return cos_subset.to(device), sin_subset.to(device)
                 
             return cos.to(device), sin.to(device)

def rotate_half(
    x: torch.Tensor
) -> torch.Tensor:
    """Rotates half the hidden dims of the input"""
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_embed(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # q, k shapes: [Batch, Heads, Seq_Len, Head_Dim]
    # cos, sin shapes: [Seq_Len_Patches, Head_Dim]

    num_tokens = q.shape[-2]
    num_patches = cos.shape[-2]
    num_prefix_tokens = num_tokens - num_patches # e.g., CLS token

    # Split: special tokens (without spatital position) vs patches
    q_prefix, q_patches = q.split([num_prefix_tokens, num_patches], dim = -2)
    k_prefix, k_patches = k.split([num_prefix_tokens, num_patches], dim = -2)

    if cos.dim() == 3:
        # [B, N, D] -> [B, 1, N, D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        # Align cos/sin dims for broadcasting: [1, 1, Seq_Len_Patches, Head_Dim]
        # [N, D] -> [1, 1, N, D]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation: x_rot = x * cos + rotate(x) * sin
    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    # Concat
    q_out = torch.cat((q_prefix, q_patches), dim = -2)
    k_out = torch.cat((k_prefix, k_patches), dim = -2)

    return q_out, k_out
