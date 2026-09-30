import math
import random
import torch
from typing import Optional, Tuple, List, Union
import warnings
import numpy as np

import copy

class MultiBlockMasking(object):
    """Generates masks for Context (Visible) and Target (Hidden) using Multi-Block Masking"""
    def __init__(self,
                 input_size: int,
                 patch_size: int,
                 prediction_ratio: float = 0.4,
                 min_block_scale: float = 0.02,
                 max_block_scale: float = 0.33,
                 aspect_ratio_min: float = 0.75,
                 aspect_ratio_max: float = 1.33,
                 num_tries: int = 10,
                 seed: Optional[int] = None) -> None:

        self.height = input_size // patch_size
        self.width = input_size // patch_size
        self.num_patches = self.height * self.width
        self.prediction_ratio = prediction_ratio

        # --- Block Geometry Parameters ---

        # Relative size of the block of patches with respect the entire image
        self.min_block_scale = min_block_scale
        self.max_block_scale = max_block_scale
        # Shape of the block of patches
        # 1.0 = Perfect Square
        # 0.75 = Wider than tall rectangle
        # 1.33 = Taller than wider rectangle
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max
        self.num_tries = num_tries

        # Randon Number Generator
        self._rng = np.random.default_rng(seed)

        # Iteration Limit
        num_target_patches = int(self.num_patches * self.prediction_ratio)
        min_block_area = max(1, int(self.num_patches * self.min_block_scale))
        self._max_rounds = max(50, 10 * math.ceil(num_target_patches / min_block_area))

    def get_rng_state(self) -> dict:
        """Snapshot of the numpy RNG (lets callers, e.g. diagnostics, leave the mask stream untouched)."""
        return copy.deepcopy(self._rng.bit_generator.state)
 
    def set_rng_state(self, state: dict) -> None:
        """Restore a snapshot taken with 'get_rng_state'."""
        self._rng.bit_generator.state = copy.deepcopy(state)

    def _sample_block_dims_batched(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples [W, W] for 'n' independent blocks.
        """

        h = np.zeros(n, dtype = np.int64)
        w = np.zeros(n, dtype = np.int64)
        pending = np.ones(n, dtype = bool)
 
        for _ in range(self.num_tries):
            if not pending.any():
                break
 
            idx = np.nonzero(pending)[0]
            m = idx.shape[0]
 
            scale = self._rng.uniform(self.min_block_scale, self.max_block_scale, size = m)
            aspect_ratio = self._rng.uniform(self.aspect_ratio_min, self.aspect_ratio_max, size = m)
            target_area = self.num_patches * scale
 
            h_try = np.sqrt(target_area * aspect_ratio).astype(np.int64)
            w_try = np.sqrt(target_area / aspect_ratio).astype(np.int64)
            h_try = np.minimum(h_try, self.height)
            w_try = np.minimum(w_try, self.width)
 
            valid = (h_try * w_try) > 0
            valid_idx = idx[valid]
            h[valid_idx] = h_try[valid]
            w[valid_idx] = w_try[valid]
            pending[valid_idx] = False
 
        if pending.any():
            h[pending] = 1
            w[pending] = 1
 
        return h, w

    def _sample_block_masks_batched(self, batch_size: int) -> np.ndarray:
        """
        Generates, for all batch, a boolean block masks [batch_size, num_patches].
        Returns (1 = Target - Masked), (0 = Context - Visible).
        """

        mask = np.zeros((batch_size, self.height, self.width), dtype = np.int32)
        mask_count = np.zeros(batch_size, dtype = np.int64)
        num_target_patches = int(self.num_patches * self.prediction_ratio)
 
        rows = np.arange(self.height)
        cols = np.arange(self.width)
 
        active = np.ones(batch_size, dtype=bool)
        round_idx = 0

        while active.any() and round_idx < self._max_rounds:
            round_idx += 1
            active_idx = np.nonzero(active)[0]
            n_active = active_idx.shape[0]
 
            h, w = self._sample_block_dims_batched(n_active)
 
            # Sampling position (Top-Left)
            top = self._rng.integers(0, self.height - h + 1)
            left = self._rng.integers(0, self.width - w + 1)
 
            # Generate rectangular region for each block using broadcasting
            # [n_active, H] AND [n_active, W] -> [n_active, H, W]
            row_sel = (rows[None, :] >= top[:, None]) & (rows[None, :] < (top + h)[:, None])
            col_sel = (cols[None, :] >= left[:, None]) & (cols[None, :] < (left + w)[:, None])
            block_region = row_sel[:, :, None] & col_sel[:, None, :]
 
            current = mask[active_idx]
            newly_masked = np.logical_and(block_region, current == 0).sum(axis=(1, 2))
 
            mask[active_idx] = (current | block_region).astype(np.int32)
            mask_count[active_idx] += newly_masked
 
            active = mask_count < num_target_patches
 
        if active.any():
            warnings.warn(
                f"MultiBlockMasking: safety limit of"
                f"{self._max_rounds} rounds reached without covering num_target_patches "
                f"across all images in the batch. Check "
                f"min_block_scale/max_block_scale/aspect_ratio if this "
                f"happened frequently."
            )
 
        return mask.reshape(batch_size, -1)

    @torch.compiler.disable
    def __call__(self,
                 batch_size: int,
                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context_idx = Indices of visible patches [B, N_ctx]
            target_idx = Indices of hidden patches to predict [B, N_tgt]
        """

        num_target = int(self.num_patches * self.prediction_ratio)

        # Block Masks for all batch.
        block_mask = self._sample_block_masks_batched(batch_size) # [B, N_patches], int32 (0/1)

        # Small Random Noise
        random_noise = self._rng.random((batch_size, self.num_patches)).astype(np.float32) * 0.1

        # Masking Priorities Array
        # The patches covered will be in [1.0, 1.1]
        # The patches uncovered will be in [0.0, 0.1]
        mask_noise_np = block_mask.astype(np.float32) + random_noise
        mask_noise = torch.from_numpy(mask_noise_np)

        # Separate Target and Context using argsort
        # First values will be the Target
        ids_shuffle = torch.argsort(mask_noise, dim = 1, descending = True)
 
        # Target (Masked blocks)
        target_idx = ids_shuffle[:, :num_target]
 
        # Context (Rest visible)
        context_idx = ids_shuffle[:, num_target:]
 
        # Sort indices again so patches appear in spatial order
        context_idx, _ = torch.sort(context_idx, dim = 1)
        target_idx, _ = torch.sort(target_idx, dim = 1)
 
        return context_idx, target_idx