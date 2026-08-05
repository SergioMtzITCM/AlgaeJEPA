import math
import random
import torch
from typing import Optional, Tuple, List, Union

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
                 num_tries: int = 10) -> None:

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

    def _sample_block_mask(self) -> torch.Tensor:
        """
        Generates a boolean mask [H, W] for a single image using block sampling logic.
        Returns 1 (True) for Target (Masked) and 0 (False) for Context (Visible).
        """
    
        mask = torch.zeros((self.height, self.width), dtype = torch.int32)
        mask_count = 0
    
        # Number of target patches
        num_target_patches = int(self.num_patches * self.prediction_ratio)
    
        while mask_count < num_target_patches:
            # Sampling block dimensions
            delta = 0
            for _ in range(self.num_tries):
                # Sampling random scale
                scale = random.uniform(self.min_block_scale, self.max_block_scale)
                # Sampling random aspect ratio
                aspect_ratio = random.uniform(self.aspect_ratio_min, self.aspect_ratio_max)
    
                # Calculate Height and Width on patches
                # Area = H * W = num_patches * scale
                # Ratio = H / W
                # H = sqrt(Area * Ratio)
                # W = sqrt(Area / Ratio)
    
                target_area = self.num_patches * scale
    
                h = int(math.sqrt(target_area * aspect_ratio))
                w = int(math.sqrt(target_area / aspect_ratio))
    
                # Limit restriction
                h = min(h, self.height)
                w = min(w, self.width)
    
                if h * w > 0:
                    break
    
            if h * w == 0:
                continue # Error on valid dimensions generation
    
            # Sampling position (Top-Left)
            # Restrict so that the block fits within the image
            top = random.randint(0, self.height - h)
            left = random.randint(0, self.width - w)
    
            # Apply block to the mask
            # Obtain how many NEW patches are being masked (avoid counting overlaps)
            block_mask = mask[top : top + h, left : left + w]
            new_masked = (block_mask == 0).sum().item()

            if new_masked > 0:
                mask[top : top + h, left : left + w] = 1
                mask_count += new_masked
    
            if mask_count >= num_target_patches:
                break
    
        return mask.flatten() # [N_patches]


    def __call__(self,
                 batch_size: int,
                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context_idx = Indices of visible patches [B, N_ctx]
            target_idx = Indices of hidden patches to predict [B, N_tgt]
        """

        num_target = int(self.num_patches * self.prediction_ratio)

        # Masking priorities array
        # Shape: [B, N_patches]
        mask_noise = torch.zeros(batch_size, self.num_patches, dtype = torch.float32)

        # Generate mask per block for each image in the batch
        for i in range(batch_size):
            # Generate the boolean mask
            block_mask = self._sample_block_mask() # [N_patches]

            # Generate small random noise 
            random_noise = torch.rand(self.num_patches) * 0.1

            # Add the block mask to the noise.
            # The patches inside the block will have value > 1.0
            # The patches out of the block will have value < 0.1
            mask_noise[i] = block_mask.float() + random_noise

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