import torch
import torch.nn as nn

from utils.masking_generator import MultiBlockMasking

from models.vit_core import ViTModel
from models.predictor_core import PredictorModel

from losses.loss import IJEPA_Loss
from losses.sigreg import SIGReg_Loss

from configs.config import BaseConfig

from typing import Tuple

class SIGReg_IJEPA(nn.Module):
    def __init__(self,
                 config: BaseConfig,
                 loss_type: str = "mse",
                 lambda_sigreg: float = 0.1) -> None:
        super().__init__()

        self.config = config

        self.encoder = ViTModel(config)

        self.predictor = PredictorModel(config)

        self.mask_generator = MultiBlockMasking(
            input_size = config.image_size,
            patch_size = config.patch_size,
            prediction_ratio = config.prediction_ratio
        )

        self.reconstruction_criterion = IJEPA_Loss(loss_type)
        self.lambda_sigreg = lambda_sigreg


    def extract_embeddings_by_indices(self,
                                      embeddings: torch.Tensor,
                                      context_idx: torch.Tensor,
                                      target_idx: torch.Tensor) -> torch.Tensor:
        
        """
        Gather Especific Embeddings Subset [B, N_subset, D]

        Args:
            embeddings: All Image Embeddings [B, N, D]
            context_idx: Indices of visible patches [B, N_ctx]
            target_idx = Indices of hidden patches to predict [B, N_tgt]

        Returns:
            context_embeddings: Embeddings of visible patches [B, N_ctx, D]
            target_embeddings: Embeddings of hidden patches to predict [B, N_tgt, D]
        """

        B, _, D = embeddings.shape

        # Expand Indices: [B, N_subset] -> [B, N_subset, D]
        context_idx_expanded = context_idx.unsqueeze(-1).expand(-1, -1, D)
        target_idx_expanded = target_idx.unsqueeze(-1).expand(-1, -1, D)

        # Gather by Indices
        context_embeddings = torch.gather(embeddings, 1, context_idx_expanded)
        target_embeddings = torch.gather(embeddings, 1, target_idx_expanded)

        return context_embeddings, target_embeddings

    def get_target_embeddings_from_patches(self,
                                           all_patch_embeddings: torch.Tensor,
                                           pixel_values: torch.Tensor,
                                           target_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute target embeddings by passing all the sequence trought Encoder in order to
        capture global context, but recycling projected patches.
        """

        B, N, D = all_patch_embeddings.shape
        device = all_patch_embeddings.device

        # Compute RoPE for all sequence
        full_idx = torch.arange(N, device = device).unsqueeze(0).expand(B, -1)
        full_cos_sin = self.encoder.rope(pixel_values, full_idx)

        # Forward pass of pixel values
        all_embeddings = self.encoder.forward_embeddings(all_patch_embeddings, full_cos_sin)

        # Extract only the target embeddings
        _, target_embeddings = self.extract_embeddings_by_indices(
            embeddings = all_embeddings,
            context_idx = target_idx,
            target_idx = target_idx
        )

        return target_embeddings

    def encode_image(self,
                     pixel_values: torch.Tensor,
                     context_idx: torch.Tensor,
                     target_idx: torch.Tensor) -> Tuple[
                                                    torch.Tensor,
                                                    torch.Tensor,
                                                    Tuple[torch.Tensor, torch.Tensor],
                                                    Tuple[torch.Tensor, torch.Tensor]
                                                  ]:
        """
        Encode the visible (context) and non visible (target) parts of the image
        using the encoder.
    
        Args:
            pixel_values: Image Tensor [Batch, C, H, W]
            context_idx: Indices of visible patches [B, N_ctx]
            target_idx = Indices of hidden patches to predict [B, N_tgt]
    
        Returns:
            context_embeddings: Embeddings of visible patches [B, N_ctx, D]
            target_embeddings: Embeddings of hidden patches to predict [B, N_tgt, D]
            contex_cos_sin: Context RoPE [B, N_ctx, Head_dim]
            target_cos_sin: Target RoPE [B, N_tgt, Head_dim]
        """

        # Unique patch projection
        all_patch_embeddings = self.encoder.get_patch_embeddings(pixel_values)

        # Compute RoPE: [B, N_susbet, D]
        context_cos_sin = self.encoder.rope(pixel_values, context_idx)
        target_cos_sin = self.encoder.rope(pixel_values, target_idx)

        # Context flow (only visible patches)
        context_embedings_raw, _ = self.extract_embeddings_by_indices(
            embeddings = all_patch_embeddings,
            context_idx = context_idx,
            target_idx = target_idx
        )
        context_embeddings = self.encoder.forward_embeddings(context_embedings_raw, context_cos_sin)

        # Target flow (forward target patches for global context and then extract them)
        target_embeddings = self.get_target_embeddings_from_patches(
            all_patch_embeddings = all_patch_embeddings,
            pixel_values = pixel_values,
            target_idx = target_idx
        )

        return context_embeddings, target_embeddings, context_cos_sin, target_cos_sin

    def forward(self,
                x_input: torch.Tensor,
                global_step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        device = x_input.device
        B = x_input.shape[0]

        # Get the Masks
        context_idx, target_idx = self.mask_generator(batch_size = B, device = device)

        context_idx = context_idx.to(device)
        target_idx = target_idx.to(device)

        # Get Context, Target Embeddings and Context, Target RoPE Using the Encoder
        context_embeddings, target_embeddings, context_cos_sin, target_cos_sin = self.encode_image(
            pixel_values = x_input,
            context_idx = context_idx,
            target_idx = target_idx
        )

        Dim = context_embeddings.shape[-1]

        # Predict the Target Embeddings Using the Predictor
        predicted_target_embeddings = self.predictor(
            context_embeddings,
            context_cos_sin,
            target_cos_sin
        )

        # Compute the Reconstruction Loss
        rec_loss = self.reconstruction_criterion(
            predicted_target_embeddings,
            target_embeddings
        )

        # Compute the SIGReg Loss
        #sigreg_loss = SIGReg_Loss(target_embeddings.reshape(-1, Dim), global_step = global_step).mean()
        sigreg_loss = SIGReg_Loss(target_embeddings, global_step = global_step)

        # Compute Total Loss
        total_loss = (1 - self.lambda_sigreg) * rec_loss + self.lambda_sigreg * sigreg_loss

        return rec_loss, sigreg_loss, total_loss