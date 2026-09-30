import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking_generator import MultiBlockMasking

from models.vit_core import ViTModel
from models.predictor_core import PredictorModel

from losses.embedding import EmbeddingLoss
from losses.sigreg import SIGReg

from configs.config import BaseConfig

from typing import Dict, Optional, Tuple, Union

class AlgaeJepa(nn.Module):
    def __init__(self,
                 config: BaseConfig,
                 loss_type: str = "mse",
                 lambda_sigreg: float = 0.1,
                 sigreg_num_slices: int = 1024,
                 sigreg_knots: int = 17,
                 sigreg_t_max: float = 5.0,
                 seed: Optional[int] = None) -> None:
        super().__init__()

        self.config = config

        self.encoder = ViTModel(config)

        self.predictor = PredictorModel(config)

        self.mask_generator = MultiBlockMasking(
            input_size = config.image_size,
            patch_size = config.patch_size,
            prediction_ratio = config.prediction_ratio,
            seed = seed
        )

        self.sigreg = SIGReg(knots = sigreg_knots, t_max = sigreg_t_max, num_slices = sigreg_num_slices)
        self.embedding_prediction_criterion = EmbeddingLoss(loss_type)
        self.lambda_sigreg = lambda_sigreg

    @staticmethod
    def _gather_tokens(embeddings: torch.Tensor,
                       idx: torch.Tensor) -> torch.Tensor:
        """Gather a token subset: embeddings [B, N, D], idx [B, N_subset] -> [B, N_subset, D]"""
 
        D = embeddings.shape[-1]
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, D)
 
        return torch.gather(embeddings, 1, idx_expanded)

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
        target_embeddings = self._gather_tokens(all_embeddings, target_idx)

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
        context_embeddings_raw = self._gather_tokens(all_patch_embeddings, context_idx)
        context_embeddings = self.encoder.forward_embeddings(context_embeddings_raw, context_cos_sin)

        # Target flow (forward target patches for global context and then extract them)
        target_embeddings = self.get_target_embeddings_from_patches(
            all_patch_embeddings = all_patch_embeddings,
            pixel_values = pixel_values,
            target_idx = target_idx
        )

        return context_embeddings, target_embeddings, context_cos_sin, target_cos_sin

    def forward(self,
                x_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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

        # Predict the Target Embeddings Using the Predictor
        predicted_target_embeddings = self.predictor(
            context_embeddings,
            context_cos_sin,
            target_cos_sin,
            target_idx = target_idx
        )

        # Compute the Target Embedding Prediction Loss
        embedding_loss = self.embedding_prediction_criterion(
            predicted_target_embeddings,
            target_embeddings
        )

        # Compute the SIGReg Loss
        sigreg_loss = self.sigreg(target_embeddings)

        # Compute Total Loss
        total_loss = (1 - self.lambda_sigreg) * embedding_loss + self.lambda_sigreg * sigreg_loss

        return embedding_loss, sigreg_loss, total_loss

    
    @torch.no_grad()
    def compute_diagnostics(self,
                            x_input: torch.Tensor,
                            probe_activations: bool = True) -> Dict[str, Union[float, str]]:
        """
        Training Health Metrics Over a Batch (Without Gradient, Eval Mode).

        They help to interpret the embedding loss:
            explained_variance   : 1 - MSE / Var(target). Fraction of the target variance explained by the
                                   prediction relative to predicting the global mean token.
            target_std_mean/min  : Standard deviation by target dimension (collapse > ~0).
            effective_rank       : exp(entropy) of the target covariance spectrum
                                   (maximum = D; dimensional collapse -> small values).
            between_image_var_frac: fraction of the total token variance stemming from
                                    differences BETWEEN images (law of total variance). Close to 1
                                    = all tokens in an image are nearly identical (spatial collapse).
            sigreg               :  reference under N(0, I) i.i.d. ≈ 1.05.
            target_abs_max, act_in_max, act_out_max: Maximum magnitudes. fp16 overflows at 65,504.
        """

        device = x_input.device
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        mask_rng_state = self.mask_generator.get_rng_state()
 
        was_training = self.training
        self.eval()
 
        act_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        handles = []
 
        if probe_activations:
            def make_hook(name: str):
                def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                    cur_in = inputs[0].detach().abs().amax().float()
                    cur_out = output.detach().abs().amax().float()
                    if name in act_stats:
                        prev_in, prev_out = act_stats[name]
                        cur_in = torch.maximum(cur_in, prev_in)
                        cur_out = torch.maximum(cur_out, prev_out)
                    act_stats[name] = (cur_in, cur_out)
                return hook
 
            for prefix, module in (("encoder", self.encoder), ("predictor", self.predictor)):
                for name, sub in module.named_modules():
                    if isinstance(sub, nn.Linear):
                        handles.append(sub.register_forward_hook(make_hook(f"{prefix}.{name}")))
 
        try:
            device = x_input.device
            B = x_input.shape[0]
 
            context_idx, target_idx = self.mask_generator(batch_size = B, device = device)
            context_idx = context_idx.to(device)
            target_idx = target_idx.to(device)
 
            context_embeddings, target_embeddings, context_cos_sin, target_cos_sin = self.encode_image(
                pixel_values = x_input,
                context_idx = context_idx,
                target_idx = target_idx
            )
            predicted = self.predictor(context_embeddings, context_cos_sin, target_cos_sin)
 
            metrics: Dict[str, Union[float, str]] = {}
 
            with torch.autocast(device_type = device.type, enabled = False):
                tgt = target_embeddings.float()                      # [B, M, D]
                prd = predicted.float()
                ctx = context_embeddings.float()
                D = tgt.shape[-1]
                flat = tgt.reshape(-1, D)                            # [B*M, D]
 
                mse = F.mse_loss(prd, tgt)
                var_dim = flat.var(dim = 0, unbiased = False)        # [D]
                baseline_mse = var_dim.mean()                        # MSE of predicting the global average token
                std_dim = var_dim.sqrt()
 
                # Effective rank (Roy & Vetterli) of the target token covariance
                centered = flat - flat.mean(dim = 0, keepdim = True)
                cov = centered.T @ centered / max(flat.shape[0] - 1, 1)
                eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
                p = eig / eig.sum().clamp_min(1e-12)
                effective_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())
 
                # Ley de la varianza total: var_total = var_entre_imagenes + media(var_dentro)
                image_mean = tgt.mean(dim = 1)                       # [B, D]
                var_between = image_mean.var(dim = 0, unbiased = False).sum()
                between_frac = var_between / var_dim.sum().clamp_min(1e-12)
 
                metrics["embed_loss"] = mse.item()
                metrics["target_var_mean"] = baseline_mse.item()
                metrics["explained_variance"] = (1.0 - mse / baseline_mse.clamp_min(1e-12)).item()
                metrics["target_std_mean"] = std_dim.mean().item()
                metrics["target_std_min"] = std_dim.min().item()
                metrics["pred_std_mean"] = prd.reshape(-1, D).std(dim = 0, unbiased = False).mean().item()
                metrics["effective_rank"] = effective_rank.item()
                metrics["between_image_var_frac"] = between_frac.item()
                metrics["sigreg"] = self.sigreg(tgt).item()
                metrics["target_abs_max"] = tgt.abs().max().item()
                metrics["context_abs_max"] = ctx.abs().max().item()
 
            if act_stats:
                in_name = max(act_stats, key = lambda k: act_stats[k][0].item())
                out_name = max(act_stats, key = lambda k: act_stats[k][1].item())
                metrics["act_in_max"] = act_stats[in_name][0].item()
                metrics["act_in_max_module"] = in_name
                metrics["act_out_max"] = act_stats[out_name][1].item()
                metrics["act_out_max_module"] = out_name
 
            return metrics
 
        finally:
            for h in handles:
                h.remove()
            self.train(was_training)
