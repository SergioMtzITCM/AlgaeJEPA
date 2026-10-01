import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class EmbeddingLoss(nn.Module):
    """
    Regression loss between 'pred' and 'target' embeddings (both with the SAME shape).
 
    'feature_dim' is the axis that holds the feature vector (hidden size / channels). It is NOT always the
    last one:
        - Algae-JEPA tokens          [B, N, D]    -> feature_dim = -1 (default)
        - Distillation, channel-first [B, C, H, W] -> feature_dim = 1  (ViT and CNN are both standardized
                                                                        to this layout by KD_Trainer)
 
    'mse' and 'l1' are element-wise means, so they do not depend on 'feature_dim'.
    'l2' and 'cosine' are defined PER FEATURE VECTOR (they reduce over 'feature_dim') and then averaged
    over all the remaining axes (batch, tokens / spatial positions).
    """
    def __init__(self,
                 loss_type: str = "mse",
                 feature_dim: int = -1) -> None:
        super().__init__()

        self.loss_type = loss_type.lower()

        if self.loss_type not in ["mse", "l1", "l2", "cosine"]:
            raise ValueError("Loss type must be: 'mse', 'l1', 'l2', or 'cosine'")

        if not isinstance(feature_dim, int):
            raise TypeError(f"'feature_dim' must be an int (received {feature_dim!r})")

        self.feature_dim = feature_dim

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                feature_dim: Optional[int] = None) -> torch.Tensor:

        # Shapes must match exactly: a silent broadcast would hide spatial / channel misalignments
        if pred.shape != target.shape:
            raise ValueError(
                f"'pred' and 'target' must have the same shape (received {tuple(pred.shape)} and {tuple(target.shape)})"
            )

        # Losses are computed in float32 (no-op when they already are)
        pred = pred.float()
        target = target.float()
 
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target)

        # 'l2' and 'cosine' reduce over the feature axis
        dim = self.feature_dim if feature_dim is None else feature_dim

        if not -pred.ndim <= dim < pred.ndim:
            raise ValueError(f"'feature_dim' = {dim} is out of range for tensors with {pred.ndim} dimensions")
 
        if self.loss_type == "l2":
            return torch.norm(pred - target, p = 2, dim = dim).mean()
        elif self.loss_type == "cosine":
            # Maximize similarity
            return (1 - F.cosine_similarity(pred, target, dim = dim)).mean()
