import torch
import torch.nn as nn
import torch.nn.functional as F

class IJEPA_Loss(nn.Module):
    def __init__(self,
                 loss_type: str = "mse"):
        super().__init__()

        self.loss_type = loss_type.lower()

        if self.loss_type not in ["mse", "l1", "l2", "cosine"]:
            raise ValueError("Loss type must be: 'mse', 'l1', 'l2', or 'cosine'")

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.loss_type == "l2":
            return torch.norm(pred - target, p = 2, dim = -1).mean()
        elif self.loss_type == "cosine":
            # Maximize similarity
            return (1 - F.cosine_similarity(pred, target, dim = -1)).mean()