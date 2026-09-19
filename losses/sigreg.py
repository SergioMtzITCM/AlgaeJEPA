import torch
import torch.nn as nn

class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer"""
    def __init__(self,
                 knots: int = 17,
                 t_max: float = 5.0,
                 num_slices: int = 1024) -> None:

        super().__init__()

        self.num_slices = num_slices

        t = torch.linspace(-t_max, t_max, knots, dtype = torch.float32)
        phi = torch.exp(-0.5 * t.square()) 

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (..., D)
        """

        flat = x.reshape(-1, x.size(-1)) # (N, D), N = B * N_p
        N = flat.size(0)

        # Random Unit Projections (Cramer-Wold)
        A = torch.rand(flat.size(-1), self.num_slices, device = flat.device, dtype = torch.float32)
        A = A / A.norm(p = 2, dim = 0)
        A = A.to(flat.dtype)

        t = self.t.to(flat.dtype)
        phi = self.phi.to(flat.dtype)

        # Empirical Characteristic Function (Euler)
        x_t = (flat @ A).unsqueeze(-1) * t # (N, num_slices, knots)
        ecf_real = x_t.cos().mean(0) # Average over 'N' samples
        ecf_imag = x_t.sin().mean(0)

        # L2 Distance Weighted by Gaussian Window 'phi(t)' integrated in 't' (Epps-Pulley)
        err = (ecf_real - phi).square() + ecf_imag.square()
        err = err * phi
        stat = torch.trapezoid(err, t, dim = -1) * N # (num_slices,)

        return stat.mean() # Average over random projections


