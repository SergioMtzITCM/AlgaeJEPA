import torch
import torch.nn as nn

class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer"""
    def __init__(self,
                 knots: int = 17,
                 t_max: float = 5.0,
                 num_slices: int = 1024) -> None:

        super().__init__()

        if knots < 2:
            raise ValueError(f"'knots' must be >= 2 (received {knots})")
        if t_max <= 0:
            raise ValueError(f"'t_max' must be > 0 (received {t_max})")
        if num_slices < 1:
            raise ValueError(f"'num_slices' must be >= 1 (received {num_slices})")

        self.num_slices = num_slices

        # [0, t_max]
        t = torch.linspace(0.0, t_max, knots, dtype = torch.float32)
        dt = t_max / (knots - 1)

        # N(0, 1) ECF and Gaussian Window w(t) = exp(-t²/2)
        phi = torch.exp(-0.5 * t.square()) 

        # Trapezoid Weights over [0, t_max]
        quad_weights = torch.full((knots,), 2.0 * dt, dtype = torch.float32)
        quad_weights[0] = dt
        quad_weights[-1] = dt

        self.register_buffer("t", t, persistent = False)
        self.register_buffer("phi", phi, persistent = False)
        self.register_buffer("weights", quad_weights * phi, persistent = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (..., D)
        """

        with torch.autocast(device_type = x.device.type, enabled = False):

            flat = x.reshape(-1, x.size(-1)).float() # (N, D), N = B * N_p
            N = flat.size(0)

            # Random Unit Projections (Cramer-Wold)
            A = torch.randn(flat.size(-1), self.num_slices, device = flat.device, dtype = torch.float32)
            A = A / A.norm(p = 2, dim = 0, keepdim = True)

            
            # Empirical Characteristic Function (Euler) and Projections
            proj = flat @ A                                   # (N, num_slices)
            x_t = proj.unsqueeze(-1) * self.t                 # (N, num_slices, knots)
            ecf_real = x_t.cos().mean(dim = 0)                # (num_slices, knots)
            ecf_imag = x_t.sin().mean(dim = 0)

            # L2 Distance Weighted by Gaussian Window 'phi(t)' integrated in 't' (Epps-Pulley)
            err = (ecf_real - self.phi).square() + ecf_imag.square()
            stat = (err * self.weights).sum(dim = -1) * N     # (num_slices,)

            return stat.mean() # Average over random projections


