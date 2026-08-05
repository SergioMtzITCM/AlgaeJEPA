import torch
import torch.distributed as dist

"""def SIGReg_Loss(x: torch.Tensor, global_step: int, num_slices: int = 256):

    # MODIFICAR
    B, N_p, D = x.shape
    
    # Slice Sampling (synced across devices)
    dev = dict(device = x.device)
    g = torch.Generator(**dev)
    g.manual_seed(global_step)
    #proj_shape = (x.size(1), num_slices)
    proj_shape = (D, num_slices)
    A = torch.randn(proj_shape, generator = g, **dev)
    A /= A.norm(p = 2, dim = 0)

    # Epps-Pulley stat
    # Integration Points
    t = torch.linspace(-5, 5, 17, **dev)
    # Theoretical CF for N(0, 1) and Gauss. window
    exp_f = torch.exp(-0.5 * t**2)

    # Empirical CF (gathered across devices)
    #x_t = (x @ A).unsqueeze(2) * t  # (N, M, T)
    #ecf = (1j * x_t).exp().mean(0)
    x_t = (x @ A).unsqueeze(-1) * t # (B, N_p, num_slices, T)
    ecf = (1j * x_t).exp().mean(1) # (B, num_slices, T)
    
    # Solo reducir si el entrenamiento distribuido está activo
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ecf, op = dist.ReduceOp.AVG)
        world_size = dist.get_world_size()
    else:
        world_size = 1

    # Weighted L2 Distance
    err = (ecf - exp_f).abs().square().mul(exp_f)

    #N = x.size(0) * world_size
    N = N_p * world_size
    
    #T = torch.trapz(err, t, dim = 1) * N # (B, num_slices)
    T = torch.trapz(err, t, dim = -1) * N # (B, num_slices)
    
    #return T
    return T.mean()"""

def SIGReg_Loss(x: torch.Tensor, global_step: int, num_slices: int = 256):
    B, N_p, D = x.shape
    
    # Slice Sampling (synced across devices)
    dev = dict(device = x.device)
    g = torch.Generator(**dev)
    g.manual_seed(global_step)
    
    proj_shape = (D, num_slices)
    A = torch.randn(proj_shape, generator = g, **dev)
    A /= A.norm(p = 2, dim = 0)

    # Epps-Pulley stat
    # Integration Points
    t = torch.linspace(-5, 5, 17, **dev)
    # Theoretical CF for N(0, 1) and Gauss. window (es puramente real)
    exp_f = torch.exp(-0.5 * t**2)

    # Empirical CF (usando Fórmula de Euler para evitar números complejos)
    x_t = (x @ A).unsqueeze(-1) * t # (B, N_p, num_slices, T)
    
    # En lugar de (1j * x_t).exp(), separamos en parte real y parte imaginaria
    ecf_real = torch.cos(x_t).mean(1) # Parte real (B, num_slices, T)
    ecf_imag = torch.sin(x_t).mean(1) # Parte imaginaria (B, num_slices, T)
    
    # Solo reducir si el entrenamiento distribuido está activo
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ecf_real, op = dist.ReduceOp.AVG)
        dist.all_reduce(ecf_imag, op = dist.ReduceOp.AVG)
        world_size = dist.get_world_size()
    else:
        world_size = 1

    # Distancia L2 ponderada
    # |(ecf_real + 1j * ecf_imag) - exp_f|^2 = (ecf_real - exp_f)^2 + (ecf_imag)^2
    err_sq = (ecf_real - exp_f).square() + ecf_imag.square()
    err = err_sq.mul(exp_f)

    N = N_p * world_size
    
    T = torch.trapz(err, t, dim = -1) * N # (B, num_slices)
    
    return T.mean()