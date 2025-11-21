import torch
import torch.nn as nn
import numpy as np
import scipy.special

class IveFunction(torch.autograd.Function):
    """
    Differentiable wrapper around scipy.special.ive(v, z)
    (exponentially scaled modified Bessel function of the first kind).
    """

    @staticmethod
    def forward(ctx, v, z):
        # Save context for backward
        ctx.v = v
        ctx.save_for_backward(z)

        z_cpu = z.detach().cpu().numpy()

        # Use specialized functions for v=0,1
        if v == 0:
            out_np = scipy.special.i0e(z_cpu)
        elif v == 1:
            out_np = scipy.special.i1e(z_cpu)
        else:
            out_np = scipy.special.ive(v, z_cpu)

        out = torch.from_numpy(out_np).to(z.device, z.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        v = ctx.v
        (z,) = ctx.saved_tensors

        z_cpu = z.detach().cpu().numpy()
        eps = 1e-8

        # Compute ive(v, z) and ive(v-1, z) on CPU with SciPy
        if v == 0:
            ive_v_np   = scipy.special.i0e(z_cpu)
            ive_vm1_np = scipy.special.ive(-1, z_cpu)  # or scipy.special.i1e for appropriate relation
        elif v == 1:
            ive_v_np   = scipy.special.i1e(z_cpu)
            ive_vm1_np = scipy.special.i0e(z_cpu)
        else:
            ive_v_np   = scipy.special.ive(v, z_cpu)
            ive_vm1_np = scipy.special.ive(v - 1, z_cpu)

        ive_v   = torch.from_numpy(ive_v_np).to(z.device, z.dtype)
        ive_vm1 = torch.from_numpy(ive_vm1_np).to(z.device, z.dtype)

        # sign(z) and safe division
        sign_z = torch.sign(z)
        z_safe = z.clone()
        z_safe = torch.where(z_safe.abs() < eps, torch.full_like(z_safe, eps), z_safe)

        # Correct derivative for scaled ive:
        # d/dz ive(v, z) = ive(v-1, z) - (v/z + sign(z)) * ive(v, z)
        dz = ive_vm1 - (v / z_safe + sign_z) * ive_v

        grad_z = grad_output * dz
        return None, grad_z


class Ive(nn.Module):
    """
    Module wrapper so you can write Ive(v)(z)
    """
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, z):
        return IveFunction.apply(self.v, z)


# Convenient alias
ive = IveFunction.apply
