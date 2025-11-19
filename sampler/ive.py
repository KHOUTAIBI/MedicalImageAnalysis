import torch
import torch.nn as nn
import numpy as np
import scipy.special

class IveFunction(torch.autograd.Function):
    """
    Differentiable wrapper around scipy.special.ive(v, z)
    using PyTorch custom autograd.
    """

    @staticmethod
    def forward(ctx, v, z):
        # Save context for backward
        ctx.v = v
        ctx.save_for_backward(z)

        # Convert to CPU NumPy
        z_cpu = z.detach().cpu().numpy()

        # Use specialized functions for speed/stability
        if v == 0:
            out_np = scipy.special.i0e(z_cpu)
        elif v == 1:
            out_np = scipy.special.i1e(z_cpu)
        else:
            out_np = scipy.special.ive(v, z_cpu)

        # Convert back to torch tensor
        out = torch.from_numpy(out_np).to(z.device, z.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        v = ctx.v
        (z,) = ctx.saved_tensors

        # d/dz ive(v, z)
        #   = ive(v-1, z) - ive(v, z)*(v+z)/z
        dz = ive(v - 1, z) - ive(v, z) * (v + z) / z # Here is the gradient of the Bessel function

        return None, grad_output * dz


class Ive(nn.Module):
    """
    Module wrapper so you can write Ive(v)(z)
    """

    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)


# Convenient alias
ive = IveFunction.apply
