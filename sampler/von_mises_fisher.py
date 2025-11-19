import math
import torch
from torch.distributions import Distribution, constraints
from ive import ive


class VonMisesFisher3D(Distribution):
    """
    Simpler von Mises–Fisher distribution on the unit sphere in R^3.

    Parameters
    ----------
    loc : tensor (..., 3)
        Mean direction, expected to be (approximately) unit-norm.
    scale : tensor (...)
        Concentration kappa > 0.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
    } # type: ignore
    support = constraints.real # type: ignore
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        assert loc.shape[-1] == 3, "This simple version only supports R^3."
        self.loc = loc / loc.norm(dim=-1, keepdim=True)  # normalize just in case
        self.scale = scale
        self.device = loc.device
        self.dtype = loc.dtype

        super().__init__(batch_shape=self.loc.shape[:-1], validate_args=validate_args)

   
    def mean(self):
        # E[X] = A_m(kappa) * mu with m=3
        m = 3
        kappa = self.scale
        A = ive(m / 2, kappa) / ive(m / 2 - 1, kappa) # type: ignore
        return self.loc * A.unsqueeze(-1)

    
    def stddev(self):
        # Not really a standard notion on the sphere; we just return kappa
        return self.scale

    def _sample_w3(self, sample_shape):
        """
        Sample w = cos(theta) for m=3.
        This is a simplified version of the original __sample_w3.
        """
        shape = sample_shape + self.scale.shape
        u = torch.rand(shape, device=self.device, dtype=self.dtype)
        # same formula as in the original code, but written more directly
        log_u = torch.log(u)
        log_1_minus_u = torch.log(1 - u)
        # numerically stable: logsumexp over [log(u), log(1-u) - 2*kappa]
        stacked = torch.stack([log_u, log_1_minus_u - 2 * self.scale], dim=0)
        log_sum = torch.logsumexp(stacked, dim=0)
        w = 1 + log_sum / self.scale
        return w

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampling.
        Returns samples of shape sample_shape + batch_shape + (3,).
        """
        w = self._sample_w3(sample_shape)  # shape: sample_shape + batch_shape

        # Sample a random vector and make it orthogonal to loc
        # Shape target: sample_shape + batch_shape + (3,)
        eps = torch.randn(sample_shape + self.loc.shape, device=self.device, dtype=self.dtype)
        # Remove component along loc
        proj = (eps * self.loc).sum(dim=-1, keepdim=True)
        v = eps - proj * self.loc
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)  # unit vector orthogonal to loc

        # Combine radial and tangential parts
        w_expanded = w.unsqueeze(-1)                                      # (..., 1)
        factor = torch.sqrt(torch.clamp(1 - w_expanded**2, 1e-10, 1.0))   # (..., 1)
        x = w_expanded * self.loc + factor * v                            # (..., 3)

        return x

    def log_prob(self, x):
        """
        log p(x) = kappa * <mu, x> - log C_3(kappa)
        """
        kappa = self.scale
        dot = (self.loc * x).sum(dim=-1)  # inner product <mu, x>

        return kappa * dot - self._log_normalization()

    def _log_normalization(self):
        """
        log C_3(kappa) for m=3:
        C_3(kappa) = kappa / (4*pi * sinh(kappa))
        So log C_3(kappa) = log(kappa) - log(4*pi) - log(sinh(kappa))
        """
        kappa = self.scale
        return (
            torch.log(kappa)
            - math.log(4 * math.pi)
            - torch.log(torch.sinh(kappa) + 1e-10)
        )
