import torch
from torch.distributions import Distribution, constraints

class SphericalUniform(Distribution):
    """
    Uniform distribution on the unit sphere S^{dim-1} in R^dim.
    """

    arg_constraints = {} # type: ignore
    support = constraints.real # type: ignore
    has_rsample = False  # sampling is not reparameterized

    def __init__(self, dim, validate_args=None, device=None, dtype=torch.float32):
        assert dim >= 2, "Dimension must be at least 2."
        self.dim = dim
        self.device = device
        self.dtype = dtype

        # No batch shape. Event shape is (dim,)
        super().__init__(batch_shape=torch.Size(),
                         event_shape=torch.Size([dim]),
                         validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        """
        Draw samples from the uniform distribution on S^{dim-1}
        by normalizing Gaussian vectors.
        """

        shape = sample_shape + (self.dim,) # type: ignore
        z = torch.randn(shape, device=self.device, dtype=self.dtype)
        return z / z.norm(dim=-1, keepdim=True)

    def log_prob(self, x):
        """
        The uniform density on the sphere is constant:
        p(x) = 1 / surface_area(S^{dim-1})
        => log p(x) = -log(surface_area)
        """

        surface_area = (
            2 * torch.pi ** (self.dim / 2)
            / torch.lgamma(torch.tensor(self.dim / 2)).exp()
        )

        log_prob = -torch.log(surface_area)

        # Return log_prob with proper shape
        return log_prob.expand(x.shape[:-1])

    def entropy(self):
        """
        Entropy of the uniform distribution on the sphere:
        H = log(surface_area(S^{dim-1}))
        """
        surface_area = (
            2 * torch.pi ** (self.dim / 2)
            / torch.lgamma(torch.tensor(self.dim / 2)).exp()
        )
        return torch.log(surface_area)
