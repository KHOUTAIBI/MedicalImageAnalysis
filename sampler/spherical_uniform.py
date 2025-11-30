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

    