import torch 
import torch.nn as nn
import torch.nn.functional as F
import geomstats as gs
import numpy as np
from sampler.spherical_uniform import *
from sampler.von_mises_fisher import *


class SphericalVAE(nn.Module):
    def __init__(self, 
                extrinsic_dim,
                latent_dim,
                sftbeta=4.5,
                encoder_width=400,
                encoder_depth=4,
                decoder_width=400,
                decoder_depth=4,
                dropout_p=0.0,          
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Variables
        self.extrinsic_dim = extrinsic_dim
        self.latent_dim = latent_dim
        self.sftbeta = sftbeta
        self.encoder_width=encoder_width
        self.encoder_depth=encoder_depth
        self.decoder_width=decoder_width
        self.decoder_depth=decoder_depth
        self.dropout_p=dropout_p

        # Encoder layers
        self.encoder_function = nn.Linear(in_features=self.extrinsic_dim, out_features=self.encoder_width)
        self.encoder_linear_layers = nn.ModuleList(
            [nn.Linear(in_features=self.encoder_width, out_features=self.encoder_width) for _ in range(encoder_depth)]
        )
        
        self.function_z_mu = nn.Linear(in_features=self.encoder_width, out_features=self.latent_dim) # mean of the Von Mises Fischer
        self.function_z_logvar = nn.Linear(in_features=self.encoder_width, out_features=1) # Kappa of Von Mises Fischer

        # Decoder layers
        self.decoder_function = nn.Linear(in_features=self.latent_dim, out_features=decoder_width)
        self.decoder_linear_layers = nn.ModuleList(
            [nn.Linear(in_features=self.decoder_width, out_features=self.decoder_width) for _ in range(decoder_depth)]
        )

        # Posteriori
        self.function_x_mu = nn.Linear(in_features=self.decoder_width, out_features=self.extrinsic_dim)
        self.dropout = nn.Dropout(p = self.dropout_p)

    
    # Encoding function
    def encode(self, x):
        """
        This function encodes x to a latent space
        
        Returns:

        mu : Mean of the multivariate Gaussians
        logvar : Vector representing the diagonal covariance of the multivariate gaussians
        """

        h = F.softplus(self.encoder_function(x), beta=self.sftbeta)

        for layer in self.encoder_linear_layers:
            h = layer(h)
            h = F.softplus(h, beta = self.sftbeta)
        
        z_mu = self.function_z_mu(h)
        z_kappa = F.softplus(self.function_z_logvar(h)) + 1

        return z_mu, z_kappa
    
    def reparemetrize(self, posteriori_params):
        """
        Here we define the reparametrization trick as seen in the paper 
        THe idea is to compute a differentiable PHI, with which he preserve the probability
        """

        z_mu, z_kappa = posteriori_params # params of Von Mises Fischer
        q_z = VonMisesFisher3D(z_mu, z_kappa) # Postetioti q(.) = p(. | z)
 
        return q_z.sample() # sample
    
    def decode(self, z):
        """
        This function decodes z sampled from the latent space

        Returns:
            Reconstructed data

        """

        h = F.softplus(z, beta = self.sftbeta)

        for layer in self.decoder_linear_layers:
            h = layer(h)
            h = F.softplus(h, beta = self.sftbeta)
        
        return self.function_x_mu(h)
    
    def forward(self, x):
        """
        Compute the output of VAE with input x

        Returns:
        mu : Mean of multivariate Gaussian
        Logvar : Variance vector of the Gaussians
        """

        posteriori_params = self.encode(x)
        z = self.reparemetrize(posteriori_params)
        x_mu = self.decode(z)
        
        # return z, x_mu and posteriori params of distribution
        return z, x_mu, posteriori_params
    
    def _elbo(self, x, x_mu, posteriori_params):
        """
        The Estimated Lower Bound of the VAE training

        Returns :

            ELBO of the VAE = E(log(p(x|z))) - KL(q(z|x) || p(z))
        """

        z_mu, z_kappa = posteriori_params
        q_z = VonMisesFisher3D(z_mu, z_kappa)
        p_z = SphericalUniform(dim = self.latent_dim - 1, device=x.device)

        KL = torch.distributions.kl.kl_divergence(q_z, p_z)
        mean = torch.mean((x - x_mu)**2)

        return mean + KL

    
    












