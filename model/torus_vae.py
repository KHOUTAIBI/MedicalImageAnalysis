import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sampler.spherical_uniform import *
from sampler.von_mises_fisher import *

import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import geomstats.backend as gs # type: ignore


class ToroidalVAE(nn.Module):
    def __init__(self,
                config,           
                *args, 
                **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Variables
        self.config = config
        self.extrinsic_dim = config["extrinsic_dim"]
        self.latent_dim = config["latent_dim"]
        self.sftbeta = config["sftbeta"]
        self.encoder_width=config["encoder_width"]
        self.encoder_depth=config["encoder_depth"]
        self.decoder_width=config["decoder_width"]
        self.decoder_depth=config["decoder_depth"]
        self.dropout_p=config["dropout_p"]
        self.device = config["device"]
        self.encoder_depth = config["encoder_depth"]
        self.decoder_depth = config["decoder_depth"]
        self.longest_radius = config["longest_radius"]
        self.shortes_radius = config["shortest_radius"]

        # Encoder layers
        self.encoder_function = nn.Linear(in_features=self.extrinsic_dim, out_features=self.encoder_width, device=self.device)
        self.encoder_linear_layers = nn.ModuleList(
            [nn.Linear(in_features=self.encoder_width, out_features=self.encoder_width, device=self.device) for _ in range(self.encoder_depth)]
        )
        
        self.function_z_theta_mu = nn.Linear(in_features=self.encoder_width, out_features=self.latent_dim, device=self.device) # mean of the Von Mises Fischer
        self.function_z_theta_logvar = nn.Linear(in_features=self.encoder_width, out_features=1, device=self.device) # Kappa of Von Mises Fischer

        self.function_z_phi_mu = nn.Linear(in_features=self.encoder_width, out_features=self.latent_dim, device = self.device)        
        self.fucntion_z_phi_logvar = nn.Linear(in_features=self.encoder_width, out_features=1, device = self.device)

        # Decoder layers
        self.decoder_function = nn.Linear(in_features=self.latent_dim, out_features=self.decoder_width, device=self.device)
        self.decoder_linear_layers = nn.ModuleList(
            [nn.Linear(in_features=self.decoder_width, out_features=self.decoder_width, device=self.device) for _ in range(self.decoder_depth)]
        )

        # Posteriori
        self.function_x_mu = nn.Linear(in_features=self.decoder_width, out_features=self.extrinsic_dim, device=self.device)
        self.dropout = nn.Dropout(p = self.dropout_p)

    # Building Torus
    def _build_torus(self, z_theta, z_phi):
        """
        Build Torus from the phis and thetas
        
        Returns :
            Built torus from angles
        """

        cos_theta = z_theta[..., 0]
        sin_theta = z_theta[..., 1]

        cos_phi = z_phi[..., 0]
        sin_phi = z_phi[..., 1]

        longest_radius = self.longest_radius
        shortest_radius = self.shortes_radius 

        x = (longest_radius - shortest_radius * cos_theta) * cos_phi
        y = (shortest_radius - shortest_radius * cos_theta) * sin_phi
        z = shortest_radius * sin_theta

        return gs.stack([x, y, z], axis=-1)

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
            
        
        z_theta_mu = self.function_z_theta_mu(h)
        z_theta_logvar = F.softplus(self.function_z_theta_logvar(h)) + 1

        z_phi_mu = self.function_z_phi_mu(h)
        z_phi_logvar = F.softplus(self.fucntion_z_phi_logvar(h)) + 1

        return z_theta_mu, z_theta_logvar, z_phi_mu, z_phi_logvar
    
    def reparemetrize(self, posteriori_params):
        """
        Here we define the reparametrization trick as seen in the paper 
        THe idea is to compute a differentiable PHI, with which he preserve the probability
        """

        z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa  = posteriori_params # params of Von Mises Fischer

        q_theta_z = VonMisesFisher3D(z_theta_mu, z_theta_kappa) # Postetioti q_theta_anlge(.) = p_theta_anlge(. | z)
        q_phi_z = VonMisesFisher3D(z_phi_mu, z_phi_kappa) # Posteriori q_phi_anlge(.) = p_phi_angle(. | z)

        z_theta = q_theta_z.rsample() # Sample Theta from Sphere
        z_phi = q_phi_z.rsample() # Sample Phi from Sphere

        # Reparametrize (Build a torus sampled from Von Mises Fischer)
        return self._build_torus(z_theta, z_phi) # reparametrize sample
    
    def decode(self, z):
        """
        This function decodes z sampled from the latent space

        Returns:
            Reconstructed data

        """
        
        z = self.decoder_function(z)
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

        posteriori_params = self.encode(x) # (z_latent, z_kappa)
        z = self.reparemetrize(posteriori_params) # reparametrize
        x_mu = self.decode(z) # x_mu (output points)
        
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

        kappa = q_z.scale  # or q_z.kappa
        kl_per_sample = self.kl_vmf_spherical_uniform(kappa)
        KL = kl_per_sample.mean()
        
        reconstruction_loss = torch.mean((x - x_mu) * (x - x_mu))   
    
        return self.config["gamma"] * reconstruction_loss + self.config["beta"] * KL


    def kl_vmf_spherical_uniform(self, kappa) -> torch.Tensor:
        """
        KL( vMF(mu, kappa) || Uniform(S^2) )
        assuming dimension = 3 (sphere S^2).
        kappa: (...,) tensor of concentrations
        returns: (...,) tensor of KLs
        """

        sinh_k = torch.sinh(kappa) 
        coth_k = torch.cosh(kappa) / sinh_k

        kl = torch.log(kappa) - torch.log(sinh_k) + kappa * coth_k - 1.0
        return kl
    
    












