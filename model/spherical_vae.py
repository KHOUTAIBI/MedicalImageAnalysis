import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sampler.spherical_uniform import *
from sampler.von_mises_fisher import *


class SphericalVAE(nn.Module):
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

        # Encoder layers
        self.encoder_function = nn.Linear(in_features=self.extrinsic_dim, out_features=self.encoder_width, device=self.device)
        self.encoder_linear_layers = nn.ModuleList(
            [nn.Linear(in_features=self.encoder_width, out_features=self.encoder_width, device=self.device) for _ in range(self.encoder_depth)]
        )
        
        self.function_z_mu = nn.Linear(in_features=self.encoder_width, out_features=self.latent_dim, device=self.device) # mean of the Von Mises Fischer
        self.function_z_logvar = nn.Linear(in_features=self.encoder_width, out_features=1, device=self.device) # Kappa of Von Mises Fischer

        # Decoder layers
        self.decoder_function = nn.Linear(in_features=self.latent_dim, out_features=self.decoder_width, device=self.device)
        self.decoder_linear_layers = nn.ModuleList(
            [nn.Linear(in_features=self.decoder_width, out_features=self.decoder_width, device=self.device) for _ in range(self.decoder_depth)]
        )

        # Posteriori
        self.function_x_mu = nn.Linear(in_features=self.decoder_width, out_features=self.extrinsic_dim, device=self.device)
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
 
        return q_z.rsample() # reparametrize sample
    
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

        eps = 1e-8
        k = torch.clamp(kappa, min=eps)
    
        # Regions of equivalences
        small = k < 1e-3
        large = k > 20.0
        mid   = ~(small | large)
    
        kl = torch.zeros_like(k)
        
        kl[small] = 0.0
        
        if mid.any():
            k_mid = k[mid]
            sinh_k = torch.sinh(k_mid) + eps
            cosh_k = torch.cosh(k_mid)
            coth_k = cosh_k / sinh_k
    
            log_sinh_k = torch.log(sinh_k)
            kl_mid = torch.log(k_mid) - log_sinh_k + k_mid * coth_k - 1.0
            kl[mid] = kl_mid
    
        
        if large.any():
            k_large = k[large]
    
            # log sinh(k) ~ k - log 2
            log_sinh_k_large = k_large - math.log(2.0)
    
            # coth(k) ~ 1 + 2 e^{-2k}
            coth_k_large = 1.0 + 2.0 * torch.exp(-2.0 * k_large)
    
            kl_large = torch.log(k_large) - log_sinh_k_large + k_large * coth_k_large - 1.0
            kl[large] = kl_large
    
        return kl
        
        












