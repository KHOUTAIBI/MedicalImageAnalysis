import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CircularVAE(nn.Module):
    def __init__(self, config):
        """
        A Robust VAE for Circular Manifolds (S1) using Projected Normal distribution.
        This serves as a differentiable, stable alternative to Von Mises-Fisher.
        """
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.extrinsic_dim = config["extrinsic_dim"]
        
        # ENCODER
        # Maps High-dim CNN features -> Parameters of a 2D Gaussian
        self.encoder_net = nn.Sequential(
            nn.Linear(self.extrinsic_dim, config["encoder_width"]),
            nn.ReLU(),
            nn.Linear(config["encoder_width"], config["encoder_width"]),
            nn.ReLU(),
            nn.Linear(config["encoder_width"], config["encoder_width"]),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(config["encoder_width"], 2) 
        self.fc_logvar = nn.Linear(config["encoder_width"], 1)

        # DECODER
        # Maps Latent Angle (cos, sin) -> Reconstructed CNN features
        self.decoder_net = nn.Sequential(
            nn.Linear(2, config["decoder_width"]),
            nn.ReLU(),
            nn.Linear(config["decoder_width"], config["decoder_width"]),
            nn.ReLU(),
            nn.Linear(config["decoder_width"], self.extrinsic_dim)
        )

    def encode(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        The Projected Normal Trick:
        1. Sample from a standard Gaussian: z_raw ~ N(mu, sigma)
        2. Project to Circle: z = z_raw / ||z_raw||
        
        This is fully differentiable.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu) # Standard Gaussian Noise
        
        # Reparameterization trick (Standard VAE step)
        z_raw = mu + eps * std
        
        # Project onto the circle (Normalization)
        # 1e-8 added for numerical stability
        z_sample = z_raw / (z_raw.norm(dim=-1, keepdim=True) + 1e-8)
        
        return z_sample, (mu, logvar)

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z_sample, params = self.reparameterize(mu, logvar)
        x_recon = self.decode(z_sample)
        
        # Return z_sample (on circle), x_recon, and distribution params
        return z_sample, x_recon, (mu, logvar)

    def _elbo(self, x, x_recon, posterior_params):
        """
        Calculates the VAE Loss (Reconstruction + Regularization)
        """
        mu, logvar = posterior_params
        
        # 1. Reconstruction Loss (MSE)
        recon_loss = torch.mean((x - x_recon)**2)
        
        # 2. Regularization (KL Proxy)
        # We implicitly regularize the pre-projected Gaussian to be close to N(0, I).
        # This prevents the latent space from exploding.
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return self.config["gamma"] * recon_loss + self.config["beta"] * kl_loss