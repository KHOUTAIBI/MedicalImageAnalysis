import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CircularVAE(nn.Module):
    def __init__(self, config):
        """
        A Robust VAE for Circular Manifolds (S1) using Projected Normal distribution.
        This serves as a differentiable alternative to Von Mises-Fisher for S1.
        """
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.extrinsic_dim = config["extrinsic_dim"]  # e.g., 512 for ResNet features
        
        # --- ENCODER ---
        # Maps high-dim CNN features -> Parameters of the distribution in 2D
        self.encoder_net = nn.Sequential(
            nn.Linear(self.extrinsic_dim, config["encoder_width"]),
            nn.ReLU(),
            nn.Linear(config["encoder_width"], config["encoder_width"]),
            nn.ReLU(),
            nn.Linear(config["encoder_width"], config["encoder_width"]),
            nn.ReLU()
        )
        
        # We predict a Mean Vector in R^2
        self.fc_mu = nn.Linear(config["encoder_width"], 2) 
        
        # We predict a Log-Variance (controls concentration/uncertainty)
        # One scalar is sufficient for isotropic concentration on the circle
        self.fc_logvar = nn.Linear(config["encoder_width"], 1)

        # --- DECODER ---
        # Maps a point on the circle (cos, sin) -> Reconstructed CNN features
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
        The "Projected Normal" Trick (Differentiable Sampling):
        1. We treat 'mu' and 'logvar' as parameters of a Gaussian in R^2.
        2. We sample a point from this Gaussian.
        3. We project that point onto the unit circle.
        
        This is topologically valid for S1 and numerically stable.
        """
        std = torch.exp(0.5 * logvar)
        
        # 1. Sample epsilon noise
        eps = torch.randn_like(mu)
        
        # 2. Apply reparameterization (z_raw is in R^2 plane)
        z_raw = mu + eps * std
        
        # 3. Project to Circle (Normalization)
        # This creates the latent variable 'z' that lies on S1
        z_circle = z_raw / (z_raw.norm(dim=-1, keepdim=True) + 1e-8)
        
        return z_circle, (mu, logvar)

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z_sample, _ = self.reparameterize(mu, logvar)
        x_recon = self.decode(z_sample)
        
        # Return params for loss calculation
        return z_sample, x_recon, (mu, logvar)

    def _elbo(self, x, x_recon, posterior_params):
        """
        Calculates the Loss Function.
        """
        mu, logvar = posterior_params
        
        # 1. Reconstruction Loss (How well do we preserve the object features?)
        recon_loss = torch.mean((x - x_recon)**2)
        
        # 2. Regularization (KL Proxy)
        # For Projected Normal, exact KL is complex. 
        # We use a standard Gaussian KL on the pre-projected parameters.
        # This effectively regularizes the latent space to be smooth.
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return self.config["gamma"] * recon_loss + self.config["beta"] * kl_loss