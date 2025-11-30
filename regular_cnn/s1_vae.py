import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class S1_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(S1_VAE, self).__init__()
        
        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Latent parameters
        # z_mean: Predicts the (x, y) vector direction
        self.z_mean = nn.Linear(hidden_dim, 2) 
        
        # z_kappa: Predicts concentration (inverse variance)
        self.z_kappa = nn.Linear(hidden_dim, 1)  
        
        # Decoder
        self.dec_fc1 = nn.Linear(2, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)
        
        # Softplus for smooth derivatives
        self.act = nn.Softplus() 

    def encode(self, x):
        h = self.act(self.enc_fc1(x))
        h = self.act(self.enc_fc2(h))
        
        # Mean Direction (mu)
        # Normalize to ensure it sits on the circle
        mean_vec = self.z_mean(h)
        mean_vec = F.normalize(mean_vec, p=2, dim=1)
        
        # Concentration (kappa)
        # Must be positive. Adding 1 ensures we don't divide by zero/get instabilities.
        kappa = F.softplus(self.z_kappa(h)) + 1.0
        
        return mean_vec, kappa

    def reparameterize(self, mean_vec, kappa):
        # We use "Projected Normal" sampling which approximates vMF and is fully differentiable.
        # Scale the mean vector by kappa (High kappa = long vector = less noise effect)
        # We broadcast kappa to match mean_vec dimensions
        scaled_mean = mean_vec * kappa
        
        # Add Gaussian noise
        noise = torch.randn_like(scaled_mean)
        
        # Normalize back to the unit circle
        # This results in a distribution concentrated around mean_vec
        z = F.normalize(scaled_mean + noise, p=2, dim=1)
        
        return z

    def decode(self, z):
        h = self.act(self.dec_fc1(z))
        h = self.act(self.dec_fc2(h))
        return self.dec_out(h)

    def forward(self, x):
        mean_vec, kappa = self.encode(x)
        z = self.reparameterize(mean_vec, kappa)
        x_recon = self.decode(z)
        return x_recon, mean_vec, kappa, z

def loss_function(recon_x, x, mean_vec, kappa):
    """
    Loss = Reconstruction + KL(VonMises || Uniform)
    """
    # Reconstruction Loss (MSE)
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence (vMF Formula)
    # Formula: KL = kappa * (I1(kappa)/I0(kappa)) - log(I0(kappa))
    
    k = kappa.squeeze()
    
    # Bessel functions of order 0 and 1
    i0 = torch.special.i0(k)
    i1 = torch.special.i1(k)
    
    # Compute KL
    # We assume a Uniform prior on S1.
    kl_per_point = k * (i1 / i0) - torch.log(i0)
    
    KLD = torch.sum(kl_per_point)

    return BCE + KLD