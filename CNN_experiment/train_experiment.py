import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our custom modules
from circular_VAE import CircularVAE
from dataset import Coil100FeatureDataset

# CONFIGURATION
config = {
    "object_id": 1,         # We tet with object of id 1
    
    "batch_size": 16,      
    "n_epochs": 1000,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    "extrinsic_dim": 512,   # (Matches ResNet18 output)
    "encoder_width": 128,
    "decoder_width": 128,
    
    # Loss Weights
    "gamma": 1.0,           # Reconstruction weight
    "beta": 0.01,           # KL div weght
    "alpha": 5.0,           # latent supervision weight
}

def train():

    dataset = Coil100FeatureDataset(root_dir="./data", object_id=config["object_id"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # init model
    model = CircularVAE(config).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    model.train()
    loss_history = []
    
    print("Training the vae...")

    for epoch in tqdm(range(config["n_epochs"]), desc="Epochs"):
        epoch_loss = 0
        for x, theta_true in dataloader:
            x = x.to(config["device"])
            theta_true = theta_true.to(config["device"])
            
            optimizer.zero_grad()
            z_sample, x_recon, params = model(x)
            
            # Base VAE Loss (Reconstruction + Regularization)
            elbo = model._elbo(x, x_recon, params)
            # Latent Supervision Loss (Alignment), z_sample is [cos_pred, sin_pred]. We convert to angle.
            pred_angle = torch.atan2(z_sample[:, 1], z_sample[:, 0])
            # Circular distance: 1 - cos(pred - true)
            latent_loss = 1 - torch.cos(pred_angle - theta_true)
        
            total_loss = elbo + config["alpha"] * latent_loss.mean()
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        loss_history.append(epoch_loss / len(dataloader))
        
    print("Training Complete.")
    visualize_results(model, dataset)

def visualize_results(model, dataset):
    model.eval()
    
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x_all, theta_all = next(iter(loader))
    x_all = x_all.to(config["device"])
    
    with torch.no_grad():
        z_sample, _, _ = model(x_all)
        
    z_vis = z_sample.cpu().numpy()
    theta_vis = theta_all.numpy()
    
    plt.figure(figsize=(15, 6))
    
    # Plot A: Topology (Latent Space)
    ax1 = plt.subplot(1, 2, 1)
    sc = ax1.scatter(z_vis[:, 0], z_vis[:, 1], c=theta_vis, cmap='hsv', s=50, edgecolor='k')
    ax1.set_title("Latent Space Representation\nTopological Check (Circle S1)")
    ax1.axis('equal')
    plt.colorbar(sc, label="True Angle")
    smooth_angles = np.linspace(0, 2 * np.pi, 360) 
    curvature_profile = compute_extrinsic_curvature(model, smooth_angles, device=config["device"])
    
    # Plot B: Geometry (Curvature Profile)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(np.degrees(smooth_angles), curvature_profile, color='orange', linewidth=2.5)
    ax2.set_title("B. Geometric Curvature Profile ($||H||$)\nExtrinsic Curvature vs Angle")
    ax2.set_xlabel("Latent Angle (Degrees)")
    ax2.set_ylabel("Curvature Norm")
    ax2.grid(True, alpha=0.3)
    
    peak_idx = np.argmax(curvature_profile)
    ax2.annotate('Max Curvature', 
                 xy=(np.degrees(smooth_angles[peak_idx]), curvature_profile[peak_idx]), 
                 xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
    
    plt.tight_layout()
    plt.show()


def compute_extrinsic_curvature(model, theta_range, device="cuda"):
    """
    Computes the Mean Curvature Vector Norm ||H|| for a range of angles.
    Corrected to handle vector-valued second derivatives.
    """
    model.eval()
    curvatures = []
    
    def decoder_wrapper(theta_tensor):
        z = torch.stack([torch.cos(theta_tensor), torch.sin(theta_tensor)], dim=-1)
        return model.decode(z).squeeze()

    def get_velocity(theta_tensor):
        return torch.autograd.functional.jacobian(decoder_wrapper, theta_tensor, create_graph=True).squeeze()

    print("Computing Riemannian Curvature profile...")
    
    for angle in tqdm(theta_range, desc="Geometry Analysis"):

        theta_t = torch.tensor([angle], dtype=torch.float32, device=device, requires_grad=True)
        tangent = get_velocity(theta_t)
        acceleration = torch.autograd.functional.jacobian(get_velocity, theta_t).squeeze()
        tangent_detached = tangent.detach()
        g = torch.dot(tangent_detached, tangent_detached)
        
        # Compute Mean Curvature Vector H
        # For 1D S1 manifold, formula simplifies to projection of acceleration orthogonal to tangent
        # H = (1/g) * (Acceleration - Tangential_Component)
        # However, paper definition 4 simplifies for 1D curves in Euclidean space essentially to:
        # H = (1/g) * Acceleration (assuming parametrization speed effects are handled by g)
        # Note: The paper's Definition 4 uses the "inverse of metric" (1/g) multiplied by the second derivative terms.
        # Since Christoffel symbols for 1D latent space (S1) are zero in this specific parameterization,
        # we can approximate it as scaling the acceleration by the inverse metric.
        
        H_vec = (1.0 / (g + 1e-8)) * acceleration
        H_norm = torch.norm(H_vec).item()
        curvatures.append(H_norm)
        
    return np.array(curvatures)

if __name__ == "__main__":
    train()