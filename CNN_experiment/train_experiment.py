import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from errors import *

# Import our custom modules
from circular_VAE import CircularVAE
from dataset import Coil100FeatureDataset

# --- CONFIGURATION ---
config = {
    "object_id": 1,         # We tet with object of id 1
    "save_path" : "./saves/circular_resnet_final.pth",
    
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
            
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), config["save_path"])
        
        loss_history.append(epoch_loss / len(dataloader))
        
    print("Training Complete.")

def visualize_results(dataset):
    

    model = CircularVAE(config).to(device=config["device"])
    model.load_state_dict(torch.load(config["save_path"]))

    model.eval()
    
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x_all, theta_all = next(iter(loader))
    x_all = x_all.to(config["device"])

    print(theta_all.shape)
    
    with torch.no_grad():
        z_sample, x_recon, (mu, logvar) = model(x_all)
        
    
    # Fixed: move to CPU for plotting
    z_vis = z_sample.cpu().numpy()
    theta_vis = theta_all.numpy()
    x_original = x_all.cpu().numpy()
    x_reconstructed = x_recon.cpu().numpy()
    
    plt.figure(figsize=(15, 6))
    
    # Plot 1: The Learned Manifold (Should be a circle normally)
    ax1 = plt.subplot(1, 2, 1)
    sc = ax1.scatter(z_vis[:, 0], z_vis[:, 1], c=theta_vis, cmap='hsv', s=50, edgecolor='k')
    ax1.set_title(f"Learned Latent Space (Object {config['object_id']})")
    ax1.set_xlabel("Latent Dim 1")
    ax1.set_ylabel("Latent Dim 2")
    ax1.axis('equal') # Essential to see true circularity
    plt.colorbar(sc, label="True Angle (Radians)")
    
    # Plot 2: Reconstruction error profile (proxy for extrinsic curvature)
    # High reconstruction error implies high curvature (hard to fit)
    error_per_point = np.mean((x_original - x_reconstructed)**2, axis=1)


    angles_mu = ( torch.atan2(mu[:, 1], mu[:, 0]) + (2 * np.pi) ) % (2 * np.pi)
    print(angles_mu.shape, theta_all.shape)
    curvature_learned = curvature_S1(angles_mu)
    curvature_true = curvature_S1(theta_all)

    print(curvature_learned.shape, curvature_true.shape)

    curvature_error = curvature_error_S1(theta_all, curvature_learned, curvature_true)
    print(f"The curvature error is: {curvature_error}")

    sort_idx = np.argsort(theta_vis)
    sorted_angles = np.degrees(theta_vis[sort_idx])
    sorted_error = error_per_point[sort_idx]
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(sorted_angles, sorted_error, color='red', linewidth=2)
    ax2.set_title("Reconstruction Error vs Rotation Angle")
    ax2.set_xlabel("Rotation (Degrees)")
    ax2.set_ylabel("MSE (Proxy for Curvature)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()   
    dataset = Coil100FeatureDataset(root_dir="./data", object_id=config["object_id"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    visualize_results(dataset=dataset)