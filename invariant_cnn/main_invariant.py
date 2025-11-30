import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from datetime import datetime

from s1_vae import S1_VAE, loss_function
from geometry import compute_curvature_profile
from data_extraction_invariant import InvariantManifoldGenerator

# Configuration
IMAGE_PATH = "tiger.jpg"
EPOCHS = 3000
LR = 1e-3
DIM_PCA = 20
HIDDEN_DIM = 64
RESULTS_DIR = "results_invariant"

def setup_results_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(RESULTS_DIR, timestamp)
    subplots_path = os.path.join(dir_path, "subplots")
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(subplots_path, exist_ok=True)
    return dir_path, subplots_path

# Drawing Helpers
def draw_extrinsic_curvature(ax, latent_sweep_angles, curvature_profile, color='blue', label='Invariant CNN'):
    ax.plot(np.degrees(latent_sweep_angles), curvature_profile, color=color, linewidth=2, label=label)
    ax.set_title(f"Extrinsic Curvature Profile ({label})")
    ax.set_xlabel("Latent Angle (Degrees)")
    ax.set_ylabel("Curvature Norm ||H||")
    ax.grid(True, alpha=0.3)
    mean_curve = np.mean(curvature_profile)
    ax.axhline(y=mean_curve, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_curve:.2f}')
    ax.legend()

def draw_latent_param(ax, learned_coords, ground_truth_angles):
    sc = ax.scatter(learned_coords[:, 0], learned_coords[:, 1], c=ground_truth_angles, cmap='hsv', s=10)
    ax.set_title("Latent Parameterization\n(Color = Ground Truth Rotation)")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.axis('equal')
    return sc

def draw_3d_manifold(ax, manifold_3d, latent_sweep_angles):
    p = ax.scatter(manifold_3d[:, 0], manifold_3d[:, 1], manifold_3d[:, 2], 
                   c=np.degrees(latent_sweep_angles), cmap='hsv', s=20)
    ax.set_title("Reconstructed Manifold (PCA 3D)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    return p

def draw_raster(ax, sorted_activity):
    sns.heatmap(sorted_activity, cmap="viridis", cbar_kws={'label': 'Activation'}, ax=ax)
    ax.set_title("Neural Activations (Sorted)")
    ax.set_xlabel("Image Rotation Angle (Index)")
    ax.set_ylabel("Neuron / PCA Component")
    ax.set_xticks([])

def draw_tuning_curves(ax, activity_matrix, ground_truth_angles, sorted_indices):
    n1, n2, n3 = sorted_indices[0], sorted_indices[len(sorted_indices)//2], sorted_indices[-1]
    sort_idx = np.argsort(ground_truth_angles)
    sorted_angles = ground_truth_angles[sort_idx]
    
    ax.plot(sorted_angles, activity_matrix[n1][sort_idx], label=f'Unit {n1}', linewidth=2)
    ax.plot(sorted_angles, activity_matrix[n2][sort_idx], label=f'Unit {n2}', linewidth=2)
    ax.plot(sorted_angles, activity_matrix[n3][sort_idx], label=f'Unit {n3}', linewidth=2)
    
    ax.set_title("Single-Unit Orientation Tuning")
    ax.set_xlabel("Rotation Angle (Degrees)")
    ax.set_ylabel("Activation Strength")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.legend()
    ax.grid(True, alpha=0.3)

def draw_polar_curvature(ax, latent_sweep_angles, curvature_profile, color='blue'):
    base_radius = 2.0
    scaled_curvature = curvature_profile + base_radius
    ax.plot(latent_sweep_angles, scaled_curvature, color=color, linewidth=3)
    ax.fill(latent_sweep_angles, scaled_curvature, color=color, alpha=0.3)
    ax.set_title("Polar Curvature Profile", pad=20)
    ax.set_rticks([base_radius, base_radius + 1])
    ax.set_rlabel_position(45)

# Plotting Drivers
def save_individual_plot(draw_func, filename, subplots_dir, figsize=(10, 8), projection=None, **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection) if projection else fig.add_subplot(111)
    res = draw_func(ax, **kwargs)
    
    if res and not projection: plt.colorbar(res, ax=ax)
    elif res and projection == '3d': plt.colorbar(res, ax=ax, label="Latent Angle")
        
    plt.tight_layout()
    plt.savefig(os.path.join(subplots_dir, filename), dpi=300)
    plt.close(fig)

def plot_main_analysis(latent_sweep_angles, curvature_profile, learned_coords, ground_truth_angles, learned_angles, vae, save_dir, subplots_dir):
    # Pre-compute 3D Manifold
    with torch.no_grad():
        z_x = np.cos(latent_sweep_angles)
        z_y = np.sin(latent_sweep_angles)
        z_sweep = torch.FloatTensor(np.stack([z_x, z_y], axis=1))
        manifold_high = vae.decode(z_sweep).cpu().numpy()
        
        from sklearn.decomposition import PCA
        manifold_3d = PCA(n_components=3).fit_transform(manifold_high)

    # Save Individual Plots
    save_individual_plot(draw_extrinsic_curvature, "A_Extrinsic_Curvature.png", subplots_dir, 
                         latent_sweep_angles=latent_sweep_angles, curvature_profile=curvature_profile, color='blue', label='Invariant CNN')
    
    save_individual_plot(draw_latent_param, "B_Latent_Parameterization.png", subplots_dir, 
                         learned_coords=learned_coords, ground_truth_angles=ground_truth_angles)
    
    save_individual_plot(draw_3d_manifold, "D_3D_Manifold.png", subplots_dir, projection='3d',
                         manifold_3d=manifold_3d, latent_sweep_angles=latent_sweep_angles)

    # Dashboard
    fig = plt.figure(figsize=(20, 6))
    
    ax1 = fig.add_subplot(1, 3, 1)
    draw_extrinsic_curvature(ax1, latent_sweep_angles, curvature_profile, color='blue', label='Invariant CNN')
    
    ax2 = fig.add_subplot(1, 3, 2)
    sc = draw_latent_param(ax2, learned_coords, ground_truth_angles)
    plt.colorbar(sc, ax=ax2, label="Rotation")
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    p = draw_3d_manifold(ax3, manifold_3d, latent_sweep_angles)
    plt.colorbar(p, ax=ax3, label="Latent Angle")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "1_Invariant_Main_Analysis.png"), dpi=300)
    plt.show()

def plot_neuroscience_dashboard(data, ground_truth_angles, curvature_profile, latent_sweep_angles, save_dir, subplots_dir):
    activity_matrix = data.cpu().numpy().T
    peak_times = np.argmax(activity_matrix, axis=1)
    sorted_indices = np.argsort(peak_times)
    sorted_activity = activity_matrix[sorted_indices, :]
    
    # Save Individual Plots
    save_individual_plot(draw_raster, "Panel1_Neural_Raster.png", subplots_dir, 
                         sorted_activity=sorted_activity)
    
    save_individual_plot(draw_tuning_curves, "Panel2_Tuning_Curves.png", subplots_dir, 
                         activity_matrix=activity_matrix, ground_truth_angles=ground_truth_angles, sorted_indices=sorted_indices)
    
    save_individual_plot(draw_polar_curvature, "Panel3_Polar_Curvature.png", subplots_dir, projection='polar',
                         latent_sweep_angles=latent_sweep_angles, curvature_profile=curvature_profile, color='blue')

    # Dashboard
    fig = plt.figure(figsize=(20, 6))
    plt.suptitle("Invariant CNN 'Orientation Cell' Analysis", fontsize=16)

    ax1 = fig.add_subplot(1, 3, 1)
    draw_raster(ax1, sorted_activity)

    ax2 = fig.add_subplot(1, 3, 2)
    draw_tuning_curves(ax2, activity_matrix, ground_truth_angles, sorted_indices)

    ax3 = fig.add_subplot(1, 3, 3, projection='polar')
    draw_polar_curvature(ax3, latent_sweep_angles, curvature_profile, color='blue')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "2_Invariant_Dashboard.png"), dpi=300)
    plt.show()

def main():
    save_dir, subplots_dir = setup_results_dir()
    
    # Generate Data
    generator = InvariantManifoldGenerator(pca_components=DIM_PCA)
    data, ground_truth_angles = generator.generate_manifold_data(IMAGE_PATH)
    
    # Train VAE
    vae = S1_VAE(input_dim=DIM_PCA, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(vae.parameters(), lr=LR)
    train_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    
    print("Training Invariant VAE...")
    vae.train()
    for epoch in range(EPOCHS):
        for x_batch in train_loader:
            optimizer.zero_grad()
            recon_x, mean, kappa, z = vae(x_batch)
            loss = loss_function(recon_x, x_batch, mean, kappa)
            loss.backward()
            optimizer.step()
        if epoch % 500 == 0: print(f"Epoch {epoch} complete")

    # Geometry Analysis
    print("Computing Geometry...")
    latent_sweep_angles, curvature_profile = compute_curvature_profile(vae, generator.device)

    # Latent Extraction
    vae.eval()
    with torch.no_grad():
        _, mean, _, _ = vae(data) 
        learned_coords = mean.cpu().numpy()
        learned_angles = np.arctan2(learned_coords[:, 1], learned_coords[:, 0])

    # Plotting
    ground_truth_angles_np = ground_truth_angles.numpy()
    
    plot_main_analysis(latent_sweep_angles, curvature_profile, learned_coords, 
                        ground_truth_angles_np, learned_angles, vae, save_dir, subplots_dir)
    
    plot_neuroscience_dashboard(data, ground_truth_angles_np, curvature_profile, 
                                latent_sweep_angles, save_dir, subplots_dir)
    
    print(f"Invariant Experiment Complete! Saved to: {save_dir}")

if __name__ == "__main__":
    main()