import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from errors import *

from circular_VAE import CircularVAE
from dataset import Coil100FeatureDataset

config = {
    "object_id": 1,
    "save_path" : "./saves/circular_resnet_final.pth",
    "batch_size": 16,
    "n_epochs": 1000,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "extrinsic_dim": 1000,
    "encoder_width": 128,
    "decoder_width": 128,
    "gamma": 1.0,
    "beta": 0.1,
    "alpha": 10.0,
    "train" : True,
}

def train():
    """
        Training : Reconstructing the inputs and imposing the latent VAE space to have a certain shape
    """
    # Defining the dataset, model and Circular VAE
    dataset = Coil100FeatureDataset(root_dir="./data", object_id=config["object_id"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    model = CircularVAE(config).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    model.train()
    loss_history = []

    print("Training the vae...")
    
    for epoch in tqdm(range(config["n_epochs"]), desc="Epochs"):
        # Epoch 
        epoch_loss = 0
        for x_true, x, theta_true in dataloader:
            
            # x and x_true
            x = x.to(config["device"])
            x_true = x_true.to(config["device"])
            theta_true = theta_true.to(config["device"])
            
            optimizer.zero_grad()
            
            # Feeding through the model
            z_sample, x_recon, params = model(x)
            
            elbo = model._elbo(x_true, x_recon, params)
            pred_angle = (torch.atan2(z_sample[..., 1], z_sample[..., 0]) + (2 * torch.pi)) % (2 * torch.pi)

            latent_loss = (1 - torch.cos(pred_angle - theta_true))**2
            total_loss = elbo + config["alpha"] * latent_loss.mean()
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(params[0])
            torch.save(model.state_dict(), config["save_path"])
        
        loss_history.append(epoch_loss / len(dataloader))
    
    print("Training Complete.")

def visualize_results(dataset):
    """
        Vizualise the results of the Reconstructions. We plot the z_sampled from the latent space and fed to the decoder. 
    """

    # Loading the Model and its weights
    model = CircularVAE(config).to(device=config["device"])
    model.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
    model.eval()
    
    # Loading data
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # getting the first element of the daatset
    x_all, x_noisy, theta_all = next(iter(loader))
    x_all = x_all.to(config["device"])
    x_noisy = x_noisy.to(config["device"])
    theta_all = theta_all.to(config["device"])

    print(theta_all.shape)

    # Feeding through the model and getting the reconstructed X
    with torch.no_grad():
        z_sample, x_recon, (mu, logvar) = model(x_noisy)
    
    # Predictions of the Logits of the model
    print(x_recon.shape)
    pred_classes = x_recon.argmax(dim=-1)
    original_pred_classes = x_all.argmax(dim=-1)
    print(pred_classes.size(), original_pred_classes.size())

    # How many correct reconstructed have we got ?
    print(f"the predicted classes after denoising are: {pred_classes} and the original predicted classes are: {original_pred_classes}")
    print(f"the accuracy of the prediction reconstruction is: {torch.sum(pred_classes == original_pred_classes) / original_pred_classes.size(0):.2}")
    
    # Plotting the Logits
    logits_recon = x_recon.cpu().detach()
    logits_true  = x_all.cpu().detach()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axes[0].imshow(logits_recon[: 20, 400:700], aspect='auto', cmap='viridis')
    axes[0].set_title("Reconstructed Logits")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(logits_true[: 20, 400:700], aspect='auto', cmap='viridis')
    axes[1].set_title("True Logits")
    plt.colorbar(im2, ax=axes[1])

    axes[0].set_xlabel("value of logit")
    axes[0].set_ylabel("index of the rotation")

    axes[1].set_xlabel("value of logit")
    axes[1].set_ylabel("index of the rotation")

    axes[0].set_yticks(np.arange(0, 20, dtype=int))
    axes[1].set_yticks(np.arange(0, 20, dtype=int))
    plt.tight_layout()
    plt.show()

    # Plotting the results 
    z_vis = z_sample.cpu().numpy()
    theta_vis = theta_all.cpu().numpy()
    x_original = x_all.cpu().numpy()
    x_reconstructed = x_recon.cpu().numpy()


    fig = plt.figure(figsize=(15, 6))

    ax1 = plt.subplot(1, 2, 1)
    sc = ax1.scatter(z_vis[:, 0], z_vis[:, 1], c=theta_vis, cmap='hsv', s=50, edgecolor='k')
    ax1.set_title(f"Learned Latent Space (Object {config['object_id']})")
    ax1.set_xlabel("Latent Dim 1")
    ax1.set_ylabel("Latent Dim 2")
    ax1.axis('equal')
    
    error_per_point = np.mean((x_original - x_reconstructed)**2, axis=1)

    angles_mu = (torch.atan2(mu[:, 1], mu[:, 0]) + (2 * np.pi)) % (2 * np.pi)
    curvature_learned = curvature_S1(angles_mu)
    curvature_true = curvature_S1(theta_all)
    curvature_error = curvature_error_S1(theta_all, curvature_learned, curvature_true)

    print(f"The curvature error is: {curvature_error}")

    # angles and error values plotted
    sort_idx = np.argsort(theta_vis)
    sorted_angles = np.mod(theta_vis[sort_idx], 2*np.pi)
    sorted_error = error_per_point[sort_idx]

    sorted_angles = np.concatenate([sorted_angles, [2*np.pi]])
    sorted_error  = np.concatenate([sorted_error,  [sorted_error[0]]])

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(sorted_angles, sorted_error, linewidth=2)

    ax2.set_xlim(0, 2*np.pi)
    ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    ax2.set_title("Reconstruction Error vs Rotation Angle")
    ax2.set_xlabel("Rotation (Radians)")
    ax2.set_ylabel("MSE (Proxy for Curvature)")
    ax2.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax1, label="True Angle (Radians)")
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    if config["train"]:
        train()
    dataset = Coil100FeatureDataset(root_dir="./data", object_id=config["object_id"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    visualize_results(dataset=dataset)
