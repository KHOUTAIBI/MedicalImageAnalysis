from model.spherical_vae import *
from model.train import *
from dataloader.data import *
from dataloader.utils import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse as arg

def main(config):
    """
        Defniing the model and Traingn + Saving params in /saves/
        Returns:
            None - The models weights will be saved in the /saves file
    
    """
    
    # loading test and train set
    train_loader, test_loader, _ = load(config)
    # loading model
    model = SphericalVAE(config)
    optimzer = Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimzer, "min")

    if config["train"]:
        train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimzer, scheduler=scheduler)
    
    if config["infer"]:

        model = SphericalVAE(config)
        state_dict = torch.load(config["save_path"], map_location=config["device"])
        model.load_state_dict(state_dict)
        model.eval()
    
        n = config["n_grid"] 
        theta = torch.linspace(0, torch.pi , n)     
        phi   = torch.linspace(0, 2 * torch.pi, n)

        Theta, Phi = torch.meshgrid(theta, phi, indexing="ij")

        z1 = torch.sin(Theta) * torch.cos(Phi)
        z2 = torch.sin(Theta) * torch.sin(Phi)
        z3 = torch.cos(Theta)

        noisy_points, labels_noisy, original_points = load_S2_synthetic_data(rotation_init_type="", embedding_dim=model.config["embedding_dim"], distortion_type="wiggle")

        z_latent, _ = model.encode(noisy_points.to(config["device"]))
        z_latent /= z_latent.norm(dim = -1, keepdim=True)

        z_latent = z_latent.detach().cpu()
        print(f"latent is  are: {z_latent}")

        n_noisy = noisy_points.shape[0]
        n_noisy = int(np.sqrt(n_noisy))

        n_original = original_points.shape[0]
        n_original = int(np.sqrt(n_original))
        X_original = original_points[:, 0].reshape(n_original, n_original)
        Y_original = original_points[:, 1].reshape(n_original, n_original)
        Z_original = original_points[:, 2].reshape(n_original, n_original)

        z_grid = torch.stack([z1, z2, z3], dim=-1)          # (n, n, 3)
        z_flat = z_grid.reshape(-1, 3).to(config["device"]) # (n*n, 3)

        X_latent = z_latent[:, 0].reshape(n_noisy, n_noisy)
        Y_latent = z_latent[:, 1].reshape(n_noisy, n_noisy)
        Z_latent = z_latent[:, 2].reshape(n_noisy, n_noisy)


        with torch.no_grad():
            x_mu = model.decode(z_flat)     

        print(torch.nn.functional.mse_loss(x_mu.cpu(), original_points))
        x_mu = x_mu.cpu().numpy()

        X = x_mu[:, 0].reshape(n, n)
        Y = x_mu[:, 1].reshape(n, n)
        Z = x_mu[:, 2].reshape(n, n)

        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.plot_wireframe(X, Y, Z, color='k', linewidth=0.5, label='sampled')
        # ax.plot_wireframe(X_original, Y_original, Z_original, color='b', linewidth = 0.5, label='original')
        ax.plot_wireframe(X_latent, Y_latent, Z_latent, color='r', linewidth = 0.5, label='latent')
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.legend()
        plt.show()
    
    else :
        ValueError("Please train or infer.")



config = {

    "dataset" : "S2_dataset", # dataset type to be used
    "batch_size" : 64,  # Batch size of the dataloader
    "n_epochs" : 600, # num_epochs
    "embedding_dim" : 3,       # embedding dims of the dataset and points
    "rotation_init_type" : "", # random init of points
    "n_angles" : 1000, # angles of points
    "n_wiggles" : 3, # number of wiggles
    "distortion_type" : "wiggle", # distortion types
    "scheduler" : False,             # Step scheduler or not / default False
    "lr" : 1e-3,                    # LR 
    "weight_decay" : 1e-6,          # Weight decay L2 regularization
    "n_grid" : 38,

    "train" : False,
    "infer" : True,
    "save_path" : "./saves/spherical_VAE_chkpt_final.pth",
 

    "extrinsic_dim": 3,        # e.g. dimension of extrinsic features (xyz)
    "latent_dim": 3,          # size of latent code
    "sftbeta": 1.0,            # smoothing / scaling factor if used
    "encoder_width": 128,      # width of MLP layers in encoder
    "encoder_depth": 4,        # number of layers in encoder
    "decoder_width": 128,      # width of MLP layers in decoder
    "decoder_depth": 4,        # number of layers in decoder
    "dropout_p": 0.1,          # dropout probability
    "device": "cuda",           # or "cpu"
    "gamma" : 1.0, # reconstruction loss
    "beta" : 0.03, # beta refularizer of KL divergence 
    "alpha" : 10.0, # regularization of latent loss
}


if __name__ == "__main__":
    main(config)