from model.spherical_vae import *
from model.train import *
from dataloader.data import *
from dataloader.utils import *
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    scheduler = CosineAnnealingLR(optimizer=optimzer, T_max=config["n_epochs"])

    if config["train"]:
        train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimzer, scheduler=scheduler)
    
    elif config["infer"]:

        model = SphericalVAE(config).to(config["device"])
        state_dict = torch.load(config["save_path"], map_location=config["device"])
        model.load_state_dict(state_dict)
        model.eval()
    
        n = config.get("grid_n", 40)    
        theta = torch.linspace(0.01, torch.pi - 0.01, n)     
        phi   = torch.linspace(0, 2 * torch.pi, n)

        Theta, Phi = torch.meshgrid(theta, phi, indexing="ij")

        z1 = torch.sin(Theta) * torch.cos(Phi)
        z2 = torch.sin(Theta) * torch.sin(Phi)
        z3 = torch.cos(Theta)

        z_grid = torch.stack([z1, z2, z3], dim=-1)          # (n, n, 3)
        z_flat = z_grid.reshape(-1, 3).to(config["device"]) # (n*n, 3)

       
        with torch.no_grad():
            x_mu = model.decode(z_flat)     

        x_mu = x_mu.cpu().numpy()

        X = x_mu[:, 0].reshape(n, n)
        Y = x_mu[:, 1].reshape(n, n)
        Z = x_mu[:, 2].reshape(n, n)

        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_wireframe(X, Y, Z, color='k', linewidth=0.5)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        plt.show()



config = {

    "dataset" : "S2_dataset", # dataset type to be used
    "batch_size" : 64,  # Batch size of the dataloader
    "n_epochs" : 1000, # num_epochs
    "embedding_dim" : 3,       # embedding dims of the dataset and points
    "rotation_init_type" : "random", # random init of points
    "n_angles" : 1000, # angles of points
    "n_wiggles" : 3, # number of wiggles
    "distortion_type" : "wiggle", # distortion types
    "scheduler" : False,             # Step scheduler or not / default False
    "lr" : 1e-4,
    "weight_decay" : 1e-6,

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
    "beta" : 0.03, # Alpha refularizer of KL divergence 
    "alpha" : 10 # regularization of latent loss
}


if __name__ == "__main__":
    main(config)