from dataloader.data import *
from dataloader.utils import *
from model.torus_vae import *
from model.train import *
from model.spherical_vae import *
from model.infer import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main(config):
    """
        Defniing the model and Traingn + Saving params in /saves/
        Returns:
            None - The models weights will be saved in the /saves file
    
    """
    
    # loading test and train set
    train_loader, test_loader, _ = load(config)
    # loading model

    if config["dataset"] == "S2_dataset" or config["dataset"] == "S1_dataset":
        model = SphericalVAE(config)

    elif config["dataset"] == "T2_dataset":
        model = ToroidalVAE(config)

    optimzer = Adam(params=model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimzer, "min")

    if config["train"]:
        train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimzer, scheduler=scheduler)
    
    if config["infer"]:
        infer(config)
        

config = {

    "dataset" : "S2_dataset", # dataset type to be used
    "longest_radius" : 1.5, # Longest radius of the torus
    "shortest_radius" : 0.5, # Shortest radius of the torus
    "batch_size" : 64,  # Batch size of the dataloader
    "n_epochs" : 1000, # num_epochs
    "embedding_dim" : 3,       # embedding dims of the dataset and points
    "rotation_init_type" : "", # random init of points
    "n_angles" : 1444, # angles of points
    "n_wiggles" : 3, # number of wiggles
    "distortion_type" : "bump", # distortion types
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
    "beta" : 0.01, # beta refularizer of KL divergence 
    "alpha" : 10.0, # regularization of latent loss
}


if __name__ == "__main__":
    main(config)