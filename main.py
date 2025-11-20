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
    optimzer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer=optimzer, T_max=config["n_epochs"])


    train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimzer, scheduler=scheduler)


config = {

    "dataset" : "S2_dataset", # dataset type to be used
    "batch_size" : 256,  # Batch size of the dataloader
    "n_epochs" : 200, # num_epochs
    "embedding_dim" : 3,       # embedding dims of the dataset and points
    "rotation_init_type" : "random", # random init of points
    "n_angles" : 1000, # angles of points
    "n_wiggles" : 3, # number of wiggles
    "distortion_type" : "wiggle", # distortion types

    "extrinsic_dim": 3,        # e.g. dimension of extrinsic features (xyz)
    "latent_dim": 3,          # size of latent code
    "sftbeta": 1.0,            # smoothing / scaling factor if used
    "encoder_width": 128,      # width of MLP layers in encoder
    "encoder_depth": 4,        # number of layers in encoder
    "decoder_width": 128,      # width of MLP layers in decoder
    "decoder_depth": 4,        # number of layers in decoder
    "dropout_p": 0.1,          # dropout probability
    "device": "cuda",           # or "cpu"
    "beta" : 1e-2, # Alpha refularizer of KL divergence 
}


if __name__ == "__main__":
    main(config)