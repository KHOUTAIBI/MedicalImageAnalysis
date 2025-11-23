from model.spherical_vae import *
from dataloader.utils import * 
from model.torus_vae import *

def infer(config):
    """
        Inference step of the VAE
    """

    if config["dataset"] == "S2_dataset":

        # S1 Inference
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

        noisy_points, labels_noisy, original_points = load_S2_synthetic_data(
            rotation_init_type="",
            embedding_dim=model.config["embedding_dim"],
            distortion_type="wiggle"
        )

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
        ax.scatter(X, Y, Z, color='k', linewidth=0.5, label='sampled')
        ax.scatter(X_original, Y_original, Z_original, color='b', linewidth = 0.5, label='original') # type: ignore
        ax.scatter(X_latent, Y_latent, Z_latent, color='r', linewidth = 0.5, label='latent')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.legend()
        plt.show()

    elif config["dataset"] == "T2_dataset":

        # T2 inference

        model = ToroidalVAE(config)
        state_dict = torch.load(config["save_path"], map_location=config["device"])
        model.load_state_dict(state_dict)
        model.eval()

        n = config["n_grid"] 

        # angles on the torus
        theta = torch.linspace(0, 2 * torch.pi, n)     # "minor" angle
        phi   = torch.linspace(0, 2 * torch.pi, n)     # "major" angle

        Theta, Phi = torch.meshgrid(theta, phi, indexing="ij")

        # torus parameters
        R = config["longest_radius"]          
        r = config["shortest_radius"]            

        # torus embedding in R^3
        z1 = (R + r * torch.cos(Theta)) * torch.cos(Phi)
        z2 = (R + r * torch.cos(Theta)) * torch.sin(Phi)
        z3 =  r * torch.sin(Theta)

        noisy_points, _, original_points = load_T2_synthetic_data(
            rotation_init_type="",
            embedding_dim=model.config["embedding_dim"]
        )

        z_theta_mu, _, z_phi_mu, _ = model.encode(noisy_points.to(config["device"]))

        z_latent = model._build_torus(z_theta_mu, z_phi_mu)

        z_latent = z_latent.detach().cpu()
        print(f"latent is  are: {z_latent}")

        n_noisy = noisy_points.shape[0]
        n_noisy = int(np.sqrt(n_noisy))

        n_original = original_points.shape[0]
        n_original = int(np.sqrt(n_original))
        X_original = original_points[:, 0].reshape(n_original, n_original)
        Y_original = original_points[:, 1].reshape(n_original, n_original)
        Z_original = original_points[:, 2].reshape(n_original, n_original)

        z_grid = torch.stack([z1, z2, z3], dim=-1)          # (n, n, 3) points on torus
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
        ax.scatter(X, Y, Z, color='k', linewidth=0.5, label='sampled')
        ax.scatter(X_original, Y_original, Z_original, color='b', linewidth = 0.5, label='original') # type: ignore
        ax.scatter(X_latent, Y_latent, Z_latent, color='r', linewidth = 0.5, label='latent')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.legend()
        plt.show()
    
    else :
        raise ValueError("Please choose between S2 and T2 !")

