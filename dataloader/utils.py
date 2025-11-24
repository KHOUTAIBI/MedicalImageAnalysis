from dataloader.data import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load(config):
    """
    Function takes a configuration file.
    This function loads the dataset in data.py, which will be used in training.

    Returns:
        train_dataloader, test_dataloader
    """
    
    # Loading Point dataset / Just points with noise added
    if config["dataset"] == "points_dataset":

        dataset, labels = load_points(n_angles=config["n_angles"])  # dataset: np.array, labels: DataFrame
        dataset = dataset.astype(np.float32)

        dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)


    # S1 Circle Dataset With dstorted poles
    elif config["dataset"] == "S1_dataset":

        dataset, labels, _ = load_S1_synthetic_data(
            rotation_init_type=config["rotation_init_type"],
            n_angles = config["n_angles"],
            n_wiggles = config["n_wiggles"],
            embedding_dim=config["embedding_dim"],
            distortion_type=config["distortion_type"]
        )

        dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0) # type: ignore
    
    # S2 Sphere with Distorted Poles
    elif config["dataset"] == "S2_dataset":
        dataset, labels, _ = load_S2_synthetic_data(
            rotation_init_type=config["rotation_init_type"],
            n_angles = config["n_angles"],
            n_wiggles = config["n_wiggles"],
            embedding_dim=config["embedding_dim"],
            distortion_type=config["distortion_type"]
        )

        dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0) # type: ignore
    
    # T2 = S2 x S2 Torus
    elif config["dataset"] == "T2_dataset":
        dataset, labels, _ = load_T2_synthetic_data(
            rotation_init_type=config["rotation_init_type"],
            n_angles = config["n_angles"],
            embedding_dim=config["embedding_dim"],
        )

        dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0) # type: ignore

    else:
        raise ValueError(f"Unknown dataset type: {config['dataset']}")

    print(f"The shape of the dataset is: {dataset.shape}")
    print(f"The shape of the labels is: {labels.shape}")

    # labels is a DataFrame (angles / radiuses) -> convert to numpy
    labels_np = labels.to_numpy().astype(np.float32)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset,
        labels_np,
        test_size=0.2,
        random_state=0,
        shuffle=True,
    )

    # Convert to torch tensors
    X_train_t = torch.from_numpy(X_train) if  isinstance(X_train, np.ndarray) else X_train
    X_test_t  = torch.from_numpy(X_test) if  isinstance(X_test, np.ndarray) else X_test
    y_train_t = torch.from_numpy(y_train) if  isinstance(y_train, np.ndarray) else y_train
    y_test_t  = torch.from_numpy(y_test) if  isinstance(y_test, np.ndarray) else y_test

    # Build TensorDatasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t, y_test_t)

    # Use batch_size from config if available, else default to 256 (default 256)
    batch_size = config["batch_size"]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, (X_test_t, y_test_t)


# ----------------------------------- Angle Loss -----------------------------------------

def latent_loss(labels, z, config):
    """
    Compute supervised latent loss for S1 or S2 datasets.
    Uses exact geodesic distance on the sphere.
    """

    dataset = config["dataset"]

    # ----------------------------------------
    # S1: z lies on S1 ⊂ R2 or S2 ⊂ R3
    # Geodesic distance:
    #   
    # ----------------------------------------
    if dataset == "S1_dataset":
        # angle from latent code (use x,y coordinates)
        
        latent_phi = torch.atan2(z[..., 1], z[..., 0])
        latent_phi = (latent_phi + (2 * torch.pi)) % (2*torch.pi)

        gt_phi = labels[:, 0]
        radius = torch.norm(z, dim=-1)
        radius_loss = (radius - 1.0)**2

        diff = torch.cos(latent_phi - gt_phi)
        latent_loss = (1 - diff)**2 + 0.1 * radius_loss
        return latent_loss.mean()

    # ----------------------------------------
    # S2: use z ∈ S² and convert labels (θ, φ) to true vec
    # True geodesic distance:
    #   d = arccos( <z, z_gt> )
    # ----------------------------------------
    elif dataset == "S2_dataset" :
        # labels: [theta_gt, phi_gt]
        theta_gt = labels[:, 0]
        phi_gt   = labels[:, 1]

        # z (64, 64, 3)
        eps = 1e-8
        z_norm = z / (z.norm(dim=-1, keepdim=True) + eps)

        x = z[..., 0]
        y = z[..., 1]
        z_comp = z_norm[..., 2].clamp(-1.0 + eps, 1.0 - eps)

        # Recover predicted spherical angles (θ̂, φ̂)
        # θ̂ = arccos(z), φ̂ = atan2(y, x)
        theta_hat = torch.acos(z_comp) # because norm of Z is one
        phi_hat   = torch.atan2(y, x)
        phi_hat = (phi_hat + (2 * torch.pi)) % (2 * torch.pi)

        # Angle differences
        dtheta = theta_gt - theta_hat
        dphi   = phi_gt - phi_hat

        # LS2 = ( 1 - cos(Δθ) + sin θ_gt sin θ̂ (1 - cos(Δφ)) )^2
        term1 = 1.0 - torch.cos(dtheta)
        term2 = torch.sin(theta_gt) * torch.sin(theta_hat) * (1.0 - torch.cos(dphi))
        ls2   = (term1 + term2) ** 2

        return ls2.mean()

    elif config["dataset"] == "T2_dataset":
        
        R = config["longest_radius"]   # major radius
        r = config["shortest_radius"]  # minor radius

        x = z[..., 0]
        y = z[..., 1]
        zc = z[..., 2]

        rho = torch.sqrt(x**2 + y**2)

        latent_phis = torch.atan2(y, x)                    # phi
        latent_thetas = torch.atan2(zc, rho - R)           # theta

        # Wrap to [0, 2π)
        latent_phis   = (latent_phis   + (2 * torch.pi)) % (2*torch.pi)
        latent_thetas = (latent_thetas + (2 * torch.pi)) % (2*torch.pi)

        # Angle losses
        thetas_loss = torch.mean(1 - torch.cos(latent_thetas - labels[:, 0]))
        phis_loss   = torch.mean(1 - torch.cos(latent_phis   - labels[:, 1]))
        torus_constraint = ((rho - R)**2 + zc**2 - r**2)**2

        latent_loss = thetas_loss + phis_loss + + 0.1 * torus_constraint             
        return latent_loss.mean()

    else:
        raise ValueError("Unknown dataset type in latent_loss()")


def curvature_S1(angles: torch.Tensor) -> torch.Tensor:
    """
    Compute the curvature vectors on S¹ for given angles.
    angles: shape (N,) or (...,)
    returns: shape (2, N) or (2, ...) depending on input
    """
    curvatures = torch.stack(
        (-torch.cos(angles), -torch.sin(angles)),
        dim=0
    )
    return curvatures


def curvature_S2(angles: torch.Tensor) -> torch.Tensor:
    """
    Compute curvature vectors on S² for given angles.
    angles: shape (N, 2) or (..., 2)
    returns: shape (3, N) or (3, ...) depending on input
    """
    thetas = angles[..., 0]
    phis   = angles[..., 1]

    curvatures = torch.stack([
        -torch.sin(thetas) * torch.cos(phis),
        -torch.sin(thetas) * torch.sin(phis),
        -torch.cos(thetas)
    ], dim=0)

    return curvatures 


def curvature_error_S1(thetas, curvature_norms_learned, curvature_norms_true):
    """
    ERROR(H, H') = integral (H - H')^2 / integral (H^2 + H'^2)
    """

    H  = curvature_norms_learned
    Ht = curvature_norms_true

    H  = H.detach().cpu().numpy()
    Ht = Ht.detach().cpu().numpy()
    thetas = np.asarray(thetas)

    diff_norm_sq = np.linalg.norm(H - Ht, axis=-1)**2       
    norm_sq = np.linalg.norm(H, axis=-1)**2 + np.linalg.norm(Ht, axis=-1)**2           

    numerator   = np.trapezoid(diff_norm_sq, thetas)
    denominator = np.trapezoid(norm_sq, thetas)

    return numerator / denominator


def integrate_S2(thetas, phis, h):
    """
    Computes: double integral h(theta, phi) sin theta  dphi dtheta
    """

    thetas = torch.unique(thetas, sorted=True)
    phis   = torch.unique(phis,   sorted=True)

    Ntheta = len(thetas)
    Nphi = len(phis)

    h = h.reshape(Ntheta, Nphi)

    inner = torch.trapz(h, phis, dim=1) * torch.sin(thetas)

    return torch.trapz(inner, thetas)


def compute_curvature_error_S2(thetas, phis, curv_norms_learned, curv_norms_true):
    """
    ERROR = As seen int eh apper in case of S2
    """

    H = curv_norms_learned.cpu()
    Ht = curv_norms_true.cpu()

    H = np.asarray(H)
    Ht = np.asarray(Ht)

    diff = integrate_S2(thetas, phis, np.linalg.norm(H - Ht, axis = -1)**2)
    norm = integrate_S2(thetas, phis, np.linalg.norm(H, axis = -1)**2 + np.linalg.norm(Ht, axis = -1)**2)

    return diff / norm





# ----------------------------------- Testing --------------------------------------------
# config = {
#     "dataset" : "S2_dataset",
#     "batch_size" : 256,
#     "n_angles" : 1500,
#     "rotation_init_type" : "random",
#     "n_wiggles" : 3,
#     "embedding_dim" : 3,
#     "distortion_type" : "wiggle"

# }
# train_loader, test_loser, _ = load(config)
    
    

     