import numpy as np
import torch

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
    thetas = np.asarray(thetas.cpu())

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

    if isinstance(curv_norms_learned, torch.Tensor):
        H = curv_norms_learned.cpu().numpy()
    else:
        H = np.asarray(curv_norms_learned)

    if isinstance(curv_norms_true, torch.Tensor):
        Ht = curv_norms_true.cpu().numpy()
    else:
        Ht = np.asarray(curv_norms_true)

    H = np.asarray(H)
    Ht = np.asarray(Ht)

    diff = integrate_S2(thetas, phis, np.linalg.norm(H - Ht, axis = -1)**2)
    norm = integrate_S2(thetas, phis, np.linalg.norm(H, axis = -1)**2 + np.linalg.norm(Ht, axis = -1)**2)

    return diff / norm