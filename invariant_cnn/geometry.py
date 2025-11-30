import torch
import torch.autograd as autograd
import numpy as np

def compute_curvature_profile(model, device, num_points=360):
    """
    Computes the Mean Curvature Vector H for the learned manifold.
    Follows Definition 4 and Section C.1 of the paper.
    """
    model.eval()
    
    # We sweep the latent space angle theta from 0 to 2pi
    thetas = torch.linspace(0, 2 * np.pi, num_points).to(device)
    
    mean_curvature_norms = []
    
    for theta in thetas:
        theta_vec = theta.unsqueeze(0)
        
        # Define Position Function: gamma(t)
        def decoder_wrt_theta(t):
            zx = torch.cos(t)
            zy = torch.sin(t)
            z_input = torch.stack([zx, zy], dim=-1) 
            return model.decode(z_input).squeeze()

        # Define Velocity Function: gamma'(t)
        def velocity_wrt_theta(t):
            return autograd.functional.jacobian(decoder_wrt_theta, t, create_graph=True).squeeze()
        
        # Velocity Vector (First Derivative)
        jac = velocity_wrt_theta(theta_vec)
        
        # Acceleration Vector (Second Derivative)
        # We calculate the Jacobian of the velocity vector w.r.t. time
        hess = autograd.functional.jacobian(velocity_wrt_theta, theta_vec).squeeze()
        
        # Compute Pullback Metric (g)
        # For a 1D curve, metric g is the squared norm of the tangent vector
        g = torch.dot(jac, jac)
        v = jac
        v_norm_sq = g
        a = hess

        if v_norm_sq.item() < 1e-6:
            mean_curvature_norms.append(0.0)
            continue

        # Project acceleration onto the normal space
        # a_tangential = (<a,v> / <v,v>) * v
        proj_scalar = torch.dot(a, v) / v_norm_sq
        a_tangential = proj_scalar * v
        
        # The normal component of acceleration (this is the extrinsic curvature direction)
        a_normal = a - a_tangential
        
        # Mean curvature vector H = a_normal / g
        H = a_normal / v_norm_sq
        
        # Store the norm (Curvature Profile)
        H_norm = torch.norm(H).item()
        mean_curvature_norms.append(H_norm)
        
    return thetas.detach().cpu().numpy(), np.array(mean_curvature_norms)