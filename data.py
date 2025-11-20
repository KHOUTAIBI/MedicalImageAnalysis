import skimage
import torch
import os
import numpy as np
import pandas as pd
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import matplotlib.pyplot as plt
import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from torch.distributions.multivariate_normal import MultivariateNormal

 

# ----------------------------------------Loading Data Points-----------------------------------------------------

def load_points(n_scalars = 1, n_angles = 1000):
    """
    This function loads a Point cloud of 1000 different angles

    Returns:
        points : array [n_scalar * angles, 3]
        labels : The labels of the points
    """

    points = []
    angles = []
    radiuses = []
    point = np.array([1, 0, 1])

    for i_angle in range(n_angles):

        # Angle of point around z
        angle = 2 * np.pi * i_angle / n_angles
        
        rot_mat_z = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1.0],
            ]
        )

        rot_point = rot_mat_z @ point
        
        # Radius of the point
        for i_scalar in range(n_scalars):

            scalar = 1 + i_scalar
            points.append(scalar * rot_point)

            angles.append(angle)
            radiuses.append(scalar)


    points = np.array(points)
    angles = np.array(angles)
    radiuses = np.array(radiuses)

    labels = pd.DataFrame({
        "angles" : angles,
        "radiuses" : radiuses
    })

    # Points and their angles + radiuses / The point will start at [1, 0, 1]
    return points, labels

# ------------------------------------------------- Bumping ---------------------------------------

def _bump(position, width, length_bump = 1000):
    """
    According to the paper, the made a function that adds A NOISY bump to the data point
    
    Parameters
    ----------
        - position (int): The index of the center of the bump.
        - width (int): The width of the bump.

    Returns
    -------
        - bump_array (numpy.ndarray): The array with the Gaussian bump.

    """

    bump_array = np.zeros(shape=(length_bump))
    
    # Defnie range of the bumps
    left = position - width // 2
    right = left + width


    # Gaussian bump, ie Gaussian 
    x = np.linspace(-1, 1, width)
    bump = np.exp(-(x**2) * 2)  # Gaussian bump
    bump /= np.max(bump)  # Normalize to maximum amplitude of 1
    
    bump_array[left : right] += bump
    
    # Zeros  
    return bump_array
    
# ---------------------------------------------- S1 (Circle/Ring) Immersion -----------------------------------------

def get_S1_immersion(distortion_type : str, 
                     radius, 
                     n_wiggles, 
                     distortion_amplitude, 
                     embedding_dim, 
                     rotation):
    
    """
    This function returns the Synthetic function 

    Returns :
        Retun the function that synthesises Bumped of Wiggled point clouds
    """
    
    # Polar coords
    def polar(angle):
        """
        Polar coords of a circle of RADIUS = 1

        Returns :
            polar_points : Polar representation of a vector of norm 1 
        """

        return gs.array([np.cos(angle), np.sin(angle)])


    

    # Immersion https://fr.wikipedia.org/wiki/Immersion_(math%C3%A9matiques) As seen here
    def synthesise_immersion(angle):

        if distortion_type == "wiggle":
            amplitude = radius * (1 + distortion_amplitude * gs.cos(n_wiggles * angle)) # cos wiggle, ir between [- distortion_angle + 1, distortion_angle + 1]

        elif distortion_type == "bump":
            amplitude = radius * (
                    1
                    + distortion_amplitude * gs.exp(-2 * (angle - gs.pi / 2) ** 2) # Gaussian Bump as seen in the paper
                    + distortion_amplitude * gs.exp(-2 * (angle - 3 * gs.pi / 2) ** 2)
                )

        else :
            raise NotImplementedError("Please choose between wiggle and bump")

        point = amplitude * polar(angle)
        # Squeeze for batch
        point = gs.squeeze(point, axis = -1)

        # Adding dimension 
        if embedding_dim > 2:

            point = gs.concatenate([point, gs.zeros(embedding_dim - 2)])

        return gs.einsum("ij,j->i", rotation, point)
 
    
    return synthesise_immersion

# ------------------------------------- Loading S1 CIRCLE Points with noise added --------------------------
def load_S1_synthetic_data(rotation_init_type : str,
                            n_angles = 1500,
                            radius = 1,
                            n_wiggles = 6,
                            distortion_amplitude = 0.6,
                            embedding_dim = 10,
                            noise_var = 0.1,
                            distortion_type = "wiggle" 
                           ):
    
    """Create "wiggly" circles with noise.

    Returns
    -------
    noisy_data : array-like, shape=[n_times, embedding_dim]
        Number of firings per time step and per cell.
    labels : pd.DataFrame, shape=[n_times, 1]
        Labels organized in 1 column: angles.
    """

    rot = torch.eye(n=embedding_dim)
    if rotation_init_type == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_S1_immersion(
        distortion_type=distortion_type,
        radius=radius,
        n_wiggles=n_wiggles,
        distortion_amplitude=distortion_amplitude,
        embedding_dim=embedding_dim,
        rotation=rot,
    )

    angles = gs.linspace(0, 2*gs.pi, n_angles)

    labels = pd.DataFrame({
        "labels" : angles
    })

    data = torch.zeros(size=(n_angles, embedding_dim))
    for idx, angle in enumerate(angles):
        data[idx] = immersion(angle)

    noise_amplitude = MultivariateNormal(loc = torch.zeros(embedding_dim), covariance_matrix=noise_var * torch.eye(n = embedding_dim)) 

    noisy_data = data + radius * noise_amplitude.sample(sample_shape=(n_angles, ))

    return noisy_data, labels 
# ---------------------------------------------- Testing the code !----------------------------------

def get_S2_synthetic_immersion(radius, distortion_amplitude, embedding_dim, rotation):

    """
    This function returns the S2 Synthetic function 

    Returns :
        Retun the function that synthesises Bumped of Wiggled point clouds
        IN THIS CASE THIS IS A SPHERE
    """

    def spherical(theta, phi):
        """
        The load the Spherical Coordinate of an Unit sphere

        Returns:
            The X,Y,Z coordinates
        """

        x = gs.sin(theta) * gs.cos(phi)
        y = gs.sin(theta) * gs.sin(phi)
        z = gs.cos(theta)
        return gs.array([x, y, z])

    def S2_synthetic_immersion(angle_pair):
        """
        Return the Immersion functions

        Returns:
            Immersion function
        """
        # Theta and phi
        theta = angle_pair[0]
        phi = angle_pair[1]

        # Aplitude modified / To have a distorted SPHERE
        amplitude = radius * (
            1
            + distortion_amplitude * gs.exp(-5 * theta**2)
            + distortion_amplitude * gs.exp(-5 * (theta - gs.pi) ** 2)
        )

        # Points 
        point = amplitude * spherical(theta, phi)
        point = gs.squeeze(point, axis=-1)

        # Embedding dim
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])

        return gs.einsum("ij,j->i", rotation, point)

    return S2_synthetic_immersion

def load_S2_synthetic_data(rotation_init_type : str,
                            n_angles = 1500,
                            radius = 1,
                            n_wiggles = 6,
                            distortion_amplitude = 0.6,
                            embedding_dim = 10,
                            noise_var = 0.1,
                            distortion_type = "wiggle"):
    
    """Create "wiggly" OR ANOTHER TPYE of Sphere with noise.

    Returns
    -------
    noisy_data : array-like, shape=[n_times, embedding_dim]
        Number of firings per time step and per cell.
    labels : pd.DataFrame, shape=[n_times, 1]
        Labels organized in 1 column: angles.
    """
    
    rotation = torch.eye(n=embedding_dim)
    if rotation_init_type == "random":
        rotation = SpecialOrthogonal(n=embedding_dim).random_point()
    
    immersion = get_S2_synthetic_immersion(radius=radius, 
                                           distortion_amplitude=distortion_amplitude,
                                           rotation=rotation,
                                           embedding_dim=embedding_dim,
                                           )

    # ! KEEP THIS AT SQRT OR IT WILL TAKE A LOOOOONG TIME TO MAKE
    sqrt_size = int(np.sqrt(n_angles))
    # Theta sphere
    thetas = gs.linspace(0.01, gs.pi, sqrt_size)
    # phi on sphere
    phis = gs.linspace(0, 2 * gs.pi, sqrt_size)

    # points
    points = torch.cartesian_prod(thetas, phis) # this is an interesting function that gpt suggested it retuns {x1,y1}, {x1,y2}... {x2, y1}...
    
    labels = pd.DataFrame({"thetas": points[:, 0], "phis": points[:, 1]})
    
    data = torch.zeros(sqrt_size * sqrt_size, embedding_dim)

    # Immersion data points
    for idx, point in enumerate(points):
        point = gs.array(point)
        data[idx] = immersion(point)

    # Adding noise and randomness to amplitudes
    noise_dist = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix = radius * noise_var * torch.eye(embedding_dim),
    )

    noisy_data = data + noise_dist.sample((sqrt_size * sqrt_size,))

    return noisy_data, labels, data


noisy_points, labels_noisy, points = load_S2_synthetic_data(rotation_init_type="random", embedding_dim=3, distortion_type="wiggle")
print(labels_noisy.head())

# # bump = _bump(position=500, width=100, length_bump=points.shape[0])
N = points.shape[0]
n = int(np.sqrt(N))
X = points[:, 0].reshape(n, n)
Y = points[:, 1].reshape(n, n)
Z = points[:, 2].reshape(n, n)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, color='k', linewidth=0.5)
# ax.scatter(noisy_points[:,0], noisy_points[:,1], noisy_points[:,2], s=3)
plt.show()