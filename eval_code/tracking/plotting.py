# helper plotting functions (2D plots)

from tracking.models_utils import get_model_properties


import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numpy as np
from scipy.stats import norm


model_abbreviations = {
    "SlidingWindowGP": "SE",
    "IntegratedGP": "iSE",
    "TwiceIntegratedGP": "iiSE",
    "DynamicsInformedIntegratedGP": "iDSE",
    "DynamicsInformedTwiceIntegratedGP": "iiDSE",
    "ConstantVelocity": "CV"
}

model_colors = {
    "SlidingWindowGP": "salmon", 
    "IntegratedGP": "mediumorchid", 
    "TwiceIntegratedGP": "green", 
    "DynamicsInformedIntegratedGP": "gold", 
    "DynamicsInformedTwiceIntegratedGP": "skyblue",
    "ConstantVelocity": "red"
}



def plot_base(gt, meas, figsize=(10, 5)):
    """Configure base plot for figures with ground truth and measurements (2D or 3D)."""
    dim = len(gt)
    if dim not in [2, 3]:
        raise ValueError('Plotting supports only 2D or 3D ground truth/meas data.')

    fig = plt.figure(figsize=figsize)
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt[0], gt[1], gt[2], label="Ground truth", linestyle="dashed", color="black")
        ax.scatter(meas[0], meas[1], meas[2], label="Measurements", color="blue", s=5)
    else:
        ax = fig.add_subplot(111)
        ax.plot(gt[0], gt[1], label="Ground truth", linestyle="dashed", color="black")
        ax.scatter(meas[0], meas[1], label="Measurements", color="blue", s=5)

    return ax


def plot_tracks(track, transition_model, measurement_model, lag=0):
    """Plot 2D or 3D tracks for one model."""
    markov_approx, dim, ndim_1d, naug = get_model_properties(transition_model)

    if lag >= ndim_1d - naug:
        print("Invalid lag. Maximum lag allowed = window size - 1.")
        return

    means = get_coordinates(transition_model, track, lag)
    means = means.T  # dimensions (dim, N)

    transition_model_1d = transition_model.model_list[0]
    model_name = transition_model_1d.__class__.__name__
    model_abbrev = model_abbreviations[model_name]
    color = model_colors[model_name]

    ax = plt.gca()
    if dim == 3 and not hasattr(ax, 'zaxis'):
        ax = plt.gcf().add_subplot(111, projection='3d')

    if dim == 3:
        ax.plot(means[0], means[1], means[2], label=f"{model_abbrev}", color=color)
    else:
        ax.plot(means[0], means[1], label=f"{model_abbrev}", color=color)


# This function was adapted from the example in the repository, licensed under the MIT license:
# https://github.com/afredgcam/iGPs.git
# Author: Fred Lydeard
def add_track_unc_stonesoup(track, transition_model, lag=0, opacity=0.3, cred_level=0.95):
    "Plot uncertainty intervals from a track."
    markov_approx, dim, ndim_1d, naug = get_model_properties(transition_model)

    if dim != 2:
        print("Warning: Uncertainty plotting is only supported in 2D.")
        return
    
    if lag >= ndim_1d - naug:
        print("Invalid lag. Maximum lag allowed = window size - 1.")
        return

    mu_x_index = ndim_1d - naug
    mu_y_index = - naug
    X = get_coordinates(transition_model, track, lag)

    if markov_approx == 1:
        # add process mean
        
        V = np.array([[state.covar[lag, 0] + state.covar[mu_x_index, mu_x_index],
                       state.covar[ndim_1d + lag, ndim_1d] + state.covar[mu_y_index, mu_y_index]] for state in track])
    else:

        V = np.array([[state.covar[lag, 0], state.covar[ndim_1d + lag, ndim_1d]] for state in track])
    
    # Compute uncertainty radius
    tail_out = (1 - cred_level) / 2
    num_sd = -norm.ppf(tail_out)

    step = 2
    X = X[::step]
    V = V[::step]
    r = num_sd * np.sqrt(V.sum(axis=1))  # Combined uncertainty from x and y variances

    if len(track) == 1:
        plt.gca().add_patch(Circle(X[0], r[0], fc=color, alpha=opacity, label='', ls=''))
        return

    elif len(track) == 0:
        return

    # Compute perpendicular vectors to trajectory
    dx = np.concatenate([[X[1, 0] - X[0, 0]], X[2:, 0] - X[:-2, 0], [X[-1, 0] - X[-2, 0]]])
    dy = np.concatenate([[X[1, 1] - X[0, 1]], X[2:, 1] - X[:-2, 1], [X[-1, 1] - X[-2, 1]]])
    l = np.hypot(dx, dy) + 1e-100
    nx = dy / l
    ny = -dx / l
    
    # Get vertices of uncertainty band
    Xp = X + (r[:, None] * np.stack([nx, ny], axis=1))
    Xn = X - (r[:, None] * np.stack([nx, ny], axis=1))

    # Construct path
    vertices = np.vstack([Xp, Xn[::-1]])
    codes = np.concatenate([[Path.MOVETO], np.full(len(X) - 1, Path.LINETO), 
                            np.full(len(X), Path.LINETO)])

    path = Path(vertices, codes)
    model_name = transition_model.model_list[0].__class__.__name__
    color = model_colors[model_name]
    plt.gca().add_patch(PathPatch(path, fc=color, alpha=opacity, label='', ls=''))


def get_coordinates(transition_model, track, lag):
    markov_approx, dim, ndim_1d, naug = get_model_properties(transition_model)
    coords = []
    for i in range(len(track)):
        state = track[i]
        c = []
        for d in range(1, dim+1):
            if markov_approx == 1:
                c.append(state.state_vector[ndim_1d*d-naug] + state.state_vector[ndim_1d*d-naug - (ndim_1d-naug-lag)])
            else:
                if i < lag:
                    # we have fewer values in the statevector than the required lag. it is currently 0. use our least recent value i instead.
                    c.append(state.state_vector[ndim_1d*(d-1) + i])
                else:
                    c.append(state.state_vector[ndim_1d*d-naug - (ndim_1d-naug-lag)])
        coords.append(c)
    coords = np.array(coords)  # shape: (N, dim)
    return coords