# helper plotting functions

from tracking.models_utils import get_model_properties

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path
from scipy.stats import norm


model_abbreviations = {
    "SimpleMarkovianGP": "SE",
    "IntegratedGP": "iSE",
    "TwiceIntegratedGP": "iiSE",
    "DynamicsInformedIntegratedGP": "iDSE",
    "DynamicsInformedTwiceIntegratedGP": "iiDSE"
}

model_colors = {
    "SimpleMarkovianGP": "salmon", 
    "IntegratedGP": "mediumorchid", 
    "TwiceIntegratedGP": "green", 
    "DynamicsInformedIntegratedGP": "gold", 
    "DynamicsInformedTwiceIntegratedGP": "skyblue"
}

def plot_base(gt_x, gt_y, meas_x, meas_y, figsize=(10, 5)):
    """Configure base plot for figures with ground truth and measurements."""
    plt.figure(figsize=figsize)
    plt.plot(gt_x, gt_y, label="Ground truth", linestyle="dashed", color="black")
    plt.scatter(meas_x, meas_y, label="Measurements", color="blue", s=5)


def plot_tracks(track, transition_model, measurement_model):
    """Plot tracks for one model."""
    x_vals, y_vals = zip(*[measurement_model.function(state) for state in track])
    x_vals = list(x_vals)
    y_vals = list(y_vals)
    transition_model_1d = transition_model.model_list[0]
    model_name = transition_model_1d.__class__.__name__
    model_abbrev = model_abbreviations[model_name]
    color = model_colors[model_name]
    markov_approx, _, _, _ = get_model_properties(transition_model)
    plt.plot(x_vals, y_vals, label=f"{model_abbrev}", color=color)


# This function was adapted from the example in the repository, licensed under the MIT license:
# https://github.com/afredgcam/iGPs.git
# Author: Fred Lydeard
def add_track_unc_stonesoup(track, transition_model, opacity=0.3, cred_level=0.95):
    "Plot uncertainty intervals from a track."
    markov_approx, dim, ndim_1d, num_aug_states = get_model_properties(transition_model)
    mu_x_index = ndim_1d - num_aug_states
    mu_y_index = - num_aug_states

    if markov_approx == 1:
        # add process mean
        X = np.array([[state.state_vector[0, 0] + state.state_vector[mu_x_index, 0],
                       state.state_vector[ndim_1d, 0] + state.state_vector[mu_y_index, 0]] for state in track])
        V = np.array([[state.covar[0, 0] + state.covar[mu_x_index, mu_x_index],
                       state.covar[ndim_1d, ndim_1d] + state.covar[mu_y_index, mu_y_index]] for state in track])
    else:
        X = np.array([[state.state_vector[0, 0], state.state_vector[ndim_1d, 0]] for state in track])
        V = np.array([[state.covar[0, 0], state.covar[ndim_1d, ndim_1d]] for state in track])
    
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
