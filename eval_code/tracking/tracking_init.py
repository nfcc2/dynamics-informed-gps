# helper functions to initialise tracking workflow

from tracking.models_utils import get_model_properties

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
import datetime
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from typing import List


def import_ground_truth_coordinates(file_path, dim=2):
    """Import specified number of ground truth position dimensions from a CSV file."""
    df = pd.read_csv(file_path)
    
    # Expected column names in order: x_position, y_position, z_position, ...
    base_dims = ['x', 'y', 'z']
    if dim > len(base_dims):
        raise ValueError(f"Can only handle up to {len(base_dims)} dimensions.")

    position_columns = [f"position_{dim}" for dim in base_dims[:dim]]

    for col in position_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: '{col}'")

    arrays = tuple(df[col].to_numpy() for col in position_columns)
    return arrays


def generate_synthetic_ground_truth(transition_model, prior, num_steps, time_interval):
    """
    Generate synthetic path using given GP transition model.
    """
    truth = GroundTruthPath(prior)
    start_time = prior.timestamp
    _, dim, ndim_1d, _ = get_model_properties(transition_model)
    gt = [[] for _ in range(dim)]
    
    for t in range(1, num_steps+1):
        for d in range(dim):
            gt[d].append(prior.state_vector[d * ndim_1d])
        covar = transition_model.covar(track=truth, time_interval=time_interval)
        noise = multivariate_normal.rvs(np.zeros(ndim_1d * dim), covar)
        noise = np.atleast_2d(noise).T
        next_state = transition_model.function(prior, noise=noise, track=truth, time_interval=time_interval)
        truth.append(GroundTruthState(next_state, timestamp=start_time + t * time_interval))
        prior = truth[-1]

    return gt


def simulate_gaussian_measurements(gt, sd):
    """Generate measurements with added Gaussian noise from ground truth coordinates."""
    meas = []
    for gt_1d in gt:
        n = len(gt_1d)
        noise = np.random.normal(0, sd, n)
        meas.append(gt_1d + noise)
    
    return meas


def generate_stonesoup_ground_truth(transition_model, gt: List[np.ndarray], time_interval) -> GroundTruthPath:
    """
    Convert raw ground truth coordinates gt = [gt_x, gt_y, ...] to Stone Soup GroundTruthState objects.

    Parameters
    ----------
    transition_model : object
        The transition model, composed of per-dimension GP models.
    gt : List[np.ndarray]
        List of ground truth coordinate arrays, one per spatial dimension.
    time_interval : datetime.timedelta
        Time interval between successive ground truth points.

    Returns
    -------
    GroundTruthPath
        A Stone Soup GroundTruthPath object populated with GroundTruthState instances.
    """
    gt = [g.copy() for g in gt]
    markov_approx, dim, ndim_1d, num_aug_states = get_model_properties(transition_model)
    truth = GroundTruthPath()
    start_time = datetime.datetime.now()
    priors = [g[0] for g in gt]

    if markov_approx == 1:
        for i in range(dim):
            gt[i] -= priors[i]

    for t in range(len(gt[0])):  # Assume all gt arrays are of equal length
        state_vector = np.zeros(ndim_1d * dim)

        for i in range(dim):
            state_vector[i * ndim_1d] = gt[i][t]

        if markov_approx == 1:
            # assume prior velocity = 0 for iiGPs
            for i in range(dim):
                state_vector[(i + 1) * ndim_1d - num_aug_states] = priors[i]

        truth.append(GroundTruthState(state_vector, timestamp=start_time + t * time_interval))

    return truth


def generate_stonesoup_measurements(measurement_model, meas: List[np.ndarray], ground_truth) -> List[Detection]:
    """
    Convert raw measurements to Stone Soup Detection objects.

    Parameters
    ----------
    measurement_model : MeasurementModel
        The measurement model to be associated with each Detection.
    meas : List[np.ndarray]
        List of measurement arrays, one per spatial dimension.
    ground_truth : GroundTruthPath
        Ground truth states to align timestamps with.

    Returns
    -------
    List[Detection]
        List of Stone Soup Detection objects.
    """
    measurements = []
    for t in range(len(ground_truth)):
        measurement_vector = np.array([meas[d][t] for d in range(len(meas))])
        measurement = Detection(measurement_vector, timestamp=ground_truth[t].timestamp, measurement_model=measurement_model)
        measurements.append(measurement)

    return measurements


def create_prior_state(transition_model, timestamp, prior: List[float], prior_var: List[float]) -> GaussianState:
    """
    Initialise Gaussian prior state vector for arbitrary-dimensional tracking.

    Parameters
    ----------
    transition_model : object
        The transition model with model_list containing per-dimension models.
    timestamp : datetime.datetime
        The timestamp for the prior state.
    prior : List[float]
        List of initial positions (one per dimension).
    prior_var : List[float]
        List of variances for initial positions (one per dimension).

    Returns
    -------
    GaussianState
        The initialized prior state as a GaussianState object.
    """
    ndim = transition_model.ndim_state
    markov_approx, dim, ndim_1d, num_aug_states = get_model_properties(transition_model)

    state_vector = np.zeros(ndim)
    prior_covar = np.zeros((ndim, ndim))

    for i in range(dim):
        prior_covar[i * ndim_1d, i * ndim_1d] = prior_var[i]

        if markov_approx == 1:
            # Set the augmented state (e.g., prior mean position or velocity)
            aug_index = (i + 1) * ndim_1d - num_aug_states
            state_vector[aug_index] = prior[i]
        else:
            state_vector[i * ndim_1d] = prior[i]

    return GaussianState(state_vector, prior_covar, timestamp=timestamp)
