# helper functions to initialise tracking workflow

from tracking.models_utils import get_model_properties

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.linear import SlidingWindowGPSE

import datetime
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd


def import_ground_truth_coordinates(file_path):
    """Import the ground truth coordinates from a csv file."""
    df = pd.read_csv(file_path)
    gt_x, gt_y = np.array(df['x']), np.array(df['y'])
    return gt_x, gt_y


def generate_synthetic_ground_truth(transition_model, prior, num_steps, time_interval):
    """Generate synthetic path using given GP transition model."""
    truth = GroundTruthPath(prior)
    start_time = prior.timestamp
    ndim_1d = transition_model.model_list[0].ndim_state
    gt_x = []
    gt_y = []
    
    for t in range(1, num_steps+1):
        gt_x.append(prior.state_vector[0])
        gt_y.append(prior.state_vector[ndim_1d])
        covar = transition_model.covar(track=truth, time_interval=time_interval)
        noise = multivariate_normal.rvs(np.zeros(ndim_1d * 2), covar)
        noise = np.atleast_2d(noise).T
        next_state = transition_model.function(prior, noise=noise, track=truth, time_interval=time_interval)
        truth.append(GroundTruthState(next_state, timestamp=start_time + t * time_interval))
        prior = truth[-1]

    return (gt_x, gt_y)


def simulate_gaussian_measurements(gt_x, gt_y, sd):
    """Generate measurements with added Gaussian noise from ground truth coordinates."""
    n = len(gt_x)
    noise_x = np.random.normal(0, sd, n)
    noise_y = np.random.normal(0, sd, n)
    meas_x = gt_x + noise_x
    meas_y = gt_y + noise_y
    
    return meas_x, meas_y


def generate_stonesoup_ground_truth(transition_model, gt_x, gt_y, time_interval):
    """Convert raw ground truth coordinates gt_x, gt_y to Stone Soup GroundTruthState objects"""
    gt_x = gt_x.copy()
    gt_y = gt_y.copy()
    markov_approx, dim, ndim_1d, num_aug_states = get_model_properties(transition_model)

    truth = GroundTruthPath()
    start_time = datetime.datetime.now()
    prior_x, prior_y = gt_x[0], gt_y[0]
    if markov_approx == 1:
        gt_x -= prior_x
        gt_y -= prior_y
    
    for t in range(len(gt_x)):
        state_vector = np.zeros(ndim_1d * dim)
        state_vector[0], state_vector[ndim_1d] = gt_x[t], gt_y[t]
        if markov_approx == 1:
            # for iiGPs, assume prior velocity = 0
            state_vector[ndim_1d - num_aug_states], state_vector[-num_aug_states] = prior_x, prior_y
        truth.append(GroundTruthState(state_vector, timestamp=start_time + t * time_interval))
    
    return truth


def generate_stonesoup_measurements(measurement_model, meas_x, meas_y, ground_truth):
    """Convert raw measurements meas_x, meas_y to Stone Soup detection objects"""
    measurements = []
    for x, y, truth in zip(meas_x, meas_y, ground_truth):
        measurement = Detection([x, y], timestamp=truth.timestamp, measurement_model=measurement_model)
        measurements.append(measurement)

    return measurements


def create_prior_state(transition_model, timestamp, prior_x, prior_y, prior_var):
    """Initialise Gaussian prior state vector."""
    ndim = transition_model.ndim_state
    markov_approx, dim, ndim_1d, num_aug_states = get_model_properties(transition_model)

    state_vector = np.zeros(ndim)
    prior_covar = np.zeros((ndim, ndim))
    prior_covar[0, 0] = prior_var
    prior_covar[ndim_1d, ndim_1d] = prior_var

    if markov_approx == 1:
        state_vector[ndim_1d - num_aug_states], state_vector[-num_aug_states] = prior_x, prior_y
    else:
        state_vector[0], state_vector[ndim_1d] = prior_x, prior_y

    prior_state = GaussianState(state_vector, prior_covar, timestamp=timestamp)
    return prior_state