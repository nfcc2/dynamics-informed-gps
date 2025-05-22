# Tracking workflow and kalman filtering

from tracking.tracking_init import generate_stonesoup_ground_truth, generate_stonesoup_measurements, create_prior_state
from tracking.models_utils import get_model_properties

from stonesoup.types.track import Track
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.predictor.gaussianprocess import GPPredictorWrapper
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.simple import SingleHypothesis

import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List

def perform_tracking(gt, meas, transition_model, measurement_model, time_interval, prior_var):
    """Carry out tracking with Kalman filtering given raw ground truth and measurement data."""

    _, dim, _, _ = get_model_properties(transition_model)

    # Step 1: Generate stone soup ground truth path
    ground_truth = generate_stonesoup_ground_truth(transition_model, gt, time_interval)
    
    # Step 2: Convert measurements to Stone Soup detections
    measurements = generate_stonesoup_measurements(measurement_model, meas, ground_truth)
    
    # Step 3: Track initialisation
    prior_state = create_prior_state(transition_model, ground_truth[0].timestamp, [gt[d][0] for d in range(dim)], [prior_var for _ in range(dim)])
    track = Track(prior_state)

    # Step 4: Define Kalman predictor and updater
    predictor = GPPredictorWrapper(KalmanPredictor(transition_model))
    updater = KalmanUpdater(measurement_model)

    # Step 5: Run tracking and calculate log likelihood
    track, log_lik = kalman_filter(predictor, updater, measurement_model, track, measurements)

    return track, log_lik

def kalman_filter(predictor, updater, measurement_model, track, measurements):
    """Carry out kalman filtering and compute marginal LL recursively."""
    log_lik = 0
    H = measurement_model.matrix()
    Cv = measurement_model.covar()
   
    for measurement in measurements[1:]:
        predicted_state = predictor.predict(track, timestamp=measurement.timestamp)
        updated_state = updater.update(SingleHypothesis(predicted_state, measurement))

        # Calculate conditional log likelihood
        m_pred = np.array(predicted_state.state_vector)
        P_pred = np.array(predicted_state.covar)
        meas = np.array(measurement.state_vector)
        S = H @ P_pred @ H.T + Cv
       
        log_lik += -0.5 * (np.linalg.slogdet(S)[1] +
                                      (meas - H @ m_pred).T @ np.linalg.pinv(S) @ (meas - H @ m_pred)).item()

        track.append(updated_state)
    return track, log_lik


def kalman_filter_dynamic_hyp(predictor, updater, measurement_model, track, measurements):
    """Carry out kalman filtering and compute marginal LL recursively."""
    log_lik = 0
    H = measurement_model.matrix()
    Cv = measurement_model.covar()
   
    for measurement in measurements[1:]:
        predicted_state = predictor.predict(track, timestamp=measurement.timestamp)
        updated_state = updater.update(SingleHypothesis(predicted_state, measurement))

        # Calculate conditional log likelihood
        m_pred = np.array(predicted_state.state_vector)
        P_pred = np.array(predicted_state.covar)
        meas = np.array(measurement.state_vector)
        S = H @ P_pred @ H.T + Cv
       
        log_lik += -0.5 * (np.linalg.slogdet(S)[1] +
                                      (meas - H @ m_pred).T @ np.linalg.pinv(S) @ (meas - H @ m_pred)).item()

        track.append(updated_state)
    return track, log_lik


def get_positions(transition_model, track, lag=0):
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


def get_variances(transition_model, track, lag=0):
    markov_approx, dim, ndim_1d, naug = get_model_properties(transition_model)
    mu_x_index = ndim_1d - naug
    mu_y_index = - naug

    if markov_approx == 1:
        V = np.array([[state.covar[lag, 0] + state.covar[mu_x_index, mu_x_index],
                       state.covar[ndim_1d + lag, ndim_1d] + state.covar[mu_y_index, mu_y_index]] for state in track])
    else:
        V = np.array([[state.covar[lag, 0], state.covar[ndim_1d + lag, ndim_1d]] for state in track])
    return V


def compute_rmse(gt, pos, lag=0) -> float:
    """
    Compute overall RMSE between the measurement model outputs and ground truth coordinates.

    Parameters
    ----------
    measurement_model : object
        Stone Soup measurement model used to map state space to measurement space.
    track : list
        List of Stone Soup State or GaussianState objects.
    gt : List[np.ndarray]
        List of ground truth coordinate arrays, one per spatial dimension (e.g., [gt_x, gt_y, ...]).

    Returns
    -------
    float
        Overall RMSE (sum of MSEs across dimensions).
    """
    if np.shape(pos)[1] != len(gt[0]):
        pos = np.array(pos).T  # Shape: (dim, N)

    if pos.shape[0] != len(gt):
        raise ValueError(f"Mismatch between measurement output dimension {pos.shape[0]} and number of ground truth arrays {len(gt)}.")

    rmse_components = [
        mean_squared_error(gt[i][:-lag], pos[i][lag:])
        for i in range(len(gt))
    ]

    return sum(rmse_components)