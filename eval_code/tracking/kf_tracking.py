# Tracking workflow and kalman filtering

from tracking.tracking_init import generate_stonesoup_ground_truth, generate_stonesoup_measurements, create_prior_state

from stonesoup.types.track import Track
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.predictor.gaussianprocess import GPPredictorWrapper
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.simple import SingleHypothesis

import numpy as np
from sklearn.metrics import mean_squared_error

def perform_tracking(gt_x, gt_y, meas_x, meas_y, transition_model, measurement_model, time_interval, prior_var):
    """Carry out tracking with Kalman filtering given raw ground truth and measurement data."""

    # Step 1: Generate stone soup ground truth path
    ground_truth = generate_stonesoup_ground_truth(transition_model, gt_x, gt_y, time_interval)
    
    # Step 2: Convert measurements to Stone Soup detections
    measurements = generate_stonesoup_measurements(measurement_model, meas_x, meas_y, ground_truth)
    
    # Step 3: Track initialisation
    prior_state = create_prior_state(transition_model, ground_truth[0].timestamp, gt_x[0], gt_y[0], prior_var)
    track = Track(prior_state)

    # Step 4: Define Kalman predictor and updater
    predictor = GPPredictorWrapper(KalmanPredictor(transition_model))
    updater = KalmanUpdater(measurement_model)

    # Step 5: Run tracking and calculate log likelihood
    track, log_lik = kalman_filter(predictor, updater, measurement_model, track, measurements)
    
    # Step 6: Compute rmse
    rmse = compute_rmse(measurement_model, track, gt_x, gt_y)

    return track, log_lik, rmse

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


def compute_rmse(measurement_model, track, gt_x, gt_y):
    pos = [measurement_model.function(state) for state in track]
    x_vals = []
    y_vals = []
    for state in pos:
        x_vals.append(state[0])
        y_vals.append(state[1])
    rmse_x = mean_squared_error(gt_x, x_vals)
    rmse_y = mean_squared_error(gt_y, y_vals)
    overall_rmse = rmse_x + rmse_y

    return overall_rmse