# Carry out tracking with GP models on mosquito trajectory data and plot figure
# with online hyperparameter tuning

# To import from stonesoup
import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import (import_ground_truth_coordinates,
                                    simulate_gaussian_measurements)
from tracking.kf_tracking import (perform_tracking, get_positions, get_variances, compute_rmse,
                                  kalman_filter)
from tracking.plotting import plot_base, plot_tracks, add_track_unc_stonesoup
from tracking.tracking_init import generate_stonesoup_ground_truth, generate_stonesoup_measurements, create_prior_state
from tracking.models_utils import get_model_properties

from stonesoup.types.track import Track
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.predictor.gaussianprocess import GPPredictorWrapper
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.simple import SingleHypothesis

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

from itertools import product
import copy
import os

# define params
tracking_models = ["SE"]
dim = 3
lag = 0  # for smoothing
num_seeds = 20

# import trajectory and hyperparameters
trajectory_idx = 3
trajectory_csv_path = f"trajectories/{trajectory_idx}.csv"
params_df = pd.read_csv(f"results/best_hyperparams_traj{trajectory_idx}.csv")

time_interval = timedelta(seconds=0.01)

# Simulated Gaussian noise standard deviation and variance
noise_sd = 0.01
noise_var = noise_sd ** 2

# onl
# ine hyperparameter estimation
q = 20  # optimise once every 20 timesteps
hyp_window = 20
# hyp = {"length_scale": [0.01, 0.05, 0.1, 0.5, 1], "kernel_variance": [0.01, 0.05, 0.1, 0.15]}
hyp_bounds = {"length_scale": [0.01, 0.2], "dynamics_coeff": [-0.5, -0.1]}
# hyp = {"dynamics_coeff": [-0.5, -0.4, -0.3, -0.2, -0.1]}
# hyp = {}
# hyp = {"length_scale": np.linspace(0.1, 1, 10)}
# hyp = {"length_scale": np.linspace(0.1, 1, 10), "kernel_variance": np.linspace(0.01, 0.2, 10)}
hyp = {"kernel_variance": np.linspace(0.01, 0.2, 10)}

# model parameters
common_model_params = {
    "window_size": 10,
    "markov_approx": 1,
    "prior_var": noise_var
}

def tracking_with_hyp(gt, meas, transition_model, measurement_model, time_interval, prior_var, model_name, q, m, hyp, model_params):
    """
    Carry out tracking with Kalman filtering given raw ground truth and measurement data.
    Tune hyperparameters every q timesteps over the previous m timesteps.
    hyp = {"length_scale": [grid search range]}
    """
    
    _, dim, _, _ = get_model_properties(transition_model)

    ground_truth = generate_stonesoup_ground_truth(transition_model, gt, time_interval)
    measurements = generate_stonesoup_measurements(measurement_model, meas, ground_truth)

    prior_state = create_prior_state(
        transition_model, ground_truth[0].timestamp,
        [gt[d][0] for d in range(dim)],
        [prior_var for _ in range(dim)]
    )
    track = Track(prior_state)

    predictor = GPPredictorWrapper(KalmanPredictor(transition_model))
    updater = KalmanUpdater(measurement_model)

    t_max = len(measurements)
    # print("length of measurements: ", t_max)
    track, log_lik = kalman_filter(predictor, updater, measurement_model, track, measurements[:max(m,q)+1])
    # print(track[-1].timestamp, measurements[m].timestamp, len(track))

    for t in range(max(m, q), t_max, q):
        # Optimise using previous m steps
        track_section = Track(copy.deepcopy(track[:t-m+1]))
        measurements_section = measurements[t - m: t + 1]

        # Optimise using last m measurements and track states
        transition_model, _, _ = optimise_hyp_grid(
            model_name, model_params, hyp, updater,
            track_section, measurements_section
        )

        # prev_length_scale = transition_model.model_list[0].kernel_params["length_scale"]
        # print(prev_length_scale)
        # prev_a = transition_model.model_list[0].dynamics_coeff

        # transition_model, _, _ = optimise_hyp_scipy(
        #     model_name, model_params, hyp_bounds, updater,
        #     track_section, measurements_section, prev_a
        # )

        print(transition_model.model_list[0].kernel_params) #, transition_model.model_list[0].dynamics_coeff)

        predictor = GPPredictorWrapper(KalmanPredictor(transition_model))
        track, ll_section = kalman_filter(predictor, updater, measurement_model, track, measurements[t: t + q+ 1])
        log_lik += ll_section

    if len(track) < t_max:
        # # Slice the last m measurements to use for optimisation
        # meas_start = t_max - m
        # last_measurements = measurements[meas_start:t_max]

        # # Slice the last m+1 track states for filtering
        # track_start = len(track) - m
        # last_track_section = Track(copy.deepcopy(track[track_start:]))

        # # Optimise using last m measurements and track states
        # transition_model, _, _ = optimise_hyp_grid(
        #     model_name, model_params, hyp, updater,
        #     last_track_section, last_measurements
        # )

        # transition_model, _, _ = optimise_hyp_scipy(
        #     model_name, model_params, hyp_bounds, updater,
        #     last_track_section, last_measurements, prev_a
        # )

        # print(transition_model.model_list[0].kernel_params)

        predictor = GPPredictorWrapper(KalmanPredictor(transition_model))

        # Use the remaining measurements to finish filtering
        remaining_measurements = measurements[len(track):]
        track, ll_section = kalman_filter(predictor, updater, measurement_model, track, remaining_measurements)
        log_lik += ll_section

    return track, log_lik




def optimise_hyp_grid(model_name, model_params, hyp, updater, track, measurements):
    """
    Find optimal hyperparameters that maximise MLL over given track and measurements.
    Returns transition model with best hyperparameters.
    hyp = {"length_scale": [grid search range], etc. for relevant hyperparameters}
    """

    # check if timestamps are correct. latest measurement timestamp should be one timestep ahead of latest track timestamp
    measurement_model = updater.measurement_model
    max_LL = -np.inf
    best_hyp = None
    best_model = None

    # Generate all combinations of hyperparameters
    keys = list(hyp.keys())
    values = list(hyp.values())
    for combo in product(*values):
        current_hyp = dict(zip(keys, combo))

        # Deep copy model_params to avoid modifying shared state
        current_model_params = copy.deepcopy(model_params)

        # Update model_params with current hyperparameters
        for h, v in current_hyp.items():
            if h in current_model_params.get("kernel_params", {}):
                current_model_params["kernel_params"][h] = v
            else:
                current_model_params[h] = v  # For non-kernel hparams

        # Create a new transition model using these hyperparameters
        transition_model = initialise_transition_model(model_name, dim=dim, **current_model_params)
        predictor = GPPredictorWrapper(KalmanPredictor(transition_model))

        # Apply Kalman filter and compute log-likelihood
        _, LL = kalman_filter(predictor, updater, measurement_model, track, measurements)

        if LL > max_LL:
            max_LL = LL
            best_hyp = current_model_params
            best_model = transition_model

    return best_model, best_hyp, max_LL

def negative_log_likelihood(length_scale, model_name, model_params, updater, track, measurements):
    # Update the model parameters with the current length_scale
    current_model_params = copy.deepcopy(model_params)
    # if "kernel_params" in current_model_params:
    current_model_params["kernel_params"]["length_scale"] = length_scale[0]
    # else:
    #     current_model_params["kernel_params"] = {"length_scale": length_scale[0]}
    # current_model_params["dynamics_coeff"] = a[0]
    
    # Initialize the transition model with updated parameters
    transition_model = initialise_transition_model(model_name, dim=dim, **current_model_params)
    predictor = GPPredictorWrapper(KalmanPredictor(transition_model))
    measurement_model = updater.measurement_model

    # Apply Kalman filter over the sliding window
    _, log_likelihood = kalman_filter(predictor, updater, measurement_model, track, measurements)

    # Return negative log-likelihood for minimization
    return -log_likelihood

def optimise_hyp_scipy(model_name, model_params, hyp_bounds, updater, track, measurements, prev_length_scale):
    """
    Optimize hyperparameters by minimizing the negative log-likelihood over a sliding window.
    hyp_bounds: dict with hyperparameter names as keys and [lower, upper] as values.
    """
    # Extract bounds for length_scale
    # length_scale_bounds = hyp_bounds.get("length_scale", [0.01, 2.0])
    # print(length_scale_bounds)
    a_bounds = hyp_bounds.get("dynamics_coeff", [-0.5, -0.1])
    print(a_bounds)
    bounds = [(a_bounds[0], a_bounds[1])]
    initial_guesses = [0.01, 0.05, 0.1, 0.2]

    # Perform optimization
    result = minimize(
        negative_log_likelihood,
        x0=prev_length_scale,
        args=(model_name, model_params, updater, track, measurements),
        bounds=bounds,
        method='Nelder-Mead'
    )

    # Update model parameters with optimized length_scale
    optimized_length_scale = result.x[0]
    print(optimized_length_scale)
    optimized_model_params = copy.deepcopy(model_params)
    # if "kernel_params" in optimized_model_params:
    #     optimized_model_params["kernel_params"]["length_scale"] = optimized_length_scale
    # else:
    #     optimized_model_params["kernel_params"] = {"length_scale": optimized_length_scale}

    optimized_model_params["dynamics_coeff"] = optimized_length_scale

    # Initialize the transition model with optimized parameters
    optimized_transition_model = initialise_transition_model(model_name, dim=dim, **optimized_model_params)

    return optimized_transition_model, optimized_model_params, -result.fun


if __name__ == "__main__":
    np.random.seed(50)
    start_time = datetime.now()
    output_data = []

    gt = import_ground_truth_coordinates(trajectory_csv_path, dim=dim)
    meas = simulate_gaussian_measurements(gt, noise_sd)

    # Plot ground truth and measurements
    # plot_base(gt, meas)

    # Carry out tracking, plot tracks with uncertainty intervals for each model
    print(f"{'Model':<10} {'Log Likelihood':<20} {'RMSE':<10}")
    print("="*40) 

    for i in range(len(tracking_models)):
        row = params_df[params_df["model"] == tracking_models[i]].iloc[0]
        rmses = []
        model_params = {
            "kernel_params": {
                "length_scale": row.get("length_scale", 0.1),
                "kernel_variance": row.get("kernel_variance", 0.1)
            },
            "gp_coeff": row.get("gp_coeff", 1),
            "dynamics_coeff": row.get("dynamics_coeff"),
            "noise_diff_coeff": row.get("noise_diff_coeff")
        }

        for seed in range(num_seeds):
            np.random.seed(seed+255)
            meas = simulate_gaussian_measurements(gt, noise_sd)

            # Remove NaNs
            model_params = {k: v for k, v in model_params.items() if pd.notna(v)}
            model_params_combined = {**common_model_params, **model_params}
            transition_model = initialise_transition_model(tracking_models[i], dim=dim, **model_params_combined)
            measurement_model = initialise_measurement_model(transition_model, noise_var)
            track, log_lik = tracking_with_hyp(gt, meas, transition_model, measurement_model, time_interval, noise_var, tracking_models[i], q, hyp_window, hyp, model_params_combined)

            pos = get_positions(transition_model, track, lag)
            
            rmse = compute_rmse(gt, pos, lag)
            rmses.append(rmse)
            print(f"{tracking_models[i]:<10} {log_lik:<20.4f} {rmse:<10.7f}")

        print(rmses)
        avg_rmse = np.mean(rmses)
        output_data.append({
            "trajectory_idx": trajectory_idx,
            "model": tracking_models[i],
            "avg_rmse": avg_rmse
        })
        print(f"Model {tracking_models[i]} | Avg RMSE: {avg_rmse:.6f}")

        # plot_tracks(transition_model, pos)
        # if tracking_models[i] in ["SE", "iDSE", "iiDSE"]:
        #     var = get_variances(transition_model, track, lag)
        #     add_track_unc_stonesoup(transition_model, pos, var)
        
    # plt.grid(True)
    # plt.xlabel("X Position")
    # plt.ylabel("Y Position")
    # plt.legend()
    # plt.show()

    # Save results
    results_df = pd.DataFrame(output_data)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/traj_5_iSE_2.csv", index=False)
    print("\nSaved RMSE results to results/average_rmse_per_model_per_trajectory.csv")

