
# to optimise hyperparameters length scale, variance, and a on synthetic data

import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import (
    generate_synthetic_ground_truth, simulate_gaussian_measurements, create_prior_state
)
from tracking.kf_tracking import perform_tracking
from tracking.plotting import plot_base

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define parameters
generating_model = "iiDSE"  # Model used for ground truth generation
tracking_model = "iSE"  # Model used for tracking/optimization

time_interval = timedelta(seconds=0.1)
num_steps = 100

# Noise parameters
noise_sd_dict = {"iDSE": 0.1, "iiSE": 0.3, "iiDSE": 0.25}
noise_sd = noise_sd_dict[generating_model]
noise_var = noise_sd ** 2

# Search space for kernel parameters
length_scales = [0.5, 1, 1.5, 2, 2.5]
kernel_variances = [0.1, 0.5, 1, 2.5, 5]
dynamics_coeffs = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]

# Common model parameters
common_model_params = {
    "window_size": 10,
    "markov_approx": 1,
    "prior_var": noise_var
}

# Model-specific parameters
generating_model_params = {
    "iDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1},
    "iiSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
    "iiDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1}
}

if __name__ == "__main__":
    # Generate ground truth data
    np.random.seed(5)
    start_time = datetime.now()
    model_params_combined = {**common_model_params, **generating_model_params[generating_model]}
    transition_model_gen = initialise_transition_model(generating_model, **model_params_combined)
    prior = create_prior_state(transition_model_gen, start_time, np.random.rand(), np.random.rand(), noise_var)
    gt_x, gt_y = generate_synthetic_ground_truth(transition_model_gen, prior, num_steps, time_interval)
    meas_x, meas_y = simulate_gaussian_measurements(gt_x, gt_y, noise_sd)

    # Choose dynamics coefficients only if applicable
    if tracking_model in ["iDSE", "iiDSE"]:
        a_values = dynamics_coeffs
    else:
        a_values = [None]  # placeholder to keep loop structure
        
    # Store results
    LL_values = np.zeros((len(length_scales), len(kernel_variances), len(a_values)))

    # Vary length scale and kernel variance
    for k, a in enumerate(a_values):
        for i, ls in enumerate(length_scales):
            for j, kv in enumerate(kernel_variances):
                tracking_model_params = {
                    "kernel_params": {"length_scale": ls, "kernel_variance": kv},
                    "gp_coeff": 1
                }
                if a is not None:
                    tracking_model_params["dynamics_coeff"] = a
                model_params_combined = {**common_model_params, **tracking_model_params}
                transition_model = initialise_transition_model(tracking_model, **model_params_combined)
                measurement_model = initialise_measurement_model(transition_model, noise_var)
                _, log_lik, _ = perform_tracking(gt_x, gt_y, meas_x, meas_y, transition_model, measurement_model, time_interval, prior_var=noise_var)
                LL_values[i, j, k] = log_lik
                print(f"Log Likelihood = {log_lik:.3f}, Length scale = {ls}, Variance = {kv}, a = {a}")


    # Find maximum LL and corresponding parameters
    max_idx = np.unravel_index(np.argmax(LL_values), LL_values.shape)
    max_ls = length_scales[max_idx[0]]
    max_kv = kernel_variances[max_idx[1]]
    max_a = a_values[max_idx[2]]
    max_LL = LL_values[max_idx]

    print("\nBest parameter combination:")
    print(f"Length scale = {max_ls}, Variance = {max_kv}, a = {max_a}, Max Log Likelihood = {max_LL:.3f}")

