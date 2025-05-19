import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import (
    generate_synthetic_ground_truth,
    simulate_gaussian_measurements,
    create_prior_state
)
from tracking.kf_tracking import perform_tracking

from datetime import datetime, timedelta
import numpy as np

# Define parameters
generating_model = "iDSE"
tracking_models = ["iSE", "iDSE", "iiSE", "iiDSE"]

time_interval = timedelta(seconds=0.1)
num_steps = 100
num_iterations = 100  # Number of iterations

# Simulated Gaussian noise standard deviation and variance
noise_sd_dict = {"iDSE": 0.1, "iiSE": 0.3, "iiDSE": 0.25}
noise_sd = noise_sd_dict[generating_model]
noise_var = noise_sd ** 2

# Common model parameters
common_model_params = {
    "window_size": 10,
    "markov_approx": 1,
    "prior_var": noise_var
}

generating_model_params = {
    "iSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
    "iDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1},
    "iiSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
    "iiDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1}
}

tracking_model_params = {
    "iDSE": {
        "iSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
        "iDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1},
        "iiSE": {"kernel_params": {"length_scale": 0.5, "kernel_variance": 5}},
        "iiDSE": {"kernel_params": {"length_scale": 0.5, "kernel_variance": 5}, "dynamics_coeff": -0.2, "gp_coeff": 1}
    }, 
    "iiSE": {
        "iSE": {"kernel_params": {"length_scale": 2.5, "kernel_variance": 5}},
        "iDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 2.5}, "dynamics_coeff": 0.2, "gp_coeff": 1},
        "iiSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
        "iiDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1}
    }, 
    "iiDSE": {
        "iSE": {"kernel_params": {"length_scale": 2.5, "kernel_variance": 5}},
        "iDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 2.5}, "dynamics_coeff": 0.1, "gp_coeff": 1},
        "iiSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
        "iiDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1}
    }
}

if __name__ == "__main__":
    # Store results
    results = {model: {"log_likelihoods": [], "rmses": []} for model in tracking_models}

    for seed in range(num_iterations):
        np.random.seed(seed)
        start_time = datetime.now()

        model_params_combined = {**common_model_params, **generating_model_params[generating_model]}
        transition_model_gen = initialise_transition_model(generating_model, **model_params_combined)
        prior = create_prior_state(transition_model_gen, start_time, np.random.rand(), np.random.rand(), noise_var)
        gt_x, gt_y = generate_synthetic_ground_truth(transition_model_gen, prior, num_steps, time_interval)
        meas_x, meas_y = simulate_gaussian_measurements(gt_x, gt_y, noise_sd)

        for model in tracking_models:
            model_params_combined = {**common_model_params, **tracking_model_params[generating_model][model]}
            transition_model = initialise_transition_model(model, **model_params_combined)
            measurement_model = initialise_measurement_model(transition_model, noise_var)
            _, log_lik, rmse = perform_tracking(gt_x, gt_y, meas_x, meas_y, transition_model, measurement_model, time_interval, prior_var=noise_var)

            results[model]["log_likelihoods"].append(log_lik)
            results[model]["rmses"].append(rmse)

    # Compute averages
    print(f"{'Model':<10} {'Avg Log Likelihood':<20} {'Avg RMSE':<10}")
    print("=" * 40)
    for model in tracking_models:
        avg_log_lik = np.mean(results[model]["log_likelihoods"])
        avg_rmse = np.mean(results[model]["rmses"])
        print(f"{model:<10} {avg_log_lik:<20.4f} {avg_rmse:<10.5f}")
