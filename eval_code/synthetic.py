# Generates synthetic data from iDSE-1, iiSE-1 or iiDSE-1
# Carry out tracking with iGP and iiGP models
# Plot tracks

# To import from stonesoup
import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import (generate_synthetic_ground_truth,
                                    simulate_gaussian_measurements,
                                    create_prior_state)
from tracking.kf_tracking import perform_tracking
from tracking.plotting import plot_base, plot_tracks, add_track_unc_stonesoup


from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

# define params
generating_model = "iDSE"
tracking_models = ["iSE", "iDSE", "iiSE", "iiDSE"]

time_interval = timedelta(seconds=0.1)
num_steps = 100

# Simulated Gaussian noise standard deviation and variance
noise_sd = 0.1
noise_var = noise_sd ** 2

# Colours for visualisation
colors = {"iSE": "mediumorchid", "iDSE": "gold", "iiSE": "coral", "iiDSE": "skyblue", "SE": "limegreen"}

# model parameters
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
    "iSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
    "iDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1},
    "iiSE": {"kernel_params": {"length_scale": 0.5, "kernel_variance": 5}},
    "iiDSE": {"kernel_params": {"length_scale": 0.5, "kernel_variance": 5}, "dynamics_coeff": -1, "gp_coeff": 1}
}

if __name__ == "__main__":
    np.random.seed(2)
    start_time = datetime.now()

    model_params_combined = {**common_model_params, **generating_model_params[generating_model]}
    transition_model_gen = initialise_transition_model(generating_model, **model_params_combined)
    prior = create_prior_state(transition_model_gen, start_time, np.random.rand(), np.random.rand(), noise_var)
    gt_x, gt_y = generate_synthetic_ground_truth(transition_model_gen, prior, 100, time_interval)
    meas_x, meas_y = simulate_gaussian_measurements(gt_x, gt_y, noise_sd)

    # Plot ground truth and measurements
    plot_base(gt_x, gt_y, meas_x, meas_y)

    # Carry out tracking, plot tracks with uncertainty intervals for each model
    print(f"{'Model':<10} {'Log Likelihood':<20} {'RMSE':<10}")
    print("="*40) 

    for i in range(len(tracking_models)):
        model_params_combined = {**common_model_params, **tracking_model_params[tracking_models[i]]}
        transition_model = initialise_transition_model(tracking_models[i], **model_params_combined)
        measurement_model = initialise_measurement_model(transition_model, noise_var)
        track, log_lik, rmse = perform_tracking(gt_x, gt_y, meas_x, meas_y, transition_model, measurement_model, time_interval, prior_var=noise_var)

        print(f"{tracking_models[i]:<10} {log_lik:<20.4f} {rmse:<10.4f}")
        plot_tracks(track, transition_model, measurement_model)
        add_track_unc_stonesoup(track, transition_model)
    
    plt.grid(True)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()


