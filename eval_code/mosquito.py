# Carry out tracking with GP models on mosquito trajectory data and plot figure

# To import from stonesoup
import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import (import_ground_truth_coordinates,
                                    simulate_gaussian_measurements)
from tracking.kf_tracking import perform_tracking
from tracking.plotting import plot_base, plot_tracks, add_track_unc_stonesoup

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

# define params
tracking_models = ["SE", "iSE", "iDSE", "iiSE", "iiDSE"]

time_interval = timedelta(seconds=0.01)
num_steps = 100

# Simulated Gaussian noise standard deviation and variance
noise_sd = 0.01
noise_var = noise_sd ** 2

# model parameters
common_model_params = {
    "window_size": 10,
    "markov_approx": 1,
    "prior_var": noise_var
}

tracking_model_params = {
    "SE": {"kernel_params": {"length_scale": 0.5, "kernel_variance": 0.5}},
    "iSE": {"kernel_params": {"length_scale": 0.1, "kernel_variance": 0.05}},
    "iDSE": {"kernel_params": {"length_scale": 0.1, "kernel_variance": 0.05}, "dynamics_coeff": -0.3, "gp_coeff": 1},
    "iiSE": {"kernel_params": {"length_scale": 0.05, "kernel_variance": 1}},
    "iiDSE": {"kernel_params": {"length_scale": 0.05, "kernel_variance": 1}, "dynamics_coeff": -0.3, "gp_coeff": 1}
}

if __name__ == "__main__":
    np.random.seed(10)
    start_time = datetime.now()

    gt = import_ground_truth_coordinates("mosquito_coordinates.csv")
    meas = simulate_gaussian_measurements(gt, noise_sd)

    # Plot ground truth and measurements
    plot_base(gt, meas)

    # Carry out tracking, plot tracks with uncertainty intervals for each model
    print(f"{'Model':<10} {'Log Likelihood':<20} {'RMSE':<10}")
    print("="*40) 

    for i in range(len(tracking_models)):
        model_params_combined = {**common_model_params, **tracking_model_params[tracking_models[i]]}
        transition_model = initialise_transition_model(tracking_models[i], dim=2, **model_params_combined)
        measurement_model = initialise_measurement_model(transition_model, noise_var)
        track, log_lik, rmse = perform_tracking(gt, meas, transition_model, measurement_model, time_interval, prior_var=noise_var)

        print(f"{tracking_models[i]:<10} {log_lik:<20.4f} {rmse:<10.7f}")
        plot_tracks(track, transition_model, measurement_model)

        if tracking_models[i] in ["SE", "iDSE", "iiDSE"]:
            add_track_unc_stonesoup(track, transition_model)
    
    plt.grid(True)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()


