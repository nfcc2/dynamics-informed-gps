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
dim = 3  # number of dimensions

time_interval = timedelta(seconds=0.1)
num_steps = 100

# Simulated Gaussian noise standard deviation and variance
noise_sd_dict = {"iDSE": 0.1, "iiSE": 0.3, "iiDSE": 0.25}
noise_sd = noise_sd_dict[generating_model]
noise_var = noise_sd ** 2

# model parameters
common_model_params = {
    "window_size": 10,
    "markov_approx": 1,
    "prior_var": noise_var
}

generating_model_params = {
    "SE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
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
    np.random.seed(5)
    start_time = datetime.now()

    model_params_combined = {**common_model_params, **generating_model_params[generating_model]}
    transition_model_gen = initialise_transition_model(generating_model, dim=dim, **model_params_combined)
    prior = create_prior_state(transition_model_gen, start_time, [np.random.rand() for _ in range(dim)], [noise_var for _ in range(dim)])
    gt = generate_synthetic_ground_truth(transition_model_gen, prior, num_steps, time_interval)
    meas = simulate_gaussian_measurements(gt, noise_sd)
    # Plot ground truth and measurements
    plot_base(gt, meas)

    # Carry out tracking, plot tracks with uncertainty intervals for each model
    print(f"{'Model':<10} {'Log Likelihood':<20} {'RMSE':<10}")
    print("="*40) 

    for i in range(len(tracking_models)):
        model_params_combined = {**common_model_params, **tracking_model_params[generating_model][tracking_models[i]]}
        transition_model = initialise_transition_model(tracking_models[i], dim=dim, **model_params_combined)
        measurement_model = initialise_measurement_model(transition_model, noise_var)
        track, log_lik, rmse = perform_tracking(gt, meas, transition_model, measurement_model, time_interval, prior_var=noise_var)

        print(f"{tracking_models[i]:<10} {log_lik:<20.4f} {rmse:<10.4f}")
        plot_tracks(track, transition_model, measurement_model)

        # only add uncertainty intervals of one iGP and one iiGP model for clarity. 
        # the iSE and iDSE have similar shapes, and similarly for the iiSE and iiDSE models
        if tracking_models[i] == "iDSE" or tracking_models[i] == "iiDSE":
            add_track_unc_stonesoup(track, transition_model)
    
    plt.grid(True)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()


