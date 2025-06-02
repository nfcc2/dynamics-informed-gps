# Generates synthetic data from iDSE-1, iiSE-1 or iiDSE-1
# Carry out tracking with iGP and iiGP models
# Plot tracks

# To import from stonesoup
import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import (generate_synthetic_ground_truth,
                                    simulate_gaussian_measurements,
                                    create_prior_state)
from tracking.kf_tracking import perform_tracking, get_positions, get_variances, compute_rmse
from tracking.plotting import plot_base, plot_tracks, add_track_unc_stonesoup


from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import time

# define params
generating_model = "iDSE"
# tracking_models = ["iSE", "iDSE", "iiSE", "iiDSE"]
tracking_models = ["iSE", "iSE", "iDSE", "iDSE", "iiSE", "iiDSE"]
markov_approxes = [1, 2, 1, 2, 1, 1]
dim = 1  # number of dimensions
# window_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # vary as needed

window_sizes = np.arange(5, 20, 1)

num_trials = 3
results = [[] for model in tracking_models]
time_interval = timedelta(seconds=0.1)
num_steps = 100

# Simulated Gaussian noise standard deviation and variance
noise_sd_dict = {"iDSE": 0.1, "iiSE": 0.3, "iiDSE": 0.25}
noise_sd = noise_sd_dict[generating_model]
noise_var = noise_sd ** 2



generating_model_params = {
    "SE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
    "iSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
    "iDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1},
    "iiSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
    "iiDSE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}, "dynamics_coeff": -0.2, "gp_coeff": 1}
}

tracking_model_params = {
    "iDSE": {
        "SE": {"kernel_params": {"length_scale": 2, "kernel_variance": 1}},
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

common_model_params = {
    "window_size": 10,
    "markov_approx": 1,
    "prior_var": noise_var
}

if __name__ == "__main__":
    np.random.seed(5)
    start_time = datetime.now()

    model_params_combined = {**common_model_params, **generating_model_params[generating_model]}
    transition_model_gen = initialise_transition_model(generating_model, dim=dim, **model_params_combined)
    prior = create_prior_state(transition_model_gen, start_time, [np.random.rand() for _ in range(dim)], [noise_var for _ in range(dim)])
    gt = generate_synthetic_ground_truth(transition_model_gen, prior, num_steps, time_interval)
    meas = simulate_gaussian_measurements(gt, noise_sd)

    # Carry out tracking, plot tracks with uncertainty intervals for each model
    print(f"{'Model':<10} {'Log Likelihood':<20} {'RMSE':<10}")
    print("="*40) 

    for window in window_sizes:
        # model parameters
        common_model_params["window_size"] = window
        for i in range(len(tracking_models)):
            if markov_approxes[i] == 2:
                common_model_params["markov_approx"] = 2
            else:
                common_model_params["markov_approx"] = 1
            times = []
            for j in range(num_trials):
                np.random.seed(j)
                model_params_combined = {**common_model_params, **tracking_model_params[generating_model][tracking_models[i]]}
                transition_model = initialise_transition_model(tracking_models[i], dim=dim, **model_params_combined)
                measurement_model = initialise_measurement_model(transition_model, noise_var)

                start = time.perf_counter()
                perform_tracking(gt, meas, transition_model, measurement_model, time_interval, prior_var=noise_var)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            results[i].append(avg_time)
    
    # Plotting
    for i in range(len(tracking_models)):
        model = tracking_models[i]
        if model == "SE":
            label = "SE"
        else:
            label = f"{model}-{markov_approxes[i]}"
        plt.plot(window_sizes, results[i], label=label, marker='o')

    plt.xlabel(r"Window size $d$")
    plt.ylabel("Tracking runtime (s)")
    # plt.title("Tracking Runtime vs Window Size")
    plt.legend()
    plt.grid(True)
    plt.show()


