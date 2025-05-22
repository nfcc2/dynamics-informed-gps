# Carry out tracking with GP models on mosquito trajectory data and plot figure

# To import from stonesoup
import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import (import_ground_truth_coordinates,
                                    simulate_gaussian_measurements)
from tracking.kf_tracking import perform_tracking, get_positions, get_variances
from tracking.plotting import plot_base, plot_tracks, add_track_unc_stonesoup

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# define params
tracking_models = ["SE", "iSE", "iDSE", "iiSE", "iiDSE"]
dim = 2
lag = 3  # for smoothing

# import trajectory and hyperparameters
trajectory_idx = 5
trajectory_csv_path = f"trajectories/{trajectory_idx}.csv"
params_df = pd.read_csv(f"results/best_hyperparams_traj{trajectory_idx}.csv")

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

if __name__ == "__main__":
    np.random.seed(50)
    start_time = datetime.now()

    gt = import_ground_truth_coordinates(trajectory_csv_path, dim=dim)
    meas = simulate_gaussian_measurements(gt, noise_sd)

    # Plot ground truth and measurements
    plot_base(gt, meas)

    # Carry out tracking, plot tracks with uncertainty intervals for each model
    print(f"{'Model':<10} {'Log Likelihood':<20} {'RMSE':<10}")
    print("="*40) 

    for i in range(len(tracking_models)):
        row = params_df[params_df["model"] == tracking_models[i]].iloc[0]

        model_params = {
            "kernel_params": {
                "length_scale": row.get("length_scale", 0.1),
                "kernel_variance": row.get("kernel_variance", 0.1)
            },
            "gp_coeff": row.get("gp_coeff", 1),
            "dynamics_coeff": row.get("dynamics_coeff"),
            "noise_diff_coeff": row.get("noise_diff_coeff")
        }

        # Remove NaNs
        model_params = {k: v for k, v in model_params.items() if pd.notna(v)}
        model_params_combined = {**common_model_params, **model_params}
        transition_model = initialise_transition_model(tracking_models[i], dim=dim, **model_params_combined)
        measurement_model = initialise_measurement_model(transition_model, noise_var)
        track, log_lik, rmse = perform_tracking(gt, meas, transition_model, measurement_model, time_interval, prior_var=noise_var)

        print(f"{tracking_models[i]:<10} {log_lik:<20.4f} {rmse:<10.7f}")

        pos = get_positions(transition_model, track, lag)
        plot_tracks(transition_model, pos)

        if tracking_models[i] in ["SE", "iDSE", "iiDSE"]:
            var = get_variances(transition_model, track, lag)
            add_track_unc_stonesoup(transition_model, pos, var)
    
    plt.grid(True)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()


