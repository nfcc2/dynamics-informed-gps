# tracking_eval.py

import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import import_ground_truth_coordinates, simulate_gaussian_measurements
from tracking.kf_tracking import perform_tracking

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os

# Configuration
tracking_models = ["SE", "iSE", "iDSE", "iiSE", "iiDSE", "CV"]
dim = 3
num_trajectories = 5
num_seeds = 20
time_interval = timedelta(seconds=0.01)
noise_sd = 0.01
noise_var = noise_sd ** 2

common_model_params = {
    "window_size": 10,
    "markov_approx": 1,
    "prior_var": noise_var
}

output_data = []

for trajectory_idx in range(1, num_trajectories + 1):
    print(f"\nProcessing trajectory {trajectory_idx}...")
    trajectory_csv_path = f"trajectories/{trajectory_idx}.csv"
    params_df = pd.read_csv(f"results/best_hyperparams_traj{trajectory_idx}.csv")

    gt = import_ground_truth_coordinates(trajectory_csv_path, dim=dim)

    for model_name in tracking_models:
        rmses = []
        row = params_df[params_df["model"] == model_name].iloc[0]

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

        for seed in range(num_seeds):
            np.random.seed(seed)
            meas = simulate_gaussian_measurements(gt, noise_sd)

            transition_model = initialise_transition_model(model_name, dim=dim, **model_params_combined)
            measurement_model = initialise_measurement_model(transition_model, noise_var)
            _, _, rmse = perform_tracking(gt, meas, transition_model, measurement_model, time_interval, prior_var=noise_var)
            rmses.append(rmse)

        avg_rmse = np.mean(rmses)
        output_data.append({
            "trajectory_idx": trajectory_idx,
            "model": model_name,
            "avg_rmse": avg_rmse
        })
        print(f"Model {model_name} | Avg RMSE: {avg_rmse:.6f}")

# Save results
results_df = pd.DataFrame(output_data)
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/average_rmse_per_model_per_trajectory.csv", index=False)
print("\nSaved RMSE results to results/average_rmse_per_model_per_trajectory.csv")
