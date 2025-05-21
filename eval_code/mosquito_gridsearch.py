import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import import_ground_truth_coordinates, simulate_gaussian_measurements
from tracking.kf_tracking import perform_tracking

import numpy as np
import pandas as pd
from datetime import timedelta
import os

# Define models and hyperparameter grids
tracking_models = ["SE", "iSE", "iDSE", "iiSE", "iiDSE", "CV"]
trajectories = [f"trajectories/{i}.csv" for i in range(1, 6)]
dim = 3
time_interval = timedelta(seconds=0.01)
noise_sd = 0.01
noise_var = noise_sd ** 2

# Common parameters
common_model_params = {
    "window_size": 10,
    "markov_approx": 1,
    "prior_var": noise_var
}

length_scales = [0.01, 0.05, 0.1, 0.5, 1]
kernel_variances = [0.01, 0.05, 0.1, 0.5, 1]
dynamics_coeffs = [-0.5, -0.4, -0.3, -0.2, -0.1]
cv_coeffs = np.linspace(0.1, 1, 10)

os.makedirs("results", exist_ok=True)

for traj_idx, path in enumerate(trajectories, start=1):
    print(f"\n== Trajectory {traj_idx} ==")
    gt = import_ground_truth_coordinates(path, dim=dim)
    meas = simulate_gaussian_measurements(gt, noise_sd)

    results = []

    for model in tracking_models:
        print(f"Model: {model}")
        best_loglik = -np.inf
        best_rmse = np.inf
        best_params = {}

        if model == "CV":
            for coeff in cv_coeffs:
                params = {"noise_diff_coeff": coeff, "kernel_params": {}}
                combined = {**common_model_params, **params}
                tm = initialise_transition_model(model, dim=dim, **combined)
                mm = initialise_measurement_model(tm, noise_var)
                _, loglik, rmse = perform_tracking(gt, meas, tm, mm, time_interval, prior_var=noise_var)

                if loglik > best_loglik:
                    best_loglik = loglik
                    best_rmse = rmse
                    best_params = params

        else:
            for ls in length_scales:
                for kv in kernel_variances:
                    for a in (dynamics_coeffs if "D" in model else [None]):
                        params = {
                            "kernel_params": {"length_scale": ls, "kernel_variance": kv},
                            "gp_coeff": 1
                        }
                        if a is not None:
                            params["dynamics_coeff"] = a
                        combined = {**common_model_params, **params}
                        tm = initialise_transition_model(model, dim=dim, **combined)
                        mm = initialise_measurement_model(tm, noise_var)
                        _, loglik, rmse = perform_tracking(gt, meas, tm, mm, time_interval, prior_var=noise_var)

                        if loglik > best_loglik:
                            best_loglik = loglik
                            best_rmse = rmse
                            best_params = params

        results.append({
            "model": model,
            "log_likelihood": best_loglik,
            "rmse": best_rmse,
            **best_params.get("kernel_params", {}),
            "dynamics_coeff": best_params.get("dynamics_coeff"),
            "gp_coeff": best_params.get("gp_coeff"),
            "noise_diff_coeff": best_params.get("noise_diff_coeff")
        })

    df = pd.DataFrame(results)
    df.to_csv(f"results/best_hyperparams_traj{traj_idx}.csv", index=False)

print("\nGrid search complete. One CSV file per trajectory saved in `results/`, including RMSE.")
