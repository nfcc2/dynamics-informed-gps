# Generates synthetic data from iDSE-1, iiSE-1 or iiDSE-1
# Compute the marginal log likelihood over a grid of length scales and kernel variances for the tracking model
# Plot a contour plot of marginal log likelihood and annotate global and local maxima

import setup

from tracking.models_utils import initialise_transition_model, initialise_measurement_model
from tracking.tracking_init import (
    generate_synthetic_ground_truth, simulate_gaussian_measurements, create_prior_state
)
from tracking.kf_tracking import perform_tracking

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define parameters
generating_model = "iiSE"
tracking_model = "iiSE"

time_interval = timedelta(seconds=0.1)
num_steps = 100

# Noise parameters
noise_sd_dict = {"iDSE": 0.1, "iiSE": 0.3, "iiDSE": 0.25}
noise_sd = noise_sd_dict[generating_model]
noise_var = noise_sd ** 2

# Search space for kernel parameters
length_scales = np.linspace(0.5, 3, 20)
kernel_variances = np.linspace(0.5, 5, 20)

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

# Function to check if the current point is a local maximum
def is_local_maxima(i, j, LL_values):
    neighbors = []
    
    # Check all valid neighbors (accounting for boundaries)
    if i > 0:  # Above
        neighbors.append(LL_values[i-1, j])
    if i < LL_values.shape[0] - 1:  # Below
        neighbors.append(LL_values[i+1, j])
    if j > 0:  # Left
        neighbors.append(LL_values[i, j-1])
    if j < LL_values.shape[1] - 1:  # Right
        neighbors.append(LL_values[i, j+1])

    # Check if current value is greater than all neighbors
    return all(LL_values[i, j] > neighbor for neighbor in neighbors)

# Generate ground truth data
np.random.seed(2)
start_time = datetime.now()
model_params_combined = {**common_model_params, **generating_model_params[generating_model]}
transition_model_gen = initialise_transition_model(generating_model, **model_params_combined)
measurement_model_gen = initialise_measurement_model(transition_model_gen, noise_var)
prior = create_prior_state(transition_model_gen, start_time, np.random.rand(), np.random.rand(), noise_var)
gt_x, gt_y = generate_synthetic_ground_truth(transition_model_gen, measurement_model_gen, prior, num_steps, time_interval)
meas_x, meas_y = simulate_gaussian_measurements(gt_x, gt_y, noise_sd)

# Store results
LL_values = np.zeros((len(length_scales), len(kernel_variances)))

# Vary length scale and kernel variance
for i, ls in enumerate(length_scales):
    for j, kv in enumerate(kernel_variances):
        tracking_model_params = {
            "kernel_params": {"length_scale": ls, "kernel_variance": kv},
            "dynamics_coeff": -0.2, "gp_coeff": 1  # fixed a
        }
        model_params_combined = {**common_model_params, **tracking_model_params}
        transition_model = initialise_transition_model(tracking_model, **model_params_combined)
        measurement_model = initialise_measurement_model(transition_model, noise_var)
        _, log_lik, _ = perform_tracking(gt_x, gt_y, meas_x, meas_y, transition_model, measurement_model, time_interval, prior_var=noise_var)
        LL_values[i, j] = log_lik

# Find maximum LL and corresponding parameters
max_idx = np.unravel_index(np.argmax(LL_values), LL_values.shape)
max_ls = length_scales[max_idx[0]]
max_kv = kernel_variances[max_idx[1]]
max_LL = LL_values[max_idx]

# Find local maxima in LL_values
local_maxima = []
for i in range(1, LL_values.shape[0] - 1):  # Avoid edges
    for j in range(1, LL_values.shape[1] - 1):  # Avoid edges
        if is_local_maxima(i, j, LL_values):
            local_maxima.append((i, j, LL_values[i, j]))

# Extract the corresponding parameters for the local maxima
local_maxima_params = [(length_scales[i], kernel_variances[j], LL_value) 
                       for i, j, LL_value in local_maxima]

# Obtain MLL for true parameters
measurement_model_gen = initialise_measurement_model(transition_model_gen, noise_var)
_, log_lik_true, _ = perform_tracking(gt_x, gt_y, meas_x, meas_y, transition_model_gen, measurement_model_gen, time_interval, prior_var=noise_var)

# Contour plot
plt.figure(figsize=(8, 6))
X, Y = np.meshgrid(kernel_variances, length_scales)
contour = plt.contourf(X, Y, LL_values, levels=20, cmap="viridis")
plt.colorbar(contour, label="Marginal log likelihood (MLL)")

plt.scatter(1, 2, color="blue", marker="o", label=r"True values: $\ell$={0:.2f}, $\sigma^2$={1:.2f}, MLL={2:.2f}".format(2, 1, log_lik_true))
plt.scatter(max_kv, max_ls, color="red", marker="x", label=r"Maximum: $\ell$={0:.2f}, $\sigma^2$={1:.2f}, MLL={2:.2f}".format(max_ls, max_kv, max_LL))

for ls, kv, ll in local_maxima_params:
    plt.scatter(kv, ls, color="purple", marker="x", label=r"Local maximum: $\ell$={0:.2f}, $\sigma^2$={1:.2f}, MLL={2:.2f}".format(ls, kv, ll))

plt.xlabel("Kernel variance $\sigma^2$")
plt.ylabel("Length scale $\ell$")
plt.legend()
plt.show()
