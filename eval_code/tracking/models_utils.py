# helper functions to initialise and access properties of stone soup transition models and measurement models

from stonesoup.base import Property
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    SlidingWindowGP, IntegratedGP, TwiceIntegratedGP, 
    DynamicsInformedIntegratedGP, DynamicsInformedTwiceIntegratedGP
)
from stonesoup.models.measurement.linear import LinearGaussian

import numpy as np
    
# Define measurement model for Markov 1 models
class LinearGaussianAugmented(LinearGaussian):
    "Measurement includes 1st augmented state"
    num_aug_states: int = Property()
    ndim_meas: int = Property()
    def matrix(self, **kwargs):
        ndim_1d = self.ndim_state // self.ndim_meas
        model_matrix = np.zeros((self.ndim_meas, self.ndim_state))

        for d in range(self.ndim_meas):
            pos_i, aug_i = d * ndim_1d, (d + 1) * ndim_1d - self.num_aug_states
            model_matrix[d, pos_i] = 1
            model_matrix[d, aug_i] = 1
        return model_matrix


# Dictionary to map string names to Stone Soup classes
MODEL_CLASSES = {
    "SE": SlidingWindowGP,
    "iSE": IntegratedGP,
    "iiSE": TwiceIntegratedGP,
    "iDSE": DynamicsInformedIntegratedGP,
    "iiDSE": DynamicsInformedTwiceIntegratedGP
}


def initialise_transition_model(model_name, window_size, markov_approx, kernel_params, dynamics_coeff=None, gp_coeff=None, prior_var=0):
    """
    Initialise a 2D model by name with the given hyperparameters.
    For SE-based models, kernel_params dict specifies length_scale and kernel_variance.
    """
    
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")
    
    ModelClass = MODEL_CLASSES[model_name]  # Get the class reference

    model_params = {
        "window_size": window_size,
        "kernel_params": kernel_params
    }
    
     # Conditionally add parameters for iGP models
    if "i" in model_name:
        model_params.update({
            "prior_var": prior_var
        })
   
    if "i" in model_name and "ii" not in model_name:
        model_params.update({
            "markov_approx": markov_approx
        })

    if "DSE" in model_name :
        model_params.update({
            "dynamics_coeff": dynamics_coeff,
            "gp_coeff": gp_coeff
        })

    return CombinedLinearGaussianTransitionModel([ModelClass(**model_params), ModelClass(**model_params)])


def initialise_measurement_model(transition_model, var):
    markov_approx, dim, ndim_1d, num_aug_states = get_model_properties(transition_model)

    if markov_approx == 1:
        return LinearGaussianAugmented(
            ndim_state=transition_model.ndim_state,
            ndim_meas=dim,
            num_aug_states=num_aug_states,
            mapping=(),  # placeholder, not needed as we construct the matrix directly
            noise_covar=np.array([[var, 0], [0, var]])
        )
    else:
        return LinearGaussian(
            ndim_state=transition_model.ndim_state,
            mapping=(0, ndim_1d),
            noise_covar=np.array([[var, 0], [0, var]])
        )


def get_model_properties(transition_model):
    """
    Returns model properties.
    markov_approx: order of markovian approximation used. 
        For the SE model, this is set to 0 as the plotting/initialisation logic for approx = 1 is different.
    dim: number of dimensions we are tracking.
    ndim_1d: dimension of state vector in 1 dimension.
    num_aug_states: For Markov 1 models, the number of augmented states (mean position or mean velocity).
    """
    dim = len(transition_model.model_list)
    transition_model_1d = transition_model.model_list[0]
    ndim_1d = transition_model_1d.ndim_state
    num_aug_states = ndim_1d - transition_model_1d.window_size

    if transition_model_1d.__class__.__name__ == "SlidingWindowGP":
        markov_approx = 0
    else:
        markov_approx = transition_model_1d.markov_approx

    return markov_approx, dim, ndim_1d, num_aug_states