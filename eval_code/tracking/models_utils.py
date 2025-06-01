# helper functions to initialise and access properties of stone soup transition models and measurement models

from stonesoup.base import Property
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    SimpleMarkovianGP, IntegratedGP, TwiceIntegratedGP, 
    DynamicsInformedIntegratedGP, DynamicsInformedTwiceIntegratedGP
)
from stonesoup.models.measurement.linear import LinearGaussian

import numpy as np
from typing import Tuple
    
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
    "SE": SimpleMarkovianGP,
    "iSE": IntegratedGP,
    "iiSE": TwiceIntegratedGP,
    "iDSE": DynamicsInformedIntegratedGP,
    "iiDSE": DynamicsInformedTwiceIntegratedGP,
    "CV": ConstantVelocity
}


def initialise_transition_model(model_name, dim, window_size, markov_approx, kernel_params, dynamics_coeff=None, gp_coeff=None, prior_var=0, noise_diff_coeff=0):
    """
    Initialise a n-dimensional model by name with the given hyperparameters.
    For SE-based models, kernel_params dict specifies length_scale and kernel_variance.
    Specify n with dim parameter.
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

    if model_name == "CV":
        model_params = {"noise_diff_coeff": noise_diff_coeff}
        
    return CombinedLinearGaussianTransitionModel([ModelClass(**model_params) for _ in range(dim)])


def initialise_measurement_model(transition_model, var):
    markov_approx, dim, ndim_1d, num_aug_states = get_model_properties(transition_model)

    if markov_approx == 1:
        return LinearGaussianAugmented(
            ndim_state=transition_model.ndim_state,
            ndim_meas=dim,
            num_aug_states=num_aug_states,
            mapping=(),  # placeholder, not needed as we construct the matrix directly
            noise_covar=np.eye(dim)*var
        )
    else:
        mapping = [i * ndim_1d for i in range(dim)]
        return LinearGaussian(
            ndim_state=transition_model.ndim_state,
            mapping=tuple(mapping),
            noise_covar=np.eye(dim)*var
        )


def get_model_properties(transition_model) -> Tuple[int, int, int, int]:
    """
    Extract properties from the transition model (assumes >1D model).

    Parameters
    ----------
    transition_model : object
        A transition model that contains a list of 1D models (e.g., SlidingWindowGP or iDSE).

    Returns
    -------
    markov_approx : int
        The order of Markovian approximation used. For SE models, this is 0.
    dim : int
        The number of spatial dimensions being tracked.
    ndim_1d : int
        The dimensionality of the state vector in a single dimension.
    num_aug_states : int
        The number of augmented states (e.g., mean velocity or position).
    """
    dim = len(transition_model.model_list)
    transition_model_1d = transition_model.model_list[0]
    ndim_1d = transition_model_1d.ndim_state
    
    name = type(transition_model_1d).__name__
    if name == "ConstantVelocity":
        return 0, dim, ndim_1d, 0

    num_aug_states = ndim_1d - transition_model_1d.window_size

    if transition_model_1d.__class__.__name__ == "SimpleMarkovianGP":
        markov_approx = 0
    else:
        markov_approx = transition_model_1d.markov_approx

    return markov_approx, dim, ndim_1d, num_aug_states
