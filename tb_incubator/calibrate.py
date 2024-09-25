import numpy as np
from typing import List
from tb_incubator.constants import compartments, infectious_compartments, model_times, age_strata
from tb_incubator.model import build_model
from tb_incubator.input import load_targets, load_param_info

from estival import targets as est
from estival import priors as esp
from estival.model import BayesianCompartmentalModel

import xarray as xr
import arviz as az


def get_bcm(params) -> BayesianCompartmentalModel:
    """
    Constructs and returns a Bayesian Compartmental Model.
    Parameters:
    - params (dict): A dictionary containing fixed parameters for the model.

    Returns:
    - BayesianCompartmentalModel: An instance of the BayesianCompartmentalModel class, ready for
      simulation and analysis. This model encapsulates the TB compartmental model, the dynamic
      and fixed parameters, prior distributions for Bayesian inference, and target data for model
      validation or calibration.
    """
    model, desc = build_model(
        compartments,
        infectious_compartments,
        age_strata,
        params,
        model_times)
    priors = get_all_priors()
    targets = get_targets()
    
    return BayesianCompartmentalModel(model, params, priors, targets)


def get_all_priors() -> List:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    priors = [
        esp.UniformPrior("contact_rate", (0.1, 10.0)),
        esp.UniformPrior("self_recovery_rate", (0.05, 0.50)),
        esp.UniformPrior("screening_scaleup_shape", (0.01, 0.5)),
        esp.UniformPrior("screening_inflection_time", (2000.0, 2018.0)),
        esp.UniformPrior("time_to_screening_end_asymp", (0.1, 20.0)),
        esp.UniformPrior("rr_infection_latent", (0.01, 1.0)),
        esp.UniformPrior("rr_infection_recovered", (0.01, 1.0)),
        esp.UniformPrior("seed_time", (1840.0, 1900.0)),
        esp.UniformPrior("seed_duration", (1.0, 20.0)),
        esp.UniformPrior("seed_rate", (1.0, 100.0)),
        esp.UniformPrior("base_sensitivity", (0.01, 1.0)),
        esp.UniformPrior("genexpert_sensitivity", (0.01, 1.0)),
        esp.UniformPrior("progression_multiplier", (0.5, 1.5)),
    ]

    return priors


def get_targets() -> list:
    """
    Loads target data for a model and constructs a list of NormalTarget instances.

    Returns:
    - list: A list of Target instances.
    """
    target_data = load_targets()

    targets = [
        est.TruncatedNormalTarget("prevalence", target_data["prevalence"], (0.0, np.inf), esp.UniformPrior("prevalence_dispersion", (0.1, target_data["prevalence"].max() * 0.1))),
        est.TruncatedNormalTarget("notification", target_data["notif2000"], (0.0, np.inf), esp.UniformPrior("notification_dispersion", (0.1, target_data["notif2000"].max() * 0.1)))
    ]

    return targets
