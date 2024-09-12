from jax import numpy as jnp
from math import log, exp
import pandas as pd
import yaml as yml
from tb_incubator.constants import project_path


def tanh_based_scaleup(t, shape, inflection_time, start_asymptote, end_asymptote=1.0):
    """
    return the function t: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :param shape: shape parameter
    :param inflection_time: inflection point
    :param start_asymptote: lowest asymptotic value
    :param end_asymptote: highest asymptotic value
    :return: a function
    """
    rng = end_asymptote - start_asymptote
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote


def triangle_wave_func(
    time: float,
    start: float,
    duration: float,
    peak: float,
) -> float:
    """Generate a peaked triangular wave function
    that starts from and returns to zero.

    Args:
        time: Model time
        start: Time at which wave starts
        duration: Duration of wave
        peak: Peak flow rate for wave

    Returns:
        The wave function
    """
    gradient = peak / (duration * 0.5)
    peak_time = start + duration * 0.5
    time_from_peak = jnp.abs(peak_time - time)
    return jnp.where(time_from_peak < duration * 0.5, peak - time_from_peak * gradient, 0.0)


def get_average_sigmoid(
    low_val, upper_val, inflection
):  # Long's code # Ragonnet, R., et al. (2019)
    """
    A sigmoidal function (x -> 1 / (1 + exp(-(x-alpha)))) is used to model a progressive increase with age.
    """
    return (log(1.0 + exp(upper_val - inflection)) - log(1.0 + exp(low_val - inflection))) / (
        upper_val - low_val
    )


def load_param_info() -> pd.DataFrame:
    """
    Load specific parameter information from a ridigly formatted yaml file, and crash otherwise.

    Returns:
        The parameters info DataFrame contains the following fields:
            value: Enough parameter values to ensure model runs, may be over-written in calibration
            descriptions: A brief reader-digestible name/description for the parameter
            units: The unit of measurement for the quantity (empty string if dimensionless)
            evidence: TeX-formatted full description of the evidence underpinning the choice of value
            abbreviations: Short names for parameters, e.g. for some plots
    """
    with open(project_path / "parameters.yaml", "r") as param_file:
        param_info = yml.safe_load(param_file)

    # Check each loaded set of keys (parameter names) are the same as the arbitrarily chosen first key
    first_key_set = param_info[list(param_info.keys())[0]].keys()
    for cat in param_info:
        working_keys = param_info[cat].keys()
        if working_keys != first_key_set:
            msg = f"Keys to {cat} category: {working_keys} - do not match first category {first_key_set}"
            raise ValueError(msg)

    return param_info


def get_param_table(param_info):
    """
    Get parameter info in a tidy pd.Dataframe format.
    """
    param_table = []
    for key in param_info["value"]:
        if isinstance(param_info["value"][key], dict):
            if key in param_info["unit"]:
                for subkey, value in param_info["value"][key].items():
                    value_str = "/".join(f"{k}: {v}" for k, v in value.items())
                    param_table.append(
                        {
                            "Parameter": f"{param_info['descriptions'][key][subkey]}",
                            "Value": value_str,
                            "Unit": param_info["unit"][key][subkey],
                            "Source": param_info["sources"][key],
                        }
                    )
        else:
            param_table.append(
                {
                    "Parameter": param_info["descriptions"][key],
                    "Value": param_info["value"][key],
                    "Unit": param_info["unit"][key],
                    "Source": param_info["sources"][key],
                }
            )

    fixed_param_table = pd.DataFrame(param_table)
    fixed_param_table = fixed_param_table.set_index("Parameter")

    return fixed_param_table
