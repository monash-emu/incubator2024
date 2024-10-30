from jax import numpy as jnp
from math import log, exp
import numpy as np
import pandas as pd

def get_next_run_number(out_path, num_priors):
    """
    Find the next available run number for a specific number of priors.
    Returns a formatted string like '01', '02', etc.
    """
    # Look for files matching the pattern with the specific number of priors
    pattern = f'calib_full_out_p{num_priors:02d}_*.nc'
    existing_files = list(out_path.glob(pattern))
    
    if not existing_files:
        return '01'
    
    # Extract run numbers from existing files
    numbers = []
    for f in existing_files:
        # Split filename and get the last part after 'p{num_priors}_'
        try:
            run_num = int(f.stem.split('_')[-1])
            numbers.append(run_num)
        except ValueError:
            continue
    
    next_num = max(numbers) + 1 if numbers else 1
    return f'{next_num:02d}'

def get_target_from_name(
    targets: list,
    name: str,
) -> pd.Series:
    """Get the data for a specific target from a set of targets from its name.

    Args:
        targets: All the targets
        name: The name of the desired target

    Returns:
        Single target to identify
    """
    return next((t.data for t in targets if t.name == name), None)

def round_sigfig(value: float, sig_figs: int) -> float:
    """
    Round a number to a certain number of significant figures,
    rather than decimal places.

    Args:
        value: Number to round
        sig_figs: Number of significant figures to round to
    """
    if np.isinf(value):
        return "infinity"
    else:
        return (
            round(value, -int(np.floor(np.log10(value))) + (sig_figs - 1)) if value != 0.0 else 0.0
        )
    
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


