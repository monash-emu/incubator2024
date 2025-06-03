from typing import Dict
from summer2.parameters import Parameter, Time, Function
from summer2.functions.time import get_sigmoidal_interpolation_function, get_linear_interpolation_function, get_time_callable
from .input import load_genexpert_util
import tb_incubator.constants as const
from .utils import tanh_based_scaleup
import tb_incubator.constants as const

compartments = const.COMPARTMENTS
infectious_compartments = const.INFECTIOUS_COMPARTMENTS
organ_strata = const.ORGAN_STRATA
model_times = const.MODEL_TIMES
agegroup_request = const.AGEGROUP_REQUEST

def get_detection_func(
    xpert_improvement: bool = True,
    covid_effects: Dict[str, bool] = None,
    xpert_util_target: float = None,
    improved_detection_multiplier: float = None
) -> Function:
    """
    Creates a time-variant TB detection function that combines multiple factors affecting case detection.
    
    This function constructs a model of TB case detection rates over time by combining:
    1. A baseline detection function 
    2. Diagnostic capacity improvements by increase of Xpert utilisation
    3. Impacts from COVID-19 (optional)
    4. Projected future improvements in detection capabilities (optional)
    
    Args:
        xpert_improvement (bool, optional): Whether to include Xpert improvement. Defaults to True.
        covid_effects (Dict[str, bool], optional): Dictionary specifying which COVID-19 
            effects to include. Currently only supports "detection_reduction". 
            Defaults to {"detection_reduction": False}.
        xpert_util_target (float, optional): Target Xpert utilisation rate to reach by 2030,
            value is between 0 and 1. Only used when xpert_improvement=True.
        improved_detection_multiplier (float, optional): Factor to scale up detection 
            capability between 2025-2030 for scenario projection. If None, no future improvement is modelled.
    
    Returns:
        Tuple containing:
            - detection_func (Function): The final combined detection function over time
            - base_detection (Function): The baseline detection function
            - diagnostic_capacity (Function): The total diagnostic capacity function
            - diagnostic_improvement (Function): The improvement in diagnostic capacity
    
    """
    detection_func = Function(
        tanh_based_scaleup,
        [
            Time,
            Parameter("screening_scaleup_shape"),
            Parameter("screening_inflection_time"),
            0.0,
            1.0 / Parameter("time_to_screening_end_asymp")
        ]
    )
    base_detection = detection_func

    ## xpert improvement
    diagnostic_capacity = Parameter("base_diagnostic_capacity") #Function(lambda: 1.0)
    
    if xpert_improvement:
        diagnostic_improvement = calculate_xpert_util_improvement(xpert_util_target)
    else:
        diagnostic_improvement = Function(lambda: 0.0)
        
    diagnostic_capacity += diagnostic_improvement
    detection_func *= diagnostic_capacity

    if covid_effects["detection_reduction"]:
        detection_func = apply_covid_effects(detection_func)

    if improved_detection_multiplier is not None:
        assert isinstance(improved_detection_multiplier, float) and improved_detection_multiplier > 0.0, "improved_detection_multiplier must be a positive float."
        detection_func = apply_future_detection_improvement(detection_func, improved_detection_multiplier)
    
    detection_func

    return detection_func, base_detection, diagnostic_capacity, diagnostic_improvement

def apply_future_detection_improvement(
    detection_func: Function, 
    improved_detection_multiplier: float
) -> Function:
    """
    Applies a projected future improvement to a detection function over time.
    
    Args:
        detection_func (Function): The base detection function to be scaled
        improved_detection_multiplier (float): Target multiplier to reach by 2030.
            For example, 1.5 would mean a 50% improvement in detection by 2030.
            Must be a positive float.
    
    Returns:
        Function: A new detection function that incorporates the projected future improvements
    """
    detection_improvement_vals = {
        2025.0: 1.0,
        2030.0: improved_detection_multiplier
    }
    improve_detection_func = get_linear_interpolation_function(
        list(detection_improvement_vals.keys()),
        list(detection_improvement_vals.values())
    )

    return detection_func * improve_detection_func

def calculate_xpert_util_improvement(
    xpert_util_target: float = None,
) -> Function:
    """
    Calculates the improvement in diagnostic capacity due to increase in Xpert utilisation.
    
    Args:
        xpert_util_target (float, optional): Target Xpert utilisation rate to reach by 2030,
            value is between 0 and 1. Used in scenario projection. Only used when xpert_improvement=True.
    
    Returns:
        Function: A time-dependent function representing the improvement in diagnostic
            capacity due to increase in Xpert utilisation over time.
    """
    utilisation = load_genexpert_util()
    genexpert_util = get_sigmoidal_interpolation_function(utilisation.index, utilisation)
    diagnostic_improvement = (1.0 - Parameter("base_diagnostic_capacity")) * Parameter("genexpert_sensitivity") * genexpert_util
        
    # Apply xpert_util_target only when xpert_improvement is True
    if xpert_util_target is not None:
        assert isinstance(xpert_util_target, float) and xpert_util_target > 0 and xpert_util_target <=1.0, "xpert_util_target must be a float between 0 and 1."
        diagnostic_improvement = apply_future_xpert_improvement(genexpert_util, diagnostic_improvement, xpert_util_target)
        
    return diagnostic_improvement

def apply_future_xpert_improvement(
    genexpert_util: Function, 
    diagnostic_improvement: Function,
    xpert_util_target: float, 
) -> Function:
    """
    Applies a projected Xpert utilisation target to scale up the diagnostic improvement.
    
    Args:
        genexpert_util (Function): The current Xpert utilisation function over time
        diagnostic_improvement (Function): The base diagnostic improvement function to scale
        xpert_util_target (float): Target utilisation rate to reach by 2030 (between 0 and 1)
    
    Returns:
        Function: Scaled diagnostic improvement function incorporating the projected utilisation increase
    
    """
    xpert_util_callable = get_time_callable(genexpert_util)
    current_util = float(xpert_util_callable(2025.0))
            
    # Create scaling function that goes from current to target utilization
    if xpert_util_target > current_util:  # Only scale up if target is higher
        util_scale_factor = get_linear_interpolation_function(
            [2025.0, 2030.0], 
            [1.0, xpert_util_target / current_util]
        )
        diagnostic_improvement *= util_scale_factor
        
    return diagnostic_improvement

def apply_covid_effects(
    detection_func: Function,
) -> Function:
    """
    Applies the impact of the COVID-19 pandemic on TB case detection.
    
    Args:
        detection_func (Function): The base TB detection function to be modified
    
    Returns:
        Function: A modified detection function incorporating COVID-19 impacts
    """
    covid_impact_vals = {
        2019.0: 1.0,
        2020.0: 1.0 - Parameter("detection_reduction"),
        2022.0: 1.0
    }

    covid_impact_func = get_linear_interpolation_function(
        list(covid_impact_vals.keys()),
        list(covid_impact_vals.values())
    )

    return detection_func * covid_impact_func