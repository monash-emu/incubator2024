from typing import Optional
from summer2.parameters import Parameter, Time, Function
from summer2.functions.time import get_sigmoidal_interpolation_function, get_linear_interpolation_function, get_time_callable
from .input import load_genexpert_util
from .utils import tanh_based_scaleup
from .constants import ImplementCDR


def get_detection_func(
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    xpert_improvement: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.NONE,
    improved_detection_multiplier: Optional[float] = None,
    xpert_util_target: Optional[float] = None,
) -> Function:
    """
    Create TB detection function with all available improvements.
    
    Args:
        covid_effects: Whether to apply COVID-19 impact on detection
        improved_detection_multiplier: Future detection improvement multiplier by 2027
        apply_diagnostic_capacity: Whether to apply diagnostic capacity scaling
        xpert_improvement: Whether to apply Xpert utilization improvements (requires apply_diagnostic_capacity=True)
        apply_cdr: Case detection rate implementation strategy
        xpert_util_target: Target Xpert utilization rate for 2027 (0-1)
    
    Returns:
        Complete detection function with all specified improvements
    """
    # Create base detection function
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

    # Apply COVID effects
    if covid_effects:
        detection_func = apply_covid_effects(detection_func)

    # Validate parameter combinations
    if xpert_improvement and not apply_diagnostic_capacity:
        raise ValueError("xpert_improvement=True requires apply_diagnostic_capacity=True")

    # Apply future detection improvements
    if improved_detection_multiplier is not None:
        _validate_positive_number(improved_detection_multiplier, "improved_detection_multiplier")
        detection_func = apply_future_detection_improvement(detection_func, improved_detection_multiplier)

    # Apply diagnostic capacity improvements
    if apply_diagnostic_capacity:
        diagnostic_capacity = Parameter("base_diagnostic_capacity")
        
        if xpert_improvement: 
            diagnostic_improvement = calculate_xpert_util_improvement(xpert_util_target=xpert_util_target)  
            diagnostic_capacity += diagnostic_improvement
            
        detection_func *= diagnostic_capacity  
    
    # Apply case detection rate improvements
    if apply_cdr == ImplementCDR.ON_DETECTION:  
        national_cdr = get_linear_interpolation_function(
            [2017.0, 2023.0],
            [Parameter("initial_notif_rate"), Parameter("latest_notif_rate")]
        )
        detection_func *= national_cdr

    return detection_func


def apply_future_detection_improvement(detection_func: Function, multiplier: float) -> Function:
    """Apply projected future improvement to detection function."""
    improve_detection_func = get_linear_interpolation_function(
        [2025.0, 2027.0],
        [1.0, multiplier]
    )
    return detection_func * improve_detection_func


def calculate_xpert_util_improvement(xpert_util_target: Optional[float] = None) -> Function:
    """Calculate improvement in diagnostic capacity due to Xpert utilisation increase."""
    utilisation = load_genexpert_util()
    genexpert_util = get_sigmoidal_interpolation_function(utilisation.index, utilisation)
    
    diagnostic_improvement = (
        (1.0 - Parameter("base_diagnostic_capacity")) * 
        Parameter("genexpert_sensitivity") * 
        genexpert_util
    )
        
    if xpert_util_target is not None:
        _validate_utilization_rate(xpert_util_target)
        diagnostic_improvement = apply_future_xpert_improvement(
            genexpert_util, diagnostic_improvement, xpert_util_target
        )
        
    return diagnostic_improvement


def apply_future_xpert_improvement(
    genexpert_util: Function, 
    diagnostic_improvement: Function,
    xpert_util_target: float, 
) -> Function:
    """Scale diagnostic improvement based on projected Xpert utilisation target."""
    xpert_util_callable = get_time_callable(genexpert_util)
    current_util = float(xpert_util_callable(2025.0))
            
    # Only scale up if target is higher than current utilization
    if xpert_util_target > current_util:
        scale_factor = xpert_util_target / current_util
        util_scale_factor = get_linear_interpolation_function(
            [2025.0, 2027.0], 
            [1.0, scale_factor]
        )
        diagnostic_improvement *= util_scale_factor
        
    return diagnostic_improvement


def apply_covid_effects(detection_func: Function) -> Function:
    """Apply COVID-19 pandemic impact on TB case detection."""
    covid_impact_func = get_linear_interpolation_function(
        [2019.0, 2020.0, 2021.0],
        [1.0, 1.0 - Parameter("detection_reduction"), 1.0]
    )
    return detection_func * covid_impact_func


def _validate_positive_number(value: float, name: str) -> None:
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value}")


def _validate_utilization_rate(value: float) -> None:
    """Validate that utilization rate is between 0 and 1."""
    if not isinstance(value, (int, float)) or not (0 < value <= 1):
        raise ValueError(f"Utilization rate must be between 0 and 1, got {value}")