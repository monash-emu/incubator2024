from typing import List, Dict
from summer2 import Stratification
from summer2 import Overwrite, Multiply
from summer2.parameters import Parameter, Time, Function
from summer2.functions.time import get_sigmoidal_interpolation_function, get_linear_interpolation_function
from .input import load_genexpert_util
import tb_incubator.constants as const
from .utils import tanh_based_scaleup
import tb_incubator.constants as const

compartments = const.compartments
infectious_compartments = const.infectious_compartments
organ_strata = const.organ_strata
model_times = const.model_times
agegroup_request = const.agegroup_request


def get_organ_strat(
    infectious_compartments: List[str],
    organ_strata: List[str],
    fixed_params: Dict[str, any],
    xpert_sensitivity: bool = True,
    covid_effects: bool = True,
) -> Stratification:
    """
    Creates and configures an organ stratification for the model. This includes defining
    adjustments for infectiousness, infection death rates, and self-recovery rates based
    on organ involvement, as well as adjusting progression rates by organ using requested
    incidence proportions.

    Args:
        infectious_compartments: A list of names of compartments that can transmit infection.
        organ_strata: A list of organ strata names for stratification (e.g., 'lung', 'extrapulmonary').
        fixed_params: A dictionary containing fixed parameters for the model, including
                      multipliers for infectiousness by organ, death rates by organ, and
                      incidence proportions for different organ involvements.

    Returns:
        A Stratification object configured with organ-specific adjustments.
    """
    strat = Stratification("organ", organ_strata, infectious_compartments)

    # Define different detection rates by organ status
    detection_adjs = {}
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

    ## xpert sensitivity
    sensitivity = Parameter("base_sensitivity")
    if xpert_sensitivity:
        utilisation = load_genexpert_util()
        genexpert_util = get_sigmoidal_interpolation_function(utilisation.index, utilisation)
        genexpert_improvement = (1.0 - Parameter("base_sensitivity")) * Parameter("genexpert_sensitivity") * genexpert_util
        sensitivity += genexpert_improvement
    
    detection_func = detection_func * sensitivity

    
    ## covid effects
    if covid_effects:
        covid_impacts = get_sigmoidal_interpolation_function(
            [2019.0, 2020.0, 2022.0], [1.0, 1.0 - Parameter("detection_reduction"), Parameter("post_covid_improvement")]
        )
        detection_func = detection_func * covid_impacts
        ### post-COVID sustained improvement
        sustained_improvement = get_linear_interpolation_function([2022.0, model_times[-1]], [1.0, Parameter("sustained_improvement")])
        detection_func = detection_func * sustained_improvement


    # Detection, self-recovery and infect death
    inf_adj, detection_adjs, infect_death_adjs, self_recovery_adjustments = {}, {}, {}, {}
    for organ_stratum in organ_strata:
        # Define infectiousness adjustment by organ status
        inf_adj_param = fixed_params[f"{organ_stratum}_infect_multiplier"]
        inf_adj[organ_stratum] = Multiply(inf_adj_param)

        # Define different natural history (self-recovery) by organ status
        #param_strat = "smear_negative" if organ_stratum == "extrapulmonary" else organ_stratum
        #self_recovery_adjustments[organ_stratum] = Overwrite(Parameter(f"{param_strat}_self_recovery"))

        # Adjust detection by organ status
        param_name = f"passive_screening_sensitivity_{organ_stratum}"
        detection_adjs[organ_stratum] = fixed_params[param_name] * detection_func

        # Calculate infection death adjustment using detection adjustments
        #infect_death_adjs[organ_stratum] = Parameter(f"{param_strat}_death_rate")
       

    # Apply the Multiply function to the detection adjustments
    detection_adjs = {k: Multiply(v) for k, v in detection_adjs.items()}
    #infect_death_adjs = {k: Overwrite(v) for k, v in infect_death_adjs.items()}

    # Set flow and infectiousness adjustments
    strat.set_flow_adjustments("detection", detection_adjs)
    #strat.set_flow_adjustments("self_recovery", self_recovery_adjustments)
    #strat.set_flow_adjustments("infect_death", infect_death_adjs)
    for comp in infectious_compartments:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    splitting_proportions = {
        "smear_positive": fixed_params["incidence_props_pulmonary"]
        * fixed_params["incidence_props_smear_positive_among_pulmonary"],
        "smear_negative": fixed_params["incidence_props_pulmonary"]
        * (1.0 - fixed_params["incidence_props_smear_positive_among_pulmonary"]),
        "extrapulmonary": 1.0 - fixed_params["incidence_props_pulmonary"],
    }
    for flow_name in ["early_activation", "late_activation"]:
        flow_adjs = {k: Multiply(v) for k, v in splitting_proportions.items()}
        strat.set_flow_adjustments(flow_name, flow_adjs)

    #organ_adjs = {
    #    "smear_positive": Multiply(1.0),
    #    "smear_negative": Multiply(1.0),
    #    "extrapulmonary": Multiply(0.0),
    #}

    #strat.set_flow_adjustments("acf_detection", organ_adjs)
    
    return strat