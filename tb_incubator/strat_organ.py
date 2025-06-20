from typing import List, Dict
from summer2 import Stratification
from summer2 import Overwrite, Multiply
from summer2.parameters import Parameter, Time, Function
from summer2.functions.time import get_sigmoidal_interpolation_function, get_linear_interpolation_function, get_time_callable
import tb_incubator.constants as const


compartments = const.COMPARTMENTS
infectious_compartments = const.INFECTIOUS_COMPARTMENTS
organ_strata = const.ORGAN_STRATA
model_times = const.MODEL_TIMES
agegroup_request = const.AGEGROUP_REQUEST


def get_organ_strat(
    fixed_params: Dict[str, any],
    detection_func: Function,
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

    # Detection, self-recovery and infect death
    inf_adj, detection_adjs, infect_death_adjs, self_recovery_adjustments = {}, {}, {}, {}

    for organ_stratum in organ_strata:
        # Define infectiousness adjustment by organ status
        inf_adj_param = fixed_params[f"{organ_stratum}_infect_multiplier"]
        inf_adj[organ_stratum] = Multiply(inf_adj_param)

        #Define different natural history (self-recovery) by organ status
        param_strat = "smear_negative" if organ_stratum == "extrapulmonary" else organ_stratum
        self_recovery_adjustments[organ_stratum] = Overwrite(Parameter(f"{param_strat}_self_recovery"))

        # Adjust detection by organ status
        param_name = f"passive_screening_sensitivity_{organ_stratum}"
        detection_adjs[organ_stratum] = fixed_params[param_name] * detection_func

        # Calculate infection death adjustment using detection adjustments
        infect_death_adjs[organ_stratum] = Parameter(f"{param_strat}_death_rate")
       

    # Apply the Multiply function to the detection adjustments
    detection_adjs = {k: Multiply(v) for k, v in detection_adjs.items()}
    infect_death_adjs = {k: Overwrite(v) for k, v in infect_death_adjs.items()}

    # Set flow and infectiousness adjustments
    strat.set_flow_adjustments("detection", detection_adjs)
    strat.set_flow_adjustments("self_recovery", self_recovery_adjustments)
    strat.set_flow_adjustments("infect_death", infect_death_adjs)
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
    
    return strat
