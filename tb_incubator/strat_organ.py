from typing import List, Dict
from summer2 import Stratification
from summer2 import Overwrite, Multiply
from summer2.parameters import Parameter, Function
from tb_incubator.constants import (INFECTIOUS_COMPARTMENTS, ORGAN_STRATA)


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
    strat = Stratification("organ", ORGAN_STRATA, INFECTIOUS_COMPARTMENTS)

    # Define different detection rates by organ status
    detection_adjs = {}

    # Detection, self-recovery and infect death
    inf_adj, detection_adjs, infect_death_adjs, self_recovery_adjustments = {}, {}, {}, {}

    for organ_stratum in ORGAN_STRATA:
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
    for comp in INFECTIOUS_COMPARTMENTS:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    splitting_proportions = {
        "smear_positive": Parameter("incidence_props_pulmonary")
        * Parameter("incidence_props_smear_positive_among_pulmonary"),
        "smear_negative": Parameter("incidence_props_pulmonary")
        * (1.0 - Parameter("incidence_props_smear_positive_among_pulmonary")),
        "extrapulmonary": 1.0 - Parameter("incidence_props_pulmonary"),
    }

    for flow_name in ["early_activation", "late_activation"]:
        flow_adjs = {k: Multiply(v) for k, v in splitting_proportions.items()}
        strat.set_flow_adjustments(flow_name, flow_adjs)
    
    return strat
