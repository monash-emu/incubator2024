from summer2 import CompartmentalModel
from typing import List
from summer2.parameters import DerivedOutput
import tb_incubator.constants as const
import numpy as np

compartments = const.compartments
infectious_compartments = const.infectious_compartments
organ_strata = const.organ_strata
latent_compartments = const.latent_compartments

def request_model_outputs(
    model: CompartmentalModel,
):
    compartments = [c.name for c in model._original_compartment_names]

    # compartment sizes
    for c in compartments:
        model.request_output_for_compartments(f"comp_size_{c}", c)
    tot_pop = model.request_output_for_compartments("total_population", compartments)
    model.request_function_output("population_log", np.log(tot_pop))

    # latent percentage
    latent_pop = model.request_output_for_compartments(
        "latent_pop", latent_compartments, save_results=False
    )
    model.request_function_output("percentage_latent", latent_pop / tot_pop * 100.0)

    # death
    deaths = model.request_output_for_flow("mortality_infectious", "TB_death")
    model.request_function_output("mortality", 1e5 * deaths / tot_pop)

    # prevalence
    # Calculate and request prevalence of pulmonary
    for organ_stratum in organ_strata:
        model.request_output_for_compartments(
            f"infectious_sizeXorgan_{organ_stratum}",
            infectious_compartments,
            strata={"organ": organ_stratum},
            save_results=False,
        )

    pulmonary_outputs = [
        f"infectious_sizeXorgan_{organ_stratum}"
        for organ_stratum in ["smear_positive", "smear_negative"]
    ]

    pulmonary_pop_size = model.request_aggregate_output("pulmonary_pop_size", pulmonary_outputs)
    model.request_function_output("prevalence_pulmonary", 1e5 * pulmonary_pop_size / tot_pop)

    # total prevalence
    infectious_population_size = model.request_output_for_compartments(
        "infectious_population_size", infectious_compartments
    )
    prevalence = model.request_function_output("prevalence", 1e5 * infectious_population_size / tot_pop)
    model.request_function_output("prevalence_log", np.log(prevalence))

    # incidence
    timings = ["early_activation", "late_activation"]
    for t in timings:
        model.request_output_for_flow(t, t, save_results=False)
    inc_raw = model.request_aggregate_output("incidence_raw", timings)
    model.request_function_output("incidence", 1e5 * inc_raw / tot_pop)

    # notification
    notifs = model.request_output_for_flow("notification", "detection")
    model.request_function_output("notification_log", np.log(notifs))

    # notification per incidence:
    model.request_function_output(
        "notification_per_incidence", notifs / DerivedOutput("incidence_raw") * 100.0
    )



