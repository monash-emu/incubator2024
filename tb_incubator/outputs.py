from summer2 import CompartmentalModel
from typing import List
from summer2.parameters import DerivedOutput
from tb_incubator.constants import latent_compartments, infectious_compartments
import numpy as np

def request_model_outputs(
    model: CompartmentalModel,
):
    compartments = [c.name for c in model._original_compartment_names]

    # compartment sizes
    for c in compartments:
        model.request_output_for_compartments(f"comp_size_{c}", c)
    tot_pop = model.request_output_for_compartments("total_population", compartments)

    # latent percentage
    latent_pop = model.request_output_for_compartments(
        "latent_pop", latent_compartments, save_results=False
    )
    model.request_function_output("percentage_latent", latent_pop / tot_pop * 100.0)

    # death
    deaths = model.request_output_for_flow("mortality_infectious", "TB_death")
    model.request_function_output("mortality", 1e5 * deaths / tot_pop)

    # prevalence
    infect_pop = model.request_output_for_compartments(
        "infect_pop", infectious_compartments, save_results=False
    )
    prevalence = model.request_function_output("prevalence", infect_pop / tot_pop * 1e5)
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



