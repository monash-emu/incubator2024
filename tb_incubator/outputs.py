from summer2 import CompartmentalModel
from typing import List
from summer2.parameters import Parameter, DerivedOutput, Function, Time
from tb_incubator.constants import latent_compartments, infectious_compartments, age_strata
from tb_incubator.utils import tanh_based_scaleup
from summer2.functions.time import get_linear_interpolation_function

def request_model_outputs(
    model: CompartmentalModel,
):
    compartments = [c.name for c in model._original_compartment_names]

    # compartment size
    for c in compartments:
        model.request_output_for_compartments(f"comp_size_{c}", c)

    # calculate and request percentage of latent population
    tot_pop = model.request_output_for_compartments("total_population", compartments, save_results=True)
    latent_pop = model.request_output_for_compartments("latent_pop_size", latent_compartments, save_results=False)
    model.request_function_output("percentage_latent", latent_pop / tot_pop * 100.0)

    # death
    deaths = model.request_output_for_flow("mortality_infectious", "TB_death")
    model.request_function_output("mortality", 1e5 * deaths / tot_pop)

    # total prevalence
    infect_pop = model.request_output_for_compartments("infectious_pop_size", infectious_compartments, save_results=False)
    model.request_function_output("prevalence", infect_pop / tot_pop * 1e5)

    # incidence
    for timing in ["early", "late"]:
        model.request_output_for_flow(
            f"{timing}_activation", f"{timing}_activation", save_results=False
        )

    inc_raw = model.request_aggregate_output(
        "incidence_raw", ["early_activation", "late_activation"], save_results=True
    )

    model.request_function_output("incidence", 1e5 * inc_raw / tot_pop)

    # notification
    notifs = model.request_output_for_flow("notification", "detection")

    # notification per incidence:
    model.request_function_output("notification_per_incidence", notifs / DerivedOutput("incidence_raw") * 100.0)

