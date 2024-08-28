import pandas as pd
from tb_incubator.constants import set_project_base_path
from summer2.parameters import Parameter


project_paths = set_project_base_path("../tb_incubator")
data_path = project_paths["DATA_PATH"]

# Add latency structures
def add_latency_flow(model):
    """
    Adds latency flows to the compartmental model, representing the progression of the disease
    through different stages of latency. This function defines three main flows: stabilization,
    early activation, and late activation.

    - Stabilization flow represents the transition of individuals from the 'early_latent' compartment
      to the 'late_latent' compartment, indicating a period where the disease does not progress or show symptoms.

    - Early activation flow represents the transition from 'early_latent' to 'infectious', modeling the
      scenario where the disease becomes active and infectious shortly after the initial infection.

    - Late activation flow represents the transition from 'late_latent' to 'infectious', modeling the
      scenario where the disease becomes active and infectious after a longer period of latency.

    Each flow is defined with a name, a rate (set to 1.0 and will be adjusted based on empirical data or model needs), and the source and destination compartments.

    Args:
        model: The compartmental model to which the latency flows are to be added.
    """
    latency_flows = [
    ["stabilisation", "early latent", "late latent"],
    ["early activation", "early latent", "infectious"],
    ["late activation", "late latent", "infectious"],
    ]
    
    for flow, source, dest in latency_flows:
        model.add_transition_flow(flow, Parameter(f"{flow} rate"), source, dest)

    description= "We added latency flows to the compartmental model, representing the progression of the disease through different stages of latency. This function defines three main flows: stabilization, early activation, and late activation."

    return description