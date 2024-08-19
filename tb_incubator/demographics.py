from summer2 import CompartmentalModel, flows
from summer2.model import _validate_flowparam
from summer2.adjust import FlowParam
from typing import Dict, Optional

def add_extra_crude_birth_flow(
    model: CompartmentalModel,
    name: str,
    birth_rate: FlowParam,
    dest: str,
    dest_strata: Optional[Dict[str, str]] = None,
    expected_flow_count: Optional[int] = None,
):
    """Identical to summer2's add_crude_birth_flow,
    except with checking that no other birth flows have been implemented
    turned off.

    Args:
        name: The name of the new flow.
        birth_rate: The fractional crude birth rate per timestep.
        dest: The name of the destination compartment.
        dest_strata (optional): A whitelist of strata to filter the destination compartments.
        expected_flow_count (optional): Used to assert that a particular number of flows
                                        are created.
    """
    _validate_flowparam(birth_rate)
    model._add_entry_flow(
        flows.CrudeBirthFlow,
        name,
        birth_rate,
        dest,
        dest_strata,
        expected_flow_count,
    )
