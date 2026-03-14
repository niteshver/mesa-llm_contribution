import random
from typing import TYPE_CHECKING

from examples.ev_model.agent import(
    AgentState,
    ChargingStationAgent,
    HouseholdAgent,
    HOUSEHOLD_TOOL_MANAGER
)
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

from mesa_llm.tools.tool_decorator import tool


from mesa_llm.tools.tool_decorator import tool
from examples.ev_model.agent import AgentState, HOUSEHOLD_TOOL_MANAGER


@tool(tool_manager=HOUSEHOLD_TOOL_MANAGER)
def purchase_vehicle(agent):
    """
    Decide whether to purchase an EV or an ICE vehicle based on utility.

    Args:
        agent: HouseholdAgent making the purchase decision.

    Returns:
        str: Description of the purchase decision.
    """

    if agent.state != AgentState.NONE_HOLDER:
        return "Agent already owns a vehicle."

    if agent.utility_ev > agent.utility_ice:
        agent.state = AgentState.EV_HOLDER
        agent.battery_level = agent.battery_capacity
        return "Agent purchased an EV."

    else:
        agent.state = AgentState.ICE_HOLDER
        return "Agent purchased an ICE vehicle."
    

@tool(tool_manager=HOUSEHOLD_TOOL_MANAGER)
def charge_ev(agent):
    """
    Charge the EV battery at the nearest charging station.

    Args:
        agent: HouseholdAgent owning an EV.

    Returns:
        str: Charging result and cost.
    """

    if agent.state != AgentState.EV_HOLDER:
        return "Agent does not own an EV."

    station = agent.find_nearest_station()

    if station.utilization_rate >= station.capacity:
        return "Charging station is full."

    station.utilization_rate += 1

    energy_needed = agent.battery_capacity - agent.battery_level
    cost = energy_needed * station.price_per_kwh

    agent.battery_level = agent.battery_capacity

    station.utilization_rate -= 1

    return f"EV charged. Cost: {cost}"