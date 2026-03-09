from shapely.geometry import Point
from typing import TYPE_CHECKING

from examples.geo_testing.agent import Citizen_tool_manager, CitizenState
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=Citizen_tool_manager)
def move_to_safe_zone(agent: "LLMAgent") -> str:
    """
    Move a migrating citizen one step toward the safe zone.
    """

    if agent.state != CitizenState.MIGRATE:
        return "Agent is not migrating."

    # current location
    x = agent.geometry.x
    y = agent.geometry.y

    # safe zone location
    safe_x, safe_y = agent.model.safe_zone

    step_size = 0.5

    new_x = x + step_size if safe_x > x else x - step_size
    new_y = y + step_size if safe_y > y else y - step_size

    agent.geometry = Point(new_x, new_y)

    return f"Agent moved toward safe zone ({safe_x},{safe_y})"