from shapely.geometry import Point
from mesa_llm.tools.tool_decorator import tool

from examples.geo_testing.agent import (
    Citizen_tool_manager,
    CitizenState,
)


@tool(tool_manager=Citizen_tool_manager)
def move_to_safe_zone(agent):
    """
    Move the citizen agent one step toward the safe zone.

    Args:
        agent: The citizen agent performing the action.

    Returns:
        Description of the movement.
    """

    if agent.state != CitizenState.MIGRATE:
        return "Agent not migrating."

    safe = agent.model.safe_zone

    x, y = agent.geometry.x, agent.geometry.y

    dx = safe.x - x
    dy = safe.y - y

    step = 0.01

    new_x = x + step * (dx / (abs(dx) + 1e-6))
    new_y = y + step * (dy / (abs(dy) + 1e-6))

    agent.geometry = Point(new_x, new_y)

    return "Agent moved toward safe zone."


@tool(tool_manager=Citizen_tool_manager)
def wander(agent):
    """
    Move the citizen randomly in the environment.

    Args:
        agent: The citizen agent performing the action.

    Returns:
        Description of the movement.
    """

    x, y = agent.geometry.x, agent.geometry.y

    step = 0.01

    new_x = x + agent.random.uniform(-step, step)
    new_y = y + agent.random.uniform(-step, step)

    agent.geometry = Point(new_x, new_y)

    return "Agent wandered randomly."