from typing import TYPE_CHECKING

from examples.spcae_settlement.agent import (
    martian_tool_manager,
    MartianAgent,
)
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=martian_tool_manager)
def die(agent: "LLMAgent") -> str:
    """
    Remove the agent from the simulation due to death.

    This tool is triggered when the agent's health reaches zero
    or survival is no longer possible.

    Effects:
    - Sets agent health to 0
    - Removes agent from the model

    Args:
        agent: The Martian agent to be removed.

    Returns:
        A confirmation message including the agent ID.
    """
    agent.health = 0
    agent.model.remove_agent(agent)

    return f"Agent {agent.unique_id} has died and was removed."


@tool(tool_manager=martian_tool_manager)
def produce_resource(agent: "LLMAgent") -> str:
    """
    Produce essential resources for the settlement.

    The agent may collaborate with a partner to increase production efficiency.
    Total production depends on the combined skill levels.

    Logic:
    - If total skill >= threshold → successful production
    - Otherwise → production fails and health penalty applies

    Effects on success:
    - Increases settlement food, water, and air
    - Improves coping capacity

    Effects on failure:
    - Reduces agent health

    Args:
        agent: The Martian agent performing the production action.

    Returns:
        A message indicating success or failure of production.
    """
    partner = agent.find_partner()

    skill_total = agent.skill_1
    if partner:
        skill_total += partner.skill_2

    threshold = 100

    if skill_total >= threshold:
        agent.model.settlement_food += 10
        agent.model.settlement_water += 15
        agent.model.settlement_air += 5

        agent.coping_capacity = min(1.5, agent.coping_capacity + 0.05)

        return (
            f"Production successful. Resources increased "
            f"(Food +10, Water +15, Air +5)."
        )
    else:
        agent.health -= 5
        return "Production failed. Health decreased by 5."