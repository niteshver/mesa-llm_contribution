from typing import TYPE_CHECKING


from examples.migration_model.agent import Citizen, CitizenState,CITIZEN_TOOL_MANAGER
from examples.migration_model.model import MigrationModel
from mesa_llm.tools.tool_decorator import tool
if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

@tool(tool_manager=CITIZEN_TOOL_MANAGER)
def move_to_safe_place(agent: "LLMAgent", state: str) -> str:
    """
    Migrate citizen to safe zone if probability condition is satisfied.

    Args:
        agent: The Citizen agent performing migration.

    Returns:
        A message describing the migration result.
    """

    if agent.state == CitizenState.MIGRATE:
        return "Already migrated."
    agent.model.grid.remove_agent(agent)
    agent.remove()

    agent.model.safe_zone.append(agent)
    agent.model.daily_migrants += 1
    agent.model.total_migrants += 1

    return f"Citizen {agent.unique_id} migrated."

  

    

        