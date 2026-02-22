import random
from examples.migration_model.agent import CitizenState,Citizen,CITIZEN_TOOL_MANAGER
from examples.migration_model.model import MigrationModel
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_decorator import tool


@tool(tool_manager=CITIZEN_TOOL_MANAGER)
def migrate(agent: "Citizen") -> str:
    """
    Migrate to safe zone.
    Removes agent from grid and moves to model.safe_zone.
    """

    if agent.state == CitizenState.MIGRATE:
        return "Already migrated."

    agent.state = CitizenState.MIGRATE

    # Remove from grid
    agent.model.grid.remove_agent(agent)

    # Remove from scheduler
    agent.remove()

    # Add to safe zone
    agent.model.safe_zone.append(agent)

    # Update counters
    agent.model.daily_migrants += 1
    agent.model.total_migrants += 1

    return f"Citizen {agent.unique_id} migrated to safe zone."           # later add stage changing details





    



