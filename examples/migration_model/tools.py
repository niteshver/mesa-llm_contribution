import random
from examples.migration_model.agent import CitizenState,Citizen,CITIZEN_TOOL_MANAGER
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_decorator import tool


@tool(tool_manager=CITIZEN_TOOL_MANAGER)
def migrate(agent: "LLMAgent", state: str) -> str:

    """
    If risk is high then the migrate 
    chance is high otherwise not

    """
    state_map = {
        "REST" : CitizenState.REST,
        "MIGRATE" : CitizenState.MIGRATE
    }
    if state not in state_map:
        raise ValueError(f"Invalid State{state}")
    agent.state = state_map[state]
    CitizenState.state = CitizenState.MIGRATE             # later add stage changing details





    



