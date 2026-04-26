from __future__ import annotations

from typing import TYPE_CHECKING

from examples.spcae_settlement.agent import martian_tool_manager
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=martian_tool_manager)
def survey_local_sector(agent: "LLMAgent") -> str:
    """Inspect the current sector and settlement status before acting."""

    return agent.model.survey_local_sector(agent)


@tool(tool_manager=martian_tool_manager)
def produce_resource(agent: "LLMAgent", resource: str) -> str:
    """Produce food, water, air, or process waste for the settlement."""

    return agent.model.produce_resource(agent, resource.lower())


@tool(tool_manager=martian_tool_manager)
def mine_minerals(agent: "LLMAgent") -> str:
    """Mine minerals in the current sector and improve settlement technology."""

    return agent.model.mine_minerals(agent)


@tool(tool_manager=martian_tool_manager)
def repair_habitat(agent: "LLMAgent") -> str:
    """Attempt to repair the strongest active habitat accident."""

    return agent.model.repair_habitat(agent)


@tool(tool_manager=martian_tool_manager)
def support_neighbor(agent: "LLMAgent", neighbor_id: int) -> str:
    """Provide morale support to a nearby colonist identified by unique id."""

    return agent.model.support_neighbor(agent, neighbor_id)
