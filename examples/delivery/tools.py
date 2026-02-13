import random
from typing import TYPE_CHECKING

from examples.delivery.agent import(
    customer,
    DeliveryAgent,
    DELIVERYAGENTSTATE,
    delivery_tool_manager,
    customer_tool_manager,
    PRODUCTSTATE
)

from mesa_llm.llm_agent import LLMAgent

@tool(tool_manager=delivery_tool_manager)
def deliver_the_product(agent: "LLMAgent", state: str) -> str:
    max_delivery_time = 10
    if agent.spent_time <= max_delivery_time:
        PRODUCTSTATE.DELIVERED()
        DeliveryAgent.earning += 10

@tool(tool_manager=customer_tool_manager)
def received_the_product(agent: "LLMAgent", state: str) -> str:
    


