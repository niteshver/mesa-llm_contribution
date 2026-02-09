import random
from typing import TYPE_CHECKING
from mesa_llm.tools.tool_decorator import tool
from examples.student.agent import student_tool_manager
if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

@tool(tool_manager=student_tool_manager)
def move_randomly(agent: "LLMAgent"):
    """
    Move the agent to a random neighboring cell within grid bounds.
    This tool is deterministic and prevents out-of-bounds movement.
    """

    grid = agent.model.grid
    x, y = agent.pos

    neighbors = grid.get_neighborhood(
        (x, y),
        moore=True,
        include_center=False
    )

    if not neighbors:
        return "No valid neighboring cells."

    new_pos = random.choice(neighbors)
    grid.move_agent(agent, new_pos)

    return f"Moved randomly to {new_pos}"
