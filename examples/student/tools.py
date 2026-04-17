from examples.student.agent import SchoolAgent,StudentAgent,school_tool_manager,student_tool_manager
from examples.student.model import SchoolModel
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.llm_agent import LLMAgent

@tool(tool_manager=student_tool_manager)
def pass_mark(agent: "LLMAgent") ->str:
    """
    """

    
