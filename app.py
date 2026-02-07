from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_space_component,
    make_plot_component
)
from examples.student.agent import StudentAgent,Strategy
from examples.student.model import StudentModel
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.reasoning import Reasoning
load_dotenv()
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping

enable_automatic_parallel_stepping(mode="threading")
import os
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"


STUDENT_COLOR = "#000000"

agent_colors = {
    Strategy.PRACTICE:  "#FE6100",
    Strategy.REVISE: "#DB28A2",
    Strategy.REST:  "#648FFF"
}

model_parameter = {
    "exam_day": 10,
    "current_day": 1,
    "pass_mark": 40,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/granite4:latest",
}


model = StudentModel(
    exam_day=model_parameter["exam_day"],
    current_day=model_parameter["current_day"],
    pass_mark=model_parameter["pass_mark"],
    reasoning=model_parameter["reasoning"],
    llm_model=model_parameter["llm_model"],
   
)

chart = make_plot_component({
    "practice": agent_colors[Strategy.PRACTICE],
    "revise": agent_colors[Strategy.REVISE],
    "rest": agent_colors[Strategy.REST],
})


if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[
            chart
        ],  
        model_params=model_parameter,
        name="Student",
    )