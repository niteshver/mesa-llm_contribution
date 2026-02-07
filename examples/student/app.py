# app.py
from dotenv import load_dotenv
import os
import warnings
import logging
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.student.agent import StudentAgent, ClusterState
from examples.student.model import StudentModel

from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping

# Suppress pydantic serialization
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message = r".*Pydantic serializer warnings.*"
)
logging.getLogger("pydantic").setLevel(logging.ERROR)
enable_automatic_parallel_stepping(mode="threading")
load_dotenv()
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"


agent_colors = {
    ClusterState.SIMILAR: "#648FFF",
    ClusterState.DISSIMILAR: "#DB28A2",
}


model_parameters = {
    "seed" : {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed"
    },
    "width": 15,
    "height": 10,
    "vision": 6,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/granite4:latest",
}

# -------------------------------------------------
# Create model instance
# -------------------------------------------------
model = StudentModel(
    width=model_parameters["width"],
    height=model_parameters["height"],
    vision=model_parameters["vision"],
    reasoning=model_parameters["reasoning"],
    llm_model=model_parameters["llm_model"],
)


def student_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 50,
        "shape": "circle",
        "color": agent_colors[agent.state],
    }
    return portrayal

space_component = make_space_component(
    student_portrayal,
    draw_grid=False,
)

chart_component = make_plot_component(
    {state.name.lower(): agent_colors[state] for state in ClusterState}
)

if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[
            space_component,
            chart_component,
        ],
        model_params=model_parameters,
        name="Student Clustering Model",
    )
