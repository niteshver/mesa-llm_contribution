import logging
import warnings

from dotenv import load_dotenv

from mesa.visualization import SolaraViz, make_plot_component
from mesa_geo.visualization import make_geospace_component

from examples.geo_testing.agent import Citizen, CitizenState
from examples.geo_testing.model import MigrationModel

from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
)

logging.getLogger("pydantic").setLevel(logging.ERROR)

enable_automatic_parallel_stepping(mode="threading")

load_dotenv()

# ---------------- Agent Colors ----------------
agent_colors = {
    CitizenState.REST: "#648FFF",
    CitizenState.MIGRATE: "#FE6100",
}

# ---------------- Model Params ----------------
model_params = {
    "seed": 42,
    "citizen": 20,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3.1:latest",
    "vision": 5,
}

# ---------------- Initialize Model ----------------
model = MigrationModel(
    citizen=model_params["citizen"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"],
)

# ---------------- Agent Portrayal ----------------
def citizen_portrayal(agent):

    portrayal = {}

    if isinstance(agent, Citizen):

        portrayal["color"] = agent_colors[agent.state]
        portrayal["radius"] = 3

    return portrayal

# ---------------- Map Component ----------------
space_component = make_geospace_component(
    citizen_portrayal,
    zoom=2,
)

# ---------------- Charts ----------------
chart_component = make_plot_component(
    {
        "rest": agent_colors[CitizenState.REST],
        "migrate": agent_colors[CitizenState.MIGRATE],
    }
)

# ---------------- Run UI ----------------
if __name__ == "__main__":

    page = SolaraViz(
        model,
        components=[
            space_component,
            chart_component,
        ],
        model_params=model_params,
        name="Conflict-Driven Migration Model",
    )

    page