import logging
import warnings

from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.geo_testing.agent import Citizen, CitizenState
from examples.geo_testing.model import MigrationModel
from mesa_llm.reasoning.react import ReActReasoning


warnings.filterwarnings("ignore")


agent_colors = {
    CitizenState.REST: "#648FFF",
    CitizenState.MIGRATE: "#FE6100",
}


def agent_portrayal(agent):

    if isinstance(agent, Citizen):

        return {
            "shape": "circle",
            "filled": True,
            "color": agent_colors[agent.state],
            "size": 6,
        }


model_params = {
    "citizen": 40,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3.1:latest",
    "vision": 5,
    "seed": 42,
}


space_component = make_space_component(agent_portrayal)

chart_component = make_plot_component(
    {
        "migrating": "red",
    }
)


model = MigrationModel(
    citizen=model_params["citizen"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"],
)


page = SolaraViz(
    model,
    components=[
        space_component,
        chart_component,
    ],
    model_params=model_params,
    name="Conflict-Driven Migration Model",
)