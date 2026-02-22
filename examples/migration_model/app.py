import logging
import warnings
import solara
from dotenv import load_dotenv

from mesa.visualization import (
    SolaraViz,
    make_space_component,
    make_plot_component,
)

from examples.migration_model.agent import Citizen, CitizenState
from examples.migration_model.model import MigrationModel
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning



warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=r".*Pydantic serializer warnings.*",)

logging.getLogger("pydantic").setLevel(logging.ERROR)

enable_automatic_parallel_stepping(mode="threading")
load_dotenv()



agent_colors = {
    CitizenState.REST: "#2600FE",
    CitizenState.MIGRATE: "#DB2E28FB",
}



model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "citizen": 20,
    "width": 10,
    "height": 10,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3.1",
    "vision": 5,
    "parallel_stepping": True,
    
}

model = MigrationModel(
    citizen=model_params["citizen"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"]["value"],
    parallel_stepping=model_params["parallel_stepping"],
)




def citizen_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "shape": "circle",
        "filled": True,
        "size": 40,
        "layer": 1,
    }

    if isinstance(agent, Citizen):
        portrayal["color"] = agent_colors[agent.state]

    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)


space_component = make_space_component(
    citizen_portrayal,
    post_process=post_process,
    draw_grid=False,
)

chart_component = make_plot_component(
    {state.name.lower(): agent_colors[state] for state in CitizenState}
)



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