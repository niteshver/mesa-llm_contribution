import logging
import warnings
from pathlib import Path

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.spcae_settlement.agent import MartianAgent, StressorAgent
from examples.spcae_settlement.model import MarsSettlementModel, MarsSiteAgent
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning

try:
    from mesa_geo.visualization import make_geospace_component
except ImportError:  # pragma: no cover
    make_geospace_component = None

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=r".*Pydantic serializer warnings.*",
)
logging.getLogger("pydantic").setLevel(logging.ERROR)

enable_automatic_parallel_stepping(mode="threading")
load_dotenv()

DATA_PATH = Path(__file__).with_name("data") / "MARS_nomenclature_center_pts.shp"

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_martians": 12,
    "width": 50,
    "height": 50,
    "reasoning": ReActReasoning,
    "llm_model": "ollama_chat/llama3.1:latest",
    "vision": 3,
    "api_base": None,
    "data_path": str(DATA_PATH),
}

model = MarsSettlementModel(
    initial_martians=model_params["initial_martians"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    api_base=model_params["api_base"],
    data_path=model_params["data_path"],
    seed=model_params["seed"]["value"],
)


def settlement_portrayal(agent):
    if agent is None:
        return None

    portrayal = {"size": 30}

    if isinstance(agent, MartianAgent):
        portrayal["color"] = "#d1495b"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 3
    elif isinstance(agent, StressorAgent):
        portrayal["color"] = "#2f1b25"
        portrayal["marker"] = "X"
        portrayal["size"] = 45
        portrayal["zorder"] = 4

    return portrayal


def mars_geo_portrayal(agent):
    if agent is None:
        return None

    if MarsSiteAgent is not None and isinstance(agent, MarsSiteAgent):
        return {
            "radius": 2,
            "color": "#7f5539",
            "opacity": 0.5,
        }

    return None


space_component = make_space_component(settlement_portrayal)
chart_component = make_plot_component(
    [
        "Population",
        "Active_Stressors",
        "Average_Health",
        "Average_Coping",
    ]
)

components = [space_component, chart_component]
if make_geospace_component is not None and model.space is not None:
    components.insert(
        0,
        make_geospace_component(
            mars_geo_portrayal,
            zoom=2,
            scroll_wheel_zoom=False,
        ),
    )

if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=components,
        model_params=model_params,
        name="Mars Settlement Model",
    )


"""run with:
cd examples/spcae_settlement
solara run app.py
"""
