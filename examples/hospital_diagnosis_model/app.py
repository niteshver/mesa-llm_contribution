import logging
import warnings

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.hospital_diagnosis_model.agent import (
    DoctorAgent,
    PatientAgent,
    PatientState,
)
from examples.hospital_diagnosis_model.model import HospitalModel
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=r".*Pydantic serializer warnings.*",
)

logging.getLogger("pydantic").setLevel(logging.ERROR)

enable_automatic_parallel_stepping(mode="threading")

load_dotenv()

DOCTOR_COLOR = "#1F1F1F"
PATIENT_AGENT_COLOR = {
    PatientState.SICK: "#0DFE00",
    PatientState.ADMITED: "#FE0000",
}

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_patients": 20,
    "initial_doctors": 5,
    "width": 50,
    "height": 50,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3.1:latest",
    "vision": 5,
    "parallel_stepping": True,
}

model = HospitalModel(
    initial_patients=model_params["initial_patients"],
    initial_doctors=model_params["initial_doctors"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"]["value"],
    parallel_stepping=model_params["parallel_stepping"],
)


def patient_portrayal(agent):
    if agent is None:
        return None

    portrayal = {"size": 50}

    if isinstance(agent, DoctorAgent):
        portrayal["color"] = DOCTOR_COLOR
    elif isinstance(agent, PatientAgent):
        portrayal["color"] = PATIENT_AGENT_COLOR[agent.state]

    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)


space_component = make_space_component(
    patient_portrayal, post_process=post_process, draw_grid=False
)

chart_component = make_plot_component(
    {state.name.lower(): PATIENT_AGENT_COLOR[state] for state in PatientState}
)

if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[space_component, chart_component],
        model_params=model_params,
        name="Hospital Diagnosis Model",
    )
