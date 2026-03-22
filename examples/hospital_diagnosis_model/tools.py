from typing import TYPE_CHECKING

from examples.hospital_diagnosis_model.agent import (
    DOCTOR_TOOL_MANAGER,
    DoctorAgent,
    HospitalAgent,
    PATIENT_TOOL_MANAGER,
    PatientAgent,
    PatientState,
)
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=PATIENT_TOOL_MANAGER)
def visit_hospital(agent: "LLMAgent") -> str:
    """
    Visit the hospital if the patient is sick.

    Args:
        agent: Provided automatically.

    Returns:
        A status message describing the result.
    """
    if not isinstance(agent, PatientAgent):
        raise TypeError("visit_hospital can only be used by PatientAgent instances.")

    if agent.state != PatientState.SICK:
        return f"Patient {agent.unique_id} is not sick."

    agent.state = PatientState.ADMITED
    return f"Patient {agent.unique_id} went to the hospital and got admitted."


@tool(tool_manager=DOCTOR_TOOL_MANAGER)
def admit_patient(agent: "LLMAgent", patient_id: int) -> str:
    """
    Admit a patient into the hospital queue.

    Args:
        patient_id: The unique id of the patient to admit.
        agent: Provided automatically.

    Returns:
        A status message describing the result.
    """
    if not isinstance(agent, DoctorAgent):
        raise TypeError("admit_patient can only be used by DoctorAgent instances.")

    patient = next(
        (
            model_agent
            for model_agent in agent.model.agents
            if isinstance(model_agent, PatientAgent)
            and model_agent.unique_id == patient_id
        ),
        None,
    )
    if patient is None:
        raise ValueError(f"No patient found with id {patient_id}")

    hospital = next(
        (
            model_agent
            for model_agent in agent.model.agents
            if isinstance(model_agent, HospitalAgent)
        ),
        None,
    )
    if hospital is None:
        raise ValueError("No hospital agent found in the model.")

    result = hospital.admit(patient)
    if result == "Patient admitted":
        patient.state = PatientState.ADMITED

    return f"{result} for patient {patient.unique_id}."
