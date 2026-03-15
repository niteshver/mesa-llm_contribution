import math
from enum import Enum
import random
import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager


PATIENT_TOOL_MANAGER = ToolManager()
DOCTOR_TOOL_MANAGER = ToolManager()


class PatientState(Enum):
    HEALTHY = "HEALTHY"
    SICK = "SICK"
    RECOVERED = "RECOVERED"
    DEATH = "DEATH"

class DoctorAgent(LLMAgent):

    def __init__(self, model, reasoning, llm_model, system_prompt, step_prompt, vision):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            step_prompt=step_prompt,
            vision=vision,
        )

        self.specialization = random.choice([
            "general_physician", "cardiologist", "neurologist"
        ])
        self.experience_level = random.randint(3, 20)
        self.accuracy_rate = random.uniform(0.7, 0.95)
        self.fatigue = 0.0

        self.memory = STLTMemory(agent=self, llm_model=llm_model, display=True)
        self.tool_manager = DOCTOR_TOOL_MANAGER

        self.internal_state.append(
            f"You have a speacialization is {self.specialization}"
        )
        self.internal_state.append(
            f"You have an experience  is{self.experience_level}"
        )

    def step(self):

        observation = self.generate_obs()

        prompt = f"""
        You are a doctor with specialization {self.specialization},
        experience level is {self.experience_level} and accuracy_rate is {self.accuracy_rate}

        Decide:
        - Diagnose disease
        - Ask for tests
        - Admit patient
        - Give treatment

        Use tools when required.
        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[
                "diagnose",
                "prescribe_test",
                "treat_patient",
                "admit_patient",
                "speak_to_patient"
            ],
        )

        self.execute_plan(plan)

class PatientAgent(LLMAgent):

    def __init__(self, model, reasoning, llm_model, system_prompt, step_prompt, vision):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            step_prompt=step_prompt,
            vision=vision,
        )

        self.age = random.randint(1, 80)
        self.symptoms = random.sample(model.symptom_list, k=2)
        self.state = PatientState.SICK
        self.wealth = random.randint(5000, 50000)

        self.memory = STLTMemory(agent=self, llm_model=llm_model, display=True)
        self.tool_manager = PATIENT_TOOL_MANAGER
        self.internal_state.append(
            f"You have an age{self.age}"
        )
        self.internal_state.append(
            f"You have an symptom {self.symptoms}"
        )
        self.internal_state.append(
            f"You have an wealth{self.wealth}"
        )


    def step(self):

        observation = self.generate_obs()
       
        prompt = f"""
        You are a patient having an age{self.age},
        symtoms is {self.symptoms} and wealth is {self.wealth}.

        Decide:
        - Visit hospital or wait
        - Talk to doctor
        - Accept or reject treatment (based on cost)

        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[
                "visit_hospital",
                "speak_to_doctor",
                "accept_treatment",
                "reject_treatment"
            ],
        )

        self.execute_plan(plan)

class HospitalAgent(mesa.Agent):

    def __init__(self, model):
        super().__init__(model)

        self.capacity = random.randint(10, 50)
        self.doctors_available = random.randint(5, 20)
        self.queue = []

    def admit(self, patient):
        if len(self.queue) < self.capacity:
            self.queue.append(patient)
            return "Patient admitted"
        return "Hospital full"