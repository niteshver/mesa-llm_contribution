from enum import Enum
import random

import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager


PATIENT_TOOL_MANAGER = ToolManager()
DOCTOR_TOOL_MANAGER = ToolManager()


class PatientState(Enum):
    SICK = "SICK"
    ADMITED = "ADMITED"


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

        self.specialization = random.choice(
            ["general_physician", "cardiologist", "neurologist"]
        )
        self.experience_level = random.randint(3, 20)
        self.accuracy_rate = random.uniform(0.7, 0.95)
        self.fatigue = 0.0

        self.memory = STLTMemory(agent=self, llm_model=llm_model, display=True)
        self.tool_manager = DOCTOR_TOOL_MANAGER

        self.internal_state.append(
            f"You have a specialization in {self.specialization}."
        )
        self.internal_state.append(
            f"You have {self.experience_level} years of experience."
        )

    def step(self):
        observation = self.generate_obs()

        prompt = f"""
        You are a doctor with specialization {self.specialization},
        experience level {self.experience_level}, and accuracy rate {self.accuracy_rate}.

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
            selected_tools=["admit_patient", "speak_to"],
        )

        self.apply_plan(plan)


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
        self.internal_state.append(f"You are {self.age} years old.")
        self.internal_state.append(f"Your symptoms are {self.symptoms}.")
        self.internal_state.append(f"Your wealth is {self.wealth}.")

    def step(self):
        observation = self.generate_obs()

        prompt = f"""
        You are a patient with age {self.age},
        symptoms {self.symptoms}, and wealth {self.wealth}.

        Decide:
        - Visit hospital or wait
        - Talk to doctor
        - Accept or reject treatment based on cost
        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["visit_hospital", "speak_to"],
        )

        self.apply_plan(plan)


class HospitalAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.capacity = random.randint(10, 50)
        self.queue = []

    def admit(self, patient):
        if len(self.queue) >= self.capacity:
            return "Hospital full"
        if patient not in self.queue:
            self.queue.append(patient)
        return "Patient admitted"
