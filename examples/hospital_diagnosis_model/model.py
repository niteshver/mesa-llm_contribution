# model.py

import random
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model

from examples.hospital_diagnosis_model.agent import (
    PatientAgent,
    DoctorAgent,
    HospitalAgent,
    PatientState,
)


class HospitalModel(Model):
    def __init__(
        self,
        initial_patients: int = 20,
        initial_doctors: int = 5,
        width: int = 20,
        height: int = 20,
        reasoning: type[Reasoning] = None,
        llm_model: str = "gpt-4o-mini",
        vision: int = 3,
        seed=None,
    ):
        super().__init__(seed=seed)

        # --- Grid ---
        self.grid = MultiGrid(width, height, torus=False)

        # --- Parameters ---
        self.initial_patients = initial_patients
        self.initial_doctors = initial_doctors

        self.symptom_list = [
            "fever", "cough", "fatigue", "headache",
            "chest_pain", "shortness_of_breath", "vomiting"
        ]

        self.disease_list = [
            "flu", "infection", "covid", "pneumonia"
        ]

        # --- Hospital ---
        self.hospital = HospitalAgent(self)
        self.grid.place_agent(self.hospital, (width // 2, height // 2))

        # --- Create Doctors ---
        for _ in range(self.initial_doctors):
            doctor = DoctorAgent(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt="You are a doctor.",
                step_prompt="Decide diagnosis and treatment.",
                vision=vision,
            )

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(doctor, (x, y))

        # --- Create Patients ---
        for _ in range(self.initial_patients):
            patient = PatientAgent(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt="You are a patient.",
                step_prompt="Decide what to do.",
                vision=vision,
            )

            # Assign disease
            patient.disease = random.choice(self.disease_list)
            patient.state = PatientState.SICK

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(patient, (x, y))

        # --- Data Collector ---
        self.datacollector = DataCollector(
            model_reporters={
                "Sick": self.count_sick,
                "Recovered": self.count_recovered,
                "Dead": self.count_dead,
            }
        )

        self.running = True

    # -----------------------------
    # 📊 Metrics
    # -----------------------------
    def count_sick(self):
        return sum(
            1 for agent in self.agents if isinstance(agent, PatientAgent)
            and agent.state == PatientState.SICK
        )

    def count_recovered(self):
        return sum(
            1 for agent in self.agents if isinstance(agent, PatientAgent)
            and agent.state == PatientState.RECOVERED
        )

    def count_dead(self):
        return sum(
            1 for agent in self.agents if isinstance(agent, PatientAgent)
            and agent.state == PatientState.DEATH
        )

    # -----------------------------
    # 🔁 Step Function
    # -----------------------------
    def step(self):
        """
        Main simulation loop:
        1. Patients act (decide hospital visit)
        2. Doctors act (diagnose & treat)
        3. Collect data
        """

        # Patients first (decision making)
        self.agents_by_type[PatientAgent].shuffle_do("step")

        # Doctors second (diagnosis)
        self.agents_by_type[DoctorAgent].shuffle_do("step")

        # Collect metrics
        self.datacollector.collect(self)

        # Optional stop condition
        if self.count_sick() == 0:
            self.running = False