import math
from enum import Enum

from mesa_geo.geoagent import GeoAgent
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager


Citizen_tool_manager = ToolManager()


class CitizenState(Enum):
    REST = 1
    MIGRATE = 2


class Citizen(GeoAgent, LLMAgent):
    """
    Citizen agent that decides whether to migrate based on perceived risk.
    """

    def __init__(
        self,
        model,
        geometry,
        crs,
        reasoning,
        llm_model,
        system_prompt,
        step_prompt,
        vision,
        internal_state=None,
    ):
        # Initialize GeoAgent
        GeoAgent.__init__(self, model, geometry, crs)

        # Initialize LLM agent
        LLMAgent.__init__(
            self,
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            step_prompt=step_prompt,
            internal_state=internal_state,
        )

        self.state = CitizenState.REST

        self.risk_proneness = self.random.random()
        self.memory_retention = self.random.random()

        self.previous_perceived_risk = 0
        self.migration_prob = 0

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3.1:latest",
            display=False,
        )

        self.tool_manager = Citizen_tool_manager

    # ---------------- Risk calculation ----------------

    def compute_event_impact(self):

        return self.model.intensity_of_event

    def update_migration_probability(self):

        event_impact = self.compute_event_impact()

        perceived_risk = (
            self.risk_proneness * event_impact
            + self.memory_retention * self.previous_perceived_risk
        )

        self.previous_perceived_risk = perceived_risk

        self.migration_prob = 1 / (
            1 + math.exp(-self.model.growth_rate * (perceived_risk - self.model.baseline_Q))
        )

    # ---------------- Migration decision ----------------

    def apply_migration(self):

        if self.random.random() <= self.migration_prob:

            self.state = CitizenState.MIGRATE
            self.model.total_migrants += 1

    # ---------------- LLM reasoning ----------------

    def explain_decision(self):

        observation = self.generate_obs()

        prompt = f"""
        You are a civilian living in a conflict zone.

        Current perceived risk: {self.previous_perceived_risk:.3f}
        Migration probability: {self.migration_prob:.3f}
        Current state: {self.state.name}

        Explain briefly why migration probability is high or low.
        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["move-one_step","move_to_safe_zone"]
        )

        self.apply_plan(plan)

    # ---------------- Agent step ----------------

    def step(self):

        self.update_migration_probability()

        self.apply_migration()

        self.explain_decision()