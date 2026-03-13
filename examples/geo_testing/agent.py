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


class Citizen(GeoAgent,LLMAgent):

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

        GeoAgent.__init__(self, model, geometry, crs)

        LLMAgent.__init__(
            self,
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            step_prompt=step_prompt,
            internal_state=internal_state or [],
        )

        self.state = CitizenState.REST

        self.risk_proneness = self.random.random()
        self.memory_retention = self.random.random()

        self.previous_risk = 0
        self.migration_prob = 0

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3.1:latest",
        )

        self.tool_manager = Citizen_tool_manager

    # ------------------------------

    def distance_to_war(self):

        dists = [
            self.geometry.distance(zone)
            for zone in self.model.war_zones
        ]

        return min(dists)

    # ------------------------------

    def compute_risk(self):

        war_dist = self.distance_to_war()

        conflict_risk = math.exp(-war_dist)

        migrating_neighbors = sum(
            1 for a in self.model.agents
            if isinstance(a, Citizen) and a.state == CitizenState.MIGRATE
        )

        social_risk = migrating_neighbors / max(1, len(self.model.agents))

        perceived_risk = (
            self.risk_proneness * conflict_risk
            + self.memory_retention * self.previous_risk
            + 0.3 * social_risk
        )

        self.previous_risk = perceived_risk

        return perceived_risk

    # ------------------------------

    def update_migration_probability(self):

        risk = self.compute_risk()

        self.migration_prob = 1 / (1 + math.exp(-4 * (risk - 0.3)))

    # ------------------------------

    def apply_migration(self):

        if self.random.random() < self.migration_prob:
            self.state = CitizenState.MIGRATE

    # ------------------------------

    def explain_decision(self):

        observation = self.generate_obs()

        prompt = f"""
You are a civilian living in a conflict region.

Risk level: {self.previous_risk}
Migration probability: {self.migration_prob}
Current state: {self.state}

If risk is high migrate toward the safe zone.
Otherwise wander locally.
Explain briefly.
"""

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["move_to_safe_zone", "wander"],
        )

        self.apply_plan(plan)

    # ------------------------------

    def step(self):

        self.update_migration_probability()

        self.apply_migration()

        self.explain_decision()