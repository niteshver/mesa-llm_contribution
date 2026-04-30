from __future__ import annotations
from enum import Enum
import random

from mesa_geo.geoagent import GeoAgent
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager


# -------------------------------
# TOOL MANAGER
# -------------------------------
martian_tool_manager = ToolManager()


# -------------------------------
# ENUMS
# -------------------------------
class Resilience(Enum):
    NEUROTIC = "neurotic"
    REACTIVE = "reactive"
    SOCIAL = "social"
    AGREEABLE = "agreeable"


class StressorType(Enum):
    HABITAT = "Habitat"
    SHIPPING = "Shipping"


# -------------------------------
# STRESSOR AGENT (MESA-GEO)
# -------------------------------
class StressorAgent(GeoAgent):
    def __init__(self, model, geometry, crs, stressor_type: StressorType):
        super().__init__(model, geometry, crs)

        self.type = stressor_type
        self.impact_strength = random.uniform(0.3, 0.6)
        self.base_damage = 5
        self.duration = random.randint(2, 4)

    def step(self):
        neighbors = self.model.space.get_neighbors_within_distance(self, 2)

        for agent in neighbors:
            if isinstance(agent, MartianAgent):

                # distance-based damage
                distance = self.geometry.distance(agent.geometry)
                distance_factor = 1 / (distance + 1)

                damage = (
                    self.impact_strength
                    * self.base_damage
                    * (1 - agent.coping_capacity)
                    * distance_factor
                )

                agent.health = max(0, agent.health - damage)

                # coping reduction
                agent.coping_capacity = max(
                    0.3,
                    agent.coping_capacity - 0.05 * self.impact_strength * distance_factor
                )

        # -----------------------------
        # SETTLEMENT IMPACT
        # -----------------------------
        if self.type == StressorType.HABITAT:
            resource = random.choice(["food", "water", "air"])

            if resource == "food":
                self.model.settlement_food *= (1 - 0.1 * self.impact_strength)
            elif resource == "water":
                self.model.settlement_water *= (1 - 0.1 * self.impact_strength)
            elif resource == "air":
                self.model.settlement_air *= (1 - 0.1 * self.impact_strength)

        elif self.type == StressorType.SHIPPING:
            self.model.shipment_blocked = True

        # -----------------------------
        # DECAY
        # -----------------------------
        self.impact_strength *= 0.8

        # -----------------------------
        # LIFETIME
        # -----------------------------
        self.duration -= 1

        if self.duration <= 0:
            self.model.space.remove_agent(self)


# -------------------------------
# MARTIAN AGENT (LLM + ABM)
# -------------------------------
class MartianAgent(GeoAgent, LLMAgent):

    def __init__(
        self,
        model,
        geometry,
        crs,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        step_prompt,
        resilience: Resilience,
        skill_1: int,
        skill_2: int,
        coping_capacity: float,
        health: float = 100,
        api_base=None,
    ):
        GeoAgent.__init__(self, model, geometry, crs)

        LLMAgent.__init__(
            self,
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=None,
            step_prompt=step_prompt,
            api_base=api_base,
        )

        # -----------------------------
        # CORE ATTRIBUTES
        # -----------------------------
        self.resilience = resilience
        self.skill_1 = skill_1
        self.skill_2 = skill_2
        self.coping_capacity = coping_capacity
        self.health = health
        self.state = "alive"

        # -----------------------------
        # NEEDS (weekly)
        # -----------------------------
        self.food_need = 10.5
        self.water_need = 28
        self.air_need = 5.88

        # -----------------------------
        # SOCIAL
        # -----------------------------
        self.partner_id = None

        # -----------------------------
        # RECOVERY
        # -----------------------------
        self.sleep_recovery = random.uniform(1, 3)

        # -----------------------------
        # MEMORY + TOOLS
        # -----------------------------
        self.memory = STLTMemory(
            agent=self,
            llm_model=llm_model,
            api_base=api_base,
            display=False,
        )

        self.tool_manager = martian_tool_manager

    # -----------------------------
    # PASSIVE RECOVERY
    # -----------------------------
    def passive_recovery(self):
        self.health = min(100, self.health + self.sleep_recovery)

    # -----------------------------
    # RESOURCE CONSUMPTION
    # -----------------------------
    def consume_resources(self):
        # Food
        if self.model.settlement_food >= self.food_need:
            self.model.settlement_food -= self.food_need
        else:
            self.health -= 10

        # Water
        if self.model.settlement_water >= self.water_need:
            self.model.settlement_water -= self.water_need
        else:
            self.health -= 10

        # Air
        if self.model.settlement_air >= self.air_need:
            self.model.settlement_air -= self.air_need
        else:
            self.health -= 10

    # -----------------------------
    # PARTNER SYSTEM
    # -----------------------------
    def find_partner(self):
        neighbors = self.model.space.get_neighbors(self)

        for agent in neighbors:
            if isinstance(agent, MartianAgent) and agent != self:
                self.partner_id = agent.unique_id

                # small coping boost
                self.coping_capacity = min(1.5, self.coping_capacity + 0.05)
                return agent

        return None

    # -----------------------------
    # INTERNAL STATE (LLM)
    # -----------------------------
    def refresh_internal_state(self):
        self.internal_state = [
            f"Health: {self.health:.2f}",
            f"Coping: {self.coping_capacity:.2f}",
            f"Skill: {self.skill_1 + self.skill_2}",
            f"Food: {self.model.settlement_food:.2f}",
            f"State: {self.state}",
        ]

    # -----------------------------
    # LLM DECISION
    # -----------------------------
    def decide(self):
        plan = self.reasoning.plan(
            obs=self.generate_obs(),
            prompt=self.step_prompt,
            selected_tools=[
                "move_one_step",
                "speak_to",
                "die",
                "produce_resource"
            ],
        )

        if plan:
            self.apply_plan(plan)

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self):

        if self.state == "dead":
            return

        # recovery
        self.passive_recovery()

        # social interaction
        self.find_partner()

        # consume resources
        self.consume_resources()

        # death condition
        if self.health <= 0:
            self.state = "dead"
            return

        # update LLM state
        self.refresh_internal_state()

        # decision making
        self.decide()

        # memory update
        self.memory.add(
            f"Health={self.health:.2f}, Coping={self.coping_capacity:.2f}"
        )