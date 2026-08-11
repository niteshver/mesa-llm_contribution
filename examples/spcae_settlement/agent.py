
import random
from enum import Enum

from mesa_geo.geoagent import GeoAgent
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.reasoning.reasoning import Observation
from mesa_llm.tools.tool_manager import ToolManager


martian_tool_manager = ToolManager()


class Resilience(Enum):
    NEUROTIC = "neurotic"
    REACTIVE = "reactive"
    SOCIAL = "social"
    AGREEABLE = "agreeable"

class MartianAgentState(Enum):
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    DEAD = "Dead"


class StressorState(Enum):
    HABITAT = "Habitat"
    SHIPPING = "Shipping"


class StressorAgent(GeoAgent):
    def __init__(self, model, geometry, crs, stressor_type: StressorState):
        super().__init__(model, geometry, crs)
        self.type = stressor_type
        self.impact_strength = random.uniform(0.3, 0.6)
        self.base_damage = 5
        self.duration = random.randint(2, 4)
        self.active = True
        self.state = StressorState.HABITAT

    def step(self):
        if not self.active:
            return

        for agent in self.model.iter_martians(
            origin=self.geometry,
            within=self.model.stressor_radius,
        ):
            distance = self.geometry.distance(agent.geometry)
            distance_factor = 1 / (distance + 1)
            damage = (
                self.impact_strength
                * self.base_damage
                * (1 - agent.coping_capacity)
                * distance_factor
            )

            agent.health = max(0, agent.health - damage)
            agent.coping_capacity = max(
                0.3,
                agent.coping_capacity - 0.05 * self.impact_strength * distance_factor,
            )

        if self.type == StressorState.HABITAT:
            resource = random.choice(["food", "water", "air"])
            if resource == "food":
                self.model.settlement_food *= 1 - 0.1 * self.impact_strength
            elif resource == "water":
                self.model.settlement_water *= 1 - 0.1 * self.impact_strength
            else:
                self.model.settlement_air *= 1 - 0.1 * self.impact_strength
        else:
            self.model.shipment_blocked = True

        self.impact_strength *= 0.8
        self.duration -= 1
        if self.duration <= 0:
            self.active = False
            self.state = "inactive"
            self.model.retire_geo_agent(self)


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
        # GeoAgent already performs Mesa agent registration, so we avoid
        # calling LLMAgent.__init__ and instead populate the required fields.
        GeoAgent.__init__(self, model, geometry, crs)

        self.model = model
        self.step_prompt = step_prompt
        self.llm = ModuleLLM(
            llm_model=llm_model,
            system_prompt=system_prompt,
            api_base=api_base,
        )
        self.vision = vision
        self.reasoning = reasoning(agent=self)
        self._current_plan = None
        self._step_display_data = {}
        self.recorder = None
        self.internal_state = []

        self.resilience = resilience
        self.skill_1 = skill_1
        self.skill_2 = skill_2
        self.coping_capacity = coping_capacity
        self.health = health
        self.state = MartianAgentState.ACTIVE
        self.partner_id = None
        self.sleep_recovery = random.uniform(1, 3)
        self.resources_produced = 0

        self.food_need = 10.5
        self.water_need = 28
        self.air_need = 5.88

        self.memory = STLTMemory(
            agent=self,
            llm_model=llm_model,
            api_base=api_base,
            display=False,
        )
        self.tool_manager = martian_tool_manager

    def _coords(self) -> tuple[float, float]:
        return (self.geometry.x, self.geometry.y)

    def _neighboring_martians(self):
        radius = self.vision if self.vision and self.vision > 0 else None
        return self.model.iter_martians(
            origin=self.geometry,
            within=radius,
            exclude=self,
        )

    def passive_recovery(self):
        self.health = min(100, self.health + self.sleep_recovery)

    def consume_resources(self):
        if self.model.settlement_food >= self.food_need:
            self.model.settlement_food -= self.food_need
        else:
            self.health -= 10

        if self.model.settlement_water >= self.water_need:
            self.model.settlement_water -= self.water_need
        else:
            self.health -= 10

        if self.model.settlement_air >= self.air_need:
            self.model.settlement_air -= self.air_need
        else:
            self.health -= 10

    def find_partner(self):
        for agent in self._neighboring_martians():
            self.partner_id = agent.unique_id
            self.coping_capacity = min(1.5, self.coping_capacity + 0.05)
            return agent
        self.partner_id = None
        return None

    def refresh_internal_state(self):
        self.internal_state = [
            f"Health: {self.health:.2f}",
            f"Coping: {self.coping_capacity:.2f}",
            f"Combined skill: {self.skill_1 + self.skill_2}",
            f"Food reserve: {self.model.settlement_food:.2f}",
            f"Water reserve: {self.model.settlement_water:.2f}",
            f"Air reserve: {self.model.settlement_air:.2f}",
            f"Position: {self._coords()}",
            f"State: {self.state}",
        ]

    def generate_obs(self) -> Observation:
        self_state = {
            "agent_unique_id": self.unique_id,
            "location": self._coords(),
            "internal_state": self.internal_state,
        }
        local_state = {
            f"{agent.__class__.__name__} {agent.unique_id}": {
                "position": (agent.geometry.x, agent.geometry.y),
                "internal_state": getattr(agent, "internal_state", []),
            }
            for agent in self._neighboring_martians()
        }
        self.memory.add_to_memory(
            type="observation",
            content={"self_state": self_state, "local_state": local_state},
        )
        return Observation(
            step=self.model.steps,
            self_state=self_state,
            local_state=local_state,
        )

    def decide(self):
        MARTIAN_PROMPT = (
        "You are a Martian colonist in a harsh environment with limited resources. "
        "Your main goal is to survive and help sustain the colony. "

        "You must manage food, water, air, and your health. "
        "If resources are low, prioritize producing resources. "
        "If a stressor (accident or damage) occurs, prioritize repair. "
        "If another agent is nearby, consider cooperation. "

        "Guidelines:\n"
        "- Survival comes first (low health or resources → act immediately)\n"
        "- Use produce_resource when resources are low\n"
        "- Use speak_to to cooperate with nearby agents\n"
        "- Use move_one_step only when needed\n"
       
        "Think logically and choose the best action based on current state.\n"
    )
        plan = self.reasoning.plan(
            obs=self.generate_obs(),
            prompt=MARTIAN_PROMPT,
            selected_tools=["speak_to", "die", "produce_resource"],
        )
        if plan:
            self.apply_plan(plan)

    def step(self):
        if self.health <= 0:
            self.state = MartianAgentState.DEAD
            self.model.remove_agent(self)
            return

        self.passive_recovery()
        self.find_partner()
        self.consume_resources()

        
        self.refresh_internal_state()
        self.decide()
        self.memory.add_to_memory(
            type="reflection",
            content={
                "summary": (
                    f"health={self.health:.2f}, coping={self.coping_capacity:.2f}"
                )
            },
        )
