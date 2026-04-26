from __future__ import annotations

from enum import Enum

import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager

martian_tool_manager = ToolManager()


class Resilience(Enum):
    NEUROTIC = "neurotic"
    REACTIVE = "reactive"
    SOCIAL = "social"
    AGREEABLE = "agreeable"


class StressorType(Enum):
    SHIPPING = "Shipping"
    HABITAT = "Habitat"


class StressorAgent(mesa.Agent):
    """Simple non-LLM stressor that persists for a few steps."""

    def __init__(
        self,
        model,
        stressor_type: StressorType,
        impacted_resource: str | None,
        impact_strength: float,
        remaining_steps: int = 4,
    ):
        super().__init__(model=model)
        self.stressor_type = stressor_type
        self.impacted_resource = impacted_resource
        self.impact_strength = impact_strength
        self.remaining_steps = remaining_steps
        self.internal_state = [
            f"stressor_type={self.stressor_type.value}",
            f"impacted_resource={self.impacted_resource or 'morale'}",
            f"impact_strength={self.impact_strength:.2f}",
        ]

    def step(self):
        if self.remaining_steps <= 0:
            self.remove()
            return

        self.remaining_steps -= 1
        self.internal_state = [
            f"stressor_type={self.stressor_type.value}",
            f"impacted_resource={self.impacted_resource or 'morale'}",
            f"impact_strength={self.impact_strength:.2f}",
            f"remaining_steps={self.remaining_steps}",
        ]

        if self.remaining_steps <= 0:
            self.remove()


class MartianAgent(LLMAgent):
    """LLM-guided colonist responsible for colony maintenance and survival."""

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        resilience: Resilience,
        coping_capacity: float,
        skill_1: int,
        skill_2: int,
        health: float = 100.0,
        weekly_food_need: float = 10.5,
        weekly_water_need: float = 28.0,
        weekly_air_need: float = 5.88,
        weekly_waste_output: float = 1.0,
        api_base: str | None = None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
            api_base=api_base,
        )
        self.resilience = resilience
        self.coping_capacity = coping_capacity
        self.skill_1 = skill_1
        self.skill_2 = skill_2
        self.health = health
        self.weekly_food_need = weekly_food_need
        self.weekly_water_need = weekly_water_need
        self.weekly_air_need = weekly_air_need
        self.weekly_waste_output = weekly_waste_output
        self.sleep_recovery = self.model.random.uniform(1.0, 3.5)
        self.current_task = "idle"
        self.last_action_summary = "No action taken yet."
        self.partner_id: int | None = None

        self.memory = STLTMemory(
            agent=self,
            llm_model=llm_model,
            api_base=api_base,
            display=True,
        )
        self.tool_manager = martian_tool_manager
        self.refresh_internal_state()

    def refresh_internal_state(self):
        sector_snapshot = self.model.get_cell_snapshot(self.pos)
        settlement_snapshot = self.model.format_resource_snapshot()
        active_stressors = self.model.describe_active_stressors()

        self.internal_state = [
            f"I am Martian {self.unique_id} with resilience profile {self.resilience.value}.",
            f"My health is {self.health:.1f}/100 and my coping capacity is {self.coping_capacity:.2f}.",
            f"My skill vector is ({self.skill_1}, {self.skill_2}).",
            f"My current task is {self.current_task}.",
            f"Settlement reserves: {settlement_snapshot}.",
            f"Current cell production capacities: {sector_snapshot}.",
            f"Active stressors: {active_stressors}.",
            f"My most recent action summary: {self.last_action_summary}.",
        ]

    def passive_recovery(self):
        self.health = min(100.0, self.health + self.sleep_recovery)
        self.coping_capacity = min(
            1.25,
            self.coping_capacity
            + self.model.resilience_recovery_bonus[self.resilience.value],
        )

    def apply_stress_load(self):
        pressure = self.model.get_psychological_pressure(self)
        self.coping_capacity = max(0.05, self.coping_capacity - pressure)

    def step(self):
        if self.health <= 0:
            self.model.remove_martian(self, reason="health_depleted")
            return

        self.current_task = "idle"
        self.partner_id = None
        self.passive_recovery()
        self.apply_stress_load()
        self.refresh_internal_state()

        observation = self.generate_obs()
        plan = self.reasoning.plan(
            obs=observation,
            prompt=self.step_prompt,
            selected_tools=[
                "move_one_step",
                "survey_local_sector",
                "produce_resource",
                "mine_minerals",
                "repair_habitat",
                "support_neighbor",
            ],
        )
        tool_results = self.apply_plan(plan)

        if tool_results:
            self.last_action_summary = " | ".join(
                result["response"] for result in tool_results
            )
        else:
            self.last_action_summary = (
                getattr(plan.llm_plan, "content", None) or "Observed and waited."
            )

        self.refresh_internal_state()
