  
import math
from enum import Enum

import mesa
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager
import litellm 
litellm._turn_on_debug()
from litellm import completion


CITIZEN_TOOL_MANAGER = ToolManager()


class CitizenState(Enum):
    REST = 1
    MIGRATE = 2


class Citizen(LLMAgent, mesa.discrete_space.CellAgent):
    """
    Conflict-driven migration agent.

    Migration decision is fully mathematical.
    LLM is used ONLY to explain the decision.
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        step_prompt,
        vision,
        internal_state=None,

        
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            step_prompt=step_prompt,
            internal_state=internal_state,
            vision=vision,
        )

        self.household_id = self.random.randint(0, self.model.num_households - 1)
        self.risk_proneness = self.random.uniform(0.5, 1.5)
        self.distance = self.random.uniform(0, 1)
        self.time_difference = self.random.uniform(0, 1)
        self.previous_perceived_risk = 0.0
        self.migration_prob = 0.0
        # self.safe_zone = []        # check 3 paramter 
        # self.daily_migrants = 0
        # self.total_migrants = 0
        self.state = CitizenState.REST
        self.tool_manager = CITIZEN_TOOL_MANAGER

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/granite4:latest",
            display=False,
        )

        # Initial internal state context
        self.internal_state.append(
            f"My household ID is {self.household_id}"
        )
        self.internal_state.append(
            f"My risk proneness is {self.risk_proneness:.3f}"
        )

    def compute_event_impact(self):
        return self.model.intensity_of_event / (
            (1 + self.model.spatial_decay * self.distance) *
            (1 + self.model.temporal_decay * self.time_difference)
            )

        
    def update_migration_probability(self):

        event_impact = self.compute_event_impact()
        total_risk = event_impact

        # 3️⃣ Perceived Behavior Control (memory)
        perceived_risk = (
            self.risk_proneness * total_risk
            + self.model.memory_retention * self.previous_perceived_risk     # memory retension not define
        )

        self.previous_perceived_risk = perceived_risk

       
        self.migration_prob = 1 / (
            1
            + math.exp(
                -self.model.growth_rate
                * (perceived_risk - self.model.baseline_Q)                  # baseline_q not define nd growth rate
            )
        )

    def apply_peer_threshold(self):

        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
        )

        if not neighbors:
            return

        migrated_neighbors = sum(
            1 for n in neighbors if n.state == CitizenState.MIGRATE
        )

        fraction = migrated_neighbors / len(neighbors)

        if fraction > self.model.threshold_phi:                     
            self.migration_prob = 1.0


    def explain_decision(self):
        

        prompt = f"""
        You are a civilian living in a conflict zone.

        Current perceived risk: {self.previous_perceived_risk:.3f}
        Migration probability: {self.migration_prob:.3f}
        Current state: {self.state.name}

        Explain briefly why migration probability is high or low.
        Do not change any decision.
        """

        observation = self.generate_obs()

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["migrate"]                           # rechck
        )
        
        

        self.apply_plan(plan)
        

    def step(self):

        # 1–4 Mathematical update
        self.update_migration_probability()

        # 7 Peer effect
        self.apply_peer_threshold()

        # LLM explanation only
        self.explain_decision()

        


        

        
