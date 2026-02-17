import math
from enum import Enum

import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager

CITIZEN_TOOL_MANAGER = ToolManager()

class CitizenState(Enum):
    REST = 3
    MIGRATE = 3

class Citizen(LLMAgent,mesa.discrete_space.cell_agent):
    """
    Building an Agent based model based on planed of
    migration. like russia ukraine war wher people have to 
    migrate from one place to other to save himself
    If above avergae of an hosuehold have agree for migrate
    then the entire household migrate.
    Migrate depend on risk nearby and neighbour if they 
    migrate then the chance of migrate is high 
    Ther are several fomula where we calculate risk of 
    migrate.
    """

    def __init(self,
            model,
            reasoning,
            llm_model,
            system_prompt,
            vision,
            internal_state,
            step_prompt,
            spatical_decay_parameter = 0.5,
            temporary_decay_paramter = 0.5,
            intensity_of_event = 0.5,
            memory_retention_prmtr = 1,
            previous_perceived_risk = 0.5,
            threshehold = 0.5
            ):
        super().__init__(
            model=model,
            reasoning = reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt   
        )
        self.distance = self.random.random()
        self.time_difference = self.random.random()
        self.memory_retention_prmtr = self.random.random()
        self.growth_rate = self.random.random()
        self.spatical_decay_prmtr = spatical_decay_parameter
        self.temporary_decay_paramter = temporary_decay_paramter
        self.intensity_of_event = intensity_of_event
        self.memory_retention_prmtr = memory_retention_prmtr
        self.previous_perceived_risk = previous_perceived_risk
        self.migration_prob = None

        self.memory = STLTMemory(agent=self,
                                display=True,
                                llm_model="Ollama/granite"
                                )
        self.threshehold = threshehold
        self.internal_state.append(
            f"on a scale of 0 to 1 , my distance b/w agent and event is {self.distance:.4f}"
        )
        self.internal_state.append(
            f"On a scale of 0 to 1, my time difference is {self.time_differnce:.4f}"
        )
        self.internal_state.append(
            f" my migration probability is {self.migration_prob:.4f}"
        )
        self.tool_manager = CITIZEN_TOOL_MANAGER
        self.system_prompt = "You are an citizen in Ukrane, and war is going on b/w Russia and Ukrane." \
                            "You have a family of some member, During this situation yo have to migrate to other place If event is " \
                            "held nearby you and intensity of event is high , Migrate decision is depend on averga of the family memeber " \
                            "and age, gender of the family member" 


    def update_migration_probability(self):
        Event_impact = self.intensity_of_event/(1+self.spatical_decay_prmtr * self.distance
                                                 *(1 + self.temporary_decay_paramter * 
                                                   self.time_difference))
        total_event = sum(Event_impact)
        risk_proneness = 1
        
        perceived_behaviour_control = (risk_proneness * total_event + self.memory_retention_prmtr 
                                       *  self.previous_perceived_risk)

        self.migration_prob = 1/1+ self.growth_rate * perceived_behaviour_control

        for item in self.internal_state:
            if item.lower().startswith("my migratio probability is"):
                self.internal_state.remove(item)
                break
        self.internal_state.append(
            f"My migration probability is {self.migration_prob:.4f}"
        )
    def step(self):
        obs = self.generate_obs()
        plan = self.reasoning.plan(
            obs=obs,selected_tools=["move_one_step", "migrate"]
        )
        self.apply_plan(plan)
        





