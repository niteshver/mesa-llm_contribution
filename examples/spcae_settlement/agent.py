import math
from enum import Enum

import mesa
import random
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager


class Stressor_State(Enum):
    Shippig = "Shipping"
    Habitat = "Habitat"

class Stressor_Agent(LLMAgent, mesa.discrete_space.CellAgent):
    def __init__(
            self,
            model,
            reasoning,
            llm_model,
            system_prompt,
            vision,
            internal_state,
            step_prompt,
            
            api_base=None,
            
    
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
        self.duration = random.randint(1,4)
        self.impact_strenght = random.normalvariate(0.2,0.5)




class Martian_State(Enum):
    Neurotic = "Nerrotic"
    Reactive = "Reactive"
    Social = "Social"
    Agreedable = "Agreedable"

class Martian_Agent(LLMAgent):

    def __init__(
            self,
            model,
            reasoning,
            llm_model,
            system_prompt,
            vision,
            internal_state,
            step_prompt,
            threshold=0.5,
            health = 100,
            death_prob = 0.01,
            accident_prob = 0.02,
            SHIPMENT_FAIL_PROB = 0.1,
            partner = None,
            # internal_state = None,
            api_base=None,

            
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
        self.p_air = 5.88
        self.p_water = 28
        self.food = 0.5
        self.p_waste = 0
        self.health = health
        self.coping_capacity = random.randint(0.84,0.98)
        self.skill_1 = random.randint(1,100)
        self.sleep = min(random.randint(4,8))
        self.alpha = random.random()
        self.skill_2 = 100 - self.skill_1
        self.waste = 1

        self.memory = STLTMemory(self,
                                 llm_model=llm_model,
                                 display = True)
        

def update_health_recover(self):
    selfhealth = min(100, self.health + self.sleep)

def waste_generation(self):
    waste = self.alpha (self.p_air + self.p_water + self.food)
    self.waste = waste
    self.internal_state.append()
    


# def production_condition(self):


def plan(self):

    prompt = """

    """
    obs = self.generate_obs


    

        





