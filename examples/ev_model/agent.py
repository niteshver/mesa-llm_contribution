import math
from enum import Enum
import random
import mesa 

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager

HOUSEHOLDAGENTTOOL_MANAGER = ToolManager()
CHARGERINGAGENTTOOL_MANAGER = ToolManager()

class AGENTSTATE(Enum):
    ICE_HOLDER = 1
    EV_HOLDER = 2
    EV_NONHOLDER = 3




class GovernmentAgent(LLMAgent):
    def __init__(
            self,
            model,
            reasoning,
            llm_model,
            system_prompt,
            step_prompt,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            step_prompt=step_prompt
        )
        self.carbon_tax = random.uniform(0,0.5)
        self.subsidy_amount = random.uniform(100,8000)
        self.risk_decay = random.uniform(0.01,0.1)
    
        subsidy = self.model.base_subsidy + self.income_targeting
        fuel_prices = self.model.base_fuel_prices + self.model.carbon_tax


class ChargingStatinAgent(LLMAgent,mesa.discrete_space.cell_agent):
    def __init__(
            self,
            model,
            reasoning,
            llm_model,
            system_prompt,
            step_prompt,
            capacity = 1,      # later check
            price_per_kwh=10,
            charging_speed=1, 
            utilization_rate=1   # later check
    ):
        
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            step_prompt=step_prompt
        )
        self.capacity = capacity
        self.price_per_kwh = price_per_kwh
        self.charging_speed = charging_speed
        self.utilization_rate = self.utilization_rate
        self.distance = random.random()
    
    def charging_cost(self):
        cost_of_charging = (self.price_per_kwh *
                            self.charging_speed * self.utilization_rate) 

class HouseholdAgent(LLMAgent, mesa.discrete_space.CellAgent):

    def __init__(self, model, reasoning, llm_model, system_prompt, step_prompt, vision):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            step_prompt=step_prompt,
            vision=vision,
        )

        # Socioeconomic attributes
        self.income = random.uniform(10000, 100000)
        self.env_awareness = random.random()
        self.annual_mileage = random.uniform(5000, 20000)

        # Technology state
        self.car_type = "ICE"
        self.state = AGENTSTATE.ICE_HOLDER

        self.total_cost_ev = 0
        self.total_cost_ice = 0
        self.utility_score = 0


        self.memory=STLTMemory(agent=self,
                               llm_model=llm_model,
                               display=False)
        
        self.tool_manager = HOUSEHOLDAGENTTOOL_MANAGER

        self.internal_state.append(
            f"My home location is {self.home_location}"
        )
        self.internal_state.append(
            f"My car type is {self.car_type}"
        )

    def calculate_to_ice(self):
        fuel_cost = (
            self.model.fuel_price
            * self.annual_mileage
            / self.model.fuel_efficiency
        )

        self.total_cost = (
            self.model.purchase_price_ice
            + fuel_cost
            + self.model.maintenance_ice
        )
        self.internal_state.append(
            f"My total cost of ICE is {self.total_cost}"
        )


    def calculate_to_ev(self):
        electricity_cost = (self.model.electricity_price * self.annual_charge
                            / self.model.ev_efficiency)
        self.total_cost = (
            self.model.purchase_price_ev - self.model.subsidy_amount +
              electricity_cost + self.model.maintenance_ev
        )
        self.internal_state.append(
            f"My total cost of ev is {self.total_cost}"
        )

   
        

    









           
        


                

                 
                
            
            

        


        

        
        

                 

