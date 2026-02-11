
import math
import random
from enum import Enum

from mesa.discrete_space import CellAgent
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager
from mesa_llm.tools.tool_decorator import tool


delivery_tool_manager = ToolManager()
customer_tool_manager = ToolManager()


class DeliveryState(Enum):
    ACTIVE = 0
    DELIVERED = 0

#class PRODUCTSTATE(Enum):


class DeliveryAgent(LLMAgent, CellAgent):
    """
    Simple delivery agent.
    Moves on a grid and delivers an order within limited time.
    LLM is used ONLY to explain decisions.
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        step_prompt,
        internal_state=None,
        vision=1,
        traffic_probability=0.5,
        speed=1.0,
        max_time=10.0,
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

        self.state = DeliveryState.ACTIVE
        self.speed = speed                  # change later 
        self.traffic_probability = traffic_probability
        self.max_time = max_time
        self.pos = random.randint(0,10)     # chnage later this model like 
        self.earnings = 0.0                 # negotation model juts like simple
        self.battery = 0.5                  # chnage later
        self.time_spent = 0.0      # change later
        

        self.tool_manager = delivery_tool_manager
        self.system_prompt = "You are a DeliveryAgent having some order to deliver at some distance" \
        "You have limited time and max time allowed to deliver the order is 10 minutes" \
        "If order not received to customer before 10 minutes, then you have to gave the order at some discount " \
        "by own money."

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/granite4:latest",
            display=False,
        )

        self.internal_state.append(f"My speed is {self.speed}")
        self.internal_state.append(f"My max delivery time is {self.max_time} minutes")
        self.internal_state.append(f"my total earning till now is {self.earnings}")

    def update_time_spent(self):

        self.speed = self.battery * (1-self.pos)
        
        self.time_spent = self.speed * self.traffic_probability

        for item in self.internal_state:
            if item.lower().startswith("my arrest probability is"):
                break
            self.internal_state.append(
                f" my time spend is {self.time_spent}"
            )

    def step(self):
        if self.time_spent >= self.max_time:
            self.update_time_spent()
            observation = self.generate_obs()
            plan = self.reasoning.plan(
                obs = observation, selected_tools=["deliver_the_product","move_one_step"] # chnage if possible
            )
class customer(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        step_prompt,
        internal_state=None,
        vision=1,
        delivery_request = 1,
        max_received_time = 10
        
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

        self.pos = random.randint(0,1) # chnage later
        self.customer_delivery_request = delivery_request
        self.max_received_time = max_received_time
        self.tool_manager = customer_tool_manager
        
        self.system_prompt = "You are a customer having some order. ," \
        " Delivery Agent have only limited time (10minuts) to deliver the order to your place." \
        "If order not received on time then you got free order."

        self.memory = STLTMemory(agent=self,
                                 llm_model="ollama/granite4:latest",
                                 display=True)
        
    def step(self):
            
            observation = self.generate_obs()
            plan = self.reasoning.plan(
                obs=observation,
                selected_tools=["received_the_product","move_one_step"]
            )
            self.apply_plan(plan)








    


        





        

        



    
   