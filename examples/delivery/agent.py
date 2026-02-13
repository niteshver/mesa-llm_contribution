
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


class DELIVERYAGENTSTATE(Enum):
    ACTIVE = 0
    
class PRODUCTSTATE(Enum):
    DELIVERED = 0


class DeliveryAgent(LLMAgent, CellAgent):
    """
    Simple delivery agent.
    Moves on a grid and delivers an order within limited time.
    LLM is used ONLY to explain decisions.
    Delivery agent is hardworking and have high energy.
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

        self.state = DELIVERYAGENTSTATE.ACTIVE
        self.speed = speed                  # change later 
        self.traffic_probability = traffic_probability
        self.max_time = max_time
        self.pos = random.randint(0,10)     # chnage later this model like 
        self.earnings = 0.0                 # negotation model juts like simple
        self.battery = 0.5                  # chnage later
        self.time_spent = 0.0      # change later
        

        self.tool_manager = delivery_tool_manager
        self.system_prompt = f"You are an delivery agent ,having battery is {self.battery} have to deliver some order to customer,"\
                             f"Your have a speed of {self.speed} and maximimum time to deliver the order is {self.max_time}"\
                             f"If the order not received on time to customer then you have to give the order to the customer with low amount."
       

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/granite4:latest",
            display=False,
        )

        self.internal_state.append(f"On having values form 0 to 10, my pos is {self.pos}")
        self.internal_state.append(f"My max delivery time is {self.max_time} minutes")
        self.internal_state.append(f"my total earning till now is {self.earnings}")

    # def total_distance(self):
    #     total_distance = 0
    #     if self.pos == 0:

    def update_time_spent(self):
        self.speed = self.battery * (self.pos) # change from {1-self.pos} to self.pos
        
        self.time_spent = self.speed * self.traffic_probability

        for item in self.internal_state:
            if item.lower().startswith("my arrest probability is"):
                break
            self.internal_state.append(
                f"my time spend is {self.time_spent}"
            )
    def update_earning(self):
        total_earning = 0
        self.earnings = self.time_spent * self.pos * 10
        if self.earnings == True:
            total_earning += self.earnings
        self.internal_state.append(f"My total earning is now {self.earnings}")

    def step(self):
        if self.time_spent <= self.max_time:       
            self.update_time_spent()
            observation = self.generate_obs()
            plan = self.reasoning.plan(
                obs = observation, selected_tools=["deliver_the_product","move_one_step"] # chnage if possible
            )
class customer(LLMAgent):
    """
    You are a customer in a mesa grid,
    waiting for our parcel, if pacel not received on time then 
    you not have to pay the price.
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
        delivery_request = 1,
        # max_received_time = 10           change later
        
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
        # self.max_received_time = max_received_time          depend on what u want
        self.tool_manager = customer_tool_manager
        
        self.system_prompt = "You are a customer having some order. ," \
                            " Delivery Agent have only limited time (10minutes) to" \
                            " deliver the order to your place." \
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








    


        





        

        



    
   