from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.delivery.agent import DeliveryAgent,customer,DeliveryState
from mesa_llm.reasoning.reasoning import Reasoning

class DeliveryModel(Model):
    def __init__(
            self,
            delivery_agent:int,
            customer:int,
            height:int,
            breath:int,
            reasoning:type[Reasoning],
            llm_model:str,
            vision:int,
            seed = None
    ):
        super().__init__(seed = seed)
        self.height = height
        self.breath = breath
        self.grid = MultiGrid(self.breath,self.height,torus = False)

        model_reporter = {

        }
        delivery_agent_prompt = "You are a delivery guy, You have to deliver the order to customer." \
        "You have limitde time, Deliver the product in that time."

        agent = DeliveryAgent.create_agents(
            self,
            n=1,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt = delivery_agent_prompt,
            viion = vision,
            internal_state = None,
            step_prompt = "Deliver the order to customer"
        )

        x = self.rng.integers(0, self.grid.width, )
        y = self.rng.integers(0, self.grid.height)
        for a, i, j in zip(agent, x, y):
            self.grid.place_agent(a, (i, j))

        agents = customer.create_agents(
            self,
            n=1,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="",
            vision=vision,
            internal_state=None,
            step_prompt="You are waiting to received your order"
        )
        x = self.rng.integers(0, self.grid.width)
        y = self.rng.integers(0, self.grid.height)
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

            
