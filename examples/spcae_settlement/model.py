from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.spcae_settlement.agent import Martian_State,Martian_Agent,Stressor_Agent,Stressor_State
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model

class Space_Model(Model):
    def __init__(
            self,
            initial_cops: int,
            initial_citizens: int,
            width: int,
            height: int,
            reasoning: type[Reasoning],
            llm_model: str,
            vision: int,
            api_base: str | None = None,
            parallel_stepping=True,
            seed=None,
    ):
        super().__init(seed=seed)
        
        self.mineral_shipment = 100
        self.settlement_air = 5.88 * self * 156
        self.settlement_water = 28 * self * 156
        self.settlement_food = 10.5 * self * 156
