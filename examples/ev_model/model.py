from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print
import random

from examples.ev_model.agent import HouseHoldAgent,ChargingStatinAgent,GovernmentAgent
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model


class EV_MODEL(Model):
    def __init__(
        self,
        initial_people: int,
        
        width: int,
        height: int,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        parallel_stepping=True,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.alpha_financial = 0.4
        self.beta_social = 0.3
        self.gamma_infrastructure = 0.2
        self.delta_environment = 0.1
        self.theta_risk = 0.2
        self.fuel_prices = 10
        self.annual_milege = 20
        self.purchase_price_ev = random.randint(34,60)
        self.purchase_price_ice = 3000
        self.maintaince_ice = 100
        self.maintaince_ev = 50
        self.ev_efficiency = 10
        self.ice_efficiency = 13
        self.electricity_price = 10
        self.rate_decay_rate = 10
        # self.base_subsidy = 3000
        # self.risk_decay_rate = 0.05
        self.base_fuel_prices = 1
        #baselineutility

        self.parallel_stepping = parallel_stepping
        self.grid = MultiGrid(self.height, self.width, torus=False)

