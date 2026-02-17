from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.migration_model.agent import Citizen,CitizenState,CITIZEN_TOOL_MANAGER
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model

class migration_model(Model):
    def __init__(
            self,
            initial_citizen: int,
            lenght: int,
            width: int,
            reasoning = type(Reasoning),
            llm_model = str,
            seed = None
    ):
        super().__init(seed = seed)
        self.lenght = lenght
        self.width = width
        self.grid = MultiGrid(self.lenght,self.width,torus = False)

        citizen_prompt = "You are a citizen of Ukrane and ur country is under war with Russia."\
                     "If possible then migrate"
        Citizen.create_agents(
            self,
            n = initial_citizen,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt = citizen_prompt,
            internal_state = None,
            step_prompt = "You have to migrate if possible"
        )
        x = s