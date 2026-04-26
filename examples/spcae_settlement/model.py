from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print
import random
from examples.spcae_settlement.agent import Martian_State,MartianAgent,Stressor_Agent,Stressor_State
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model
import mesa_geo as mg




class SpaceModel(Model):
    def __init__(self, n_martians, width, height, seed=None):
        super().__init__(seed=seed)
        geojson_regions = "data/MARS_nomenclature_center_pts.shp"

        self.num_agents = n_martians
        self.grid = MultiGrid(width, height, torus=False)
        self.space = mg.GeoSpace(warn_crs_conversion=False)
        ac = mg.AgentCreator(MartianAgent, model=self)
        neighbourhood_agents = ac.from_file(geojson_regions)
        self.space.add_agents(neighbourhood_agents)

        # -------------------------------
        # SETTLEMENT RESOURCES
        # -------------------------------
        self.settlement_air = 5.88 * n_martians * 156
        self.settlement_water = 28 * n_martians * 156
        self.settlement_food = 10.5 * n_martians * 156
        self.settlement_waste = 0

        # -------------------------------
        # CREATE MARTIANS
        # -------------------------------
        for i in range(self.num_agents):
            agent = MartianAgent(i, self)
            self.schedule.add(agent)

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.running = True

    # -------------------------------
    # RANDOM STRESSOR EVENT
    # -------------------------------
    def create_stressor(self):
        if random.random() < 0.1:  # 10% chance
            stressor = Stressor_Agent(self.next_id(), self)
            self.schedule.add(stressor)

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(stressor, (x, y))

    # -------------------------------
    # STEP
    # -------------------------------
    def step(self):
        print(f"\n--- Step {self.steps} ---")
        print(f"Agents: {self.schedule.get_agent_count()}")
        print(f"Food: {self.settlement_food:.2f}")

        self.create_stressor()
        self.schedule.step()

if __name__ == "__main__":
   
    from examples.spcae_settlement.model import Model

    for _ in range(5):
        Model.step()