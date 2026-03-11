from mesa.datacollection import DataCollector
from mesa.model import Model

from shapely.geometry import Point

from mesa_geo import GeoSpace
from mesa_geo.geoagent import AgentCreator

import geopandas as gpd
import mesa
from mesa.model import Model
from examples.geo_testing.agent import Citizen, CitizenState
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model


@record_model(output_dir="recordings")
class MigrationModel(Model):

    def __init__(
        self,
        citizen: int,
        reasoning: type[Reasoning],
        llm_model: str,
        internal_state,
        vision: int,
        seed=None,
    ):
        super().__init__(seed=seed)

        # ---------------- Load GeoJSON ----------------

        self.gdf = gpd.read_file("data/TorontoNeighbourhoods.geojson")

        if self.gdf.crs is None:
            self.gdf = self.gdf.set_crs("EPSG:4326")

        # ---------------- GeoSpace ----------------

        self.space = GeoSpace(crs=self.gdf.crs)

        # ---------------- Model parameters ----------------

        self.intensity_of_event = 1.5
        self.spatial_decay = 0.5
        self.temporal_decay = 0.3
        self.growth_rate = 3.0
        self.baseline_Q = 0.4

        self.total_migrants = 0

        # ---------------- Data collection ----------------

        self.datacollector = DataCollector(
            model_reporters={
                "rest": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, Citizen) and a.state == CitizenState.REST
                ),
                "migrate": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, Citizen) and a.state == CitizenState.MIGRATE
                ),
                "total_migrants": lambda m: m.total_migrants,
            },
            agent_reporters={
                "state": "state",
                "migration_prob": "migration_prob",
            },
        )

        # ---------------- Citizen creator ----------------

        citizen_prompt = (
            "You are a citizen in a conflict region. "
            "Decide whether to migrate based on perceived risk."
        )

        creator = AgentCreator(
            Citizen,
            model=self,
            crs=self.space.crs,
            agent_kwargs={
                "reasoning": reasoning,
                "llm_model": llm_model,
                "system_prompt": citizen_prompt,
                "vision": vision,
                "internal_state": None,
                "step_prompt": "Assess risk and decide whether to migrate.",
            },
        )

        # ---------------- Spawn citizens ----------------

        for i in range(citizen):

            region = self.gdf.sample(1).iloc[0]

            x, y = region.geometry.centroid.coords.xy

            point = Point(x[0], y[0])

            agent = creator.create_agent(point)

            self.space.add_agents(agent)

    # ---------------- Step ----------------

    def step(self):

        self.agents_by_type[Citizen].shuffle_do("step")

        self.datacollector.collect(self)