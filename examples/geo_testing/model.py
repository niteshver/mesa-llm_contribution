from shapely.geometry import Point
import geopandas as gpd

from mesa.model import Model
from mesa.datacollection import DataCollector

from mesa_geo import GeoSpace
from mesa_geo.geoagent import AgentCreator

from examples.geo_testing.agent import Citizen, CitizenState


class MigrationModel(Model):

    def __init__(
        self,
        citizen,
        reasoning,
        llm_model,
        vision,
        seed=None,
    ):
        super().__init__(seed=seed)

        

        gdf = gpd.read_file("data/TorontoNeighbourhoods.geojson")

        print("Dataset CRS:", gdf.crs)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        self.space = GeoSpace(crs=gdf.crs)

        self.safe_zone = Point(-79.38, 43.65)

        self.war_zones = [
            Point(-79.4, 43.7).buffer(0.02),
            Point(-79.35, 43.66).buffer(0.02),
        ]

        citizen_prompt = "You are a civilian deciding whether to flee war."

        creator = AgentCreator(
            Citizen,
            model=self,
            crs=gdf.crs,
            agent_kwargs={
                "reasoning": reasoning,
                "llm_model": llm_model,
                "system_prompt": citizen_prompt,
                "vision": vision,
                "internal_state": [],
                "step_prompt": "Assess risk and decide migration.",
            },
        )

        for _ in range(citizen):

            region = gdf.sample(1).iloc[0]

            centroid = region.geometry.centroid

            agent = creator.create_agent(centroid)

            self.space.add_agents(agent)

        self.datacollector = DataCollector(
            model_reporters={
                "migrating": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, Citizen)
                    and a.state == CitizenState.MIGRATE
                )
            }
        )

    def step(self):

        self.agents_by_type[Citizen].shuffle_do("step")

        self.datacollector.collect(self)