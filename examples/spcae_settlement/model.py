from __future__ import annotations

from pathlib import Path

from mesa import Model
from mesa.datacollection import DataCollector
from mesa_geo import GeoSpace
from rich import print
from shapely.geometry import Point

from examples.spcae_settlement.agent import (
    MartianAgent,
    Resilience,
    StressorAgent,
    StressorType,
)
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


@record_model(output_dir="recordings")
class SpaceModel(Model):
    """Mars settlement model backed by Mesa-Geo points."""

    def __init__(
        self,
        initial_martians: int,
        width: int,
        height: int,
        data_path: str | None,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        api_base: str | None = None,
        parallel_stepping: bool = True,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.parallel_stepping = parallel_stepping
        self.space = GeoSpace(warn_crs_conversion=False)
        self.shipment_blocked = False
        self.stressor_radius = 2.0
        self.active_crs = "EPSG:4326"

        # Settlement-level reserves.
        self.settlement_food = 10.5 * initial_martians * 12
        self.settlement_water = 28 * initial_martians * 12
        self.settlement_air = 5.88 * initial_martians * 12
        self.settlement_waste = 0.0

        martian_points = self._seed_points(data_path, initial_martians)
        martians = [
            MartianAgent(
                model=self,
                geometry=point,
                crs=self.active_crs,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt="You are a Martian colonist helping your settlement survive.",
                vision=vision,
                step_prompt=(
                    "Check your health, nearby colonists, and settlement reserves. "
                    "Use tools only when they help the colony survive."
                ),
                resilience=self.random.choice(list(Resilience)),
                skill_1=self.random.randint(25, 100),
                skill_2=self.random.randint(25, 100),
                coping_capacity=self.random.uniform(0.7, 1.0),
                api_base=api_base,
            )
            for point in martian_points
        ]
        self.space.add_agents(martians)

        self.datacollector = DataCollector(
            model_reporters={
                "Population": lambda m: sum(
                    1
                    for agent in m.agents
                    if isinstance(agent, MartianAgent) and agent.state != "dead"
                ),
                "Active_Stressors": lambda m: sum(
                    1
                    for agent in m.agents
                    if isinstance(agent, StressorAgent) and agent.active
                ),
                "Average_Health": lambda m: m._average_martian_attr("health"),
                "Average_Coping": lambda m: m._average_martian_attr(
                    "coping_capacity"
                ),
            },
            agent_reporters={
                "health": lambda a: getattr(a, "health", None),
                "coping_capacity": lambda a: getattr(a, "coping_capacity", None),
                "resources_produced": lambda a: getattr(a, "resources_produced", None),
            },
        )

    def _seed_points(
        self,
        data_path: str | None,
        initial_martians: int,
    ) -> list[Point]:
        points: list[Point] = []

        if data_path:
            source = Path(data_path)
            if source.exists() and gpd is not None:
                try:
                    gdf = gpd.read_file(source)
                    geometries = [
                        geom.centroid if not isinstance(geom, Point) else geom
                        for geom in gdf.geometry
                        if geom is not None and not geom.is_empty
                    ]
                    if geometries:
                        self.active_crs = str(gdf.crs or self.active_crs)
                        sample_size = min(initial_martians, len(geometries))
                        points.extend(self.random.sample(geometries, k=sample_size))
                except Exception:
                    # Fall back to synthetic points when the partial shapefile
                    # cannot be read in the local environment.
                    points = []

        while len(points) < initial_martians:
            points.append(
                Point(
                    self.random.uniform(0, self.width),
                    self.random.uniform(0, self.height),
                )
            )

        return points

    def _average_martian_attr(self, attr: str) -> float:
        values = [
            getattr(agent, attr)
            for agent in self.agents
            if isinstance(agent, MartianAgent) and agent.state != "dead"
        ]
        return sum(values) / len(values) if values else 0.0

    def iter_martians(
        self,
        origin: Point,
        within: float | None = None,
        exclude: MartianAgent | None = None,
    ):
        for agent in self.agents:
            if not isinstance(agent, MartianAgent) or agent.state == "dead":
                continue
            if exclude is not None and agent is exclude:
                continue
            if within is None or origin.distance(agent.geometry) <= within:
                yield agent

    def retire_geo_agent(self, agent) -> None:
        try:
            self.space.remove_agent(agent)
        except Exception:
            pass

    def remove_agent(self, agent) -> None:
        agent.state = "dead"
        if hasattr(agent, "active"):
            agent.active = False
        self.retire_geo_agent(agent)

    def maybe_spawn_stressor(self) -> None:
        active_martians = [
            agent
            for agent in self.agents
            if isinstance(agent, MartianAgent) and agent.state != "dead"
        ]
        if not active_martians or self.random.random() >= 0.15:
            return

        anchor = self.random.choice(active_martians)
        stressor = StressorAgent(
            model=self,
            geometry=anchor.geometry,
            crs=self.active_crs,
            stressor_type=self.random.choice(list(StressorType)),
        )
        self.space.add_agents([stressor])

    def step(self):
        """Execute one model step."""
        self.shipment_blocked = False
        self.maybe_spawn_stressor()

        print(
            f"\n[bold purple] step {self.steps} "
            "────────────────────────────────────────────────────────────────[/bold purple]"
        )
        print(
            f"population={sum(1 for a in self.agents if isinstance(a, MartianAgent) and a.state != 'dead')} "
            f"food={self.settlement_food:.2f} "
            f"water={self.settlement_water:.2f} "
            f"air={self.settlement_air:.2f}"
        )

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    from examples.spcae_settlement.app import model

    for _ in range(5):
        model.step()
