from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import struct

from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.spcae_settlement.agent import (
    MartianAgent,
    Resilience,
    StressorAgent,
    StressorType,
)
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model

try:
    import mesa_geo as mg
    from shapely.geometry import Point
except ImportError:  # pragma: no cover
    mg = None
    Point = None

RESOURCE_KEYS = ("food", "water", "air", "waste", "minerals")


@dataclass(frozen=True)
class MarsPoint:
    index: int
    x: float
    y: float


@dataclass(frozen=True)
class MarsShapeSummary:
    total_points: int
    bounds: tuple[float, float, float, float]
    projected_counts: dict[tuple[int, int], int]


if mg is not None:

    class MarsSiteAgent(mg.GeoAgent):
        """Display-only Mars reference point used by the optional GeoSpace view."""

        def __init__(self, model, geometry, crs, site_index: int):
            super().__init__(model=model, geometry=geometry, crs=crs)
            self.site_index = site_index
            self.atype = "reference_site"

        def step(self):
            return None

else:  # pragma: no cover
    MarsSiteAgent = None


def load_point_shapefile(path: str | Path) -> list[MarsPoint]:
    """Parse a point-only `.shp` file directly."""

    shp_path = Path(path)
    if not shp_path.exists():
        return []

    raw_points: list[MarsPoint] = []

    with shp_path.open("rb") as handle:
        header = handle.read(100)
        if len(header) != 100:
            raise ValueError(f"Invalid shapefile header in {shp_path}")

        shape_type = struct.unpack("<i", header[32:36])[0]
        if shape_type not in {0, 1}:
            raise ValueError(
                f"Unsupported shape type {shape_type} in {shp_path}; expected Point."
            )

        point_index = 0
        while True:
            record_header = handle.read(8)
            if not record_header:
                break
            if len(record_header) != 8:
                raise ValueError(f"Corrupt shapefile record header in {shp_path}")

            _record_number, content_length_words = struct.unpack(">2i", record_header)
            content = handle.read(content_length_words * 2)
            if len(content) != content_length_words * 2:
                raise ValueError(f"Corrupt shapefile record payload in {shp_path}")

            record_shape_type = struct.unpack("<i", content[:4])[0]
            if record_shape_type == 0:
                continue
            if record_shape_type != 1:
                raise ValueError(
                    f"Unsupported record type {record_shape_type} in {shp_path}"
                )

            x, y = struct.unpack("<2d", content[4:20])
            raw_points.append(MarsPoint(index=point_index, x=x, y=y))
            point_index += 1

    return raw_points


def project_points_to_grid(
    points: list[MarsPoint],
    width: int,
    height: int,
) -> MarsShapeSummary:
    if not points:
        return MarsShapeSummary(
            total_points=0,
            bounds=(0.0, 0.0, 0.0, 0.0),
            projected_counts={},
        )

    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)

    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)

    counts: Counter[tuple[int, int]] = Counter()
    for point in points:
        grid_x = min(
            width - 1,
            max(0, int(((point.x - min_x) / x_span) * (width - 1))),
        )
        grid_y = min(
            height - 1,
            max(0, int(((point.y - min_y) / y_span) * (height - 1))),
        )
        counts[(grid_x, grid_y)] += 1

    return MarsShapeSummary(
        total_points=len(points),
        bounds=(min_x, min_y, max_x, max_y),
        projected_counts=dict(counts),
    )


def build_optional_geospace(model, points: list[MarsPoint], max_points: int = 300):
    if mg is None or Point is None or MarsSiteAgent is None:
        return None, []

    geospace = mg.GeoSpace(warn_crs_conversion=False)
    if not points:
        return geospace, []

    stride = max(1, math.ceil(len(points) / max_points))
    display_points = points[::stride]
    display_agents = [
        MarsSiteAgent(
            model=model,
            geometry=Point(point.x, point.y),
            crs="EPSG:4326",
            site_index=point.index,
        )
        for point in display_points
    ]
    geospace.add_agents(display_agents)
    return geospace, display_agents


@record_model(output_dir="recordings")
class MarsSettlementModel(Model):
    """Mars colony model mixing Mesa, Mesa-LLM, and optional Mesa-Geo context."""

    def __init__(
        self,
        initial_martians: int = 12,
        width: int = 50,
        height: int = 50,
        reasoning: type[Reasoning] | None = None,
        llm_model: str = "ollama_chat/llama3.2:latest",
        vision: int = 3,
        api_base: str | None = None,
        data_path: str | None = None,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grid = MultiGrid(self.width, self.height, torus=False)

        self.initial_martians = initial_martians
        self.reasoning = reasoning
        self.llm_model = llm_model
        self.vision = vision
        self.api_base = api_base

        self.shipment_frequency = 78
        self.shipment_failure_probability = 0.12
        self.habitat_accident_probability = 0.05
        self.new_arrival_probability = 0.35
        self.random_mortality_probability = 0.002

        self.technology = 0.50
        self.energy = 1.0
        self.shipments_received = 0
        self.shipping_disasters = 0
        self.habitat_accidents = 0
        self.habitat_repairs = 0
        self.martian_deaths = 0

        self.weekly_needs = {
            "food": 10.5,
            "water": 28.0,
            "air": 5.88,
        }
        self.resource_pool = {
            "food": self.weekly_needs["food"] * initial_martians * 156,
            "water": self.weekly_needs["water"] * initial_martians * 156,
            "air": self.weekly_needs["air"] * initial_martians * 156,
            "waste": 0.0,
            "minerals": 0.0,
        }
        self.shipment_payload = {
            "food": self.weekly_needs["food"] * initial_martians * 78,
            "minerals": 100.0,
        }

        self.base_patch_capacity = {
            "food": 0.5,
            "water": 28.0,
            "air": 5.88,
            "waste": 3.0,
            "minerals": 2.0,
        }
        self.resilience_recovery_bonus = {
            Resilience.NEUROTIC.value: 0.002,
            Resilience.REACTIVE.value: 0.004,
            Resilience.SOCIAL.value: 0.007,
            Resilience.AGREEABLE.value: 0.006,
        }
        self.resilience_stress_multiplier = {
            Resilience.NEUROTIC.value: 1.35,
            Resilience.REACTIVE.value: 1.10,
            Resilience.SOCIAL.value: 0.85,
            Resilience.AGREEABLE.value: 0.95,
        }
        self.social_support_bonus = {
            Resilience.NEUROTIC.value: 0.018,
            Resilience.REACTIVE.value: 0.026,
            Resilience.SOCIAL.value: 0.040,
            Resilience.AGREEABLE.value: 0.034,
        }
        self.skill_requirements = {
            "food": self._roll_skill_requirement(),
            "water": self._roll_skill_requirement(),
            "air": self._roll_skill_requirement(),
            "waste": self._roll_skill_requirement(),
            "mining": self._roll_skill_requirement(),
            "accident": self._roll_skill_requirement(),
        }

        self.data_path = Path(
            data_path
            or Path(__file__).with_name("data") / "MARS_nomenclature_center_pts.shp"
        )
        self.mars_points = load_point_shapefile(self.data_path)
        self.mars_shape_summary = project_points_to_grid(
            self.mars_points, self.width, self.height
        )
        self.geo_space, self.geo_reference_agents = build_optional_geospace(
            self, self.mars_points
        )
        self.space = self.geo_space

        self.base_cell_capacities = self._build_cell_capacities()
        self.available_cell_capacities = {}
        self.reset_cell_capacities()

        self._spawn_initial_martians()

        self.datacollector = DataCollector(
            model_reporters={
                "Population": lambda m: len(m.martians),
                "Active_Stressors": lambda m: len(m.active_stressors),
                "Settlement_Food": lambda m: m.resource_pool["food"],
                "Settlement_Water": lambda m: m.resource_pool["water"],
                "Settlement_Air": lambda m: m.resource_pool["air"],
                "Settlement_Waste": lambda m: m.resource_pool["waste"],
                "Settlement_Minerals": lambda m: m.resource_pool["minerals"],
                "Technology": lambda m: m.technology,
                "Average_Health": lambda m: m.average_health,
                "Average_Coping": lambda m: m.average_coping,
                "Shipments_Received": lambda m: m.shipments_received,
                "Shipping_Disasters": lambda m: m.shipping_disasters,
                "Habitat_Accidents": lambda m: m.habitat_accidents,
                "Habitat_Repairs": lambda m: m.habitat_repairs,
            },
            agent_reporters={
                "health": lambda a: getattr(a, "health", None),
                "coping_capacity": lambda a: getattr(a, "coping_capacity", None),
                "current_task": lambda a: getattr(a, "current_task", None),
                "resilience": lambda a: getattr(a, "resilience", None).value
                if isinstance(a, MartianAgent)
                else None,
            },
        )

    @property
    def martians(self) -> list[MartianAgent]:
        return [agent for agent in self.agents if isinstance(agent, MartianAgent)]

    @property
    def active_stressors(self) -> list[StressorAgent]:
        return [
            agent
            for agent in self.agents
            if isinstance(agent, StressorAgent) and agent.remaining_steps > 0
        ]

    @property
    def average_health(self) -> float:
        if not self.martians:
            return 0.0
        return sum(agent.health for agent in self.martians) / len(self.martians)

    @property
    def average_coping(self) -> float:
        if not self.martians:
            return 0.0
        return sum(agent.coping_capacity for agent in self.martians) / len(
            self.martians
        )

    def _roll_skill_requirement(self) -> dict[str, int]:
        return {
            "skill_1": self.random.randint(25, 95),
            "skill_2": self.random.randint(25, 95),
        }

    def _build_cell_capacities(self) -> dict[tuple[int, int], dict[str, float]]:
        capacities = {}
        projected_counts = self.mars_shape_summary.projected_counts
        max_count = max(projected_counts.values(), default=1)

        for x in range(self.width):
            for y in range(self.height):
                density = projected_counts.get((x, y), 0)
                density_factor = density / max_count if max_count else 0.0
                capacities[(x, y)] = {
                    "food": self.base_patch_capacity["food"]
                    * (1.0 + 0.35 * density_factor),
                    "water": self.base_patch_capacity["water"]
                    * (0.90 + 0.25 * density_factor),
                    "air": self.base_patch_capacity["air"]
                    * (0.95 + 0.20 * density_factor),
                    "waste": self.base_patch_capacity["waste"]
                    * (1.0 + 0.50 * density_factor),
                    "minerals": self.base_patch_capacity["minerals"]
                    * (1.0 + 2.5 * density_factor),
                }
        return capacities

    def reset_cell_capacities(self):
        self.available_cell_capacities = {
            cell: values.copy() for cell, values in self.base_cell_capacities.items()
        }

    def _spawn_initial_martians(self):
        resilience_cycle = [
            Resilience.NEUROTIC,
            Resilience.REACTIVE,
            Resilience.SOCIAL,
            Resilience.AGREEABLE,
        ]
        reasoning = self.reasoning
        if reasoning is None:
            raise ValueError("MarsSettlementModel requires a Reasoning class.")

        system_prompt = (
            "You are a colonist in a Mars mining settlement. "
            "Protect human life first, stabilize the habitat second, "
            "and mine minerals only when life support is secure."
        )
        step_prompt = (
            "Review your health, coping capacity, settlement reserves, nearby colonists, "
            "and active stressors. Prioritize repairs and life-support production when "
            "resources are strained. Use mining only when the colony is stable."
        )

        for index in range(self.initial_martians):
            skill_1 = self.random.randint(0, 100)
            resilience = resilience_cycle[index % len(resilience_cycle)]
            agent = MartianAgent(
                model=self,
                reasoning=reasoning,
                llm_model=self.llm_model,
                system_prompt=system_prompt,
                vision=self.vision,
                internal_state=None,
                step_prompt=step_prompt,
                resilience=resilience,
                coping_capacity=self.random.uniform(0.84, 0.98),
                skill_1=skill_1,
                skill_2=100 - skill_1,
                api_base=self.api_base,
            )
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.grid.place_agent(agent, (x, y))

    def add_new_martians(self, count: int = 4):
        resilience_options = list(Resilience)
        for _ in range(count):
            skill_1 = self.random.randint(0, 100)
            agent = MartianAgent(
                model=self,
                reasoning=self.reasoning,
                llm_model=self.llm_model,
                system_prompt=(
                    "You are a newly arrived Mars settler joining an operating colony."
                ),
                vision=self.vision,
                internal_state=None,
                step_prompt=(
                    "Integrate into the colony quickly. Help with urgent repairs or "
                    "resource production before taking on mining."
                ),
                resilience=self.random.choice(resilience_options),
                coping_capacity=self.random.uniform(0.84, 0.98),
                skill_1=skill_1,
                skill_2=100 - skill_1,
                api_base=self.api_base,
            )
            position = (
                self.random.randrange(self.width),
                self.random.randrange(self.height),
            )
            self.grid.place_agent(agent, position)

    def get_cell_snapshot(self, pos: tuple[int, int] | None) -> str:
        if pos is None:
            return "unplaced"

        snapshot = self.available_cell_capacities.get(pos, {})
        density = self.mars_shape_summary.projected_counts.get(pos, 0)
        return (
            f"food={snapshot.get('food', 0.0):.2f}, "
            f"water={snapshot.get('water', 0.0):.2f}, "
            f"air={snapshot.get('air', 0.0):.2f}, "
            f"waste={snapshot.get('waste', 0.0):.2f}, "
            f"minerals={snapshot.get('minerals', 0.0):.2f}, "
            f"mars_sites={density}"
        )

    def format_resource_snapshot(self) -> str:
        return ", ".join(
            f"{resource}={self.resource_pool[resource]:.2f}"
            for resource in RESOURCE_KEYS
        )

    def describe_active_stressors(self) -> str:
        if not self.active_stressors:
            return "none"
        return "; ".join(
            (
                f"{stressor.stressor_type.value}"
                f"[resource={stressor.impacted_resource or 'morale'},"
                f" impact={stressor.impact_strength:.2f},"
                f" ttl={stressor.remaining_steps}]"
            )
            for stressor in self.active_stressors
        )

    def get_neighboring_martians(
        self,
        agent: MartianAgent,
        radius: int = 3,
    ) -> list[MartianAgent]:
        if agent.pos is None:
            return []

        neighbors = self.grid.get_neighbors(
            agent.pos,
            moore=True,
            include_center=False,
            radius=radius,
        )
        return [
            neighbor for neighbor in neighbors if isinstance(neighbor, MartianAgent)
        ]

    def get_psychological_pressure(self, agent: MartianAgent) -> float:
        base_pressure = 0.002
        total_pressure = base_pressure
        multiplier = self.resilience_stress_multiplier[agent.resilience.value]

        for stressor in self.active_stressors:
            if stressor.stressor_type is StressorType.SHIPPING:
                total_pressure += 0.008 * multiplier
            else:
                total_pressure += 0.014 * multiplier

        if self.resource_pool["food"] < self.weekly_needs["food"] * max(
            1, len(self.martians)
        ):
            total_pressure += 0.01 * multiplier

        return total_pressure

    def survey_local_sector(self, agent: MartianAgent) -> str:
        neighbors = self.get_neighboring_martians(agent, radius=3)
        neighbor_ids = [neighbor.unique_id for neighbor in neighbors]
        return (
            f"Sector {agent.pos}: {self.get_cell_snapshot(agent.pos)}. "
            f"Nearby martians={neighbor_ids or 'none'}. "
            f"Settlement reserves: {self.format_resource_snapshot()}. "
            f"Stressors: {self.describe_active_stressors()}."
        )

    def _pair_fit(
        self,
        first: MartianAgent,
        second: MartianAgent | None,
        requirement: dict[str, int],
    ) -> float:
        total_skill_1 = first.skill_1 + (second.skill_1 if second else 0)
        total_skill_2 = first.skill_2 + (second.skill_2 if second else 0)
        return min(
            total_skill_1 / max(requirement["skill_1"], 1),
            total_skill_2 / max(requirement["skill_2"], 1),
        )

    def _best_partner(
        self,
        agent: MartianAgent,
        task: str,
    ) -> tuple[MartianAgent | None, float]:
        requirement = self.skill_requirements[task]
        best_partner = None
        best_fit = self._pair_fit(agent, None, requirement)

        for candidate in self.get_neighboring_martians(agent, radius=3):
            fit = self._pair_fit(agent, candidate, requirement)
            if fit > best_fit:
                best_fit = fit
                best_partner = candidate

        return best_partner, best_fit

    def _consume_cell_capacity(
        self,
        pos: tuple[int, int],
        resource: str,
        amount: float,
    ):
        self.available_cell_capacities[pos][resource] = max(
            0.0, self.available_cell_capacities[pos][resource] - amount
        )

    def produce_resource(self, agent: MartianAgent, resource: str) -> str:
        if resource not in {"food", "water", "air", "waste"}:
            return f"{resource} is not a valid production target."
        if agent.pos is None:
            return f"Martian {agent.unique_id} is not on the settlement grid."

        available = self.available_cell_capacities[agent.pos][resource]
        if available <= 0:
            return f"No {resource} capacity remains in sector {agent.pos} this week."

        partner, fit = self._best_partner(agent, resource)
        throughput = min(1.40, 0.45 + 0.45 * self.technology + 0.35 * fit)
        used_capacity = min(available, available * min(1.0, 0.55 + 0.25 * fit))
        amount = used_capacity * throughput

        agent.current_task = f"produce_{resource}"
        agent.partner_id = partner.unique_id if partner else None

        if fit < 0.75:
            agent.coping_capacity = max(0.05, agent.coping_capacity - 0.015)
            return (
                f"Martian {agent.unique_id} attempted {resource} production but the "
                f"skill fit ({fit:.2f}) was too low."
            )

        self._consume_cell_capacity(agent.pos, resource, used_capacity)
        if resource == "waste":
            processed = min(self.resource_pool["waste"], amount)
            self.resource_pool["waste"] -= processed
            result_amount = processed
        else:
            self.resource_pool[resource] += amount
            self.resource_pool["waste"] += amount * 0.04
            result_amount = amount

        return (
            f"Martian {agent.unique_id} produced {result_amount:.2f} units of {resource}"
            f"{' with Martian ' + str(partner.unique_id) if partner else ' solo'}."
        )

    def mine_minerals(self, agent: MartianAgent) -> str:
        if agent.pos is None:
            return f"Martian {agent.unique_id} is not on the settlement grid."

        available = self.available_cell_capacities[agent.pos]["minerals"]
        if available <= 0:
            return f"No mineral capacity remains in sector {agent.pos} this week."

        partner, fit = self._best_partner(agent, "mining")
        if fit < 0.70:
            agent.current_task = "mine_minerals"
            agent.coping_capacity = max(0.05, agent.coping_capacity - 0.01)
            return (
                f"Martian {agent.unique_id} attempted mining but did not have enough "
                f"combined capability."
            )

        used_capacity = min(available, available * min(1.0, 0.60 + 0.30 * fit))
        yield_amount = used_capacity * (1.0 + 0.75 * self.technology)
        self._consume_cell_capacity(agent.pos, "minerals", used_capacity)
        self.resource_pool["minerals"] += yield_amount
        self.resource_pool["waste"] += yield_amount * 0.05
        self.technology = min(1.50, self.technology + 0.0025 * yield_amount)

        agent.current_task = "mine_minerals"
        agent.partner_id = partner.unique_id if partner else None
        return (
            f"Martian {agent.unique_id} mined {yield_amount:.2f} minerals"
            f"{' with Martian ' + str(partner.unique_id) if partner else ' solo'}. "
            f"Technology is now {self.technology:.3f}."
        )

    def repair_habitat(self, agent: MartianAgent) -> str:
        habitat_stressors = [
            stressor
            for stressor in self.active_stressors
            if stressor.stressor_type is StressorType.HABITAT
        ]
        if not habitat_stressors:
            return "No active habitat accident needs repair right now."

        partner, fit = self._best_partner(agent, "accident")
        target = max(habitat_stressors, key=lambda stressor: stressor.impact_strength)
        agent.current_task = "repair_habitat"
        agent.partner_id = partner.unique_id if partner else None

        if fit < 0.85:
            agent.coping_capacity = max(0.05, agent.coping_capacity - 0.02)
            return (
                f"Martian {agent.unique_id} could not stabilize the habitat accident "
                f"affecting {target.impacted_resource}."
            )

        recovered_amount = target.impact_strength * min(1.5, 0.75 + 0.35 * fit)
        if target.impacted_resource:
            self.resource_pool[target.impacted_resource] += recovered_amount
        target.remaining_steps = 0
        self.habitat_repairs += 1

        return (
            f"Martian {agent.unique_id} repaired the habitat accident on "
            f"{target.impacted_resource} and recovered {recovered_amount:.2f} units."
        )

    def support_neighbor(self, agent: MartianAgent, neighbor_id: int) -> str:
        neighbors = self.get_neighboring_martians(agent, radius=3)
        neighbor = next(
            (candidate for candidate in neighbors if candidate.unique_id == neighbor_id),
            None,
        )
        if neighbor is None:
            return (
                f"Martian {neighbor_id} is not within support range of Martian "
                f"{agent.unique_id}."
            )

        bonus = (
            self.social_support_bonus[agent.resilience.value]
            + self.social_support_bonus[neighbor.resilience.value]
        ) / 2
        agent.current_task = "support_neighbor"
        agent.partner_id = neighbor.unique_id
        neighbor.partner_id = agent.unique_id
        agent.coping_capacity = min(1.25, agent.coping_capacity + bonus * 0.5)
        neighbor.coping_capacity = min(1.25, neighbor.coping_capacity + bonus)
        agent.health = min(100.0, agent.health + 0.8)
        neighbor.health = min(100.0, neighbor.health + 1.5)

        return (
            f"Martian {agent.unique_id} supported Martian {neighbor.unique_id}. "
            f"Coping improved by {bonus:.3f}."
        )

    def apply_periodic_shipment(self):
        if (self.steps + 1) % self.shipment_frequency != 0:
            return

        if self.random.random() < self.shipment_failure_probability:
            self.shipping_disasters += 1
            StressorAgent(
                model=self,
                stressor_type=StressorType.SHIPPING,
                impacted_resource=None,
                impact_strength=1.0,
                remaining_steps=4,
            )
            return

        self.shipments_received += 1
        self.resource_pool["food"] += self.shipment_payload["food"]
        self.resource_pool["minerals"] += self.shipment_payload["minerals"]
        self.technology = min(1.50, self.technology + 0.05)

        if self.random.random() < self.new_arrival_probability:
            self.add_new_martians(count=4)

    def apply_habitat_accident(self):
        if self.random.random() >= self.habitat_accident_probability:
            return

        impacted_resource = self.random.choice(["food", "water", "air", "minerals"])
        self.resource_pool[impacted_resource] *= 0.5
        impact_strength = (
            self.weekly_needs.get(impacted_resource, 8.0)
            * max(2, len(self.martians))
            * 0.12
        )
        StressorAgent(
            model=self,
            stressor_type=StressorType.HABITAT,
            impacted_resource=impacted_resource,
            impact_strength=impact_strength,
            remaining_steps=4,
        )
        self.habitat_accidents += 1

    def apply_stressor_penalties(self):
        for stressor in self.active_stressors:
            if (
                stressor.stressor_type is StressorType.HABITAT
                and stressor.impacted_resource
            ):
                self.resource_pool[stressor.impacted_resource] = max(
                    0.0,
                    self.resource_pool[stressor.impacted_resource]
                    - stressor.impact_strength,
                )

    def apply_weekly_consumption(self):
        martians = list(self.martians)
        for martian in martians:
            self.resource_pool["waste"] += martian.weekly_waste_output
            deficit_penalty = 0.0

            for resource, need in self.weekly_needs.items():
                available = self.resource_pool[resource]
                consumed = min(available, need)
                self.resource_pool[resource] -= consumed
                deficit = need - consumed
                if deficit > 0:
                    if resource == "food":
                        deficit_penalty += 8.0 * (deficit / need)
                    elif resource == "water":
                        deficit_penalty += 12.0 * (deficit / need)
                    else:
                        deficit_penalty += 18.0 * (deficit / need)

            if deficit_penalty > 0:
                martian.health -= deficit_penalty
                martian.coping_capacity = max(
                    0.05, martian.coping_capacity - 0.02 * deficit_penalty
                )

            if self.random.random() < self.random_mortality_probability:
                martian.health = 0

            if martian.health <= 0:
                self.remove_martian(martian, reason="mortality")

    def remove_martian(self, martian: MartianAgent, reason: str):
        if martian.pos is not None:
            self.grid.remove_agent(martian)
        martian.last_action_summary = f"Removed from colony because {reason}."
        martian.remove()
        self.martian_deaths += 1

    def step(self):
        print(
            f"\n[bold red]Mars settlement step {self.steps} "
            f"population={len(self.martians)} "
            f"resources=({self.format_resource_snapshot()})[/bold red]"
        )

        self.reset_cell_capacities()
        self.apply_periodic_shipment()
        self.apply_habitat_accident()
        self.apply_stressor_penalties()
        self.agents.shuffle_do("step")
        self.apply_weekly_consumption()
        self.datacollector.collect(self)

        if not self.martians:
            self.running = False


if __name__ == "__main__":
    from mesa_llm.reasoning.react import ReActReasoning

    model = MarsSettlementModel(
        initial_martians=8,
        width=30,
        height=30,
        reasoning=ReActReasoning,
        llm_model="ollama_chat/llama3.2:latest",
        vision=3,
    )

    for _ in range(5):
        model.step()
