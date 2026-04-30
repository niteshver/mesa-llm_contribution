import random
from mesa import Model
import mesa_geo as mg
from mesa_geo import GeoSpace, AgentCreator
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import Reasoning
from examples.spcae_settlement.agent import MartianAgent, Resilience
from mesa.datacollection import DataCollector

class SpaceModel(Model):
    """
    Mars settlement model
    """

    def __init__(
        self,
        n_agents,
        geo_file,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        api_base: str | None = None,
        parallel_stepping=True,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.num_agents = n_agents
        self.parallel_stepping = parallel_stepping


        # -------------------------
        # GEO SPACE
        # -------------------------
        self.space = GeoSpace(warn_crs_conversion=False)

        self.agent_creator = AgentCreator(
            MartianAgent,
            model=self,
            agent_kwargs={
                "reasoning": reasoning,
                "llm_model": llm_model,
                "system_prompt": "You are a Martian colonist.",
                "vision": 2,
                "internal_state": [],
                "step_prompt": "Decide your next move.",
                "resilience": random.choice(list(Resilience)),
                "skill_1": random.randint(0, 100),
                "skill_2": random.randint(0, 100),
                "coping_capacity": random.uniform(0.8, 1.0),
                "api_base": api_base,
            },
        )

        self.agents = self.agent_creator.from_file(geo_file)
        self.space.add_agents(self.agents)

        # -------------------------
        # SETTLEMENT RESOURCES
        # -------------------------
        self.settlement_food = 10.5 * n_agents * 156
        self.settlement_water = 28 * n_agents * 156
        self.settlement_air = 5.88 * n_agents * 156
        self.settlement_waste = 0

        self.running = True

        model_reporter = {

            "neurotic": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, MartianAgent) and agent.state == Resilience.NEUROTIC
            ),
            "reactive": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, MartianAgent) and agent.state == Resilience.REACTIVE
            ),

            "social": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, MartianAgent) and agent.state == Resilience.SOCIAL
            ),
            "agreedable": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, MartianAgent) and agent.state == Resilience.AGREEABLE
            ),
        }
        agent_reporters = {
            "resources_produce": lambda a: getattr(a, "resources_produce", None),
            "consume_resources": lambda a: getattr(a, "consume_resources", None),
        }

        self.data_collector = DataCollector(
            model_reporters=model_reporter,agent_reporters=agent_reporters
        )

    # -------------------------
    # REMOVE AGENT
    # --------------

    # -------------------------
    # STEP
    # -------------------------
    def step(self):
        print(f"\n--- Step ---")
        print(f"Agents: {len(self.agents)}")
        print(f"Food: {self.settlement_food:.2f}")
        print(self.step)

        self.agent.shuffle_do()


if __name__ == "__main__":
    """
    run the model without the solara integration with:
    conda activate mesa-llm && python -m examples.epstein_civil_violence.model
    """
    from examples.spcae_settlement.app import model

    for _ in range(5):
        model.step()