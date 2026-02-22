from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid


from examples.migration_model.agent import Citizen,CitizenState,CITIZEN_TOOL_MANAGER
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.recording.record_model import record_model

# @record_model(output_dir="recordings")
class MigrationModel(Model):
    def __init__(
        self,
        citizen: int,
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
        self.intensity_of_event = 0.5
        self.spatial_decay = 0.7
        self.temporal_decay = 0.5
        self.memory_retention = 1
        self.growth_rate = 0.7
        self.baseline_Q = 1
        self.threshold_phi = 0.1
        self.num_households = 1
        self.safe_zone = []        # check 3 paramter 
        self.daily_migrants = 0
        self.total_migrants = 0
        self.parallel_stepping = parallel_stepping

        self.grid = MultiGrid(width, height, torus=False)

        # ---------------- Data Collection ----------------
        model_reporters = {
            "rest": lambda m: sum(
                1 for a in m.agents
                if isinstance(a, Citizen) and a.state == CitizenState.REST
            ),
            "migrate": lambda m: sum(
                1 for a in m.agents
                if isinstance(a,citizen) and a.state == CitizenState.MIGRATE

            ),
            "daily_migrants": lambda m: m.daily_migrants,
            "total_migrants": lambda m: m.total_migrants,
            }
        agent_reporters = {
            "migration_prob": lambda a: getattr(a, "migration_prob", None)
        }

        self.datacollector = DataCollector(model_reporters=model_reporters,agent_reporters=agent_reporters)

        # ---------------- Create Agents ----------------
        citizen_prompt = (
            "You are a citizen in a conflict region. "
            "Decide whether to migrate based on perceived risk."
        )

        agents = Citizen.create_agents(
            self,
            n=citizen,
            reasoning=ReActReasoning,
            llm_model=llm_model,
            system_prompt=citizen_prompt,
            vision=vision,
            internal_state=None,
            step_prompt="Assess risk and decide whether to migrate.",
        )

        x = self.rng.integers(0, width, size=(citizen,))
        y = self.rng.integers(0, height, size=(citizen,))

        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

    def step(self):

        # Reset daily counter
        # self.daily_migrants = 0

        self.agents.shuffle_do("step")

        self.datacollector.collect(self)

if __name__ == "__main__":
    
    from examples.migration_model.app import model
    
    model = MigrationModel(
        citizen=10,
        width=10,
        height=10,
        reasoning=ReActReasoning,
        llm_model="ollama/granite4:latest",
        vision=2,
    )
                                                    # for testing
    for _ in range(10):
        model.step()
        