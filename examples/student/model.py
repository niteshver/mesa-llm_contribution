from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid

from mesa_llm.reasoning.reasoning import Reasoning
from examples.student.agent import StudentAgent, ClusterState


class StudentModel(Model):
    """
    Grid-based student clustering model.

    Students compare internal states (score, stress, motivation)
    with neighbors and either cluster or drift apart.
    """

    def __init__(
        self,
        width: int,
        height: int,
        llm_model: str,
        reasoning: type[Reasoning],
        vision: int,
        similarity_threshold: float = 0.6,  # 
        seed=None,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.vision = vision
        self.similarity_threshold = similarity_threshold

        self.grid = MultiGrid(self.width, self.height, torus=False)

        # ---------------- Data collection ----------------
        model_reporters = {
            "similar" : lambda m: sum(
                1
                for a in m.agents
                if isinstance(a, StudentAgent)
                and a.state == ClusterState.SIMILAR 
            ),
            "dissimilar": lambda m: sum(
                1
                for a in m.agents
                if isinstance(a, StudentAgent)
                and a.state == ClusterState.DISSIMILAR
            ),
        }

        self.datacollector = DataCollector(
            model_reporters=model_reporters
        )

        # ---------------- Create agents ----------------
        system_prompt = (
            "You are a student with a score, stress, and motivation. "
            "You observe nearby students and decide whether you are "
            "similar to them or not. You may move if you are dissimilar."
        )

        agents = StudentAgent.create_agents(
            self,
            n=10,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=None,
            step_prompt="Observe neighbors, decide similarity, and move if needed.",
        )

        # ---------------- Place agents randomly ----------------
        x_positions = self.rng.integers(0, self.grid.width, size=len(agents))
        y_positions = self.rng.integers(0, self.grid.height, size=len(agents))

        for agent, x, y in zip(agents, x_positions, y_positions):
            self.grid.place_agent(agent, (x, y))

    # ======================================================
    # Model step
    # ======================================================
    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

if __name__ == "__main__":
    """
    Run the model 
    """
    for _ in range(5):
        StudentModel.step()