from enum import Enum
import math

import mesa
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager



class ClusterState(Enum):
    SIMILAR = "similar"
    DISSIMILAR = "dissimilar"


student_tool_manager = ToolManager()

class StudentAgent(LLMAgent, mesa.discrete_space.CellAgent):
    """
    Students exist on a grid.
    Each student has (score, stress, motivation).

    Students compare themselves to nearby students:
    - If most neighbors are similar → they cluster.
    - If most neighbors are dissimilar → they drift away.

    The LLM is used ONLY to explain the outcome.
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state=None,
        step_prompt=None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )

        # ---------------- Internal attributes ----------------
        self.score = self.random.uniform(0, 100)
        self.stress = self.random.uniform(0, 1)
        self.motivation = self.random.uniform(0, 1)

        self.state = ClusterState.SIMILAR

        # ---------------- Memory (optional) ----------------
        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/granite4:latest",
            display=False,
        )

        self.tool_manager = student_tool_manager

        # Initial internal description
        self.internal_state.append(
            f"My score is {self.score:.1f}, stress is {self.stress:.2f}, motivation is {self.motivation:.2f}"
        )


    def similarity_distance(self, other):
        """
        Compute distance between two students
        Lower distance = more similar
        """
        return (
            abs(self.score - other.score) / 100
            + abs(self.stress - other.stress)
            + abs(self.motivation - other.motivation)
        )


    def evaluate_neighbors(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.vision,
        )

        similar = 0
        dissimilar = 0

        for agent in neighbors:
            if not isinstance(agent, StudentAgent):
                continue

            distance = self.similarity_distance(agent)

            if distance < self.model.similarity_threshold:
                similar += 1
            else:
                dissimilar += 1

        return similar, dissimilar



    def move(self, similar, dissimilar):
        if similar >= dissimilar:
            self.state = ClusterState.SIMILAR
            return  # stay put (cluster stabilizes)

        self.state = ClusterState.DISSIMILAR

        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
        )

        empty_cells = [
            cell for cell in possible_steps
            if self.model.grid.is_cell_empty(cell)
        ]

        if empty_cells:
            new_pos = self.random.choice(empty_cells)
            self.model.grid.move_agent(self, new_pos)

    # ======================================================
    # LLM explanation (NO DECISIONS)
    # ======================================================
    def explain(self, similar, dissimilar):
        prompt = f"""
        You are a student in a social clustering simulation.

        Your internal state:
        - Score: {self.score:.1f}
        - Stress: {self.stress:.2f}
        - Motivation: {self.motivation:.2f}

        Nearby students:
        - Similar neighbors: {similar}
        - Dissimilar neighbors: {dissimilar}

        Current state: {self.state.value}

        Explain in simple language why this outcome makes sense.
        Do not suggest changes.
        Do not compute numbers.
        """

        obs = self.generate_obs()

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=obs,
            selected_tools=[],
        )

        self.apply_plan(plan)


    def step(self):
        similar, dissimilar = self.evaluate_neighbors()
        self.move(similar, dissimilar)
        self.explain(similar, dissimilar)
