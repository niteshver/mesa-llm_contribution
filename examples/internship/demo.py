import mesa
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa.model import Model
from mesa_llm.reasoning.react import ReActReasoning

class SimpleAgent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3.1:latest",
            display=True
        )

    def step(self):
        observation = {"step": self.model.steps}

        prompt = """
        This is a new simulation step.
        Based on the current step number, explain how you reason about
        the situation and what you might consider doing next.
        Focus on describing your reasoning process clearly.
        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[]
        )

        print(plan)

class SimpleModel(Model):

    def __init__(self, seed=None):
        super().__init__(seed=seed)

        agent1 = SimpleAgent.create_agents(
            model=self,
            n=1,
            reasoning=ReActReasoning,
            llm_model="ollama/llama3.1:latest",
            system_prompt="""
            You are an educational content generator AI.

            Generate content based on the given input.

            Input:
            {
            "grade": <number>,
            "topic": "<topic>"
            }

            Output (STRICT JSON ONLY):
            {
            "explanation": "Write a simple explanation suitable for the given grade.",
            "mcqs": [
                {
                "question": "Question based on topic",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Correct option letter (A/B/C/D)"
                },
                {
                "question": "Question based on topic",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Correct option letter (A/B/C/D)"
                },
                {
                "question": "Question based on topic",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Correct option letter (A/B/C/D)"
                }
            ]
            }

            Rules:
            - Keep language appropriate for the given grade.
            - Ensure concepts are correct.
            - Exactly 3 MCQs.
            - No extra text outside JSON.

            """,
            internal_state=""
        )
        agent_2 = SimpleAgent.create_agents(
            model=self,
            n=1,
            reasoning = ReActReasoning,
            llm_model = "ollama/llama3.1:latest",
            system_prompt = """
            You are an educational content reviewer AI.

            Evaluate the given content.

            Input:
            {
            "grade": <number>,
            "topic": "<topic>",
            "explanation": "...",
            "mcqs": [...]
            }

            Output (STRICT JSON ONLY):
            {
            "status": "pass" or "fail",
            "feedback": [
                "Issue 1 (if any)",
                "Issue 2 (if any)"
            ]
            }

            Rules:
            - Check age appropriateness, correctness, and clarity.
            - Ensure questions match the explanation.
            - If no issues → status = "pass", feedback = []
            - If issues → status = "fail"
            - No extra text outside JSON.

             """
                                           

        )

    def step(self):
        print(f"\nModel step {self.steps}")
        self.agents.shuffle_do("step")

if __name__ == "__main__":
    model = SimpleModel()

    for _ in range(3):
        model.step()