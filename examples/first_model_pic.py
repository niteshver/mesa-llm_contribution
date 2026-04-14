"""
A minimal Mesa-LLM agent-based model for demonstration and testing of LLM usage tracking and budget control.
"""
from mesa.model import Model
from mesa_llm.llm_agent import LLMAgent

from mesa_llm.reasoning.react import ReActReasoning

class PicAgent(LLMAgent):
    def step(self):
        # Provide a non-empty prompt for realistic reasoning
        prompt = "Describe your environment and possible actions."
        obs = "You are in a simple grid world."
        plan = self.reasoning.plan(prompt=prompt, obs=obs)
        self.apply_plan(plan)
        print("Reasoning output:", plan)
        # Simulate LLM usage for budget tracking
        usage = {'prompt_tokens': 20, 'completion_tokens': 30, 'total_tokens': 50}
        self.llm.usage_tracker.update_usage(usage, self.llm.llm_model)
        print("Usage summary:", self.llm.usage_tracker.get_usage_summary())

class PicModel(Model):
    def __init__(self, budget_limit=1.0, token_limit=10000):
        super().__init__()
        self.agent = PicAgent(
            model=self,
            reasoning=ReActReasoning,
            llm_model="ollama/llama3.1:latest",
            system_prompt="You are a simple agent in a grid world. Describe your environment and possible actions each step.",
            budget_limit=budget_limit,
            token_limit=token_limit,
        )

    def step(self):
        self.agent.step()

if __name__ == "__main__":
    model = PicModel(budget_limit=1.0, token_limit=10000)
    for i in range(5):
        try:
            print(f"\n--- Step {i+1} ---")
            model.step()
        except Exception as e:
            print(f"Budget or token limit reached at step {i+1}: {e}")
            break
