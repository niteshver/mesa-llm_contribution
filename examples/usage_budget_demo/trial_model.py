
from mesa.model import Model
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.usage_tracker import UsageTracker, BudgetExceededError, TokenLimitExceededError

class TrialBudgetModel(Model):
    def __init__(self, budget_limit=0.001, token_limit=1000):
        super().__init__()
        self.agent = LLMAgent(
            model=self,
            reasoning=lambda agent: None,  # Dummy reasoning
            llm_model="openai/gpt-4o-mini",
            system_prompt="You are a helpful agent.",
            budget_limit=budget_limit,
            token_limit=token_limit,
        )

    def step(self):
        pass  # No scheduler, just a single agent for demo

if __name__ == "__main__":
    # Set a very low budget and token limit to trigger errors
    model = TrialBudgetModel(budget_limit=0.00005, token_limit=150)
    agent = model.agent
    for i in range(5):
        try:
            usage = {'prompt_tokens': 50, 'completion_tokens': 50, 'total_tokens': 100}
            agent.llm.usage_tracker.update_usage(usage, agent.llm.llm_model)
            print(f"Step {i+1}: Usage updated.")
            print(agent.llm.usage_tracker.get_usage_summary())
        except Exception as e:
            print(f"Step {i+1}: Exception - {e}")
            break
