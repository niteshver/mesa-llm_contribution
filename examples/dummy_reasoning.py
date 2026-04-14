from mesa_llm.reasoning.reasoning import Plan

class DummyReasoning:
    def __init__(self, agent):
        self.agent = agent
    def plan(self, obs=None, ttl=1, selected_tools=None):
        return Plan(step=0, llm_plan="Dummy plan output", ttl=ttl)
