from mesa.model import Model
from mesa_llm.reasoning.react import ReActReasoning

from examples.internship.agents import GeneratorAgent, ReviewerAgent
from examples.internship.pipeline import AgentPipeline

GENERATOR_PROMPT = """
You are an educational content generator AI.

Generate content based on the given grade and topic.

STRICT JSON ONLY — no markdown, no preamble, no extra text:
{
  "explanation": "Clear, age-appropriate explanation.",
  "mcqs": [
    {"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "A"},
    {"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "B"},
    {"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "C"}
  ]
}

Rules:
- Match language complexity to the grade level.
- Exactly 3 MCQs with exactly 4 options each.
- All facts and answers must be correct.
- Output ONLY JSON.
"""

REVIEWER_PROMPT = """
You are an educational content reviewer AI.

Evaluate content for age appropriateness, factual correctness, and clarity.

STRICT JSON ONLY — no markdown, no preamble, no extra text:

If content is good:
{"status": "pass", "feedback": []}

If content has issues:
{"status": "fail", "feedback": ["Issue 1", "Issue 2"]}

Rules:
- Be specific in feedback (mention which sentence or question has the issue).
- Output ONLY JSON.
"""


class SimpleModel(Model):

    def __init__(self, llm_model: str = "ollama/llama3.1:latest"):
        super().__init__()

        self.generator = GeneratorAgent(
            model=self,
            reasoning=ReActReasoning,
            llm_model=llm_model,
            system_prompt=GENERATOR_PROMPT,
        )

        self.reviewer = ReviewerAgent(
            model=self,
            reasoning=ReActReasoning,
            llm_model=llm_model,
            system_prompt=REVIEWER_PROMPT,
        )

        self.pipeline = AgentPipeline(self.generator, self.reviewer)

    def step(self):
        pass  # Pipeline is driven by pipeline.run(), not Mesa steps


# ── Quick CLI test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    model = SimpleModel()
    result = model.pipeline.run(grade=4, topic="Types of angles")
    print(json.dumps(result, indent=2))
