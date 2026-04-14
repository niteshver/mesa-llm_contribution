import json
import re
from mesa_llm.llm_agent import LLMAgent


def parse_json(text) -> dict | None:
    """Strip markdown fences and extract the first JSON object found."""
    text = str(text)
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


class GeneratorAgent(LLMAgent):

    def generate(self, grade: int, topic: str, feedback: list = None) -> str:
        feedback_block = ""
        if feedback:
            issues = "\n".join(f"- {f}" for f in feedback)
            feedback_block = f"\n\nFix ALL of these issues from the previous review:\n{issues}"

        observation = {"grade": grade, "topic": topic}

        prompt = f"""Generate educational content for Grade {grade} on: "{topic}".{feedback_block}

Return ONLY valid JSON — no markdown, no extra text:
{{
  "explanation": "Simple explanation suitable for Grade {grade}.",
  "mcqs": [
    {{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "A"}},
    {{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "B"}},
    {{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "C"}}
  ]
}}"""

        output = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[]
        )

        return str(output)

    def step(self):
        pass  # Pipeline-driven, not step-driven


class ReviewerAgent(LLMAgent):

    def review(self, grade: int, topic: str, content: dict) -> str:
        observation = {"grade": grade, "topic": topic, "content": content}

        prompt = f"""Review this Grade {grade} educational content on "{topic}":

{json.dumps(content, indent=2)}

Check:
1. Is the language appropriate for Grade {grade}?
2. Are the facts and MCQ answers correct?
3. Is the explanation clear?
4. Do MCQs test only what the explanation covers?

Return ONLY valid JSON — no markdown, no extra text:

If all good:
{{"status": "pass", "feedback": []}}

If issues found:
{{"status": "fail", "feedback": ["Specific issue 1", "Specific issue 2"]}}"""

        output = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[]
        )

        return str(output)

    def step(self):
        pass