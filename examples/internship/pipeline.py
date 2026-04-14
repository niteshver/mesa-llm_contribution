from agents import parse_json


class AgentPipeline:

    def __init__(self, generator, reviewer):
        self.generator = generator
        self.reviewer = reviewer

    def run(self, grade: int, topic: str, on_status=None) -> dict:
        """
        Run: Generator → Reviewer → Refine (if fail, max 1 pass)

        Returns:
            {
                "generator_output": dict,
                "review":           dict,
                "refined_output":   dict | None,
                "final_review":     dict | None,
                "refined_needed":   bool
            }
        """

        def status(msg):
            if on_status:
                on_status(msg)

        # ── Step 1: Generate ──────────────────────────────────────────
        status("🤖 Generator Agent is drafting content…")
        raw_gen = self.generator.generate(grade, topic)
        gen_json = parse_json(raw_gen)

        if not gen_json:
            return {"error": "Generator returned invalid JSON", "raw": raw_gen}

        # ── Step 2: Review ────────────────────────────────────────────
        status("🔍 Reviewer Agent is evaluating the draft…")
        raw_review = self.reviewer.review(grade, topic, gen_json)
        review_json = parse_json(raw_review)

        if not review_json:
            return {"error": "Reviewer returned invalid JSON", "raw": raw_review}

        refined_json = None
        final_review_json = None
        refined_needed = review_json.get("status") == "fail"

        # ── Step 3: Refine (one pass) ─────────────────────────────────
        if refined_needed:
            feedback = review_json.get("feedback", [])
            status("🔄 Review failed — refining with feedback…")

            raw_refined = self.generator.generate(grade, topic, feedback=feedback)
            refined_json = parse_json(raw_refined)

            if refined_json:
                status("🔍 Reviewer evaluating the refined content…")
                raw_final = self.reviewer.review(grade, topic, refined_json)
                final_review_json = parse_json(raw_final)
            else:
                refined_json = {"error": "Refined output was invalid JSON", "raw": raw_refined}

        else:
            status("✅ Content passed review on first attempt!")

        return {
            "generator_output": gen_json,
            "review": review_json,
            "refined_output": refined_json,
            "final_review": final_review_json,
            "refined_needed": refined_needed,
        }
