import streamlit as st

from examples.internship.agents import GeneratorAgent
from examples.internship.model import SimpleModel

st.set_page_config(page_title="AI Agent Pipeline", page_icon="🎓", layout="wide")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🎓 AI Educational Content Pipeline")
st.markdown("**Generator Agent** → **Reviewer Agent** → **Refinement** *(if needed)*")
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns([1, 2, 1])

with col_a:
    grade = st.number_input("Grade Level", min_value=1, max_value=12, value=4)

with col_b:
    topic = st.text_input("Topic", value="Types of angles")

with col_c:
    st.write("")
    st.write("")
    run = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run:
    if not topic.strip():
        st.error("Please enter a topic.")
        st.stop()

    status_text = st.empty()
    progress = st.progress(0)

    steps = {"n": 0}

    def on_status(msg):
        steps["n"] += 1
        progress.progress(min(steps["n"] * 22, 90))
        status_text.info(msg)

    try:
        model = SimpleModel()
        result = model.pipeline.run(grade=int(grade), topic=topic, on_status=on_status)
    except Exception as e:
        status_text.error(f"Pipeline error: {e}")
        st.exception(e)
        st.stop()

    progress.progress(100)

    # ── Error ─────────────────────────────────────────────────────────────────
    if "error" in result:
        status_text.error(result["error"])
        st.code(result.get("raw", ""), language="text")
        st.stop()

    status_text.success("✅ Pipeline complete!")
    st.divider()

    # ── Helper renderers ──────────────────────────────────────────────────────
    def show_content(data: dict, header: str):
        st.subheader(header)
        explanation = data.get("explanation", "*(missing)*")
        st.info(explanation)

        mcqs = data.get("mcqs", [])
        if not mcqs:
            st.warning("No MCQs found.")
            return

        for i, q in enumerate(mcqs, 1):
            with st.expander(f"Q{i}: {q.get('question', '—')}"):
                for opt in q.get("options", []):
                    st.write(opt)
                st.success(f"Answer: **{q.get('answer', '?')}**")

    def show_review(data: dict, header: str):
        st.subheader(header)
        status = data.get("status", "unknown")

        if status == "pass":
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

        feedback = data.get("feedback", [])
        if feedback:
            for f in feedback:
                st.warning(f"• {f}")
        else:
            st.markdown("*No issues found.*")

    # ── Pass 1 ────────────────────────────────────────────────────────────────
    st.markdown("### Pass 1 — Generation & Review")
    left, right = st.columns(2)

    with left:
        show_content(result["generator_output"], "📝 Generator Output")

    with right:
        show_review(result["review"], "🔍 Reviewer Feedback")

    # ── Pass 2 (refinement) ───────────────────────────────────────────────────
    if result["refined_needed"]:
        st.divider()
        st.markdown("### Pass 2 — Refined Content")
        left2, right2 = st.columns(2)

        refined = result.get("refined_output") or {}
        final_review = result.get("final_review") or {}

        with left2:
            if "error" in refined:
                st.error(refined["error"])
            else:
                show_content(refined, "🔄 Refined Generator Output")

        with right2:
            if final_review:
                show_review(final_review, "🔍 Final Review")

        if final_review.get("status") == "pass":
            st.balloons()
            st.success("🎉 Content passed after refinement!")
        else:
            st.warning("⚠️ Still has issues after one refinement pass.")

    else:
        st.success("✅ Passed on first attempt — no refinement needed.")
