"""Streamlit UI for the RAG-Enhanced AI Music Recommender."""

import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from src.agent import MusicRecommenderAgent
from src.guardrails import setup_logging, validate_user_input

load_dotenv()
setup_logging()

st.set_page_config(page_title="AI Music Recommender", page_icon="🎵", layout="wide")

st.title("AI Music Recommender")
st.markdown(
    "A RAG-enhanced music recommendation system with an agentic workflow. "
    "Describe what you're in the mood for and the AI will find the perfect tracks."
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Required for AI features. Get one at platform.openai.com",
    )
    show_agent_steps = st.checkbox("Show agent reasoning steps", value=True)
    show_scores = st.checkbox("Show algorithmic scores", value=True)

    st.divider()
    st.header("Quick Prompts")
    quick_prompts = [
        "Upbeat pop for a morning run",
        "Chill lo-fi beats for studying",
        "High-energy rock for the gym",
        "Moody synthwave for a night drive",
        "Sad blues for a rainy evening",
        "EDM bangers for a party",
    ]
    for prompt in quick_prompts:
        if st.button(prompt, use_container_width=True):
            st.session_state["user_input"] = prompt

# --- Initialize agent ---
if api_key:
    if "agent" not in st.session_state or st.session_state.get("_api_key") != api_key:
        with st.spinner("Initializing AI agent and building knowledge index..."):
            client = OpenAI(api_key=api_key)
            st.session_state["agent"] = MusicRecommenderAgent(client)
            st.session_state["_api_key"] = api_key
else:
    st.warning("Enter your OpenAI API key in the sidebar to enable AI features.")
    st.stop()

# --- Main input ---
user_input = st.text_area(
    "What kind of music are you looking for?",
    value=st.session_state.get("user_input", ""),
    placeholder="e.g., 'I want something energetic and danceable for a Friday night party'",
    height=80,
)

context = st.text_input(
    "Additional context (optional)",
    placeholder="e.g., 'I usually listen to indie pop and electronic'",
)

if st.button("Get Recommendations", type="primary", use_container_width=True):
    if not user_input.strip():
        st.error("Please describe what kind of music you're looking for.")
    else:
        valid, msg = validate_user_input(user_input)
        if not valid:
            st.error(f"Invalid input: {msg}")
        else:
            agent = st.session_state["agent"]

            with st.spinner("AI agent is analyzing your request..."):
                result = agent.run(user_input, context=context if context else None)

            if not result.success:
                st.error(f"Agent error: {result.error}")
            else:
                # --- Agent steps ---
                if show_agent_steps:
                    st.subheader("Agent Reasoning Pipeline")
                    for step in result.steps:
                        icon = "✅" if step.status in ("completed", "passed") else "❌"
                        with st.expander(
                            f"{icon} Step: {step.name}  ({step.duration_ms:.0f}ms)",
                            expanded=False,
                        ):
                            st.write(f"**Status:** {step.status}")
                            st.write(f"**Details:** {step.details}")
                            if step.output_data:
                                st.json(step.output_data)

                # --- Recommendations ---
                st.subheader("Your Recommendations")

                confidence_color = (
                    "green" if result.confidence >= 0.7
                    else "orange" if result.confidence >= 0.5
                    else "red"
                )
                st.markdown(
                    f"**Overall Confidence:** "
                    f":{confidence_color}[{result.confidence:.0%}]"
                )

                for i, rec in enumerate(result.recommendations):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### #{i+1} — {rec['title']}")
                            st.markdown(f"**Artist:** {rec.get('artist', 'Unknown')}")
                            st.markdown(rec.get("explanation", ""))
                        with col2:
                            if show_scores:
                                st.metric("Score", f"{rec.get('score', 0):.2f}")
                            conf = rec.get("confidence", 0)
                            st.metric("Match", f"{conf:.0%}")
                            st.caption(
                                f"{rec.get('genre', '')} | {rec.get('mood', '')} | "
                                f"energy {rec.get('energy', 0):.2f}"
                            )
                        st.divider()

# --- Footer ---
st.caption(
    "Built for CodePath AI110 — extends the Module 3 Music Recommender Simulation "
    "with RAG retrieval, agentic workflow, and reliability evaluation."
)
