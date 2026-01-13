"""
Model Comparison Page
=====================

Detailed pairwise comparison and win rate analysis.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.components.model_loader import load_results_data, get_test_prompts
from dashboard.components.charts import create_win_rate_heatmap, create_pairwise_bars
from dashboard.components.cards import render_response_card, render_prompt_viewer

st.set_page_config(
    page_title="Model Comparison | RLHF Dashboard",
    page_icon="",
    layout="wide",
)

st.title(" Model Comparison")
st.markdown("Pairwise win rates and head-to-head analysis")
st.markdown("---")

# Load data
try:
    data = load_results_data()
    pairwise = data["pairwise"]
    full_results = data["full_results"]
except Exception as e:
    st.error(f"Error loading results data: {e}")
    st.stop()

# Win Rate Matrix
st.markdown("### Win Rate Matrix")
st.markdown("Each cell shows the win rate of the **row model** against the **column model**.")

fig_heatmap = create_win_rate_heatmap(pairwise)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")

# Pairwise Comparison Bars
st.markdown("### Pairwise Comparison Results")
st.markdown("Detailed wins, losses, and ties for each matchup.")

fig_bars = create_pairwise_bars(pairwise)
st.plotly_chart(fig_bars, use_container_width=True)

# Win/Loss/Tie summary table
st.markdown("#### Summary Table")

import pandas as pd

table_data = []
for key, counts in pairwise.items():
    models = [k for k in counts.keys() if k != "tie"]
    if len(models) >= 2:
        total = sum(counts.values())
        table_data.append({
            "Matchup": key.replace("_", " ").upper(),
            f"{models[0].upper()} Wins": counts.get(models[0], 0),
            f"{models[1].upper()} Wins": counts.get(models[1], 0),
            "Ties": counts.get("tie", 0),
            f"{models[0].upper()} Win Rate": f"{counts.get(models[0], 0) / total * 100:.1f}%",
        })

if table_data:
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

# Sample Responses Viewer
st.markdown("### Sample Responses Viewer")
st.markdown("Explore individual examples and see how each model responded.")

# Get prompts for dropdown
prompts = get_test_prompts(data)

# Sample selector
col1, col2 = st.columns([1, 3])

with col1:
    sample_options = [f"Sample {p['index'] + 1}: {p['preview']}" for p in prompts]
    selected = st.selectbox(
        "Select a sample",
        options=range(len(prompts)),
        format_func=lambda i: sample_options[i],
    )

# Display selected sample
if selected is not None and selected < len(full_results):
    result = full_results[selected]

    st.markdown("#### Prompt")
    prompt_text = result.get("prompt", "N/A")
    # Remove truncation if present
    if prompt_text.endswith("..."):
        prompt_text = prompts[selected]["full"]
    render_prompt_viewer(prompt_text)

    st.markdown("#### Model Responses")

    # Get rewards and find winner
    rewards = {
        "base": result.get("base_reward", 0),
        "dpo": result.get("dpo_reward", 0),
        "ppo": result.get("ppo_reward", 0),
        "grpo": result.get("grpo_reward", 0),
    }
    winner = max(rewards.keys(), key=lambda k: rewards[k])

    # Display in 2x2 grid
    col1, col2 = st.columns(2)

    with col1:
        render_response_card(
            model_name="base",
            response=result.get("base_response", "N/A"),
            reward=rewards["base"],
            is_winner=(winner == "base"),
        )
        st.markdown("")
        render_response_card(
            model_name="ppo",
            response=result.get("ppo_response", "N/A"),
            reward=rewards["ppo"],
            is_winner=(winner == "ppo"),
        )

    with col2:
        render_response_card(
            model_name="dpo",
            response=result.get("dpo_response", "N/A"),
            reward=rewards["dpo"],
            is_winner=(winner == "dpo"),
        )
        st.markdown("")
        render_response_card(
            model_name="grpo",
            response=result.get("grpo_response", "N/A"),
            reward=rewards["grpo"],
            is_winner=(winner == "grpo"),
        )

    # Show reward summary
    st.markdown("#### Reward Scores")
    reward_df = pd.DataFrame([
        {"Model": k.upper(), "Reward": f"{v:.4f}", "Winner": "" if k == winner else ""}
        for k, v in sorted(rewards.items(), key=lambda x: x[1], reverse=True)
    ])
    st.dataframe(reward_df, use_container_width=True, hide_index=True)
