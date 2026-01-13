"""
Results Summary Page
====================

Displays high-level KPIs and visualizations from Experiment J.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.components.model_loader import load_results_data
from dashboard.components.charts import (
    create_reward_bar_chart,
    create_reward_distribution,
    create_response_length_chart,
)
from dashboard.components.cards import render_stats_table

st.set_page_config(
    page_title="Results Summary | RLHF Dashboard",
    page_icon="",
    layout="wide",
)

st.title(" Results Summary")
st.markdown("Overview of Experiment J: DPO vs PPO vs GRPO on 100 held-out test samples")
st.markdown("---")

# Load data
try:
    data = load_results_data()
    comparison = data["comparison"]
    model_stats = comparison["model_stats"]
except Exception as e:
    st.error(f"Error loading results data: {e}")
    st.stop()

# KPI Cards
st.markdown("### Key Metrics")

col1, col2, col3, col4 = st.columns(4)

# Find best model
best_model = max(model_stats.keys(), key=lambda k: model_stats[k]["mean_reward"])
best_reward = model_stats[best_model]["mean_reward"]

# Calculate PPO win rate vs base
pairwise = comparison.get("pairwise_comparisons", {})
ppo_vs_base = pairwise.get("ppo_vs_base", {})
ppo_win_rate = ppo_vs_base.get("ppo", 0) / 100 * 100 if ppo_vs_base else 0

with col1:
    st.metric(
        label="Best Model",
        value=best_model.upper(),
        delta=f"Reward: {best_reward:.4f}",
    )

with col2:
    st.metric(
        label="PPO Win Rate vs Base",
        value=f"{ppo_win_rate:.0f}%",
        delta=f"{ppo_vs_base.get('ppo', 0)} wins",
    )

with col3:
    st.metric(
        label="Test Samples",
        value="100",
    )

with col4:
    timestamp = comparison.get("timestamp", "N/A")
    if timestamp != "N/A":
        # Format timestamp
        date_part = timestamp.split("T")[0] if "T" in timestamp else timestamp[:10]
        st.metric(label="Experiment Date", value=date_part)
    else:
        st.metric(label="Experiment Date", value="N/A")

st.markdown("---")

# Mean Reward Bar Chart
st.markdown("### Mean Reward by Model")
fig_bar = create_reward_bar_chart(model_stats)
st.plotly_chart(fig_bar, use_container_width=True)

# Two columns for additional charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Reward Distribution")
    fig_dist = create_reward_distribution(data["full_results"])
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.markdown("### Average Response Length")
    fig_len = create_response_length_chart(model_stats)
    st.plotly_chart(fig_len, use_container_width=True)

st.markdown("---")

# Statistics Table
st.markdown("### Detailed Statistics")
render_stats_table(model_stats)

# Key findings
st.markdown("---")
st.markdown("### Key Findings")

# Determine rankings
models_by_reward = sorted(model_stats.keys(), key=lambda k: model_stats[k]["mean_reward"], reverse=True)

st.markdown(f"""
Based on the evaluation of 100 held-out test samples:

1. **{models_by_reward[0].upper()}** achieves the highest mean reward ({model_stats[models_by_reward[0]]['mean_reward']:.4f})
2. **{models_by_reward[1].upper()}** ranks second ({model_stats[models_by_reward[1]]['mean_reward']:.4f})
3. **{models_by_reward[2].upper()}** ranks third ({model_stats[models_by_reward[2]]['mean_reward']:.4f})
4. **{models_by_reward[3].upper()}** ranks fourth ({model_stats[models_by_reward[3]]['mean_reward']:.4f})

The reward scores are computed using a trained reward model on the HH-RLHF dataset.
""")
