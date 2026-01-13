"""
Training Curves Page
====================

Visualize training dynamics for each alignment method.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.components.model_loader import load_training_history
from dashboard.components.charts import create_training_curve, create_multi_metric_training_chart

st.set_page_config(
    page_title="Training Curves | RLHF Dashboard",
    page_icon="",
    layout="wide",
)

st.title(" Training Curves")
st.markdown("Visualize training dynamics and metrics over time")
st.markdown("---")

# Model selector
model_option = st.radio(
    "Select Model",
    options=["DPO", "PPO", "GRPO"],
    horizontal=True,
)

model_name = model_option.lower()

# Load training history
history = load_training_history(model_name)

if history is None or len(history) == 0:
    st.warning(f"No training history found for {model_option}. Make sure training logs exist.")
    st.stop()

st.markdown(f"### {model_option} Training Metrics")
st.markdown(f"Showing {len(history)} recorded training steps.")

# Common metrics
st.markdown("#### Loss Curve")
fig_loss = create_training_curve(history, "loss", f"{model_option} Training Loss")
if fig_loss:
    st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info("Loss metric not available in training history.")

# Model-specific metrics
if model_name == "dpo":
    st.markdown("---")
    st.markdown("#### DPO-Specific Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig_chosen = create_training_curve(history, "rewards/chosen", "Chosen Rewards")
        if fig_chosen:
            st.plotly_chart(fig_chosen, use_container_width=True)

    with col2:
        fig_rejected = create_training_curve(history, "rewards/rejected", "Rejected Rewards")
        if fig_rejected:
            st.plotly_chart(fig_rejected, use_container_width=True)

    # Reward margins
    fig_margins = create_training_curve(history, "rewards/margins", "Reward Margins (Chosen - Rejected)")
    if fig_margins:
        st.plotly_chart(fig_margins, use_container_width=True)

    # Multi-metric view
    st.markdown("#### Combined View")
    metrics = ["rewards/chosen", "rewards/rejected", "rewards/margins"]
    fig_multi = create_multi_metric_training_chart(history, metrics)
    if fig_multi:
        st.plotly_chart(fig_multi, use_container_width=True)

elif model_name == "ppo":
    st.markdown("---")
    st.markdown("#### PPO-Specific Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig_reward = create_training_curve(history, "objective/kl", "KL Divergence")
        if fig_reward:
            st.plotly_chart(fig_reward, use_container_width=True)
        else:
            fig_reward = create_training_curve(history, "ppo/mean_non_score_reward", "Mean Reward")
            if fig_reward:
                st.plotly_chart(fig_reward, use_container_width=True)

    with col2:
        fig_kl = create_training_curve(history, "objective/entropy", "Entropy")
        if fig_kl:
            st.plotly_chart(fig_kl, use_container_width=True)

    # Policy loss
    fig_policy = create_training_curve(history, "ppo/loss/policy", "Policy Loss")
    if fig_policy:
        st.plotly_chart(fig_policy, use_container_width=True)

    # Value loss
    fig_value = create_training_curve(history, "ppo/loss/value", "Value Loss")
    if fig_value:
        st.plotly_chart(fig_value, use_container_width=True)

elif model_name == "grpo":
    st.markdown("---")
    st.markdown("#### GRPO-Specific Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig_reward = create_training_curve(history, "mean_reward", "Mean Reward")
        if fig_reward:
            st.plotly_chart(fig_reward, use_container_width=True)

    with col2:
        fig_kl = create_training_curve(history, "kl_divergence", "KL Divergence")
        if fig_kl:
            st.plotly_chart(fig_kl, use_container_width=True)

    # Policy loss
    fig_policy = create_training_curve(history, "policy_loss", "Policy Loss")
    if fig_policy:
        st.plotly_chart(fig_policy, use_container_width=True)

# Learning rate schedule
st.markdown("---")
st.markdown("#### Learning Rate Schedule")
fig_lr = create_training_curve(history, "learning_rate", "Learning Rate")
if fig_lr:
    st.plotly_chart(fig_lr, use_container_width=True)
else:
    st.info("Learning rate not recorded in training history.")

# Raw data expander
with st.expander("View Raw Training History"):
    import pandas as pd
    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True)
