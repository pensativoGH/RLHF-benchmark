"""
RLHF Experiment Dashboard
=========================

Main entry point for the Streamlit dashboard.
Provides overview and navigation to different analysis pages.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="RLHF Experiment Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    # Header
    st.title(" RLHF Experiment Dashboard")
    st.markdown("---")

    # Overview
    st.markdown("""
    Welcome to the **RLHF Training & Evaluation Dashboard**. This dashboard provides:

    - **Results Summary**: Comprehensive metrics from Experiment J comparing DPO, PPO, and GRPO
    - **Model Comparison**: Win rate matrices and pairwise analysis
    - **Training Curves**: Visualize training dynamics for each method
    - **Interactive Generation**: Generate responses from any model in real-time
    """)

    # Quick status
    st.markdown("### Quick Status")

    col1, col2, col3, col4 = st.columns(4)

    # Check data availability
    results_dir = Path(__file__).parent.parent / "results" / "experiment_j"
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

    with col1:
        experiment_j_ready = (results_dir / "comparison_results.json").exists()
        status = "" if experiment_j_ready else ""
        st.metric("Experiment J Data", status)

    with col2:
        models_ready = all([
            (checkpoints_dir / "dpo_model" / "final").exists(),
            (checkpoints_dir / "ppo_model" / "final").exists(),
            (checkpoints_dir / "grpo_model" / "final").exists(),
        ])
        status = "" if models_ready else ""
        st.metric("Model Checkpoints", status)

    with col3:
        reward_model_ready = (checkpoints_dir / "reward_model" / "best").exists()
        status = "" if reward_model_ready else ""
        st.metric("Reward Model", status)

    with col4:
        st.metric("Test Samples", "100")

    st.markdown("---")

    # Navigation guide
    st.markdown("### Navigation")
    st.markdown("""
    Use the **sidebar** to navigate between pages:

    | Page | Description |
    |------|-------------|
    |  **Results Summary** | KPIs, reward bar charts, model statistics |
    |  **Model Comparison** | Win rate heatmap, pairwise analysis, sample viewer |
    |  **Training Curves** | Loss curves, reward progression over training |
    |  **Interactive Generation** | Generate responses from models in real-time |
    """)

    # Project info
    st.markdown("---")
    st.markdown("### About This Project")

    st.markdown("""
    This dashboard visualizes results from training and evaluating three alignment methods:

    - **DPO** (Direct Preference Optimization): Reference-free offline alignment
    - **PPO** (Proximal Policy Optimization): Online RL with reward model feedback
    - **GRPO** (Group Relative Policy Optimization): Group-based reward normalization

    All methods were trained on the **Anthropic HH-RLHF** dataset using **Qwen2-0.5B** as the base model
    with LoRA adapters for parameter efficiency.
    """)

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #6b7280; font-size: 12px;">'
        'Built with Streamlit | RLHF Experiments'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
