"""Card components for the RLHF dashboard."""

import streamlit as st


# Model color scheme
MODEL_COLORS = {
    "base": ("#e0e7ff", "#3730a3"),   # Light indigo bg, dark indigo text
    "dpo": ("#fef3c7", "#92400e"),    # Light amber bg, dark amber text
    "ppo": ("#d1fae5", "#065f46"),    # Light emerald bg, dark emerald text
    "grpo": ("#fce7f3", "#9d174d"),   # Light pink bg, dark pink text
}


def render_kpi_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Render a KPI metric card."""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color,
    )


def render_model_badge(model_name: str):
    """Render a colored badge for a model."""
    bg_color, text_color = MODEL_COLORS.get(model_name.lower(), ("#f3f4f6", "#374151"))
    st.markdown(
        f'<span style="background-color: {bg_color}; color: {text_color}; '
        f'padding: 4px 12px; border-radius: 16px; font-weight: 600; font-size: 14px;">'
        f'{model_name.upper()}</span>',
        unsafe_allow_html=True,
    )


def render_response_card(
    model_name: str,
    response: str,
    reward: float = None,
    is_winner: bool = False,
    show_full: bool = False,
):
    """Render a response card with optional reward score."""
    bg_color, text_color = MODEL_COLORS.get(model_name.lower(), ("#f3f4f6", "#374151"))

    # Winner styling
    border_style = "3px solid #10b981" if is_winner else "1px solid #e5e7eb"
    winner_badge = " (Winner)" if is_winner else ""

    with st.container():
        # Header with model name
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f'<span style="background-color: {bg_color}; color: {text_color}; '
                f'padding: 4px 12px; border-radius: 16px; font-weight: 600;">'
                f'{model_name.upper()}{winner_badge}</span>',
                unsafe_allow_html=True,
            )
        with col2:
            if reward is not None:
                reward_color = "#10b981" if reward > 0 else "#ef4444"
                st.markdown(
                    f'<span style="color: {reward_color}; font-weight: 600;">'
                    f'Score: {reward:.4f}</span>',
                    unsafe_allow_html=True,
                )

        # Response content
        st.markdown(
            f'<div style="border: {border_style}; border-radius: 8px; '
            f'padding: 12px; margin-top: 8px; background-color: #fafafa; '
            f'max-height: {"none" if show_full else "200px"}; overflow-y: auto;">'
            f'{response}</div>',
            unsafe_allow_html=True,
        )


def render_prompt_viewer(prompt: str, max_chars: int = 500):
    """Render an expandable prompt display."""
    with st.expander("View Full Prompt", expanded=False):
        st.markdown(
            f'<div style="background-color: #f8fafc; padding: 12px; '
            f'border-radius: 8px; border-left: 4px solid #6366f1; '
            f'font-family: monospace; white-space: pre-wrap;">{prompt}</div>',
            unsafe_allow_html=True,
        )


def render_comparison_result(
    prompt: str,
    responses: dict,
    rewards: dict,
):
    """Render a full comparison view for one example."""
    # Find winner
    winner = max(rewards.keys(), key=lambda k: rewards[k])

    # Show prompt
    render_prompt_viewer(prompt)

    # Show responses in 2x2 grid
    col1, col2 = st.columns(2)

    models = ["base", "dpo", "ppo", "grpo"]
    for i, model in enumerate(models):
        with col1 if i % 2 == 0 else col2:
            if model in responses:
                render_response_card(
                    model_name=model,
                    response=responses.get(model, "N/A"),
                    reward=rewards.get(model),
                    is_winner=(model == winner),
                )
            st.markdown("<br>", unsafe_allow_html=True)


def render_stats_table(model_stats: dict):
    """Render a formatted statistics table."""
    import pandas as pd

    models = ["base", "dpo", "ppo", "grpo"]

    data = []
    for model in models:
        stats = model_stats.get(model, {})
        data.append({
            "Model": model.upper(),
            "Mean Reward": f"{stats.get('mean_reward', 0):.4f}",
            "Std": f"{stats.get('std_reward', 0):.4f}",
            "Min": f"{stats.get('min_reward', 0):.4f}",
            "Max": f"{stats.get('max_reward', 0):.4f}",
            "Median": f"{stats.get('median_reward', 0):.4f}",
            "Avg Length": f"{stats.get('mean_length', 0):.1f}",
        })

    df = pd.DataFrame(data)

    # Highlight best values
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def render_generation_status(model_name: str, status: str):
    """Render generation status indicator."""
    status_colors = {
        "pending": "#6b7280",
        "generating": "#f59e0b",
        "complete": "#10b981",
        "error": "#ef4444",
    }

    color = status_colors.get(status, "#6b7280")
    icon = {"pending": "", "generating": "", "complete": "", "error": ""}.get(status, "")

    st.markdown(
        f'<span style="color: {color};">{icon} {model_name.upper()}: {status.title()}</span>',
        unsafe_allow_html=True,
    )
