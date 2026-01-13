"""Chart components for the RLHF dashboard."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# Color scheme for models
MODEL_COLORS = {
    "base": "#6366f1",   # Indigo
    "dpo": "#f59e0b",    # Amber
    "ppo": "#10b981",    # Emerald
    "grpo": "#ec4899",   # Pink
}


def create_reward_bar_chart(model_stats: dict) -> go.Figure:
    """Create bar chart of mean rewards with error bars."""
    models = ["base", "dpo", "ppo", "grpo"]

    data = {
        "Model": [m.upper() for m in models],
        "Mean Reward": [model_stats[m]["mean_reward"] for m in models],
        "Std": [model_stats[m]["std_reward"] for m in models],
    }
    df = pd.DataFrame(data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Model"],
        y=df["Mean Reward"],
        error_y=dict(type='data', array=df["Std"], visible=True),
        marker_color=[MODEL_COLORS[m] for m in models],
        text=[f"{v:.4f}" for v in df["Mean Reward"]],
        textposition='outside',
    ))

    fig.update_layout(
        title="Mean Reward Score by Model",
        xaxis_title="Model",
        yaxis_title="Mean Reward",
        showlegend=False,
        height=400,
    )

    return fig


def create_win_rate_heatmap(pairwise_data: dict) -> go.Figure:
    """Create 4x4 win rate matrix heatmap."""
    models = ["base", "dpo", "ppo", "grpo"]
    n = len(models)

    # Initialize matrix with None (diagonal)
    matrix = [[None for _ in range(n)] for _ in range(n)]
    text_matrix = [["-" for _ in range(n)] for _ in range(n)]

    # Mapping of comparisons to matrix positions
    comparisons = {
        "dpo_vs_base": ("dpo", "base"),
        "ppo_vs_base": ("ppo", "base"),
        "grpo_vs_base": ("grpo", "base"),
        "dpo_vs_ppo": ("dpo", "ppo"),
        "dpo_vs_grpo": ("dpo", "grpo"),
        "ppo_vs_grpo": ("ppo", "grpo"),
    }

    for key, (m1, m2) in comparisons.items():
        if key in pairwise_data:
            counts = pairwise_data[key]
            total = sum(counts.values())

            i1, i2 = models.index(m1), models.index(m2)

            # Win rate of m1 vs m2
            win_rate_m1 = counts.get(m1, 0) / total * 100
            win_rate_m2 = counts.get(m2, 0) / total * 100

            matrix[i1][i2] = win_rate_m1
            matrix[i2][i1] = win_rate_m2
            text_matrix[i1][i2] = f"{win_rate_m1:.0f}%"
            text_matrix[i2][i1] = f"{win_rate_m2:.0f}%"

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[m.upper() for m in models],
        y=[m.upper() for m in models],
        colorscale="RdYlGn",
        zmin=0,
        zmax=100,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="Row %{y} vs Column %{x}: %{z:.1f}% win rate<extra></extra>",
    ))

    fig.update_layout(
        title="Win Rate Matrix (Row Model vs Column Model)",
        xaxis_title="Opponent",
        yaxis_title="Model",
        height=400,
    )

    return fig


def create_pairwise_bars(pairwise_data: dict) -> go.Figure:
    """Create grouped bar chart for all pairwise comparisons."""
    comparisons = [
        ("dpo_vs_base", "DPO vs BASE"),
        ("ppo_vs_base", "PPO vs BASE"),
        ("grpo_vs_base", "GRPO vs BASE"),
        ("dpo_vs_ppo", "DPO vs PPO"),
        ("dpo_vs_grpo", "DPO vs GRPO"),
        ("ppo_vs_grpo", "PPO vs GRPO"),
    ]

    data = []
    for key, label in comparisons:
        if key in pairwise_data:
            counts = pairwise_data[key]
            models = [k for k in counts.keys() if k != "tie"]
            data.append({
                "Comparison": label,
                f"{models[0].upper()} Wins": counts.get(models[0], 0),
                f"{models[1].upper()} Wins": counts.get(models[1], 0),
                "Ties": counts.get("tie", 0),
            })

    df = pd.DataFrame(data)

    fig = go.Figure()

    # Add bars for each category
    colors = ["#10b981", "#ef4444", "#6b7280"]  # Green, Red, Gray

    for i, col in enumerate(df.columns[1:]):
        fig.add_trace(go.Bar(
            name=col.split()[0] if "Wins" in col else col,
            x=df["Comparison"],
            y=df[col],
            marker_color=colors[i % len(colors)],
            text=df[col],
            textposition='auto',
        ))

    fig.update_layout(
        title="Pairwise Comparison Results",
        xaxis_title="Matchup",
        yaxis_title="Count",
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_training_curve(history: list, metric: str = "loss", title: str = None) -> go.Figure:
    """Create line chart for training metrics over steps."""
    if not history:
        return None

    steps = []
    values = []

    for entry in history:
        if metric in entry:
            steps.append(entry.get("step", len(steps)))
            values.append(entry[metric])

    if not values:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps,
        y=values,
        mode='lines+markers',
        name=metric,
        line=dict(color="#6366f1", width=2),
        marker=dict(size=6),
    ))

    fig.update_layout(
        title=title or f"{metric.replace('_', ' ').title()} over Training",
        xaxis_title="Step",
        yaxis_title=metric.replace("_", " ").title(),
        height=350,
    )

    return fig


def create_reward_distribution(full_results: list) -> go.Figure:
    """Create box plots of reward distributions by model."""
    models = ["base", "dpo", "ppo", "grpo"]

    data = []
    for result in full_results:
        for model in models:
            reward_key = f"{model}_reward"
            if reward_key in result:
                data.append({
                    "Model": model.upper(),
                    "Reward": result[reward_key],
                })

    df = pd.DataFrame(data)

    fig = px.box(
        df,
        x="Model",
        y="Reward",
        color="Model",
        color_discrete_map={m.upper(): MODEL_COLORS[m] for m in models},
    )

    fig.update_layout(
        title="Reward Score Distribution by Model",
        xaxis_title="Model",
        yaxis_title="Reward Score",
        showlegend=False,
        height=400,
    )

    return fig


def create_response_length_chart(model_stats: dict) -> go.Figure:
    """Create bar chart of average response lengths."""
    models = ["base", "dpo", "ppo", "grpo"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[m.upper() for m in models],
        y=[model_stats[m]["mean_length"] for m in models],
        marker_color=[MODEL_COLORS[m] for m in models],
        text=[f"{model_stats[m]['mean_length']:.0f}" for m in models],
        textposition='outside',
    ))

    fig.update_layout(
        title="Average Response Length by Model",
        xaxis_title="Model",
        yaxis_title="Average Length (words)",
        showlegend=False,
        height=350,
    )

    return fig


def create_multi_metric_training_chart(history: list, metrics: list) -> go.Figure:
    """Create multi-line chart for multiple training metrics."""
    if not history:
        return None

    fig = go.Figure()

    colors = ["#6366f1", "#10b981", "#f59e0b", "#ec4899"]

    for i, metric in enumerate(metrics):
        steps = []
        values = []

        for entry in history:
            if metric in entry:
                steps.append(entry.get("step", len(steps)))
                values.append(entry[metric])

        if values:
            fig.add_trace(go.Scatter(
                x=steps,
                y=values,
                mode='lines+markers',
                name=metric.replace("_", " ").title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
            ))

    fig.update_layout(
        title="Training Metrics over Time",
        xaxis_title="Step",
        yaxis_title="Value",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
