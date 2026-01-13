"""
Interactive Generation Page
===========================

Generate responses from models in real-time.
"""

import streamlit as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.components.model_loader import (
    load_all_models,
    load_reward_model,
    load_results_data,
    get_test_prompts,
    generate_response,
    load_config,
)
from dashboard.components.cards import render_response_card

st.set_page_config(
    page_title="Interactive Generation | RLHF Dashboard",
    page_icon="",
    layout="wide",
)

st.title(" Interactive Generation")
st.markdown("Generate responses from different models in real-time")
st.markdown("---")

# Initialize session state
if "generated_responses" not in st.session_state:
    st.session_state.generated_responses = {}
if "generation_times" not in st.session_state:
    st.session_state.generation_times = {}

# Sidebar configuration
with st.sidebar:
    st.markdown("### Configuration")

    # Model selection
    st.markdown("#### Models to Generate")
    use_all = st.checkbox("Generate from all models", value=True)

    if use_all:
        selected_models = ["base", "dpo", "ppo", "grpo"]
    else:
        selected_models = []
        if st.checkbox("Base", value=True):
            selected_models.append("base")
        if st.checkbox("DPO", value=True):
            selected_models.append("dpo")
        if st.checkbox("PPO", value=True):
            selected_models.append("ppo")
        if st.checkbox("GRPO", value=True):
            selected_models.append("grpo")

    st.markdown("---")

    # Generation parameters
    st.markdown("#### Generation Parameters")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    max_tokens = st.slider("Max New Tokens", 32, 512, 256, 32)

    st.markdown("---")

    # Score with reward model
    score_responses = st.checkbox("Score with Reward Model", value=True)

    st.markdown("---")

    # Clear cache button
    if st.button("Clear Model Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared! Models will be reloaded on next generation.")

# Main content area
# Load test prompts
try:
    data = load_results_data()
    prompts = get_test_prompts(data)
except Exception as e:
    st.error(f"Error loading test prompts: {e}")
    st.stop()

# Prompt selector
st.markdown("### Select a Test Prompt")

col1, col2 = st.columns([2, 1])

with col1:
    prompt_options = [f"#{p['index'] + 1}: {p['preview']}" for p in prompts]
    selected_idx = st.selectbox(
        "Choose from 100 test examples",
        options=range(len(prompts)),
        format_func=lambda i: prompt_options[i],
        key="prompt_selector",
    )

with col2:
    st.markdown("")
    st.markdown("")
    generate_btn = st.button(
        "Generate Responses",
        type="primary",
        disabled=len(selected_models) == 0,
    )

# Show selected prompt
if selected_idx is not None:
    full_prompt = prompts[selected_idx]["full"]

    with st.expander("View Full Prompt", expanded=True):
        st.markdown(
            f'<div style="background-color: #f8fafc; padding: 12px; '
            f'border-radius: 8px; border-left: 4px solid #6366f1; '
            f'white-space: pre-wrap; font-family: monospace; max-height: 300px; overflow-y: auto;">'
            f'{full_prompt}</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# Generation
if generate_btn:
    if len(selected_models) == 0:
        st.warning("Please select at least one model.")
    else:
        # Load models
        with st.spinner("Loading models (first time may take ~30 seconds)..."):
            try:
                config = load_config()
                models, tokenizer, _ = load_all_models(device=config.get("device", "mps"))

                if score_responses:
                    reward_model = load_reward_model(device=config.get("device", "mps"))
                else:
                    reward_model = None

            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.stop()

        # Generate responses
        st.session_state.generated_responses = {}
        st.session_state.generation_times = {}
        rewards = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, model_name in enumerate(selected_models):
            status_text.text(f"Generating from {model_name.upper()}...")

            start_time = time.time()

            try:
                response = generate_response(
                    model=models[model_name],
                    tokenizer=tokenizer,
                    prompt=full_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    device=config.get("device", "mps"),
                )
                st.session_state.generated_responses[model_name] = response
                st.session_state.generation_times[model_name] = time.time() - start_time

                # Score if requested
                if reward_model is not None:
                    reward = reward_model.score(full_prompt, response, tokenizer)
                    rewards[model_name] = reward

            except Exception as e:
                st.session_state.generated_responses[model_name] = f"Error: {e}"
                st.session_state.generation_times[model_name] = 0
                rewards[model_name] = None

            progress_bar.progress((i + 1) / len(selected_models))

        status_text.text("Generation complete!")
        progress_bar.empty()

        # Store rewards in session state
        st.session_state.rewards = rewards

        st.rerun()

# Display results
if st.session_state.generated_responses:
    st.markdown("### Generated Responses")

    responses = st.session_state.generated_responses
    times = st.session_state.generation_times
    rewards = st.session_state.get("rewards", {})

    # Find winner if we have rewards
    winner = None
    if rewards and any(r is not None for r in rewards.values()):
        valid_rewards = {k: v for k, v in rewards.items() if v is not None}
        if valid_rewards:
            winner = max(valid_rewards.keys(), key=lambda k: valid_rewards[k])

    # Display in 2x2 grid
    models_to_show = list(responses.keys())

    if len(models_to_show) == 1:
        # Single model - full width
        model = models_to_show[0]
        render_response_card(
            model_name=model,
            response=responses[model],
            reward=rewards.get(model),
            is_winner=(model == winner),
            show_full=True,
        )
        st.caption(f"Generation time: {times.get(model, 0):.2f}s")

    elif len(models_to_show) == 2:
        # Two models - side by side
        col1, col2 = st.columns(2)
        for i, model in enumerate(models_to_show):
            with col1 if i == 0 else col2:
                render_response_card(
                    model_name=model,
                    response=responses[model],
                    reward=rewards.get(model),
                    is_winner=(model == winner),
                )
                st.caption(f"Generation time: {times.get(model, 0):.2f}s")

    else:
        # 3-4 models - 2x2 grid
        col1, col2 = st.columns(2)

        for i, model in enumerate(models_to_show):
            with col1 if i % 2 == 0 else col2:
                render_response_card(
                    model_name=model,
                    response=responses[model],
                    reward=rewards.get(model),
                    is_winner=(model == winner),
                )
                st.caption(f"Generation time: {times.get(model, 0):.2f}s")
                st.markdown("")

    # Summary table
    if rewards and any(r is not None for r in rewards.values()):
        st.markdown("---")
        st.markdown("### Reward Score Summary")

        import pandas as pd

        summary_data = []
        for model in models_to_show:
            reward = rewards.get(model)
            summary_data.append({
                "Model": model.upper(),
                "Reward Score": f"{reward:.4f}" if reward is not None else "N/A",
                "Gen Time (s)": f"{times.get(model, 0):.2f}",
                "Winner": "" if model == winner else "",
            })

        # Sort by reward
        summary_data.sort(key=lambda x: float(x["Reward Score"]) if x["Reward Score"] != "N/A" else -999, reverse=True)

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    # No results yet
    st.info("Select a prompt and click 'Generate Responses' to see model outputs.")

    # Show existing results from Experiment J for reference
    st.markdown("---")
    st.markdown("### Pre-computed Results (from Experiment J)")
    st.markdown("Below are the responses that were generated during the benchmark comparison.")

    if selected_idx is not None and selected_idx < len(data["full_results"]):
        result = data["full_results"][selected_idx]

        rewards = {
            "base": result.get("base_reward", 0),
            "dpo": result.get("dpo_reward", 0),
            "ppo": result.get("ppo_reward", 0),
            "grpo": result.get("grpo_reward", 0),
        }
        winner = max(rewards.keys(), key=lambda k: rewards[k])

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
