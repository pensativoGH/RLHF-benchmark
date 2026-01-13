"""Model loading utilities with Streamlit caching."""

import json
import sys
from pathlib import Path

import streamlit as st
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.reward_model import RewardModel


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "benchmark_comparison.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading base model and tokenizer...")
def _load_base_model(model_name: str, device: str):
    """Load the base model and tokenizer (cached)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    return model, tokenizer


@st.cache_resource(show_spinner="Loading aligned model...")
def _load_aligned_model(base_model_name: str, adapter_path: str, device: str):
    """Load a model with LoRA adapter (cached)."""
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(device)
    model.eval()

    return model


@st.cache_resource(show_spinner="Loading all models (this may take a minute)...")
def load_all_models(config_path: str = None, device: str = "mps"):
    """Load and cache all models for generation."""
    config = load_config(config_path)
    base_name = config["models"]["base"]

    models = {}

    # Load base model and tokenizer
    base_model, tokenizer = _load_base_model(base_name, device)
    models["base"] = base_model

    # Load aligned models with adapters
    checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"

    for model_name in ["dpo", "ppo", "grpo"]:
        adapter_path = str(checkpoint_dir / f"{model_name}_model" / "final")
        models[model_name] = _load_aligned_model(base_name, adapter_path, device)

    return models, tokenizer, config


@st.cache_resource(show_spinner="Loading reward model...")
def load_reward_model(config_path: str = None, device: str = "mps"):
    """Load and cache the reward model."""
    config = load_config(config_path)
    checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "reward_model" / "best"

    reward_model = RewardModel.load(
        checkpoint_path=str(checkpoint_path),
        model_name=config["models"]["base"],
        device=device,
    )

    return reward_model


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "mps",
) -> str:
    """Generate a response for a given prompt."""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return response.strip()


@st.cache_data
def load_results_data():
    """Load all results data from JSON files."""
    results_dir = Path(__file__).parent.parent.parent / "results" / "experiment_j"

    data = {}

    # Load comparison results
    with open(results_dir / "comparison_results.json", 'r') as f:
        data["comparison"] = json.load(f)

    # Load pairwise matrix
    with open(results_dir / "pairwise_matrix.json", 'r') as f:
        data["pairwise"] = json.load(f)

    # Load full results (100 examples)
    with open(results_dir / "full_results.json", 'r') as f:
        data["full_results"] = json.load(f)

    return data


@st.cache_data
def load_training_history(model_name: str):
    """Load training history for a specific model."""
    results_dir = Path(__file__).parent.parent.parent / "results"

    model_dirs = {
        "dpo": "m2_dpo",
        "ppo": "m3_ppo",
        "grpo": "m4_grpo",
    }

    if model_name not in model_dirs:
        return None

    history_path = results_dir / model_dirs[model_name] / "logs" / "training_history.json"

    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)

    return None


def get_test_prompts(data: dict = None) -> list:
    """Get list of test prompts from full results."""
    if data is None:
        data = load_results_data()

    prompts = []
    for i, item in enumerate(data["full_results"]):
        prompt = item.get("prompt", "")
        # Create preview (first 80 chars)
        preview = prompt[:80].replace("\n", " ")
        if len(prompt) > 80:
            preview += "..."
        prompts.append({"index": i, "preview": preview, "full": prompt})

    return prompts
