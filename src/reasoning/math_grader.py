"""
Math Answer Grading for RLVR

Provides functions to:
1. Extract answers from \boxed{...} format
2. Normalize math expressions for comparison
3. Grade answers against ground truth
4. Compute binary RLVR rewards
"""

import re
from fractions import Fraction
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \boxed{...} format in model response.

    Handles nested braces by counting brace depth.

    Args:
        text: Model response text.

    Returns:
        Extracted answer string, or None if no \boxed{} found.

    Examples:
        >>> extract_boxed_answer("The answer is \\boxed{42}")
        '42'
        >>> extract_boxed_answer("Therefore \\boxed{\\frac{1}{2}}")
        '\\frac{1}{2}'
        >>> extract_boxed_answer("No boxed answer here")
        None
    """
    # Find all \boxed{ occurrences and extract the last one
    # (the final answer is typically at the end)
    pattern = r'\\boxed\{'

    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Use the last \boxed{} occurrence
    last_match = matches[-1]
    start_idx = last_match.end()

    # Count braces to find matching closing brace
    depth = 1
    idx = start_idx
    while idx < len(text) and depth > 0:
        if text[idx] == '{':
            depth += 1
        elif text[idx] == '}':
            depth -= 1
        idx += 1

    if depth != 0:
        return None  # Unmatched braces

    # Extract content between braces
    content = text[start_idx:idx - 1]
    return content.strip()


def normalize_answer(text: str) -> str:
    """
    Normalize a math answer for comparison.

    Handles:
    - Whitespace normalization
    - LaTeX command removal (frac, sqrt, etc.)
    - Fraction conversion (1/2 -> 0.5 for numeric comparison)
    - Common formatting variations

    Args:
        text: Raw answer string.

    Returns:
        Normalized answer string.
    """
    if text is None:
        return ""

    result = text.strip()

    # Remove common LaTeX formatting that doesn't change value
    result = result.replace("\\left", "")
    result = result.replace("\\right", "")
    result = result.replace("\\,", " ")
    result = result.replace("\\;", " ")
    result = result.replace("\\!", "")
    result = result.replace("\\quad", " ")
    result = result.replace("\\qquad", " ")

    # Handle \text{} by extracting content
    result = re.sub(r'\\text\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', result)

    # Handle \dfrac and \tfrac same as \frac
    result = result.replace("\\dfrac", "\\frac")
    result = result.replace("\\tfrac", "\\frac")

    # Normalize whitespace
    result = " ".join(result.split())

    # Remove trailing periods, commas
    result = result.rstrip(".,;")

    return result


def latex_frac_to_decimal(text: str) -> Optional[float]:
    """
    Try to convert LaTeX fraction to decimal.

    Args:
        text: String potentially containing \frac{a}{b}.

    Returns:
        Decimal value if conversion successful, None otherwise.
    """
    # Match \frac{numerator}{denominator}
    match = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}$', text.strip())
    if match:
        try:
            num = float(match.group(1))
            denom = float(match.group(2))
            if denom != 0:
                return num / denom
        except ValueError:
            pass
    return None


def try_parse_number(text: str) -> Optional[float]:
    """
    Try to parse text as a number.

    Handles:
    - Integers: 42
    - Decimals: 3.14
    - Fractions: 1/2
    - Negative numbers: -5
    - Scientific notation: 1e-3

    Args:
        text: String to parse.

    Returns:
        Float value if parseable, None otherwise.
    """
    text = text.strip()

    # Try direct float parse
    try:
        return float(text)
    except ValueError:
        pass

    # Try fraction format a/b
    if "/" in text and "\\" not in text:
        parts = text.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass

    # Try LaTeX fraction
    decimal = latex_frac_to_decimal(text)
    if decimal is not None:
        return decimal

    return None


def answers_match(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """
    Check if two answers are mathematically equivalent.

    Tries multiple comparison strategies:
    1. Exact string match (after normalization)
    2. Numeric comparison (if both parseable as numbers)
    3. Symbolic comparison (for fractions, expressions)

    Args:
        predicted: Predicted answer.
        ground_truth: Ground truth answer.
        tolerance: Tolerance for numeric comparison.

    Returns:
        True if answers match, False otherwise.
    """
    # Normalize both
    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(ground_truth)

    # Exact string match
    if pred_norm == truth_norm:
        return True

    # Case-insensitive comparison for text answers
    if pred_norm.lower() == truth_norm.lower():
        return True

    # Try numeric comparison
    pred_num = try_parse_number(pred_norm)
    truth_num = try_parse_number(truth_norm)

    if pred_num is not None and truth_num is not None:
        if abs(pred_num - truth_num) < tolerance:
            return True
        # Also check relative tolerance for large numbers
        if truth_num != 0 and abs((pred_num - truth_num) / truth_num) < tolerance:
            return True

    # Handle LaTeX fractions vs decimal comparison
    pred_frac = latex_frac_to_decimal(pred_norm)
    truth_frac = latex_frac_to_decimal(truth_norm)

    if pred_frac is not None and truth_num is not None:
        if abs(pred_frac - truth_num) < tolerance:
            return True

    if pred_num is not None and truth_frac is not None:
        if abs(pred_num - truth_frac) < tolerance:
            return True

    if pred_frac is not None and truth_frac is not None:
        if abs(pred_frac - truth_frac) < tolerance:
            return True

    return False


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """
    Grade a predicted answer against ground truth.

    Args:
        predicted: Predicted answer (already extracted from \boxed{}).
        ground_truth: Ground truth answer.

    Returns:
        True if correct, False if wrong.
    """
    if predicted is None or ground_truth is None:
        return False

    return answers_match(predicted, ground_truth)


def reward_rlvr(response: str, ground_truth: str) -> float:
    """
    Compute binary RLVR reward for a model response.

    Reward is 1.0 if:
    - Response contains a \boxed{} answer
    - The answer matches ground truth

    Reward is 0.0 if:
    - No \boxed{} in response (format violation)
    - Answer is wrong

    Args:
        response: Full model response text.
        ground_truth: Ground truth answer.

    Returns:
        1.0 for correct, 0.0 for wrong or bad format.
    """
    extracted = extract_boxed_answer(response)
    if extracted is None:
        return 0.0  # Format violation
    return 1.0 if grade_answer(extracted, ground_truth) else 0.0


def compute_advantages(rewards: list[float], epsilon: float = 1e-4) -> list[float]:
    """
    Compute normalized advantages from rewards.

    Advantages are computed as: (reward - mean) / (std + epsilon)

    This is used in GRPO to determine which rollouts to reinforce.

    Args:
        rewards: List of rewards for a group of rollouts.
        epsilon: Small constant for numerical stability.

    Returns:
        List of normalized advantages.
    """
    if len(rewards) == 0:
        return []

    if len(rewards) == 1:
        return [0.0]  # Single rollout has zero advantage

    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = variance ** 0.5

    advantages = [(r - mean) / (std + epsilon) for r in rewards]
    return advantages


# Test cases
if __name__ == "__main__":
    print("Testing extract_boxed_answer:")
    test_cases = [
        ("The answer is \\boxed{42}", "42"),
        ("Therefore \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("We get \\boxed{x^2 + 1}", "x^2 + 1"),
        ("Step 1: \\boxed{5} Final: \\boxed{10}", "10"),  # Should get last
        ("No boxed answer", None),
        ("\\boxed{nested {braces} work}", "nested {braces} work"),
    ]

    for text, expected in test_cases:
        result = extract_boxed_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:40]}...' -> '{result}' (expected '{expected}')")

    print("\nTesting normalize_answer:")
    test_cases = [
        ("  42  ", "42"),
        ("\\frac{1}{2}", "\\frac{1}{2}"),
        ("\\dfrac{1}{2}", "\\frac{1}{2}"),
        ("x \\quad y", "x y"),
    ]

    for text, expected in test_cases:
        result = normalize_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' -> '{result}' (expected '{expected}')")

    print("\nTesting answers_match:")
    test_cases = [
        ("42", "42", True),
        ("0.5", "1/2", True),
        ("\\frac{1}{2}", "0.5", True),
        ("\\frac{1}{2}", "\\frac{2}{4}", True),
        ("3.14159", "3.14159", True),
        ("42", "43", False),
        ("yes", "YES", True),
    ]

    for pred, truth, expected in test_cases:
        result = answers_match(pred, truth)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{pred}' vs '{truth}' -> {result} (expected {expected})")

    print("\nTesting reward_rlvr:")
    test_cases = [
        ("The answer is \\boxed{42}", "42", 1.0),
        ("The answer is \\boxed{42}", "43", 0.0),
        ("No boxed answer here", "42", 0.0),
        ("Therefore \\boxed{0.5}", "1/2", 1.0),
    ]

    for response, truth, expected in test_cases:
        result = reward_rlvr(response, truth)
        status = "✓" if result == expected else "✗"
        print(f"  {status} response with '{extract_boxed_answer(response)}' vs '{truth}' -> {result}")

    print("\nTesting compute_advantages:")
    rewards = [0, 0, 1, 0]
    advantages = compute_advantages(rewards)
    print(f"  Rewards: {rewards}")
    print(f"  Advantages: {[f'{a:.3f}' for a in advantages]}")

    rewards = [1, 1, 1, 1]
    advantages = compute_advantages(rewards)
    print(f"  Rewards: {rewards}")
    print(f"  Advantages: {[f'{a:.3f}' for a in advantages]}")
