"""任务专用：三种演化策略与初始种群相关的提示词模板。"""

from typing import Dict, List, Tuple

from tasks.min_max_layout_16.constants import TARGET_FUNCTION_NAME

_FN = TARGET_FUNCTION_NAME


def implement_modification_prompt_strategy(parent_individual: Dict[str, str]) -> Tuple[str, str]:
    if not isinstance(parent_individual, dict):
        raise TypeError("parent_individual must be a dict with keys 'thought' and 'code'")
    if "thought" not in parent_individual or "code" not in parent_individual:
        raise TypeError("parent_individual must contain 'thought' and 'code' keys")

    parent_thought = parent_individual.get("thought") or ""
    parent_code = parent_individual.get("code") or ""

    system_prompt = (
        "You are an expert in geometric optimization, numerical methods, and Python programming. "
        "Your task is to produce a concise improvement to a provided Python constructor function "
        "that generates a 16-point planar layout. Focus on targeted, verifiable improvements "
        "that preserve the core idea of the parent. Reply with a short strategic thought and "
        "the complete, runnable constructor code."
    )

    stub = (
        "# (Parent provided no code.)\n"
        "import numpy as np\n\n"
        f"def {_FN}():\n"
        "    n = 16\n"
        "    d = 2\n"
        "    np.random.seed(42)\n"
        "    points = np.random.randn(n, d)\n"
        "    return points"
    )

    user_prompt_parts = [
        "Parent Individual:",
        "",
        "Strategic Thought:",
        parent_thought if parent_thought.strip() else "(No explicit strategic thought provided.)",
        "",
        "Constructor Code:",
        "```python",
        parent_code.strip() if parent_code.strip() else stub,
        "```",
        "",
        "Modification Request:",
        "- Provide a targeted improvement to the constructor code above while preserving its core idea.",
        "- Explain, in one or two sentences, the strategic rationale for the change (this will be the new 'Strategic Thought').",
        f"- Return a single complete, runnable Python constructor function named `{_FN}` with no required arguments.",
        "- The function must return a numpy array of shape (16, 2).",
        "- Keep imports minimal; include them in the code block if needed.",
        "- If you introduce randomness, set a seed or document how reproducibility is achieved.",
        "- Provide the strategic thought and code using explicit section markers exactly as follows:",
        "",
        "Response format (required):",
        "Strategic Thought: <one or two sentence description>",
        "",
        "```python",
        "# Improved constructor code here",
        "```",
        "",
        "Notes:",
        "- Do not include additional unrelated commentary outside the required sections.",
        "- Keep the code clear and focused on increasing the min/max pairwise distance ratio while preserving the parent's core idea.",
    ]

    return system_prompt, "\n".join(user_prompt_parts)


def implement_exploration_prompt_strategy(parent_individuals: List[Dict[str, str]]) -> Tuple[str, str]:
    if not isinstance(parent_individuals, list):
        raise TypeError("parent_individuals must be a list of dicts with 'thought' and 'code' keys")
    if len(parent_individuals) < 2:
        raise ValueError("At least two parent individuals are required for exploration/synthesis")

    for idx, p in enumerate(parent_individuals):
        if not isinstance(p, dict):
            raise TypeError(f"Parent at index {idx} is not a dict")
        if "thought" not in p or "code" not in p:
            raise TypeError(f"Parent at index {idx} must contain 'thought' and 'code' keys")

    system_prompt = (
        "You are an expert in geometric optimization, numerical methods, and Python programming. "
        "Your task is synthesis: given several parent strategies and their constructor code, "
        "analyze their strengths and produce a single novel constructor that merges, hybridizes, "
        f"or innovates upon the best features. Provide a brief strategic thought and a complete, "
        f"runnable constructor function named `{_FN}()` with no required arguments."
    )

    user_parts: List[str] = ["Parents (for synthesis):", ""]

    for i, parent in enumerate(parent_individuals, start=1):
        thought = parent.get("thought", "").strip() or "(No explicit strategic thought provided.)"
        code = parent.get("code", "").strip() or ("# (No code provided for this parent)")
        user_parts.extend(
            [
                f"Parent {i}:",
                "Strategic Thought:",
                thought,
                "",
                "Constructor Code:",
                "```python",
                code,
                "```",
                "",
            ]
        )

    user_parts.extend(
        [
            "Synthesis Request:",
            "- Analyze the parents above. Summarize the strongest ideas and merge them into a single, "
            "novel strategy and constructor.",
            "- The output must NOT be a trivial concatenation or minor edit; create an original, coherent approach.",
            "- Provide a brief strategic thought (one or two sentences) explaining the new approach.",
            f"- Provide a complete, runnable Python constructor function named `{_FN}()` "
            "that returns a numpy array of shape (16, 2). The function must accept no required arguments.",
            "- Include any necessary imports inside the code block.",
            "- If randomness is used, document reproducibility or set a seed.",
            "",
            "Response format (required):",
            "Strategic Thought: <one or two sentence description>",
            "",
            "```python",
            "# Improved/synthesized constructor code here",
            "```",
            "",
            "Notes:",
            "- Use explicit section markers exactly as above to facilitate downstream parsing.",
            "- Avoid extra commentary outside the required sections.",
        ]
    )

    return system_prompt, "\n".join(user_parts)


def implement_simplification_prompt_strategy(parent_individual: Dict[str, str]) -> Tuple[str, str]:
    if not isinstance(parent_individual, dict):
        raise TypeError("parent_individual must be a dict with 'thought' and 'code' keys")
    if "thought" not in parent_individual or "code" not in parent_individual:
        raise TypeError("parent_individual must contain 'thought' and 'code' keys")

    parent_thought = parent_individual.get("thought") or ""
    parent_code = parent_individual.get("code") or ""

    stub = (
        "# Parent provided no code. Provide a simple, clear constructor below.\n"
        "import numpy as np\n\n"
        f"def {_FN}():\n"
        "    n = 16\n"
        "    d = 2\n"
        "    np.random.seed(42)\n"
        "    points = np.random.randn(n, d)\n"
        "    return points"
    )

    system_prompt = (
        "You are an expert in algorithmic simplification, numerical stability, and Python programming. "
        "Your task is to simplify the provided strategy description and constructor code while maintaining "
        "or improving its effectiveness for maximizing the minimum-to-maximum pairwise distance ratio "
        "for 16-point planar layouts."
    )

    user_prompt_lines = [
        "Parent Individual (for simplification):",
        "",
        "Strategic Thought:",
        parent_thought if parent_thought.strip() else "(No explicit strategic thought provided.)",
        "",
        "Constructor Code:",
        "```python",
        parent_code.strip() if parent_code.strip() else stub,
        "```",
        "",
        "Simplification Request:",
        "- Simplify the strategic thought to a concise (one or two sentence) description that preserves the core idea.",
        "- Simplify the constructor code to be clearer, shorter, and more maintainable while preserving or improving performance.",
        "- Avoid introducing unnecessary complexity; prefer readability and numerical stability.",
        "- If you remove or replace parts of the parent's approach, briefly justify the change in one sentence.",
        f"- Provide a single complete, runnable Python constructor function named `{_FN}()` with no required arguments.",
        "- Ensure the function returns a numpy array of shape (16, 2).",
        "",
        "Response format (required):",
        "Strategic Thought: <concise simplified thought>",
        "",
        "```python",
        "# Simplified constructor code here",
        "```",
        "",
        "Notes:",
        "- Place the strategic thought and the code exactly in the format above to enable reliable parsing.",
        "- Do not include unrelated commentary outside these sections.",
        "- If randomness is used, document reproducibility or set a seed.",
    ]

    return system_prompt, "\n".join(user_prompt_lines)


INITIAL_POPULATION_SYSTEM_PROMPT = (
    "You are an expert in geometric optimization and evolutionary algorithms. "
    "Your task is to generate Python constructor functions for 16-point planar layouts "
    "that maximize the minimum-to-maximum pairwise distance ratio. "
    "Return both your strategic thought and the constructor code. "
    f"Provide valid, executable Python defining def {_FN}(): ..."
)

INITIAL_DIVERSITY_INSTRUCTIONS = [
    "Generate a constructor that uses a distinct geometric or probabilistic approach.",
    "Please invent a new strategy for the 16-point layout, differing from the seed.",
    "Try a novel method (e.g., grid, circle, or hybrid) for point initialization.",
    "Change both the code and the strategic thought for more diversity.",
    "Explore a mathematically innovative or unconventional initialization.",
    "Emphasize symmetry or lattice-based arrangements while maintaining randomness.",
    "Use an optimization-heuristic approach (e.g., repulsion-based initialization).",
]
