import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: extract_thought_and_code_sections

Description:
    Extracts and separates the 'strategic thought' and candidate code from a raw
    LLM response string. Supports structured responses with explicit markers and
    semi-structured responses (function definitions or plain code).

Constraints:
    - Python 3.9+
    - Uses only 're' and 'typing' from the standard library.
    - No file I/O, no code execution, and no simulated API calls.
    - Raises ValueError if neither thought nor code can be reliably extracted.
"""

import re
from typing import Dict


def extract_thought_and_code_sections(
    llm_response: str, target_function_name: str = "_entrypoint"
) -> Dict[str, str]:
    """
    Extracts the 'strategic thought' and candidate code from the LLM's raw response string.

    Args:
        llm_response: The raw string content returned by the LLM API.

    Returns:
        A dictionary with 'thought' and 'code' keys containing their respective content.

    Raises:
        ValueError: If neither a thought nor code section can be reliably extracted.
    """
    if not isinstance(llm_response, str):
        raise TypeError("llm_response must be a string")

    text = llm_response.strip()
    thought = ""
    code = ""

    # 1) Try to find fenced code blocks first (```python ... ``` or ``` ... ```)
    code_block_re = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.IGNORECASE | re.DOTALL)
    m_code_block = code_block_re.search(text)
    if m_code_block:
        code = m_code_block.group(1).strip()
        # Everything before the code block is candidate thought
        before = text[: m_code_block.start()].strip()
        # Try to extract explicit thought marker from 'before'
        thought_marker_re = re.compile(
            r"(?:Strategic\s*Thought|Thought|Strategy)\s*[:\-]\s*(.*)",
            re.IGNORECASE | re.DOTALL,
        )
        m_thought = thought_marker_re.search(before)
        if m_thought:
            thought = m_thought.group(1).strip()
        else:
            thought = before
        return {"thought": thought, "code": code}

    # 2) If no fenced block, look for 'Code:' marker
    code_label_re = re.compile(r"(?:^|\n)\s*Code\s*[:\-]\s*(.*)", re.IGNORECASE | re.DOTALL)
    m_code_label = code_label_re.search(text)
    if m_code_label:
        code = m_code_label.group(1).strip()
        # Text before 'Code:' is candidate thought
        before = text[: m_code_label.start()].strip()
        thought_marker_re = re.compile(
            r"(?:Strategic\s*Thought|Thought|Strategy)\s*[:\-]\s*(.*)",
            re.IGNORECASE | re.DOTALL,
        )
        m_thought = thought_marker_re.search(before)
        if m_thought:
            thought = m_thought.group(1).strip()
        else:
            thought = before
        # If code is empty after trimming, treat as not found
        if code:
            return {"thought": thought, "code": code}

    # 3) If still no code, search for a Python function definition for the expected task entry
    #    or any 'def ' occurrence; prefer the specific constructor name if present.
    escaped_name = re.escape(target_function_name)
    specific_def_re = re.compile(
        rf"(def\s+{escaped_name}\s*\(.*?\):[\s\S]*)", re.IGNORECASE
    )
    m_specific_def = specific_def_re.search(text)
    if m_specific_def:
        code = m_specific_def.group(1).strip()
        before = text[: m_specific_def.start()].strip()
        thought_marker_re = re.compile(
            r"(?:Strategic\s*Thought|Thought|Strategy)\s*[:\-]\s*(.*)",
            re.IGNORECASE | re.DOTALL,
        )
        m_thought = thought_marker_re.search(before)
        if m_thought:
            thought = m_thought.group(1).strip()
        else:
            thought = before
        return {"thought": thought, "code": code}

    # 4) If no specific def, try to find any function definition as code candidate
    any_def_re = re.compile(r"(def\s+\w+\s*\(.*?\):[\s\S]*)", re.IGNORECASE)
    m_any_def = any_def_re.search(text)
    if m_any_def:
        code = m_any_def.group(1).strip()
        before = text[: m_any_def.start()].strip()
        thought_marker_re = re.compile(
            r"(?:Strategic\s*Thought|Thought|Strategy)\s*[:\-]\s*(.*)",
            re.IGNORECASE | re.DOTALL,
        )
        m_thought = thought_marker_re.search(before)
        if m_thought:
            thought = m_thought.group(1).strip()
        else:
            thought = before
        return {"thought": thought, "code": code}

    # 5) If still no code, check if there's an explicit thought section only
    thought_only_re = re.compile(
        r"(?:Strategic\s*Thought|Thought|Strategy)\s*[:\-]\s*(.*)", re.IGNORECASE | re.DOTALL
    )
    m_thought_only = thought_only_re.search(text)
    if m_thought_only:
        thought = m_thought_only.group(1).strip()
        code = ""
        return {"thought": thought, "code": code}

    # 6) If the entire text looks like code (contains common Python tokens), treat as code
    if any(tok in text for tok in ("def ", "import ", "np.", "numpy", "return ")):
        code = text
        thought = ""
        return {"thought": thought, "code": code.strip()}

    # 7) Unable to extract meaningful thought or code
    raise ValueError("Unable to extract 'thought' or 'code' from the provided LLM response.")



import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
"""
Module: validate_constructor_syntax

Description:
    Validates that a provided candidate code string is syntactically valid Python
    and contains the expected task entry function with no required arguments.

Constraints:
    - Python 3.9+
    - Uses only 'ast' and 're' from the standard library.
    - No file I/O, no code execution, and no virtual/simulated API calls.
    - Raises SyntaxError for syntax problems and ValueError for missing/invalid signature.
"""

import ast
import re
from typing import Dict


def validate_constructor_syntax(
    code_str: str, function_name: str = "_entrypoint"
) -> None:
    """
    Validates that the code string is syntactically correct Python and
    contains a function named `function_name` with no required arguments.

    Args:
        code_str: The code string to validate.
        function_name: Entry-point function name required in the code (task-specific).

    Raises:
        SyntaxError: If the code is not valid Python.
        ValueError: If the required function is missing or has an invalid signature.
    """
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")

    src = code_str.strip()
    if not src:
        raise ValueError("Provided code string is empty")

    # 1) Syntax check via AST parsing
    try:
        module_ast = ast.parse(src)
    except SyntaxError as se:
        # Re-raise SyntaxError with the original details for caller inspection
        raise SyntaxError(f"Syntax error while parsing code: {se.msg} (line {se.lineno})") from se

    # 2) Search AST for the required function definition
    found = False
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            found = True
            func_def: ast.FunctionDef = node  # type: ignore

            # Check positional args: require no positional-only or positional args
            # In Python 3.8+, ast.arguments has posonlyargs attribute
            posonly_count = len(getattr(func_def.args, "posonlyargs", ()))
            pos_args_count = len(func_def.args.args)
            vararg = func_def.args.vararg
            kwonly_count = len(func_def.args.kwonlyargs)
            kwarg = func_def.args.kwarg

            # Determine if there are any required positional or keyword-only args
            # For positional args, if any exist, they are considered required unless defaults provided.
            # Simpler strict rule per spec: require zero positional args and no varargs or kwonlyargs.
            if posonly_count != 0 or pos_args_count != 0:
                raise ValueError(
                    f"Function {function_name!r} must accept no positional arguments."
                )
            if vararg is not None:
                raise ValueError(
                    f"Function {function_name!r} must not accept *args (varargs)."
                )
            # Allow keyword-only args only if none are required (i.e., they must have defaults).
            if kwonly_count != 0:
                # Check if all kwonlyargs have defaults (kw_defaults parallel list)
                kw_defaults = func_def.args.kw_defaults
                # kw_defaults list aligns with kwonlyargs; a None in kw_defaults indicates a required kw-only arg
                if any(d is None for d in kw_defaults):
                    raise ValueError(
                        f"Function {function_name!r} must not have required keyword-only arguments."
                    )
            if kwarg is not None:
                # Allow **kwargs? It's safer to disallow for simplicity per spec.
                raise ValueError(
                    f"Function {function_name!r} must not accept **kwargs."
                )

            # Signature passed the checks
            return None

    if not found:
        raise ValueError(f"Function {function_name!r} not found in provided code.")



