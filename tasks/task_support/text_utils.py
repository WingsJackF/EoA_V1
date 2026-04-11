import ast
import re


def first_function_name(code: str) -> str:
    m = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else "heuristics_v2"


def ensure_target_function_name(code: str, target_function_name: str) -> str:
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if not isinstance(target_function_name, str):
        raise TypeError("target_function_name must be a string")

    src = code.strip()
    if not src or not target_function_name:
        return src

    try:
        module_ast = ast.parse(src)
    except SyntaxError:
        return src

    top_level_functions = [
        node.name for node in module_ast.body if isinstance(node, ast.FunctionDef)
    ]
    if not top_level_functions:
        return src
    if target_function_name in top_level_functions:
        return src

    alias_source_name = top_level_functions[0]
    alias_line = f"{target_function_name} = {alias_source_name}"
    if re.search(rf"^{re.escape(alias_line)}\s*$", src, re.MULTILINE):
        return src
    return f"{src}\n\n{alias_line}\n"
