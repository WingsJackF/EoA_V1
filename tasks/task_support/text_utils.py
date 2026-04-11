import re


def first_function_name(code: str) -> str:
    m = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else "heuristics_v2"
