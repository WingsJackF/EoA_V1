"""任务专用：适应度评估（原 embed_builtin_evaluator_code 逻辑）。"""

import importlib
import os
import sys
import tempfile
import time
from typing import Any, Dict

import numpy as np
import scipy as sp

from tasks.min_max_layout_16.constants import TARGET_FUNCTION_NAME


def run_evaluation(program_code: str) -> Dict[str, Any]:
    """
    执行候选代码中的目标函数，计算 min/max 距离比相关指标。
    program_code 须定义 TARGET_FUNCTION_NAME（无必需参数）。
    """
    num_points = 16
    dimension = 2
    benchmark = 1 / 12.889266112

    with tempfile.TemporaryDirectory() as temp_dir:
        program_path = os.path.join(temp_dir, "program.py")
        with open(program_path, "w", encoding="utf-8") as f:
            f.write(program_code)

        module_name = "program"

        try:
            sys.path.insert(0, temp_dir)
            program = importlib.import_module(module_name)
            importlib.reload(program)

            start_time = time.time()
            ctor = getattr(program, TARGET_FUNCTION_NAME)
            points = ctor()
            end_time = time.time()
            eval_time = end_time - start_time

            if not isinstance(points, np.ndarray):
                points = np.array(points)

            if points.shape != (num_points, dimension):
                raise ValueError(
                    f"Invalid shapes: points = {points.shape}, expected {(num_points, dimension)}"
                )

            pairwise_distances = sp.spatial.distance.pdist(points)
            min_distance = np.min(pairwise_distances)
            max_distance = np.max(pairwise_distances)

            inv_ratio_squared = (min_distance / max_distance) ** 2 if max_distance > 0 else 0

            return {
                "min_max_ratio": float(inv_ratio_squared),
                "combined_score": float(inv_ratio_squared / benchmark),
                "eval_time": float(eval_time),
            }
        except Exception as e:
            return {"combined_score": 0.0, "error": str(e)}
        finally:
            if temp_dir in sys.path:
                try:
                    sys.path.remove(temp_dir)
                except ValueError:
                    pass
