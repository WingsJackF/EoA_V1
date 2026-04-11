import numpy as np
import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from tqdm import tqdm

def solve_tsp_instance(coords):
    """求解单个 TSP 实例的近似最优解"""
    num_nodes = len(coords)
    # 1. 计算距离矩阵并放大精度（OR-Tools 需要整数）
    # 使用欧式距离: sqrt((x1-x2)^2 + (y1-y2)^2)
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    # 放大 10^6 倍以保留 6 位小数精度
    scaling_factor = 1000000
    int_dist_matrix = (dist_matrix * scaling_factor).astype(np.int64)

    # 2. 初始化 OR-Tools 路由模型
    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 3. 配置搜索参数（使用 GREEDY_DESCENT 结合局部搜索提升精度）
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # 设置单例求解时限（秒），TSP100 建议 1-2 秒
    search_parameters.time_limit.seconds = 2 

    # 4. 求解
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        # 返回真实长度（除回缩放因子）
        return solution.ObjectiveValue() / scaling_factor
    else:
        return None

def process_datasets(data_dir, file_list):
    """遍历所有数据集文件并计算 Gap 基准"""
    summary_results = {}

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"跳过文件: {file_name} (不存在)")
            continue
        
        print(f"\n正在处理: {file_name} ...")
        # 加载数据 (Shape: [Instances, Nodes, 2])
        dataset = np.load(file_path)
        num_instances = dataset.shape[0]
        
        lengths = []
        # 为了节省时间，你可以先测试前 100 个实例
        test_range = min(num_instances, 100) 
        
        for i in tqdm(range(test_range)):
            length = solve_tsp_instance(dataset[i])
            if length:
                lengths.append(length)
        
        avg_optimal = np.mean(lengths)
        summary_results[file_name] = avg_optimal
        print(f"-> {file_name} 的平均理论最优估值: {avg_optimal:.6f}")

    return summary_results

if __name__ == "__main__":
    # 配置你的路径
    DATA_DIR = "."  # 你的数据文件夹路径
    FILES_TO_TEST = [
        "val20_dataset.npy",
        "val50_dataset.npy",
        "val100_dataset.npy",
        "val200_dataset.npy"
    ]

    # 执行计算
    results = process_datasets(DATA_DIR, FILES_TO_TEST)

    # 最后打印对比表所需的基准值
    print("\n" + "="*30)
    print(" 最终计算结果 (用于计算 Gap)")
    print("="*30)
    for file, val in results.items():
        print(f"{file:20} | Average Optimal: {val:.6f}")