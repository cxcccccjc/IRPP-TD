import numpy as np
import math
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TruthDiscoveryResult:
    """Truth Discovery结果"""
    aggregated_truth: List[float]  # 聚合真值
    worker_weights: List[float]  # 工人权重（未归一化）
    algorithm_time: float  # 算法运行时间（秒）
    iterations: int  # 迭代次数
    converged: bool  # 是否收敛


class TruthDiscoveryEngine:
    """Truth Discovery算法引擎"""

    def __init__(self, epsilon: float = 1e-6, max_iterations: int = 100):
        """
        初始化Truth Discovery引擎

        Parameters:
        -----------
        epsilon : float
            收敛阈值
        max_iterations : int
            最大迭代次数
        """
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def discover_truth(self, workers_data: Dict[int, List[float]]) -> TruthDiscoveryResult:
        """
        Truth Discovery核心算法

        Parameters:
        -----------
        workers_data : Dict[int, List[float]]
            工人数据字典 {worker_id: data_vector}

        Returns:
        --------
        TruthDiscoveryResult
            包含聚合真值、权重、运行时间等信息
        """
        if len(workers_data) < 2:
            raise ValueError("至少需要2个工人的数据")

        # 准备数据
        worker_ids = list(workers_data.keys())
        data_matrix = np.array([workers_data[wid] for wid in worker_ids], dtype=np.float64)
        m, d = data_matrix.shape  # m个工人，d维数据

        # 初始化权重和真值
        weights = np.ones(m, dtype=np.float64)  # 初始权重为1（不归一化）
        truth = np.ones(d, dtype=np.float64)  # 初始真值

        # 开始计时
        start_time = time.perf_counter()

        # 初始化真值（加权平均）
        truth = self._update_truth(weights, data_matrix)

        # 迭代优化
        iterations = 0
        converged = False

        for epoch in range(self.max_iterations):
            iterations = epoch + 1

            # 更新权重
            weights = self._update_weights(weights, truth, data_matrix)

            # 更新真值
            new_truth = self._update_truth(weights, data_matrix)

            # 检查收敛
            truth_error = np.subtract(new_truth, truth)
            convergence_measure = np.sum(np.power(truth_error, 2))

            if convergence_measure < self.epsilon:
                truth = new_truth
                converged = True
                break
            else:
                truth = new_truth

        # 结束计时
        algorithm_time = time.perf_counter() - start_time

        return TruthDiscoveryResult(
            aggregated_truth=truth.tolist(),
            worker_weights=weights.tolist(),  # 返回未归一化的权重
            algorithm_time=algorithm_time,
            iterations=iterations,
            converged=converged
        )

    def _update_truth(self, weights: np.ndarray, data_matrix: np.ndarray) -> np.ndarray:
        """
        更新真值估计（利用权重计算数据）

        Parameters:
        -----------
        weights : np.ndarray
            工人权重
        data_matrix : np.ndarray
            工人数据矩阵

        Returns:
        --------
        np.ndarray
            更新后的真值
        """
        truth = np.matmul(weights, data_matrix)
        # 按权重和进行归一化
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            truth = truth / weight_sum
        return truth

    def _update_weights(self, weights: np.ndarray, truth: np.ndarray, data_matrix: np.ndarray) -> np.ndarray:
        """
        权重更新（不归一化）

        Parameters:
        -----------
        weights : np.ndarray
            当前权重
        truth : np.ndarray
            当前真值估计
        data_matrix : np.ndarray
            工人数据矩阵

        Returns:
        --------
        np.ndarray
            更新后的权重（未归一化）
        """
        m = data_matrix.shape[0]
        un_log_1 = 0  # 总距离平方和
        un_log_2 = np.zeros(m)  # 每个工人的距离平方

        # 计算每个工人数据与真值的距离平方
        for i in range(m):
            for j in range(data_matrix.shape[1]):
                distance_sq = self._square_of_distance(data_matrix[i][j], truth[j])
                un_log_2[i] += distance_sq
            un_log_1 += un_log_2[i]

        # 避免除零和对数零值
        un_log_1 = max(un_log_1, 1e-10)
        un_log_2 = np.maximum(un_log_2, 1e-10)

        # 更新权重（直接使用对数比值，不归一化）
        new_weights = np.zeros(m)
        for i in range(m):
            new_weights[i] = math.log(un_log_1 / un_log_2[i])

        return new_weights

    def _square_of_distance(self, data_1: float, data_2: float) -> float:
        """计算平方距离"""
        return math.pow(data_2 - data_1, 2)


def discover_truth(workers_data: Dict[int, List[float]],
                   epsilon: float = 1e-3,
                   max_iterations: int = 100) -> TruthDiscoveryResult:
    """
    便捷函数：执行Truth Discovery算法

    Parameters:
    -----------
    workers_data : Dict[int, List[float]]
        工人数据字典 {worker_id: data_vector}
    epsilon : float
        收敛阈值
    max_iterations : int
        最大迭代次数

    Returns:
    --------
    TruthDiscoveryResult
        Truth Discovery结果
    """
    engine = TruthDiscoveryEngine(epsilon, max_iterations)
    return engine.discover_truth(workers_data)


def get_truth_and_weights(workers_data: Dict[int, List[float]]) -> Tuple[List[float], Dict[int, float], float]:
    """
    快速获取真值、权重和运行时间

    Parameters:
    -----------
    workers_data : Dict[int, List[float]]
        工人数据字典

    Returns:
    --------
    Tuple[List[float], Dict[int, float], float]
        (聚合真值, {工人ID: 权重}, 算法运行时间)
    """
    result = discover_truth(workers_data)
    worker_ids = list(workers_data.keys())
    weights_dict = dict(zip(worker_ids, result.worker_weights))
    return result.aggregated_truth, weights_dict, result.algorithm_time


# 测试函数
def _test_truth_discovery():
    """算法测试"""
    test_data = {
        9: [996.44, 280.13, 79.98, 1240.16, 205.58],
        16: [995.92, 279.68, 80.06, 1234.21, 206.16],
        19: [995.91, 279.66, 79.27, 1233.27, 204.96],
        21: [994.35, 279.04, 79.95, 1228.54, 206.65],
        23: [991.26, 279.48, 79.95, 1238.37, 205.97],
        65: [819.52, 278.32, 80.18, 1238.07, 206.25],  # 异常数据
        85: [1186.66, 313.1, 80.07, 1399.89, 231.91],  # 异常数据
    }

    result = discover_truth(test_data)

    print("Truth Discovery结果:")
    print(f"聚合真值: {[f'{x:.4f}' for x in result.aggregated_truth]}")
    print(f"算法运行时间: {result.algorithm_time:.6f} 秒")
    print(f"迭代次数: {result.iterations}")
    print(f"是否收敛: {result.converged}")

    print("\n工人权重:")
    worker_ids = list(test_data.keys())
    for i, worker_id in enumerate(worker_ids):
        print(f"工人{worker_id}: {result.worker_weights[i]:.6f}")


if __name__ == "__main__":
    _test_truth_discovery()

"""

调用策略

from truth_discovery import discover_truth

workers_data = {
    9: [996.44, 280.13, 79.98, 1240.16, 205.58],
    16: [995.92, 279.68, 80.06, 1234.21, 206.16],
    # ... 其他数据
}

result = discover_truth(workers_data)
print(f"聚合真值: {result.aggregated_truth}")
print(f"工人真实权重: {result.worker_weights}")
print(f"运行时间: {result.algorithm_time}")

"""