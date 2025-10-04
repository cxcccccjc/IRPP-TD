import numpy as np
import time
from typing import Tuple, Dict, Any


class ABODDetector:
    """简化的ABOD异常检测算法实现，只包含核心功能"""

    def __init__(self, n_neighbors: int = 5, method: str = 'fast'):
        self.n_neighbors = n_neighbors
        self.method = method

    def calculate_angle_variances(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算所有点的角度方差

        Parameters:
        -----------
        X : np.ndarray
            数据矩阵，每行是一个数据点

        Returns:
        --------
        Tuple[np.ndarray, Dict[str, Any]]
            角度方差数组和算法详细信息
        """
        start_time = time.perf_counter()
        n_samples = X.shape[0]
        angle_variances = np.zeros(n_samples)

        total_comparisons = 0
        total_wcos_calculations = 0

        for i in range(n_samples):
            angle_variances[i], point_stats = self._calculate_single_point_variance(X, i)
            total_comparisons += point_stats['comparisons']
            total_wcos_calculations += point_stats['wcos_calculations']

        end_time = time.perf_counter()
        computation_time = end_time - start_time

        details = {
            'computation_time': computation_time,
            'n_samples': n_samples,
            'n_features': X.shape[1],
            'method': self.method,
            'n_neighbors': self.n_neighbors,
            'total_comparisons': total_comparisons,
            'total_wcos_calculations': total_wcos_calculations,
            'avg_time_per_sample': computation_time / n_samples if n_samples > 0 else 0
        }

        return angle_variances, details

    def calculate_single_point_variance(self, target_point: np.ndarray, other_points: np.ndarray) -> float:
        """
        计算单个点相对于其他点的角度方差

        Parameters:
        -----------
        target_point : np.ndarray
            目标点（一维数组）
        other_points : np.ndarray
            其他点的数据矩阵，每行是一个点

        Returns:
        --------
        float
            目标点的角度方差
        """
        if other_points.shape[0] < 2:
            return 0.0

        # 构建完整的数据矩阵，目标点放在第一行
        X = np.vstack([target_point.reshape(1, -1), other_points])

        # 计算目标点（索引0）的角度方差
        variance, _ = self._calculate_single_point_variance(X, 0)
        return variance

    def _calculate_single_point_variance(self, X: np.ndarray, target_index: int) -> Tuple[float, Dict[str, int]]:
        """
        内部方法：计算指定索引点的角度方差

        Parameters:
        -----------
        X : np.ndarray
            数据矩阵
        target_index : int
            目标点索引

        Returns:
        --------
        Tuple[float, Dict[str, int]]
            角度方差和统计信息
        """
        n_samples = X.shape[0]
        curr_pt = X[target_index, :]
        wcos_list = []

        # 获取其他点的索引
        other_indices = [j for j in range(n_samples) if j != target_index]

        comparisons = 0
        wcos_calculations = 0

        if len(other_indices) == 0:
            return 0.0, {'comparisons': 0, 'wcos_calculations': 0}

        # 如果使用fast方法，只考虑最近的邻居
        if self.method == 'fast' and len(other_indices) > self.n_neighbors:
            distances = [(np.linalg.norm(curr_pt - X[j, :]), j) for j in other_indices]
            distances.sort()
            other_indices = [idx for _, idx in distances[:self.n_neighbors]]

        # 计算所有点对的加权余弦值
        for j in range(len(other_indices)):
            for k in range(j + 1, len(other_indices)):
                comparisons += 1
                a_idx, b_idx = other_indices[j], other_indices[k]
                a, b = X[a_idx, :], X[b_idx, :]

                # 跳过与当前点相同的点
                if np.allclose(a, curr_pt, rtol=1e-10) or np.allclose(b, curr_pt, rtol=1e-10):
                    continue

                # 计算加权余弦
                wcos = self._calculate_wcos(curr_pt, a, b)
                if not np.isnan(wcos) and not np.isinf(wcos):
                    wcos_list.append(wcos)
                    wcos_calculations += 1

        # 计算方差
        variance = np.var(wcos_list) if len(wcos_list) > 1 else 0.0

        return variance, {'comparisons': comparisons, 'wcos_calculations': wcos_calculations}

    def _calculate_wcos(self, curr_pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """计算加权余弦值"""
        a_curr = a - curr_pt
        b_curr = b - curr_pt

        norm_a = np.linalg.norm(a_curr, 2)
        norm_b = np.linalg.norm(b_curr, 2)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        # 计算加权余弦值
        dot_product = np.dot(a_curr, b_curr)
        wcos = dot_product / (norm_a ** 2) / (norm_b ** 2)

        return wcos


# 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = ABODDetector(n_neighbors=5, method='fast')

    # 示例数据
    data_matrix = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [1.2, 1.9, 3.2],
        [10.0, 20.0, 30.0]  # 异常点
    ])

    print("测试功能一：计算所有点的角度方差")
    variances, details = detector.calculate_angle_variances(data_matrix)
    print(f"角度方差: {variances}")
    print(f"计算时间: {details['computation_time']:.6f} 秒")
    print()

    print("测试功能二：计算单个点的角度方差")
    target_point = np.array([10.0, 20.0, 30.0])  # 明显的异常点
    other_points = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [1.2, 1.9, 3.2]
    ])

    single_variance = detector.calculate_single_point_variance(target_point, other_points)
    print(f"单点角度方差: {single_variance}")
