import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class DataQuality(Enum):
    """数据质量分类枚举"""
    HIGH = "high_quality"
    UNCERTAIN = "uncertain"
    LOW = "low_quality"


@dataclass
class WorkerResult:
    """工人评估结果"""
    worker_id: int
    angle_variance: float
    data_quality: Optional[DataQuality] = None
    submitted_data: Optional[List[float]] = None


@dataclass
class AssessmentResult:
    """评估结果（包含运行时间）"""
    results: Union[List[WorkerResult], WorkerResult]
    execution_time: float  # 运行时间（秒）
    algorithm_details: Dict[str, Any]  # 算法详细信息


class ABODDetector:
    """ABOD异常检测算法实现"""

    def __init__(self, n_neighbors: int = 5, method: str = 'fast'):
        self.n_neighbors = n_neighbors
        self.method = method

    def calculate_angle_variances(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """计算每个点的角度方差，返回结果和详细信息"""
        start_time = time.perf_counter()

        n_samples = X.shape[0]
        angle_variances = np.zeros(n_samples)

        # 算法详细信息统计
        total_comparisons = 0
        total_wcos_calculations = 0

        for i in range(n_samples):
            curr_pt = X[i, :]
            wcos_list = []

            # 获取其他所有点
            other_indices = [j for j in range(n_samples) if j != i]

            if len(other_indices) == 0:
                angle_variances[i] = 0.0
                continue

            # 如果使用fast方法，只考虑最近的邻居
            if self.method == 'fast' and len(other_indices) > self.n_neighbors:
                distances = []
                for j in other_indices:
                    dist = np.linalg.norm(curr_pt - X[j, :])
                    distances.append((dist, j))
                distances.sort()
                other_indices = [idx for _, idx in distances[:self.n_neighbors]]

            # 计算所有点对的加权余弦值
            for j in range(len(other_indices)):
                for k in range(j + 1, len(other_indices)):
                    total_comparisons += 1
                    a_idx, b_idx = other_indices[j], other_indices[k]
                    a, b = X[a_idx, :], X[b_idx, :]

                    # 跳过与当前点相同的点
                    if np.allclose(a, curr_pt, rtol=1e-10) or np.allclose(b, curr_pt, rtol=1e-10):
                        continue

                    # 计算加权余弦
                    wcos = self._calculate_wcos(curr_pt, a, b)
                    if not np.isnan(wcos) and not np.isinf(wcos):
                        wcos_list.append(wcos)
                        total_wcos_calculations += 1

            # 计算方差
            if len(wcos_list) > 1:
                angle_variances[i] = np.var(wcos_list)
            else:
                angle_variances[i] = 0.0

        end_time = time.perf_counter()
        computation_time = end_time - start_time

        # 算法详细信息
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

    def calculate_single_point_angle_variance(self, X: np.ndarray, target_index: int) -> Tuple[float, Dict[str, Any]]:
        """
        计算指定点的角度方差

        Parameters:
        -----------
        X : np.ndarray
            数据矩阵，每行是一个数据点
        target_index : int
            目标点的索引（在数据矩阵中的行号）

        Returns:
        --------
        Tuple[float, Dict[str, Any]]
            目标点的角度方差值和算法详细信息
        """
        start_time = time.perf_counter()

        n_samples = X.shape[0]

        # 验证目标索引
        if target_index < 0 or target_index >= n_samples:
            raise ValueError(f"目标索引 {target_index} 超出范围 [0, {n_samples - 1}]")

        if n_samples < 2:
            raise ValueError("至少需要2个数据点才能计算角度方差")

        curr_pt = X[target_index, :]
        wcos_list = []

        # 获取其他所有点的索引
        other_indices = [j for j in range(n_samples) if j != target_index]

        # 算法详细信息统计
        total_comparisons = 0
        total_wcos_calculations = 0

        if len(other_indices) == 0:
            angle_variance = 0.0
        else:
            # 如果使用fast方法，只考虑最近的邻居
            if self.method == 'fast' and len(other_indices) > self.n_neighbors:
                distances = []
                for j in other_indices:
                    dist = np.linalg.norm(curr_pt - X[j, :])
                    distances.append((dist, j))
                distances.sort()
                other_indices = [idx for _, idx in distances[:self.n_neighbors]]

            # 计算所有点对的加权余弦值
            for j in range(len(other_indices)):
                for k in range(j + 1, len(other_indices)):
                    total_comparisons += 1
                    a_idx, b_idx = other_indices[j], other_indices[k]
                    a, b = X[a_idx, :], X[b_idx, :]

                    # 跳过与当前点相同的点
                    if np.allclose(a, curr_pt, rtol=1e-10) or np.allclose(b, curr_pt, rtol=1e-10):
                        continue

                    # 计算加权余弦
                    wcos = self._calculate_wcos(curr_pt, a, b)
                    if not np.isnan(wcos) and not np.isinf(wcos):
                        wcos_list.append(wcos)
                        total_wcos_calculations += 1

            # 计算方差
            if len(wcos_list) > 1:
                angle_variance = np.var(wcos_list)
            else:
                angle_variance = 0.0

        end_time = time.perf_counter()
        computation_time = end_time - start_time

        # 算法详细信息
        details = {
            'computation_time': computation_time,
            'target_index': target_index,
            'n_samples': n_samples,
            'n_features': X.shape[1],
            'method': self.method,
            'n_neighbors': self.n_neighbors,
            'other_points_count': len(other_indices),
            'total_comparisons': total_comparisons,
            'total_wcos_calculations': total_wcos_calculations,
            'wcos_values_count': len(wcos_list)
        }

        return angle_variance, details

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


class WorkerQualityAssessor:
    """工人数据质量评估器"""

    def __init__(self, u1_threshold: float = 1e-6, u2_threshold: float = 1e-8,
                 n_neighbors: int = 8, method: str = 'fast'):
        """
        初始化评估器

        Parameters:
        -----------
        u1_threshold : float
            高质量数据的阈值，大于此值为高质量数据
        u2_threshold : float
            低质量数据的阈值，小于此值为低质量数据
        n_neighbors : int
            ABOD算法的邻居数量参数
        method : str
            ABOD算法的方法 ('fast' 或 'default')
        """
        self.u1_threshold = u1_threshold
        self.u2_threshold = u2_threshold
        self.detector = ABODDetector(n_neighbors=n_neighbors, method=method)

    def _classify_data_quality(self, angle_variance: float) -> DataQuality:
        """根据角度方差分类数据质量"""
        if angle_variance > self.u1_threshold:
            return DataQuality.HIGH
        elif angle_variance < self.u2_threshold:
            return DataQuality.LOW
        else:
            return DataQuality.UNCERTAIN

    def assess_multiple_workers(self, worker_data: Dict[int, List[float]],
                                include_classification: bool = True) -> AssessmentResult:
        """
        功能一：评估多个工人的数据质量

        Parameters:
        -----------
        worker_data : Dict[int, List[float]]
            工人数据字典，格式：{worker_id: [data_values]}
        include_classification : bool
            是否包含数据质量分类

        Returns:
        --------
        AssessmentResult
            评估结果，包含每个工人的结果和运行时间
        """
        overall_start_time = time.perf_counter()

        if len(worker_data) < 2:
            raise ValueError("至少需要2个工人的数据才能进行ABOD分析")

        # 准备数据
        data_prep_start = time.perf_counter()
        worker_ids = list(worker_data.keys())
        data_matrix = np.array([worker_data[worker_id] for worker_id in worker_ids])
        data_prep_time = time.perf_counter() - data_prep_start

        # 计算角度方差
        angle_variances, algorithm_details = self.detector.calculate_angle_variances(data_matrix)

        # 生成结果
        result_gen_start = time.perf_counter()
        results = []
        for i, worker_id in enumerate(worker_ids):
            angle_variance = angle_variances[i]
            quality = self._classify_data_quality(angle_variance) if include_classification else None

            result = WorkerResult(
                worker_id=worker_id,
                angle_variance=angle_variance,
                data_quality=quality,
                submitted_data=worker_data[worker_id]
            )
            results.append(result)

        # 按角度方差排序（从大到小）
        results.sort(key=lambda x: x.angle_variance, reverse=True)
        result_gen_time = time.perf_counter() - result_gen_start

        overall_end_time = time.perf_counter()
        total_execution_time = overall_end_time - overall_start_time

        # 合并详细信息
        algorithm_details.update({
            'data_preparation_time': data_prep_time,
            'result_generation_time': result_gen_time,
            'total_execution_time': total_execution_time,
            'classification_enabled': include_classification,
            'thresholds': {
                'u1_threshold': self.u1_threshold,
                'u2_threshold': self.u2_threshold
            }
        })

        return AssessmentResult(
            results=results,
            execution_time=total_execution_time,
            algorithm_details=algorithm_details
        )

    def assess_single_worker(self, target_worker_data: List[float],
                             other_workers_data: List[List[float]],
                             include_classification: bool = True) -> AssessmentResult:
        """
        功能二：评估单个工人相对于其他工人的数据质量

        Parameters:
        -----------
        target_worker_data : List[float]
            目标工人的数据
        other_workers_data : List[List[float]]
            其他工人的数据列表
        include_classification : bool
            是否包含数据质量分类

        Returns:
        --------
        AssessmentResult
            目标工人的评估结果，包含运行时间
        """
        overall_start_time = time.perf_counter()

        if len(other_workers_data) < 1:
            raise ValueError("至少需要1个其他工人的数据作为对比")

        # 准备数据矩阵（目标工人数据放在第一行）
        data_prep_start = time.perf_counter()
        all_data = [target_worker_data] + other_workers_data
        data_matrix = np.array(all_data)
        data_prep_time = time.perf_counter() - data_prep_start

        # 计算角度方差
        angle_variances, algorithm_details = self.detector.calculate_angle_variances(data_matrix)

        # 目标工人的角度方差是第一个
        target_angle_variance = angle_variances[0]

        # 分类
        classification_start = time.perf_counter()
        quality = self._classify_data_quality(target_angle_variance) if include_classification else None
        classification_time = time.perf_counter() - classification_start

        result = WorkerResult(
            worker_id=-1,  # 单个评估时不指定ID
            angle_variance=target_angle_variance,
            data_quality=quality,
            submitted_data=target_worker_data
        )

        overall_end_time = time.perf_counter()
        total_execution_time = overall_end_time - overall_start_time

        # 合并详细信息
        algorithm_details.update({
            'data_preparation_time': data_prep_time,
            'classification_time': classification_time,
            'total_execution_time': total_execution_time,
            'classification_enabled': include_classification,
            'target_worker_position': 0,
            'other_workers_count': len(other_workers_data),
            'thresholds': {
                'u1_threshold': self.u1_threshold,
                'u2_threshold': self.u2_threshold
            }
        })

        return AssessmentResult(
            results=result,
            execution_time=total_execution_time,
            algorithm_details=algorithm_details
        )

    def get_statistics(self, results: List[WorkerResult]) -> Dict[str, Any]:
        """获取评估结果统计信息"""
        angle_variances = [r.angle_variance for r in results]

        quality_counts = {
            DataQuality.HIGH.value: 0,
            DataQuality.UNCERTAIN.value: 0,
            DataQuality.LOW.value: 0
        }

        for result in results:
            if result.data_quality:
                quality_counts[result.data_quality.value] += 1

        return {
            'total_workers': len(results),
            'quality_distribution': quality_counts,
            'angle_variance_stats': {
                'mean': float(np.mean(angle_variances)),
                'std': float(np.std(angle_variances)),
                'min': float(np.min(angle_variances)),
                'max': float(np.max(angle_variances)),
                'median': float(np.median(angle_variances))
            }
        }


def create_assessor(u1_threshold: float = 1e-9, u2_threshold: float = 1e-11,
                    n_neighbors: int = 8, method: str = 'fast') -> WorkerQualityAssessor:
    """便捷函数：创建评估器实例"""
    return WorkerQualityAssessor(u1_threshold, u2_threshold, n_neighbors, method)


def print_performance_details(assessment_result: AssessmentResult, title: str = "性能详情"):
    """打印性能详情的便捷函数"""
    print(f"\n=== {title} ===")
    details = assessment_result.algorithm_details
    print(f"总执行时间: {assessment_result.execution_time:.4f} 秒")
    print(f"ABOD核心计算时间: {details['computation_time']:.4f} 秒")

    if 'data_preparation_time' in details:
        print(f"数据准备时间: {details['data_preparation_time']:.4f} 秒")
    if 'result_generation_time' in details:
        print(f"结果生成时间: {details['result_generation_time']:.4f} 秒")
    if 'classification_time' in details:
        print(f"分类处理时间: {details['classification_time']:.4f} 秒")

    print(f"样本数量: {details['n_samples']}")
    print(f"特征维度: {details['n_features']}")
    print(f"计算方法: {details['method']}")
    print(f"邻居数量: {details['n_neighbors']}")
    print(f"总比较次数: {details['total_comparisons']}")
    print(f"有效WCOS计算次数: {details['total_wcos_calculations']}")
    print(f"平均每样本时间: {details['avg_time_per_sample']:.6f} 秒")


# 测试函数
def _test_module():
    """模块测试函数"""
    print("测试ABOD工人数据质量评估模块（包含运行时间）...")

    # 测试数据
    test_worker_data = {
        9: [996.44, 280.13, 79.98, 1240.16, 205.58],
        16: [995.92, 279.68, 80.06, 1234.21, 206.16],
        19: [995.91, 279.66, 79.27, 1233.27, 204.96],
        21: [994.35, 279.04, 79.95, 1228.54, 206.65],
        23: [991.26, 279.48, 79.95, 1238.37, 205.97],
        65: [819.52, 278.32, 80.18, 1238.07, 206.25],  # 异常数据
        85: [1186.66, 313.1, 80.07, 1399.89, 231.91],  # 异常数据
    }

    # 创建评估器
    assessor = create_assessor()

    # 功能一测试
    print("\n=== 功能一测试：多工人评估 ===")
    multi_result = assessor.assess_multiple_workers(test_worker_data)

    print("评估结果：")
    for result in multi_result.results[:5]:
        print(f"工人{result.worker_id}: 角度方差={result.angle_variance:.2e}, 质量={result.data_quality.value}")

    print_performance_details(multi_result, "多工人评估性能")

    # 功能二测试
    print("\n=== 功能二测试：单工人评估 ===")
    target_data = [819.52, 278.32, 80.18, 1238.07, 206.25]  # 明显异常的数据
    other_data = [
        [996.44, 280.13, 79.98, 1240.16, 205.58],
        [995.92, 279.68, 80.06, 1234.21, 206.16],
        [995.91, 279.66, 79.27, 1233.27, 204.96]
    ]

    single_result = assessor.assess_single_worker(target_data, other_data)
    result = single_result.results
    print(f"目标工人: 角度方差={result.angle_variance:.2e}, 质量={result.data_quality.value}")

    print_performance_details(single_result, "单工人评估性能")


if __name__ == "__main__":
    _test_module()

'''

使用方法说明

from abod_worker_assessment import create_assessor, print_performance_details

# 创建评估器
assessor = create_assessor()

# 功能一：多工人评估（带运行时间）
worker_data = {1: [1,2,3,4,5], 2: [1.1,2.1,3.1,4.1,5.1]}
multi_result = assessor.assess_multiple_workers(worker_data)

print(f"多工人评估耗时: {multi_result.execution_time:.4f} 秒")
print_performance_details(multi_result)  # 详细性能信息

# 功能二：单工人评估（带运行时间）
target_data = [1,2,3,4,5]
others_data = [[1.1,2.1,3.1,4.1,5.1], [0.9,1.9,2.9,3.9,4.9]]
single_result = assessor.assess_single_worker(target_data, others_data)

print(f"单工人评估耗时: {single_result.execution_time:.4f} 秒")
print_performance_details(single_result)

'''