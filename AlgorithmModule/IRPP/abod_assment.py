import numpy as np
import random
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict
from enum import Enum

# 导入封装的ABOD检测器
from abod import ABODDetector

# ========== 参数设置区域 ==========
TRUE_DATA = [100, 200, 300, 400, 500]  # 真实数据
N_WORKERS = 10000  # 工人数量

# 方法一的阈值
METHOD1_U1_THRESHOLD = 1e-3
METHOD1_U2_THRESHOLD = 1e-5

# 方法二的阈值
METHOD2_U1_THRESHOLD = 5e-7
METHOD2_U2_THRESHOLD = 5e-12

# 方法三离群点检测的参数
DISTANCE_THRESHOLD = 10  # r = 4 距离阈值
PROPORTION_THRESHOLD = 0.31  # μ = 0.31 比例阈值

# 方法三三分类阈值
METHOD3_HIGH_PROPORTION = 0.1  # 高比例阈值（优异工人）
METHOD3_LOW_PROPORTION = 0.0005  # 低比例阈值（恶意工人）

RANDOM_SEED = 42  # 随机种子


# ===============================

class WorkerType(Enum):
    EXCELLENT = "excellent"  # 优异工人
    UNCERTAIN = "uncertain"  # 不确定工人
    MALICIOUS = "malicious"  # 恶意工人


@dataclass
class Worker:
    worker_id: int
    worker_type: WorkerType
    submitted_data: List[float]


class WorkerDataGenerator:
    """工人数据生成器"""

    def __init__(self, true_data: List[float], seed: int = 42):
        self.true_data = true_data
        np.random.seed(seed)
        random.seed(seed)

    def generate_workers(self, n: int) -> List[Worker]:
        workers = []

        # 计算各类型工人数量
        excellent_count = int(n * 0.5)  # 前50%
        uncertain_count = int(n * 0.25)  # 50%-75%
        malicious_count = n - excellent_count - uncertain_count  # 75%-100%

        print(f"工人分布: 优异 {excellent_count}, 不确定 {uncertain_count}, 恶意 {malicious_count}")

        worker_id = 1

        # 生成优异工人 (误差1%以内)
        for i in range(excellent_count):
            data = [val * (1 + np.random.uniform(-0.01, 0.01)) for val in self.true_data]
            workers.append(Worker(worker_id, WorkerType.EXCELLENT, data))
            worker_id += 1

        # 生成不确定工人 (误差2%-5%)
        for i in range(uncertain_count):
            data = [val * (1 + np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.05))
                    for val in self.true_data]
            workers.append(Worker(worker_id, WorkerType.UNCERTAIN, data))
            worker_id += 1

        # 生成恶意工人 (误差10%-30%)
        for i in range(malicious_count):
            data = [val * (1 + np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3))
                    for val in self.true_data]
            workers.append(Worker(worker_id, WorkerType.MALICIOUS, data))
            worker_id += 1

        return workers


class OutlierDetector:
    """离群点检测器 - 支持三分类"""

    def __init__(self, distance_threshold: float = 4.0, proportion_threshold: float = 0.31,
                 high_proportion: float = 0.7, low_proportion: float = 0.15):
        self.distance_threshold = distance_threshold  # r = 4
        self.proportion_threshold = proportion_threshold  # μ = 0.31 (用于离群点检测)
        self.high_proportion = high_proportion  # 高比例阈值（优异工人）
        self.low_proportion = low_proportion  # 低比例阈值（恶意工人）

    def euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算两点间的欧几里得距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    def outlier_detection_with_classification(self, data_points: List[List[float]], worker_ids: List[int]) -> Tuple[
        Set[int], Dict[int, float], Dict[int, WorkerType]]:
        """
        离群点检测模块 - 支持三分类

        Returns:
            outlier_indices: 离群点的索引集合
            proportions: 每个点的邻近点比例
            classifications: 每个点的分类结果
        """
        n = len(data_points)
        if n <= 2:
            return set(), {}, {}

        outlier_indices = set()
        proportions = {}
        classifications = {}

        for i, point_i in enumerate(data_points):
            # 计算与其他点的距离
            close_points_count = 0

            for j, point_j in enumerate(data_points):
                if i != j:
                    dist = self.euclidean_distance(point_i, point_j)
                    if dist <= self.distance_threshold:  # r = 4
                        close_points_count += 1

            # 计算比例
            proportion = close_points_count / (n - 1)
            proportions[i] = proportion

            # 三分类逻辑
            if proportion >= self.high_proportion:
                # 高比例 -> 优异工人（周围有很多相似的点）
                classifications[i] = WorkerType.EXCELLENT
            elif proportion <= self.low_proportion:
                # 低比例 -> 恶意工人（周围很少相似的点，孤立）
                classifications[i] = WorkerType.MALICIOUS
            else:
                # 中等比例 -> 不确定工人
                classifications[i] = WorkerType.UNCERTAIN

            # 原始离群点检测逻辑（用于统计）
            if proportion <= self.proportion_threshold:  # μ = 0.31
                outlier_indices.add(i)

        return outlier_indices, proportions, classifications


class WorkerQualityAssessor:
    """工人质量评估器"""

    def __init__(self, u1_threshold: float, u2_threshold: float, method_name: str = ""):
        self.u1_threshold = u1_threshold
        self.u2_threshold = u2_threshold
        self.method_name = method_name

    def classify_worker(self, angle_variance: float) -> WorkerType:
        """根据角度方差分类工人质量"""
        if angle_variance > self.u1_threshold:
            return WorkerType.EXCELLENT
        elif angle_variance < self.u2_threshold:
            return WorkerType.MALICIOUS
        else:
            return WorkerType.UNCERTAIN

    def get_threshold_info(self) -> str:
        """返回阈值信息"""
        return f"u1={self.u1_threshold:.0e}, u2={self.u2_threshold:.0e}"


def method1_batch_detection(workers: List[Worker], detector: ABODDetector, assessor: WorkerQualityAssessor) -> Tuple[
    float, float]:
    """方法一：批量检测所有工人"""
    print(f"\n=== 方法一：批量检测所有工人 ===")
    print(f"使用阈值: {assessor.get_threshold_info()}")

    # 准备数据矩阵
    data_matrix = np.array([worker.submitted_data for worker in workers])

    # 批量计算所有工人的角度方差
    start_time = time.perf_counter()
    angle_variances, details = detector.calculate_angle_variances(data_matrix)
    print(angle_variances)
    end_time = time.perf_counter()

    computation_time = end_time - start_time

    # 使用评估器进行分类并计算准确率
    correct_predictions = 0
    type_stats = {
        WorkerType.EXCELLENT: {'correct': 0, 'total': 0,
                               'predicted_as': {'excellent': 0, 'uncertain': 0, 'malicious': 0}},
        WorkerType.UNCERTAIN: {'correct': 0, 'total': 0,
                               'predicted_as': {'excellent': 0, 'uncertain': 0, 'malicious': 0}},
        WorkerType.MALICIOUS: {'correct': 0, 'total': 0,
                               'predicted_as': {'excellent': 0, 'uncertain': 0, 'malicious': 0}}
    }

    for i, worker in enumerate(workers):
        predicted_type = assessor.classify_worker(angle_variances[i])
        actual_type = worker.worker_type

        type_stats[actual_type]['total'] += 1
        type_stats[actual_type]['predicted_as'][predicted_type.value] += 1

        if predicted_type == actual_type:
            correct_predictions += 1
            type_stats[actual_type]['correct'] += 1

    accuracy = correct_predictions / len(workers)

    print(f"角度方差计算时间: {computation_time:.4f} 秒")
    print(f"识别准确率: {accuracy:.2%}")

    # 显示各类型分类统计
    print("\n各类型工人分类统计:")
    for worker_type, stats in type_stats.items():
        if stats['total'] > 0:
            type_accuracy = stats['correct'] / stats['total']
            print(f"{worker_type.value}: 准确率 {type_accuracy:.1%} ({stats['correct']}/{stats['total']})")
            print(
                f"  预测为: 优异{stats['predicted_as']['excellent']}, 不确定{stats['predicted_as']['uncertain']}, 恶意{stats['predicted_as']['malicious']}")

    return computation_time, accuracy


def method2_single_detection(workers: List[Worker], detector: ABODDetector, assessor: WorkerQualityAssessor) -> Tuple[
    float, float]:
    """方法二：单点检测每个工人"""
    print(f"\n=== 方法二：单点检测每个工人 ===")
    print(f"使用阈值: {assessor.get_threshold_info()}")

    n = len(workers)
    excellent_worker_count = int(n * 0.5)
    excellent_workers = [w for w in workers if w.worker_id <= excellent_worker_count]

    # 确定参考工人数量
    if n < 100:
        reference_count = min(10, len(excellent_workers))
        print(f"总工人数 {n} < 100，从前50%工人中选择 {reference_count} 个作为参考")
    else:
        reference_count = min(int(math.sqrt(n)), len(excellent_workers))
        print(f"总工人数 {n} >= 100，从前50%工人中选择 √{n} = {reference_count} 个作为参考")

    start_time = time.perf_counter()

    worker_variances = {}
    correct_predictions = 0
    type_stats = {
        WorkerType.EXCELLENT: {'correct': 0, 'total': 0,
                               'predicted_as': {'excellent': 0, 'uncertain': 0, 'malicious': 0}},
        WorkerType.UNCERTAIN: {'correct': 0, 'total': 0,
                               'predicted_as': {'excellent': 0, 'uncertain': 0, 'malicious': 0}},
        WorkerType.MALICIOUS: {'correct': 0, 'total': 0,
                               'predicted_as': {'excellent': 0, 'uncertain': 0, 'malicious': 0}}
    }

    # 对每个工人进行单点检测
    for worker in workers:
        # 随机选择参考工人
        if reference_count >= len(excellent_workers):
            selected_reference_workers = excellent_workers
        else:
            selected_reference_workers = random.sample(excellent_workers, reference_count)

        # 准备参考数据
        reference_data = np.array([w.submitted_data for w in selected_reference_workers])
        target_point = np.array(worker.submitted_data)

        # 计算该工人的角度方差
        angle_variance = detector.calculate_single_point_variance(target_point, reference_data)
        worker_variances[worker.worker_id] = angle_variance

        # 使用评估器进行分类
        predicted_type = assessor.classify_worker(angle_variance)
        actual_type = worker.worker_type

        type_stats[actual_type]['total'] += 1
        type_stats[actual_type]['predicted_as'][predicted_type.value] += 1

        if predicted_type == actual_type:
            correct_predictions += 1
            type_stats[actual_type]['correct'] += 1

    end_time = time.perf_counter()
    computation_time = end_time - start_time
    accuracy = correct_predictions / len(workers)

    print(f"角度方差计算时间: {computation_time:.4f} 秒")
    print(f"识别准确率: {accuracy:.2%}")

    # 显示各类型分类统计
    print("\n各类型工人分类统计:")
    for worker_type, stats in type_stats.items():
        if stats['total'] > 0:
            type_accuracy = stats['correct'] / stats['total']
            print(f"{worker_type.value}: 准确率 {type_accuracy:.1%} ({stats['correct']}/{stats['total']})")
            print(
                f"  预测为: 优异{stats['predicted_as']['excellent']}, 不确定{stats['predicted_as']['uncertain']}, 恶意{stats['predicted_as']['malicious']}")

    return computation_time, accuracy


def method3_outlier_detection(workers: List[Worker], outlier_detector: OutlierDetector) -> Tuple[float, float]:
    """
    方法三：离群点检测 - 三分类版本
    基于邻近点比例进行三分类

    Returns:
    --------
    Tuple[float, float]
        (计算时间, 识别准确率)
    """
    print(f"\n=== 方法三：离群点检测（三分类） ===")
    print(f"使用参数: 距离阈值 r={outlier_detector.distance_threshold}")
    print(
        f"三分类阈值: 高比例≥{outlier_detector.high_proportion} (优异), 低比例≤{outlier_detector.low_proportion} (恶意)")

    # 准备数据
    data_points = [worker.submitted_data for worker in workers]
    worker_ids = [worker.worker_id for worker in workers]

    # 执行离群点检测和三分类
    start_time = time.perf_counter()
    outlier_indices, proportions, classifications = outlier_detector.outlier_detection_with_classification(data_points,
                                                                                                           worker_ids)
    end_time = time.perf_counter()

    computation_time = end_time - start_time

    # 计算三分类准确率
    correct_predictions = 0
    type_stats = {
        WorkerType.EXCELLENT: {'correct': 0, 'total': 0,
                               'predicted_as': {'excellent': 0, 'uncertain': 0, 'malicious': 0}},
        WorkerType.UNCERTAIN: {'correct': 0, 'total': 0,
                               'predicted_as': {'excellent': 0, 'uncertain': 0, 'malicious': 0}},
        WorkerType.MALICIOUS: {'correct': 0, 'total': 0,
                               'predicted_as': {'excellent': 0, 'uncertain': 0, 'malicious': 0}}
    }

    for i, worker in enumerate(workers):
        actual_type = worker.worker_type
        predicted_type = classifications[i]

        type_stats[actual_type]['total'] += 1
        type_stats[actual_type]['predicted_as'][predicted_type.value] += 1

        if predicted_type == actual_type:
            correct_predictions += 1
            type_stats[actual_type]['correct'] += 1

    accuracy = correct_predictions / len(workers)

    print(f"离群点检测计算时间: {computation_time:.4f} 秒")
    print(f"识别准确率: {accuracy:.2%}")
    print(f"传统离群点数量: {len(outlier_indices)} (比例≤{outlier_detector.proportion_threshold})")

    # 显示三分类统计
    print("\n各类型工人分类统计:")
    for worker_type, stats in type_stats.items():
        if stats['total'] > 0:
            type_accuracy = stats['correct'] / stats['total']
            print(f"{worker_type.value}: 准确率 {type_accuracy:.1%} ({stats['correct']}/{stats['total']})")
            print(
                f"  预测为: 优异{stats['predicted_as']['excellent']}, 不确定{stats['predicted_as']['uncertain']}, 恶意{stats['predicted_as']['malicious']}")

    # 显示比例分布统计
    proportion_ranges = {
        'high': [p for p in proportions.values() if p >= outlier_detector.high_proportion],
        'medium': [p for p in proportions.values() if
                   outlier_detector.low_proportion < p < outlier_detector.high_proportion],
        'low': [p for p in proportions.values() if p <= outlier_detector.low_proportion]
    }

    print(f"\n邻近点比例分布:")
    print(f"  高比例 (≥{outlier_detector.high_proportion}): {len(proportion_ranges['high'])} 个")
    print(
        f"  中等比例 ({outlier_detector.low_proportion}-{outlier_detector.high_proportion}): {len(proportion_ranges['medium'])} 个")
    print(f"  低比例 (≤{outlier_detector.low_proportion}): {len(proportion_ranges['low'])} 个")

    # 显示部分详细结果
    if len(workers) <= 20:
        print("\n详细结果 (前10个工人):")
        print("ID\t实际类型\t\t预测类型\t\t邻近比例\t正确")
        print("-" * 70)
        for i in range(min(10, len(workers))):
            worker = workers[i]
            predicted_type = classifications[i]
            proportion = proportions[i]
            is_correct = predicted_type == worker.worker_type
            print(
                f"{worker.worker_id}\t{worker.worker_type.value:15}\t{predicted_type.value:15}\t{proportion:.3f}\t\t{'✓' if is_correct else '✗'}")

    return computation_time, accuracy


def analyze_results(workers: List[Worker], method1_results: Tuple[float, float],
                    method2_results: Tuple[float, float], method3_results: Tuple[float, float]):
    """分析和对比三种方法的结果"""
    time1, accuracy1 = method1_results
    time2, accuracy2 = method2_results
    time3, accuracy3 = method3_results

    print("\n" + "=" * 80)
    print("三种方法对比结果:")
    print("=" * 80)

    print(f"方法一（ABOD批量检测）:")
    print(f"  阈值设置: u1={METHOD1_U1_THRESHOLD:.0e}, u2={METHOD1_U2_THRESHOLD:.0e}")
    print(f"  计算时间: {time1:.4f} 秒")
    print(f"  识别准确率: {accuracy1:.2%}")

    print(f"\n方法二（ABOD单点检测）:")
    print(f"  阈值设置: u1={METHOD2_U1_THRESHOLD:.0e}, u2={METHOD2_U2_THRESHOLD:.0e}")
    print(f"  计算时间: {time2:.4f} 秒")
    print(f"  识别准确率: {accuracy2:.2%}")

    print(f"\n方法三（离群点检测三分类）:")
    print(f"  参数设置: r={DISTANCE_THRESHOLD}, 高≥{METHOD3_HIGH_PROPORTION}, 低≤{METHOD3_LOW_PROPORTION}")
    print(f"  计算时间: {time3:.4f} 秒")
    print(f"  识别准确率: {accuracy3:.2%}")

    print(f"\n综合性能对比:")
    fastest_method = min(enumerate([time1, time2, time3]), key=lambda x: x[1])
    most_accurate = max(enumerate([accuracy1, accuracy2, accuracy3]), key=lambda x: x[1])

    method_names = ["方法一(ABOD批量)", "方法二(ABOD单点)", "方法三(离群点检测)"]
    print(f"  最快方法: {method_names[fastest_method[0]]} ({fastest_method[1]:.4f} 秒)")
    print(f"  最准确方法: {method_names[most_accurate[0]]} ({most_accurate[1]:.2%})")

    print(f"\n时间比较:")
    print(f"  方法二/方法一: {time2 / time1:.2f}")
    print(f"  方法三/方法一: {time3 / time1:.2f}")
    print(f"  方法三/方法二: {time3 / time2:.2f}")


def main():
    """主函数"""
    print("工人数据质量评估系统 - 三种方法对比（均为三分类）")
    print("=" * 60)
    print(f"真实数据: {TRUE_DATA}")
    print(f"数据维度: {len(TRUE_DATA)}")
    print(f"工人数量: {N_WORKERS}")
    print(f"方法一阈值: u1={METHOD1_U1_THRESHOLD:.0e}, u2={METHOD1_U2_THRESHOLD:.0e}")
    print(f"方法二阈值: u1={METHOD2_U1_THRESHOLD:.0e}, u2={METHOD2_U2_THRESHOLD:.0e}")
    print(f"方法三参数: r={DISTANCE_THRESHOLD}, 高≥{METHOD3_HIGH_PROPORTION}, 低≤{METHOD3_LOW_PROPORTION}")

    # 生成工人数据
    generator = WorkerDataGenerator(TRUE_DATA, RANDOM_SEED)
    workers = generator.generate_workers(N_WORKERS)

    # 显示部分工人数据样本
    print(f"\n工人数据样本 (前5个):")
    print("ID\t类型\t\t\t提交数据")
    print("-" * 50)
    for worker in workers[:5]:
        data_str = [f"{x:.2f}" for x in worker.submitted_data]
        print(f"{worker.worker_id}\t{worker.worker_type.value:15}\t{data_str}")

    # 创建检测器
    abod_detector = ABODDetector(n_neighbors=5, method='fast')
    outlier_detector = OutlierDetector(DISTANCE_THRESHOLD, PROPORTION_THRESHOLD,
                                       METHOD3_HIGH_PROPORTION, METHOD3_LOW_PROPORTION)

    # 为前两种方法创建不同的评估器
    assessor1 = WorkerQualityAssessor(METHOD1_U1_THRESHOLD, METHOD1_U2_THRESHOLD, "方法一")
    assessor2 = WorkerQualityAssessor(METHOD2_U1_THRESHOLD, METHOD2_U2_THRESHOLD, "方法二")

    # 执行三种方法
    method1_results = method1_batch_detection(workers, abod_detector, assessor1)
    method2_results = method2_single_detection(workers, abod_detector, assessor2)
    method3_results = method3_outlier_detection(workers, outlier_detector)

    # 分析结果
    analyze_results(workers, method1_results, method2_results, method3_results)


if __name__ == "__main__":
    main()
