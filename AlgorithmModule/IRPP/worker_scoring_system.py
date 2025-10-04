import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
import time


@dataclass
class WorkerHistory:
    """工人历史数据"""
    worker_id: int
    alpha_h: int  # 高质量数据提交次数
    alpha_u: int  # 不确定质量数据提交次数
    alpha_l: int  # 低质量数据提交次数

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.alpha_h, self.alpha_u, self.alpha_l)


@dataclass
class WorkerScore:
    """工人综合评分结果"""
    worker_id: int
    phi_h: float  # 高质量数据概率
    phi_u: float  # 不确定质量数据概率
    phi_l: float  # 低质量数据概率
    uncertainty: float  # 不确定度
    reputation_score: float  # 综合信誉评分
    computation_time: float  # 计算时间


class WorkerScorer:
    """工人综合评分系统"""

    def __init__(self, k_h: float = 1.0, k_u: float = 1.0, k_l: float = 1.0,
                 w_h: float = 1.0, w_u: float = 0.25, w_l: float = -2.5):
        """
        初始化评分系统

        Parameters:
        -----------
        k_h, k_u, k_l : float
            Dirichlet分布调整系数 (默认都为1.0)
        w_h, w_u, w_l : float
            信誉评分权重 (默认: w_h=1.0, w_u=0.25, w_l=-2.5)
        """
        # Dirichlet分布参数
        self.k_h = k_h
        self.k_u = k_u
        self.k_l = k_l

        # 信誉评分权重
        self.w_h = w_h
        self.w_u = w_u
        self.w_l = w_l

    def _calculate_uncertainty(self, alpha_tilde: Tuple[int, int, int],
                               alpha: Tuple[int, int, int]) -> float:
        """计算不确定度"""
        combined_alpha = tuple(a_t + a for a_t, a in zip(alpha_tilde, alpha))
        total_alpha = sum(combined_alpha)

        if total_alpha <= 3:
            return 1.0

        uncertainty = 1.0 / np.sqrt(total_alpha - 3)
        return min(uncertainty, 1.0)

    def _calculate_quality_probabilities(self, worker_history: WorkerHistory,
                                         alpha_tilde: Tuple[int, int, int] = (1, 1, 1)) -> Tuple[
        float, float, float, float]:
        """
        计算工人的质量概率和不确定度

        Returns:
        --------
        Tuple[float, float, float, float]
            phi_h, phi_u, phi_l, uncertainty
        """
        alpha = worker_history.to_tuple()

        # 计算不确定度
        uncertainty = self._calculate_uncertainty(alpha_tilde, alpha)

        # 使用 (1 - 不确定度) 作为信心系数
        confidence = 1.0 - uncertainty

        # 计算调整后的参数
        a_h, a_u, a_l = alpha
        tilde_a_h, tilde_a_u, tilde_a_l = alpha_tilde

        # 计算分母
        denominator = (self.k_h * tilde_a_h + self.k_h * a_h) + \
                      (self.k_u * tilde_a_u + self.k_u * a_u) + \
                      (self.k_l * tilde_a_l + self.k_l * a_l) + 3

        # 计算 phi_h 和 phi_l
        phi_h = confidence * (self.k_h * tilde_a_h + self.k_h * a_h + 1) / denominator
        phi_l = confidence * (self.k_l * tilde_a_l + self.k_l * a_l + 1) / denominator

        # 计算 phi_u
        phi_u = 1 - phi_h - phi_l

        # 确保概率值在有效范围内
        phi_h = max(0.0, min(1.0, phi_h))
        phi_l = max(0.0, min(1.0, phi_l))
        phi_u = max(0.0, min(1.0, phi_u))

        # 归一化以确保总和为1
        total = phi_h + phi_u + phi_l
        if total > 0:
            phi_h /= total
            phi_u /= total
            phi_l /= total

        return phi_h, phi_u, phi_l, uncertainty

    def _calculate_reputation_score(self, phi_h: float, phi_u: float, phi_l: float) -> float:
        """
        计算综合信誉评分

        公式：reputation_score = w_h * phi_h + w_u * phi_u + w_l * phi_l
        """
        return self.w_h * phi_h + self.w_u * phi_u + self.w_l * phi_l

    def score_multiple_workers(self, workers_data: Dict[int, List[int]]) -> List[WorkerScore]:
        """
        【主要功能】批量评估多个工人

        Parameters:
        -----------
        workers_data : Dict[int, List[int]]
            工人数据字典 {worker_id: [alpha_h, alpha_u, alpha_l]}

        Returns:
        --------
        List[WorkerScore]
            所有工人的评分结果列表
        """
        start_time = time.perf_counter()

        results = []

        # 计算每个工人的评分
        for worker_id, history_data in workers_data.items():
            if len(history_data) != 3:
                raise ValueError(f"工人{worker_id}的历史数据必须包含3个元素: [alpha_h, alpha_u, alpha_l]")

            worker_start_time = time.perf_counter()

            worker_history = WorkerHistory(worker_id, history_data[0], history_data[1], history_data[2])

            # 计算质量概率和不确定度
            phi_h, phi_u, phi_l, uncertainty = self._calculate_quality_probabilities(worker_history)

            # 计算信誉评分
            reputation_score = self._calculate_reputation_score(phi_h, phi_u, phi_l)

            worker_end_time = time.perf_counter()
            worker_computation_time = worker_end_time - worker_start_time

            result = WorkerScore(
                worker_id=worker_id,
                phi_h=phi_h,
                phi_u=phi_u,
                phi_l=phi_l,
                uncertainty=uncertainty,
                reputation_score=reputation_score,
                computation_time=worker_computation_time
            )
            results.append(result)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f"批量评估完成，总耗时: {total_time:.4f} 秒")
        print(f"平均每个工人: {total_time / len(workers_data):.6f} 秒")

        return results

    def score_single_worker(self, worker_id: int, history_data: List[int]) -> WorkerScore:
        """
        【次要功能】评估单个工人

        Parameters:
        -----------
        worker_id : int
            工人ID
        history_data : List[int]
            历史数据 [alpha_h, alpha_u, alpha_l]

        Returns:
        --------
        WorkerScore
            包含三元评分、不确定度和综合信誉评分的结果
        """
        start_time = time.perf_counter()

        if len(history_data) != 3:
            raise ValueError("历史数据必须包含3个元素: [alpha_h, alpha_u, alpha_l]")

        worker_history = WorkerHistory(worker_id, history_data[0], history_data[1], history_data[2])

        # 计算质量概率和不确定度
        phi_h, phi_u, phi_l, uncertainty = self._calculate_quality_probabilities(worker_history)

        # 计算信誉评分
        reputation_score = self._calculate_reputation_score(phi_h, phi_u, phi_l)

        end_time = time.perf_counter()
        computation_time = end_time - start_time

        return WorkerScore(
            worker_id=worker_id,
            phi_h=phi_h,
            phi_u=phi_u,
            phi_l=phi_l,
            uncertainty=uncertainty,
            reputation_score=reputation_score,
            computation_time=computation_time
        )


def create_worker_scorer(k_h: float = 1.0, k_u: float = 1.0, k_l: float = 1.0,
                         w_h: float = 1.0, w_u: float = 0.25, w_l: float = -2) -> WorkerScorer:
    """
    便捷函数：创建工人评分系统

    Parameters:
    -----------
    k_h, k_u, k_l : float
        Dirichlet分布调整系数 (默认都为1.0)
    w_h, w_u, w_l : float
        信誉评分权重 (默认: w_h=1.0, w_u=0.25, w_l=-2.5)

    Returns:
    --------
    WorkerScorer
        评分系统实例
    """
    return WorkerScorer(k_h, k_u, k_l, w_h, w_u, w_l)


def print_worker_results(results: List[WorkerScore], title: str = "工人综合评分结果"):
    """打印工人评分结果"""
    print(f"\n=== {title} ===")
    print(f"{'工人ID':<8} {'φ_h':<8} {'φ_u':<8} {'φ_l':<8} {'不确定度':<10} {'信誉评分':<12}")
    print("-" * 65)

    for result in results:
        print(f"{result.worker_id:<8} {result.phi_h:<8.3f} {result.phi_u:<8.3f} "
              f"{result.phi_l:<8.3f} {result.uncertainty:<10.4f} {result.reputation_score:<12.4f}")


# 测试函数
def _test_worker_scorer():
    """测试工人评分系统"""
    print("测试工人综合评分系统...")

    # 创建评分系统（使用指定参数）
    scorer = create_worker_scorer()

    print(f"系统参数:")
    print(f"- Dirichlet系数: k_h={scorer.k_h}, k_u={scorer.k_u}, k_l={scorer.k_l}")
    print(f"- 信誉权重: w_h={scorer.w_h}, w_u={scorer.w_u}, w_l={scorer.w_l}")
    print(f"- 信誉评分公式: {scorer.w_h}×φ_h + {scorer.w_u}×φ_u + {scorer.w_l}×φ_l")

    # 【主要功能测试】批量工人评估
    print(f"\n=== 主要功能：批量工人评估测试 ===")
    test_data = {
        1: [25, 5, 2],  # 优秀工人
        2: [15, 8, 4],  # 良好工人
        3: [8, 12, 8],  # 一般工人
        4: [3, 6, 12],  # 较差工人
        5: [1, 2, 1],  # 新工人
        6: [40, 8, 2],  # 资深优秀工人
        7: [5, 15, 20],  # 低质量工人
    }

    results = scorer.score_multiple_workers(test_data)
    print_worker_results(results)

    # 按信誉评分排序
    sorted_results = sorted(results, key=lambda x: x.reputation_score, reverse=True)
    print_worker_results(sorted_results, "按信誉评分排序结果")

    # 【次要功能测试】单个工人评估
    print(f"\n=== 次要功能：单个工人评估测试 ===")
    single_result = scorer.score_single_worker(worker_id=99, history_data=[20, 5, 3])
    print(f"工人99 历史数据[20,5,3]:")
    print(f"  φ_h={single_result.phi_h:.3f}, φ_u={single_result.phi_u:.3f}, φ_l={single_result.phi_l:.3f}")
    print(f"  不确定度={single_result.uncertainty:.4f}")
    print(f"  信誉评分={single_result.reputation_score:.4f}")
    print(f"  计算时间={single_result.computation_time * 1000:.2f}ms")


if __name__ == "__main__":
    _test_worker_scorer()


"""

使用评分系统示例

from worker_comprehensive_scoring import create_worker_scorer

# 创建评分系统（使用指定参数）
scorer = create_worker_scorer()

# 【主要功能】批量工人评估
workers_data = {
    1: [12, 2, 1],    # 工人1历史数据
    2: [8, 5, 3],     # 工人2历史数据
    3: [5, 8, 6],     # 工人3历史数据
    4: [2, 4, 15]     # 工人4历史数据
}

results = scorer.score_multiple_workers(workers_data)
for result in results:
    print(f"工人{result.worker_id}: φ_h={result.phi_h:.3f}, φ_u={result.phi_u:.3f}, "
          f"φ_l={result.phi_l:.3f}, 不确定度={result.uncertainty:.4f}, "
          f"信誉评分={result.reputation_score:.4f}")

# 【次要功能】单个工人评估
single_result = scorer.score_single_worker(worker_id=99, history_data=[15, 3, 2])


print(f"工人99: 信誉评分={single_result.reputation_score:.4f}")

"""