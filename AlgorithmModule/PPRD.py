import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class CrowdSensingTruthDiscovery:
    def __init__(self,
                 tau: float = 1.0,
                 t0: float = 0.5,
                 epsilon: float = 1e-6,
                 xi: float = 1e-6,
                 max_iterations: int = 100,
                 distance_metric: str = 'absolute'):
        """
        群智感知真相发现算法

        Args:
            tau: 控制因子，用于将声誉映射为可靠度
            t0: 任务的声誉门槛
            epsilon: 收敛阈值
            xi: 质量评估里用于平滑的小常数
            max_iterations: 最大迭代次数
            distance_metric: 距离函数类型 ('absolute' 或 'squared')
        """
        self.tau = tau
        self.t0 = t0
        self.epsilon = epsilon
        self.xi = xi
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric
        self.eps_small = 1e-12  # 防止log(0)的小常数

    def distance_function(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """距离函数"""
        if self.distance_metric == 'absolute':
            return np.abs(a - b)
        elif self.distance_metric == 'squared':
            return (a - b) ** 2
        else:
            raise ValueError("distance_metric must be 'absolute' or 'squared'")

    def load_data(self, file_path: str) -> Tuple[np.ndarray, Dict[int, float], Dict[int, List]]:
        """
        加载数据文件

        Returns:
            X: 观测矩阵 (K x M x D) - K个用户，M个任务，每个观测D维
            worker_reputations: 用户声誉字典
            task_true_values: 真实值字典（用于验证）
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        task_data = data['task_worker_data']

        # 收集所有工人ID和任务ID
        all_workers = set()
        all_tasks = []

        for task_id, task_info in task_data.items():
            all_tasks.append(int(task_id))
            for submission in task_info['worker_submissions']:
                all_workers.add(submission['worker_id'])

        all_workers = sorted(list(all_workers))
        all_tasks = sorted(all_tasks)

        K = len(all_workers)  # 用户数量
        M = len(all_tasks)  # 任务数量

        # 获取观测维度（假设所有观测维度相同）
        first_task = task_data[str(all_tasks[0])]
        D = len(first_task['task_true_data'])  # 观测维度

        # 初始化观测矩阵
        X = np.full((K, M, D), np.nan)

        # 创建工人ID到索引的映射
        worker_to_idx = {worker_id: idx for idx, worker_id in enumerate(all_workers)}
        task_to_idx = {task_id: idx for idx, task_id in enumerate(all_tasks)}

        # 填充观测矩阵
        for task_id, task_info in task_data.items():
            task_idx = task_to_idx[int(task_id)]
            for submission in task_info['worker_submissions']:
                worker_idx = worker_to_idx[submission['worker_id']]
                X[worker_idx, task_idx, :] = np.array(submission['submitted_data'])

        # 初始化工人声誉（这里随机初始化，实际应用中应该有历史声誉）
        np.random.seed(42)  # 保证可重现性
        worker_reputations = {}
        for worker_id in all_workers:
            # 根据参与任务数量和随机因子初始化声誉
            base_reputation = 0.6 + 0.3 * np.random.random()
            worker_reputations[worker_id] = base_reputation

        # 收集真实值用于验证
        task_true_values = {}
        for task_id, task_info in task_data.items():
            task_true_values[int(task_id)] = task_info['task_true_data']

        return X, worker_reputations, task_true_values, all_workers, all_tasks

    def compute_reliability(self, reputations: np.ndarray) -> np.ndarray:
        """
        步骤1：计算可靠度

        Args:
            reputations: 声誉数组

        Returns:
            gamma: 可靠度数组
        """
        gamma = np.zeros_like(reputations)
        mask = reputations >= self.t0
        gamma[mask] = (reputations[mask] - self.t0) / self.tau
        return gamma

    def iterative_truth_discovery(self, X: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        步骤2：迭代真相发现

        Args:
            X: 观测矩阵 (K, M, D)
            gamma: 可靠度数组 (K,)

        Returns:
            x_star: 最终真值 (M, D)
            weights: 最终权重 (K,)
            iterations: 实际迭代次数
        """
        K, M, D = X.shape

        # 初始化真值（使用每个任务每个维度的中位数）
        x_current = np.nanmedian(X, axis=0)  # (M, D)

        for iteration in range(self.max_iterations):
            x_previous = x_current.copy()

            # 2.1 计算每个用户的聚合距离
            d_k = np.zeros(K)
            for k in range(K):
                for m in range(M):
                    if not np.isnan(X[k, m]).any():  # 用户k参与了任务m
                        distances = self.distance_function(X[k, m], x_current[m])
                        d_k[k] += np.sum(distances)  # 对所有维度求和

            # 防止d_k为0的情况
            d_k = np.maximum(d_k, self.eps_small)

            # 2.2 计算加权和S
            S = np.sum(gamma * d_k)
            S = max(S, self.eps_small)  # 防止S为0

            # 2.3 计算权重
            omega = np.zeros(K)
            for k in range(K):
                if gamma[k] > 0:  # 只对可靠用户计算权重
                    gamma_d = gamma[k] * d_k[k]
                    gamma_d = max(gamma_d, self.eps_small)
                    omega[k] = np.log(S) - np.log(gamma_d)
                else:
                    omega[k] = 0  # 不可靠用户权重为0

            # 归一化权重（可选，避免数值问题）
            if np.sum(gamma) > 0:
                min_omega = np.min(omega[gamma > 0])
                omega = omega - min_omega

            # 2.4 更新真值
            for m in range(M):
                numerator = np.zeros(D)
                denominator = 0

                for k in range(K):
                    if not np.isnan(X[k, m]).any() and gamma[k] > 0:  # 用户k参与了任务m且可靠
                        weight = gamma[k] * omega[k]
                        numerator += weight * X[k, m]
                        denominator += weight

                if denominator > self.eps_small:
                    x_current[m] = numerator / denominator
                # 如果分母为0，保持当前值不变

            # 2.5 检查收敛
            diff = np.linalg.norm(x_current - x_previous)
            if diff < self.epsilon:
                return x_current, omega, iteration + 1

        return x_current, omega, self.max_iterations

    def quality_assessment(self, X: np.ndarray, x_star: np.ndarray) -> np.ndarray:
        """
        步骤3：质量评估

        Args:
            X: 观测矩阵 (K, M, D)
            x_star: 最终真值 (M, D)

        Returns:
            Q: 质量矩阵 (K, M)
        """
        K, M, D = X.shape
        Q = np.zeros((K, M))

        for m in range(M):
            distances = np.zeros(K)

            # 计算每个用户对任务m的距离
            for k in range(K):
                if not np.isnan(X[k, m]).any():
                    dist_vector = self.distance_function(X[k, m], x_star[m])
                    distances[k] = np.sum(dist_vector)  # 对所有维度求和
                else:
                    distances[k] = np.inf  # 未参与的用户距离设为无穷大

            # 计算质量分数
            valid_mask = distances != np.inf
            if np.sum(valid_mask) > 0:
                inv_distances = 1.0 / (distances + self.xi)
                inv_distances[~valid_mask] = 0  # 未参与用户的倒数距离为0

                sum_inv_distances = np.sum(inv_distances)
                if sum_inv_distances > 0:
                    Q[:, m] = inv_distances / sum_inv_distances

        return Q

    def reputation_update(self, gamma: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        步骤4：声誉更新

        Args:
            gamma: 当前可靠度 (K,)
            Q: 质量矩阵 (K, M)

        Returns:
            new_reputations: 更新后的声誉 (K,)
        """
        K, M = Q.shape

        # 计算每个用户的平均质量
        q_k = np.zeros(K)
        for k in range(K):
            valid_tasks = Q[k, :] > 0  # 用户参与的任务
            if np.sum(valid_tasks) > 0:
                q_k[k] = np.mean(Q[k, valid_tasks])

        # 使用Sigmoid函数更新声誉
        new_reputations = 1.0 / (1.0 + np.exp(-(gamma * q_k - 1)))

        return new_reputations

    def fit(self, file_path: str) -> Dict:
        """
        运行完整的真相发现算法

        Args:
            file_path: 数据文件路径

        Returns:
            results: 包含所有结果的字典
        """
        print("正在加载数据...")
        X, worker_reputations, task_true_values, all_workers, all_tasks = self.load_data(file_path)

        K, M, D = X.shape
        print(f"数据加载完成：{K}个用户，{M}个任务，每个观测{D}维")

        # 转换声誉为数组
        reputation_array = np.array([worker_reputations[worker_id] for worker_id in all_workers])

        print("\n步骤1：计算可靠度...")
        gamma = self.compute_reliability(reputation_array)
        reliable_users = np.sum(gamma > 0)
        print(f"可靠用户数量：{reliable_users}/{K}")

        print("\n步骤2：迭代真相发现...")
        x_star, final_weights, iterations = self.iterative_truth_discovery(X, gamma)
        print(f"迭代收敛，共{iterations}轮")

        print("\n步骤3：质量评估...")
        Q = self.quality_assessment(X, x_star)

        print("\n步骤4：声誉更新...")
        new_reputations = self.reputation_update(gamma, Q)

        # 计算准确性指标（如果有真实值）
        accuracies = {}
        if task_true_values:
            print("\n计算准确性指标...")
            for i, task_id in enumerate(all_tasks):
                if task_id in task_true_values:
                    true_value = np.array(task_true_values[task_id])
                    predicted_value = x_star[i]

                    # 计算各种误差指标
                    mae = np.mean(np.abs(predicted_value - true_value))
                    mse = np.mean((predicted_value - true_value) ** 2)
                    rmse = np.sqrt(mse)

                    accuracies[task_id] = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'true_value': true_value.tolist(),
                        'predicted_value': predicted_value.tolist()
                    }

        # 整理结果
        results = {
            'final_truth_values': x_star,
            'user_reliabilities': dict(zip(all_workers, gamma)),
            'user_final_weights': dict(zip(all_workers, final_weights)),
            'user_quality_scores': Q,
            'initial_reputations': dict(zip(all_workers, reputation_array)),
            'updated_reputations': dict(zip(all_workers, new_reputations)),
            'task_accuracies': accuracies,
            'convergence_iterations': iterations,
            'algorithm_params': {
                'tau': self.tau,
                't0': self.t0,
                'epsilon': self.epsilon,
                'xi': self.xi,
                'distance_metric': self.distance_metric
            }
        }

        return results

    def print_summary(self, results: Dict):
        """打印结果摘要"""
        print("\n" + "=" * 50)
        print("群智感知真相发现算法结果摘要")
        print("=" * 50)

        print(f"收敛迭代次数：{results['convergence_iterations']}")
        print(f"算法参数：τ={self.tau}, t₀={self.t0}, ε={self.epsilon}")

        # 可靠度统计
        reliabilities = list(results['user_reliabilities'].values())
        reliable_count = sum(1 for r in reliabilities if r > 0)
        print(f"\n可靠用户统计：{reliable_count}/{len(reliabilities)} 个用户可靠")
        print(f"平均可靠度：{np.mean(reliabilities):.4f}")

        # 声誉变化统计
        initial_rep = list(results['initial_reputations'].values())
        updated_rep = list(results['updated_reputations'].values())
        rep_change = np.array(updated_rep) - np.array(initial_rep)
        print(f"\n声誉更新统计：")
        print(f"平均声誉变化：{np.mean(rep_change):.4f}")
        print(f"声誉提升用户：{sum(1 for c in rep_change if c > 0.01)} 个")
        print(f"声誉下降用户：{sum(1 for c in rep_change if c < -0.01)} 个")

        # 准确性统计
        if results['task_accuracies']:
            maes = [acc['mae'] for acc in results['task_accuracies'].values()]
            rmses = [acc['rmse'] for acc in results['task_accuracies'].values()]
            print(f"\n准确性统计：")
            print(f"平均MAE：{np.mean(maes):.4f}")
            print(f"平均RMSE：{np.mean(rmses):.4f}")
            print(f"最佳MAE：{np.min(maes):.4f}")
            print(f"最差MAE：{np.max(maes):.4f}")


# 使用示例
if __name__ == "__main__":
    # 初始化算法
    algorithm = CrowdSensingTruthDiscovery(
        tau=1.0,
        t0=0.3,  # 降低门槛让更多用户参与
        epsilon=1e-6,
        xi=1e-6,
        max_iterations=100,
        distance_metric='absolute'
    )

    # 注意：你需要将file_path替换为你的实际文件路径
    file_path = "/AlgorithmModule/Scene_1_Number_of_Workers_27.json"

    try:
        # 运行算法
        results = algorithm.fit(file_path)

        # 打印摘要
        algorithm.print_summary(results)

        # 可以进一步分析结果
        print("\n前5个任务的真相发现结果：")
        for i in range(min(5, len(results['final_truth_values']))):
            print(f"任务{i + 1}：{results['final_truth_values'][i]}")

        print("\n声誉变化最大的5个用户：")
        initial_rep = results['initial_reputations']
        updated_rep = results['updated_reputations']
        rep_changes = {user_id: updated_rep[user_id] - initial_rep[user_id]
                       for user_id in initial_rep.keys()}
        top_changes = sorted(rep_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        for user_id, change in top_changes:
            direction = "↑" if change > 0 else "↓"
            print(
                f"用户{user_id}：{initial_rep[user_id]:.4f} → {updated_rep[user_id]:.4f} ({direction}{abs(change):.4f})")

    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
        print("请确保文件路径正确，或者将数据文件放在当前目录下")
    except Exception as e:
        print(f"运行出错：{e}")
