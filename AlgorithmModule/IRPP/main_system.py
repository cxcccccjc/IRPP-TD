# main_system.py
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

# 导入必要的模块
from worker_scoring_system import create_worker_scorer, print_worker_results
from abod_worker_assessment import create_assessor, print_performance_details, DataQuality
from truth_discovery import discover_truth, get_truth_and_weights


@dataclass
class Worker:
    """工人类"""
    id: int
    alpha_h: int  # 高质量数据提交次数
    alpha_u: int  # 不确定质量数据提交次数
    alpha_l: int  # 低质量数据提交次数

    def __init__(self, worker_id: int):
        self.id = worker_id
        self.alpha_h = 0
        self.alpha_u = 0
        self.alpha_l = 0

    def get_triplet(self) -> List[int]:
        """获取三元组评分"""
        return [self.alpha_h, self.alpha_u, self.alpha_l]

    def update_quality(self, quality: str):
        """根据数据质量评估结果更新三元组"""
        if quality == DataQuality.HIGH.value:
            self.alpha_h += 1
        elif quality == DataQuality.UNCERTAIN.value:
            self.alpha_u += 1
        elif quality == DataQuality.LOW.value:
            self.alpha_l += 1

    def __str__(self):
        return f"Worker({self.id}): [{self.alpha_h}, {self.alpha_u}, {self.alpha_l}]"


@dataclass
class TaskResult:
    """单次任务结果"""
    task_id: int
    participating_workers: List[int]
    high_quality_workers: List[int]
    data_quality_results: Dict[int, str]
    aggregated_truth: Optional[List[float]]
    truth_discovery_time: float
    true_data: Optional[List[float]]
    error: Optional[float]
    iterations: Optional[int] = None


@dataclass
class ErrorStatistics:
    """误差统计类"""
    mae: float  # 平均绝对误差
    mse: float  # 均方误差
    rmse: float  # 均方根误差

    def __str__(self):
        return f"MAE: {self.mae:.4f}, MSE: {self.mse:.4f}, RMSE: {self.rmse:.4f}"


class MCSSystem:
    """MCS主系统"""

    def __init__(self, data_file_path: str, reputation_threshold: float = 0.3):
        """
        初始化MCS系统

        Parameters:
        -----------
        data_file_path : str
            数据文件路径
        reputation_threshold : float
            高质量工人评分阈值
        """
        self.data_file_path = data_file_path
        self.reputation_threshold = reputation_threshold
        self.workers: Dict[int, Worker] = {}
        self.task_data = None
        self.task_results: List[TaskResult] = []

        # 新增：每10次任务的时间记录
        self.truth_discovery_time_records: Dict[int, float] = {}  # {任务批次: 累积时间}

        # 初始化模块
        self.scorer = create_worker_scorer()
        self.assessor = create_assessor()

        print(f"MCS系统初始化完成")
        print(f"数据文件路径: {data_file_path}")
        print(f"高质量工人阈值: {reputation_threshold}")

    def load_task_data(self) -> bool:
        """加载任务数据"""
        try:
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.task_data = data['task_worker_data']
                print(f"成功加载 {len(self.task_data)} 个任务的数据")
                return True
        except FileNotFoundError:
            print(f"错误: 文件不存在 {self.data_file_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"错误: JSON解析失败 {e}")
            return False
        except Exception as e:
            print(f"错误: 加载数据失败 {e}")
            return False

    def calculate_error_statistics(self, predicted: List[float], true_data: List[float]) -> ErrorStatistics:
        """计算MAE、MSE、RMSE误差"""
        predicted_array = np.array(predicted)
        true_array = np.array(true_data)

        # 计算各种误差
        mae = np.mean(np.abs(predicted_array - true_array))
        mse = np.mean((predicted_array - true_array) ** 2)
        rmse = np.sqrt(mse)

        return ErrorStatistics(mae=mae, mse=mse, rmse=rmse)

    def run_all_tasks(self):
        """执行所有100次任务"""
        if not self.task_data:
            print("请先加载任务数据")
            return

        print("\n" + "=" * 80)
        print("开始执行MCS全流程 - 100次任务")
        print("=" * 80)

        # 按任务ID顺序处理
        task_ids = sorted(self.task_data.keys(), key=int)

        # 用于记录每10次任务的累积时间
        current_batch_time = 0.0

        for i, task_id in enumerate(task_ids, 1):
            print(f"\n{'=' * 20} 任务 {task_id} ({i}/100) {'=' * 20}")
            task_result = self.process_single_task(task_id)
            self.task_results.append(task_result)

            # 累积当前批次的真相发现时间
            current_batch_time += task_result.truth_discovery_time

            # 每10次任务记录一次累积时间
            if i % 10 == 0:
                batch_number = i // 10
                self.truth_discovery_time_records[batch_number] = current_batch_time
                print(f"已完成 {i}/100 任务，当前10次批次累积真相发现时间: {current_batch_time:.6f}秒")
                current_batch_time = 0.0  # 重置计数器

        print("\n" + "=" * 80)
        print("所有任务执行完成!")
        self.print_final_statistics()

    def process_single_task(self, task_id: str) -> TaskResult:
        """处理单个任务"""
        task_info = self.task_data[task_id]
        worker_submissions = task_info['worker_submissions']
        true_data = task_info.get('task_true_data', None)

        # 收集本次任务的工人数据
        current_workers = {}  # {worker_id: submitted_data}
        participating_workers = []

        for submission in worker_submissions:
            worker_id = submission['worker_id']
            submitted_data = submission['submitted_data']
            current_workers[worker_id] = submitted_data
            participating_workers.append(worker_id)

            # 确保工人存在于系统中
            if worker_id not in self.workers:
                self.workers[worker_id] = Worker(worker_id)

        print(f"参与工人: {len(participating_workers)}个")

        # 步骤1: 工人评分
        print("步骤1: 工人信誉评分...")
        high_quality_workers = self._score_and_predict_workers(participating_workers)

        # 步骤2: 数据质量评估
        print("步骤2: 数据质量评估...")
        quality_results = self._assess_data_quality(current_workers)

        # 步骤3: 更新工人历史
        print("步骤3: 更新工人历史...")
        self._update_worker_histories(quality_results)

        # 步骤4: 收集数据并进行真相发现
        print("步骤4: 真相发现...")
        aggregated_truth, td_time, iterations= self._truth_discovery(current_workers, high_quality_workers, quality_results)

        # 计算误差
        error = None
        if aggregated_truth and true_data:
            error = np.linalg.norm(np.array(aggregated_truth) - np.array(true_data))
            print(f"与真实数据误差: {error:.4f}")

        return TaskResult(
            task_id=int(task_id),
            participating_workers=participating_workers,
            high_quality_workers=high_quality_workers,
            data_quality_results=quality_results,
            aggregated_truth=aggregated_truth,
            truth_discovery_time=td_time,
            true_data=true_data,
            error=error,
            iterations=iterations
        )

    def _score_and_predict_workers(self, worker_ids: List[int]) -> List[int]:
        """工人评分和预测高质量工人"""
        # 准备工人历史数据
        workers_data = {}
        for worker_id in worker_ids:
            workers_data[worker_id] = self.workers[worker_id].get_triplet()

        # 使用评分系统
        results = self.scorer.score_multiple_workers(workers_data)

        # 预测高质量工人
        high_quality_workers = []
        print("工人评分结果:")
        for result in results:
            reputation_score = result.reputation_score
            print(f"  工人{result.worker_id}: φ_h={result.phi_h:.3f}, φ_u={result.phi_u:.3f}, "
                  f"φ_l={result.phi_l:.3f}, 不确定度={result.uncertainty:.4f}, "
                  f"信誉评分={reputation_score:.4f}")

            if reputation_score > self.reputation_threshold:
                high_quality_workers.append(result.worker_id)

        print(f"预测高质量工人 (阈值>{self.reputation_threshold}): {high_quality_workers}")
        return high_quality_workers

    def _assess_data_quality(self, worker_data: Dict[int, List[float]]) -> Dict[int, str]:
        """评估工人数据质量"""
        # 使用ABOD评估器的assess_multiple_workers方法
        assessment_result = self.assessor.assess_multiple_workers(worker_data, include_classification=True)

        print(f"数据质量评估耗时: {assessment_result.execution_time:.4f}秒")

        # 提取质量结果 - 根据actual模块返回的results格式
        quality_results = {}
        print("数据质量评估结果:")
        for result in assessment_result.results:
            worker_id = result.worker_id
            quality = result.data_quality.value if result.data_quality else DataQuality.UNCERTAIN.value
            quality_results[worker_id] = quality
            print(f"  工人{worker_id}: {quality} (方差: {result.angle_variance:.2e})")

        # 打印详细性能信息
        print_performance_details(assessment_result)

        return quality_results

    def _update_worker_histories(self, quality_results: Dict[int, str]):
        """更新工人历史三元组"""
        print("更新工人历史:")
        for worker_id, quality in quality_results.items():
            old_triplet = self.workers[worker_id].get_triplet().copy()
            self.workers[worker_id].update_quality(quality)
            new_triplet = self.workers[worker_id].get_triplet()
            print(f"  工人{worker_id}: {old_triplet} -> {new_triplet} ({quality})")

    def _truth_discovery(self, worker_data: Dict[int, List[float]],
                         high_quality_workers: List[int],
                         quality_results: Dict[int, str]) -> Tuple[Optional[List[float]], float]:
        """真相发现过程"""
        # 收集用于真相发现的数据
        td_data = {}

        for worker_id, data in worker_data.items():
            quality = quality_results[worker_id]

            # 收集条件：
            # 1. 所有高质量数据
            # 2. 被预测为高质量工人且数据为高质量或不确定
            if quality == DataQuality.HIGH.value:
                td_data[worker_id] = data
                print(f"  收集工人{worker_id}数据: 高质量")
            elif worker_id in high_quality_workers and quality == DataQuality.UNCERTAIN.value:
                td_data[worker_id] = data
                print(f"  收集工人{worker_id}数据: 潜在高质量工人+不确定数据")

        if not td_data:
            print("警告: 没有数据用于真相发现")
            return None, 0.0

        print(f"用于真相发现的工人数: {len(td_data)}, ID: {list(td_data.keys())}")

        # 执行真相发现
        start_time = time.perf_counter()

        try:
            result = discover_truth(td_data)
            td_time = time.perf_counter() - start_time

            print(f"真相发现耗时: {td_time:.6f}秒")
            print(f"聚合真值: {[f'{x:.4f}' for x in result.aggregated_truth[:3]]}...")
            print(f"是否收敛: {result.converged}, 迭代次数: {result.iterations}")

            return result.aggregated_truth, td_time, result.iterations

        except Exception as e:
            td_time = time.perf_counter() - start_time
            print(f"真相发现失败: {e}")
            return None, td_time, None

    def calculate_comprehensive_error_statistics(self) -> Optional[ErrorStatistics]:
        """计算所有任务的综合误差统计"""
        valid_results = [
            result for result in self.task_results
            if result.aggregated_truth is not None and result.true_data is not None
        ]

        if not valid_results:
            return None

        all_predicted = []
        all_true = []

        for result in valid_results:
            all_predicted.extend(result.aggregated_truth)
            all_true.extend(result.true_data)

        return self.calculate_error_statistics(all_predicted, all_true)

    def print_final_statistics(self):
        """打印最终统计信息"""
        print("\n" + "=" * 60)
        print("最终统计信息")
        print("=" * 60)

        # 工人统计
        print(f"总工人数: {len(self.workers)}")

        total_high = sum(worker.alpha_h for worker in self.workers.values())
        total_uncertain = sum(worker.alpha_u for worker in self.workers.values())
        total_low = sum(worker.alpha_l for worker in self.workers.values())
        total_submissions = total_high + total_uncertain + total_low

        if total_submissions > 0:
            print(f"总提交次数: {total_submissions}")
            print(f"  高质量: {total_high} ({total_high / total_submissions * 100:.1f}%)")
            print(f"  不确定: {total_uncertain} ({total_uncertain / total_submissions * 100:.1f}%)")
            print(f"  低质量: {total_low} ({total_low / total_submissions * 100:.1f}%)")
        else:
            print("没有数据提交记录")

        # 真相发现统计
        successful_td = sum(1 for result in self.task_results if result.aggregated_truth is not None)
        total_td_time = sum(result.truth_discovery_time for result in self.task_results)
        avg_td_time = total_td_time / len(self.task_results) if self.task_results else 0

        print(f"\n真相发现统计:")
        print(f"成功次数: {successful_td} / {len(self.task_results)}")
        print(f"总耗时: {total_td_time:.4f}秒")
        print(f"平均耗时: {avg_td_time:.6f}秒")

        # 新增：每10次任务的真相发现时间记录
        print(f"\n每10次任务真相发现时间记录:")
        for batch, batch_time in self.truth_discovery_time_records.items():
            start_task = (batch - 1) * 10 + 1
            end_task = batch * 10
            print(f"  任务 {start_task}-{end_task}: {batch_time:.6f}秒")

        # 综合误差统计 (MAE, MSE, RMSE)
        comprehensive_errors = self.calculate_comprehensive_error_statistics()
        if comprehensive_errors:
            print(f"\n综合误差统计:")
            print(f"  {comprehensive_errors}")

            # 新增：前50个任务的误差统计
            first_50_results = [result for result in self.task_results[:50]
                                if result.aggregated_truth is not None and result.true_data is not None]
            if first_50_results:
                first_50_predicted = []
                first_50_true = []
                for result in first_50_results:
                    first_50_predicted.extend(result.aggregated_truth)
                    first_50_true.extend(result.true_data)

                first_50_errors = self.calculate_error_statistics(first_50_predicted, first_50_true)
                print(f"\n前50个任务误差统计:")
                print(f"  {first_50_errors}")
                print(f"  有效任务数: {len(first_50_results)}/50")
            else:
                print(f"\n前50个任务误差统计: 无有效数据")

            # 新增：后50个任务的误差统计
            last_50_results = [result for result in self.task_results[50:]
                               if result.aggregated_truth is not None and result.true_data is not None]
            if last_50_results:
                last_50_predicted = []
                last_50_true = []
                for result in last_50_results:
                    last_50_predicted.extend(result.aggregated_truth)
                    last_50_true.extend(result.true_data)

                last_50_errors = self.calculate_error_statistics(last_50_predicted, last_50_true)
                print(f"\n后50个任务误差统计:")
                print(f"  {last_50_errors}")
                print(f"  有效任务数: {len(last_50_results)}/50")
            else:
                print(f"\n后50个任务误差统计: 无有效数据")

        else:
            print(f"\n综合误差统计: 无有效数据")

        # 原有误差统计 (保持兼容)
        errors = [result.error for result in self.task_results if result.error is not None]
        if errors:
            print(f"\n欧几里得距离误差统计:")
            print(f"平均误差: {np.mean(errors):.4f}")
            print(f"误差标准差: {np.std(errors):.4f}")
            print(f"最小误差: {np.min(errors):.4f}")
            print(f"最大误差: {np.max(errors):.4f}")

        iterations_data = [result.iterations for result in self.task_results
                           if hasattr(result, 'iterations') and result.iterations is not None]

        if iterations_data:
            print(f"\n真相发现迭代次数统计:")
            print(f"总迭代次数: {sum(iterations_data)}")
            print(f"平均迭代次数: {np.mean(iterations_data):.2f}")
            print(f"迭代次数标准差: {np.std(iterations_data):.2f}")
            print(f"最大迭代次数: {max(iterations_data)}")
            print(f"最小迭代次数: {min(iterations_data)}")
            print(f"有迭代数据的任务数: {len(iterations_data)}/100")
        else:
            print(f"\n真相发现迭代次数统计: 无有效迭代数据")

        # 显示部分工人详情
        print(f"\n工人详情 (按总任务数排序，前10个):")
        sorted_workers = sorted(self.workers.items(), key=lambda x: sum(x[1].get_triplet()), reverse=True)
        for worker_id, worker in sorted_workers[:10]:
            total_tasks = sum(worker.get_triplet())
            if total_tasks > 0:
                print(f"  {worker} - 总任务: {total_tasks}")


def main():
    """主程序入口"""
    # 配置参数
    data_file_path = r"D:\py\IRPP\AlgorithmModule\Scene_3_Number_of_Workers_27.json"
    reputation_threshold = 0.3  # 可调整

    try:
        # 创建MCS系统
        mcs = MCSSystem(data_file_path, reputation_threshold)

        # 加载数据并运行
        if mcs.load_task_data():
            mcs.run_all_tasks()
        else:
            print("系统启动失败")

    except ImportError as e:
        print(f"模块导入失败: {e}")
        print("请确保以下文件存在并且可以正确导入:")
        print("- worker_comprehensive_scoring.py")
        print("- abod_worker_assessment.py")
        print("- truth_discovery.py")
    except Exception as e:
        print(f"系统运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
