import configparser
import pickle
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import copy
import math
import json
from datetime import datetime

# 创建ConfigParser对象
config = configparser.ConfigParser()

# 读取配置文件
config.read('config.ini')

# 获取配置参数
Maximum_ceiling_for_system_workers = config['Main'].getint('Maximum_ceiling_for_system_workers')
Number_of_workers = config['Main'].getint('Number_of_workers')
Number_of_tasks = config['Main'].getint('Number_of_tasks')
Grid_size = config['Main'].getint('Grid_size')
Lower_horizontal_coordinate = config['Main'].getint('Lower_horizontal_coordinate')
Upper_horizontal_coordinate = config['Main'].getint('Upper_horizontal_coordinate')
Lower_vertical_coordinate = config['Main'].getint('Lower_vertical_coordinate')
Upper_vertical_coordinate = config['Main'].getint('Upper_vertical_coordinate')
DataSelect = config['Main'].getint('DataSelect')


class Worker:
    def __init__(self, worker_id: int):
        self.id = worker_id
        self.trajectory = None
        self.submitted_data = None
        self.error_probability = self._calculate_error_probability()
        self.submission_count = 0
        self.task_history = []
        self.current_position = None  # 工人当前位置
        self.is_recruited = False  # 是否被招募参与当前任务

    def _calculate_error_probability(self) -> float:
        """根据工人ID计算犯错概率"""
        if 51 <= self.id <= 60:
            return 0.1  # 错误率
        elif 61 <= self.id <= 70:
            return 0.3  # 错误率
        elif 71 <= self.id <= 80:
            return 0.5  # 错误率
        elif 81 <= self.id <= 90:
            return 0.7  # 错误率
        elif 91 <= self.id <= 100:
            return 0.9  # 错误率
        else:
            return 0.0  # 其他工人不犯错

    def get_distance_to_task(self, task) -> float:
        """计算工人当前位置到任务中心的距离"""
        if not self.current_position:
            return float('inf')

        dx = self.current_position[0] - task.center_x
        dy = self.current_position[1] - task.center_y
        return math.sqrt(dx * dx + dy * dy)

    def update_position(self, trajectory_data: List):
        """更新工人当前位置（从轨迹中随机选择一个位置）"""
        if trajectory_data:
            trajectory = random.choice(trajectory_data)
            if len(trajectory) > 1:
                # 从轨迹点中随机选择一个作为当前位置
                points = trajectory[1:]
                if points:
                    random_point = random.choice(points)
                    try:
                        self.current_position = (float(random_point[0]), float(random_point[1]))
                    except (ValueError, TypeError, IndexError):
                        self.current_position = None

    def execute_task(self, task, trajectory_data: List, task_true_data: List[float],
                     dataset_bounds: Dict, force_recruit: bool = False) -> bool:
        """
        执行任务，基于任务的真实数据生成提交数据

        Args:
            task: 任务对象
            trajectory_data: 轨迹数据
            task_true_data: 任务的真实数据（固定的一组真实值）
            dataset_bounds: 数据集边界
            force_recruit: 是否强制招募
        """
        # 如果是强制招募，直接执行
        if force_recruit:
            self.is_recruited = True
            # 为被招募的工人分配轨迹
            if not self.trajectory:
                self.trajectory = random.choice(trajectory_data)
            # 基于任务真实数据生成提交数据
            self.submitted_data = self._generate_submitted_data_from_truth(task_true_data)
            self.submission_count += 1
            self.task_history.append(task.id)
            return True

        # 正常执行流程
        self.trajectory = random.choice(trajectory_data)
        self.is_recruited = False

        # 检查轨迹是否在数据集边界内且在任务范围内
        if (self._is_trajectory_in_dataset_bounds(dataset_bounds) and
                self._is_trajectory_in_task_range(task)):
            # 基于任务真实数据生成提交数据
            self.submitted_data = self._generate_submitted_data_from_truth(task_true_data)
            self.submission_count += 1
            self.task_history.append(task.id)
            return True
        return False

    def _generate_submitted_data_from_truth(self, true_data: List[float]) -> List[float]:
        """
        基于任务的真实数据生成工人的提交数据
        每个数据点都可能有误差（根据工人的error_probability）

        Args:
            true_data: 任务的真实数据
        """
        submitted_data = []

        for true_value in true_data:
            if random.random() < self.error_probability:
                # 工人犯错：添加较大的误差
                error_min = abs(true_value) * 0.05
                error_max = abs(true_value) * 0.20

                if random.random() < 0.5:
                    error_value = true_value - random.uniform(error_min, error_max)
                else:
                    error_value = true_value + random.uniform(error_min, error_max)

                submitted_data.append(round(error_value, 2))
            else:
                # 工人没犯错：添加小的正态噪声
                std_dev = abs(true_value) * 0.01 / 3
                normal_value = np.random.normal(true_value, std_dev)

                min_val = true_value * 0.99
                max_val = true_value * 1.01
                normal_value = max(min_val, min(max_val, normal_value))

                submitted_data.append(round(normal_value, 2))

        return submitted_data

    def _safe_convert_to_float(self, value) -> float:
        """安全地将值转换为浮点数"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _is_trajectory_in_dataset_bounds(self, bounds: Dict) -> bool:
        """检查轨迹是否在数据集边界内"""
        if not self.trajectory:
            return False

        points = self.trajectory[1:]

        for point in points:
            try:
                if len(point) != 2:
                    continue

                x = self._safe_convert_to_float(point[0])
                y = self._safe_convert_to_float(point[1])

                if (bounds['Lower_horizontal_coordinate'] <= x <= bounds['Upper_horizontal_coordinate'] and
                        bounds['Lower_vertical_coordinate'] <= y <= bounds['Upper_vertical_coordinate']):
                    return True
            except (IndexError, TypeError):
                continue
        return False

    def _is_trajectory_in_task_range(self, task) -> bool:
        """检查轨迹是否在任务范围内"""
        if not self.trajectory:
            return False

        points = self.trajectory[1:]

        for point in points:
            try:
                if len(point) != 2:
                    continue

                x = self._safe_convert_to_float(point[0])
                y = self._safe_convert_to_float(point[1])

                if (task.x_min <= x <= task.x_max and
                        task.y_min <= y <= task.y_max):
                    return True
            except (IndexError, TypeError):
                continue
        return False


class Task:
    def __init__(self, task_id: int, center_x: float, center_y: float, true_data: List[float], size: float = 1000):
        self.id = task_id
        self.center_x = center_x
        self.center_y = center_y
        self.size = size
        self.x_min = center_x - size / 2
        self.x_max = center_x + size / 2
        self.y_min = center_y - size / 2
        self.y_max = center_y + size / 2
        self.true_data = true_data  # 任务的真实数据（固定的一组真实值）
        self.participating_workers = []
        self.submission_data_list = []
        self.submission_count = 0
        self.recruited_workers = []  # 被招募的工人列表

    def add_worker(self, worker: Worker):
        """添加参与此任务的工人"""
        self.participating_workers.append(worker)
        self.submission_data_list.append(worker.submitted_data)
        self.submission_count += 1

        if worker.is_recruited:
            self.recruited_workers.append(worker.id)


class CrowdSensingSystem:
    def __init__(self, trajectory_file_path: str, choice: int = 1, task_range_threshold: float = 1200):
        self.workers = [Worker(i) for i in range(1, 101)]
        self.tasks = []
        self.dataset_choice = choice
        self.min_workers_per_task = 10  # 每个任务最少工人数量
        self.task_range_threshold = task_range_threshold  # 任务范围大小阈值参数

        # 数据集文件路径
        self.dataset_paths = {
            1: r"D:\py\IRPP\DataProcess\ProcessData\Climate_processed.csv",
            2: r"D:\py\IRPP\DataProcess\ProcessData\Traffic_processed.csv",
            3: r"D:\py\IRPP\DataProcess\ProcessData\Water_processed.csv"
        }

        # 数据集信息
        self.dataset_info = {
            1: {
                'name': '天气数据集',
                'columns': ['p (mbar)', 'Tpot (K)', 'rh (%)', 'rho (g/m**3)', 'wd (deg)']
            },
            2: {
                'name': '交通流量数据集',
                'columns': ['中型车流量', '大车流量', '微型车流量', '长车流量', '车流量', '轻型车流量']
            },
            3: {
                'name': '水质检测数据集',
                'columns': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
                            'Trihalomethanes', 'Turbidity']
            }
        }

        # 加载真实数据集
        self.real_datasets = self._load_real_datasets()

        # 数据集边界限制
        self.dataset_bounds = {
            'Lower_horizontal_coordinate': 0,
            'Upper_horizontal_coordinate': 5800,
            'Lower_vertical_coordinate': 0,
            'Upper_vertical_coordinate': 4300
        }

        # 加载轨迹数据
        self.trajectory_data = self._load_trajectory_data(trajectory_file_path)

        # 初始化所有工人的位置
        self._initialize_worker_positions()

    def _initialize_worker_positions(self):
        """初始化所有工人的当前位置"""
        for worker in self.workers:
            worker.update_position(self.trajectory_data)

    def _load_real_datasets(self) -> Dict[int, pd.DataFrame]:
        """加载真实数据集"""
        datasets = {}

        for dataset_id, file_path in self.dataset_paths.items():
            try:
                df = pd.read_csv(file_path)
                numeric_df = df.select_dtypes(include=[np.number])

                if not numeric_df.empty:
                    datasets[dataset_id] = numeric_df
                    print(f"成功加载数据集{dataset_id}: {self.dataset_info[dataset_id]['name']}")
                    print(f"  数据形状: {numeric_df.shape}")
                    print(f"  列名: {list(numeric_df.columns)}")
                    print(f"  数据样本: {numeric_df.iloc[0].values.tolist()}")
                else:
                    print(f"警告: 数据集{dataset_id}中没有数值列")
                    datasets[dataset_id] = pd.DataFrame()

            except FileNotFoundError:
                print(f"错误: 找不到文件 {file_path}")
                datasets[dataset_id] = pd.DataFrame()
            except Exception as e:
                print(f"加载数据集{dataset_id}时出错: {e}")
                datasets[dataset_id] = pd.DataFrame()

        return datasets

    def _get_task_true_data(self, dataset_choice: int, real_datasets: Dict) -> List[float]:
        """
        为任务获取一组真实数据（从数据集中随机抽取一个样本）
        每个任务只调用一次，确保任务的真实数据是固定的
        """
        if dataset_choice in real_datasets and not real_datasets[dataset_choice].empty:
            sample_row = real_datasets[dataset_choice].sample(n=1).iloc[0]
            return [float(val) for val in sample_row.values]
        else:
            print(f"警告: 数据集{dataset_choice}不存在或为空，使用默认数据")
            if dataset_choice == 1:
                return [996.52, 265.4, 93.3, 1307.75, 152.3]
            elif dataset_choice == 2:
                return [1556.0, 912.0, 10926.0, 2761.0, 17604.0, 1449.0]
            else:
                return [8.316765884214679, 214.3733940856225, 22018.41744077529, 8.05933237743854,
                        356.88613564305666, 363.2665161642437, 18.436524495493305, 100.34167436508008,
                        4.628770536837084]

    def _load_trajectory_data(self, file_path: str) -> List:
        """加载轨迹数据"""
        try:
            with open(file_path, 'rb') as f:
                trajectory_data = pickle.load(f)
            print(f"成功加载 {len(trajectory_data)} 条轨迹数据")

            cleaned_data = []
            valid_count = 0
            for trajectory in trajectory_data:
                try:
                    if len(trajectory) >= 2:
                        vehicle_id = trajectory[0]
                        points = []

                        for point in trajectory[1:]:
                            if isinstance(point, (tuple, list)) and len(point) == 2:
                                try:
                                    x = float(point[0])
                                    y = float(point[1])
                                    if (-10000 <= x <= 10000) and (-10000 <= y <= 10000):
                                        points.append((x, y))
                                except (ValueError, TypeError):
                                    continue

                        if len(points) >= 2:
                            cleaned_trajectory = [vehicle_id] + points
                            cleaned_data.append(cleaned_trajectory)
                            valid_count += 1

                except (IndexError, TypeError):
                    continue

            print(f"清理后保留 {valid_count} 条有效轨迹数据")

            if valid_count == 0:
                print("没有有效的轨迹数据，生成模拟数据")
                return self._generate_mock_trajectory_data()

            return cleaned_data

        except FileNotFoundError:
            print(f"警告: 轨迹文件 {file_path} 未找到，使用模拟数据")
            return self._generate_mock_trajectory_data()
        except Exception as e:
            print(f"加载轨迹文件时出错: {e}，使用模拟数据")
            return self._generate_mock_trajectory_data()

    def _generate_mock_trajectory_data(self) -> List:
        """生成模拟轨迹数据"""
        mock_data = []
        for i in range(200):
            vehicle_id = f'veh{i}'
            points = []
            start_x = random.uniform(
                self.dataset_bounds['Lower_horizontal_coordinate'],
                self.dataset_bounds['Upper_horizontal_coordinate']
            )
            start_y = random.uniform(
                self.dataset_bounds['Lower_vertical_coordinate'],
                self.dataset_bounds['Upper_vertical_coordinate']
            )

            for j in range(5):
                x = max(self.dataset_bounds['Lower_horizontal_coordinate'],
                        min(self.dataset_bounds['Upper_horizontal_coordinate'],
                            start_x + random.uniform(-600, 600)))
                y = max(self.dataset_bounds['Lower_vertical_coordinate'],
                        min(self.dataset_bounds['Upper_vertical_coordinate'],
                            start_y + random.uniform(-200, 200)))
                points.append((x, y))
            trajectory = [vehicle_id] + points
            mock_data.append(trajectory)
        print(f"生成了 {len(mock_data)} 条模拟轨迹数据")
        return mock_data

    def create_tasks(self, num_tasks: int = 100):
        """创建任务，每个任务都有固定的真实数据"""
        self.tasks = []
        for i in range(1, num_tasks + 1):
            center_x = random.uniform(
                self.dataset_bounds['Lower_horizontal_coordinate'] + self.task_range_threshold / 2,
                self.dataset_bounds['Upper_horizontal_coordinate'] - self.task_range_threshold / 2
            )
            center_y = random.uniform(
                self.dataset_bounds['Lower_vertical_coordinate'] + self.task_range_threshold / 2,
                self.dataset_bounds['Upper_vertical_coordinate'] - self.task_range_threshold / 2
            )

            # 为每个任务获取一组固定的真实数据
            task_true_data = self._get_task_true_data(self.dataset_choice, self.real_datasets)

            # 创建任务时传入真实数据
            task = Task(i, center_x, center_y, task_true_data, self.task_range_threshold)
            self.tasks.append(task)

        print(f"创建了 {num_tasks} 个任务，任务范围大小: {self.task_range_threshold}")
        print(f"使用数据集: {self.dataset_info[self.dataset_choice]['name']}")

        if self.dataset_choice in self.real_datasets and not self.real_datasets[self.dataset_choice].empty:
            actual_columns = list(self.real_datasets[self.dataset_choice].columns)
            print(f"实际数据列: {', '.join(actual_columns)}")
        else:
            print(f"预期数据列: {', '.join(self.dataset_info[self.dataset_choice]['columns'])}")

    def _recruit_additional_workers(self, task: Task, current_participants: List[Worker]) -> List[Worker]:
        """为任务招募额外的工人"""
        needed_workers = self.min_workers_per_task - len(current_participants)
        if needed_workers <= 0:
            return []

        # 获取当前未参与此任务的工人
        participating_worker_ids = {worker.id for worker in current_participants}
        available_workers = [worker for worker in self.workers
                             if worker.id not in participating_worker_ids and worker.current_position]

        if not available_workers:
            print(f"  警告: 没有足够的可用工人为任务{task.id}进行招募")
            return []

        # 计算每个可用工人到任务的距离
        worker_distances = []
        for worker in available_workers:
            distance = worker.get_distance_to_task(task)
            if distance != float('inf'):
                worker_distances.append((worker, distance))

        # 按距离排序
        worker_distances.sort(key=lambda x: x[1])

        # 招募最近的工人
        recruited_workers = []
        for worker, distance in worker_distances[:needed_workers]:
            # 强制执行任务，使用任务的固定真实数据
            success = worker.execute_task(task, self.trajectory_data,
                                          task.true_data, self.dataset_bounds, force_recruit=True)
            if success:
                recruited_workers.append(worker)
                print(f"  招募工人{worker.id}参与任务{task.id}，距离: {distance:.2f}")

        return recruited_workers

    def execute_all_tasks(self):
        """执行所有任务"""
        results = {}
        total_recruited = 0

        for task in self.tasks:
            print(f"\r执行任务 {task.id}/100...", end='', flush=True)

            # 第一阶段：正常任务执行
            task_results = {
                'task_id': task.id,
                'task_range': (task.x_min, task.y_min, task.x_max, task.y_max),
                'task_size': task.size,
                'task_true_data': task.true_data,  # 记录任务的真实数据
                'participating_workers': [],
                'all_submitted_data': [],
                'submission_count': 0,
                'workers_who_submitted': [],
                'recruited_workers': [],
                'recruitment_needed': False
            }

            # 正常执行流程
            current_participants = []
            for worker in self.workers:
                # 使用任务的固定真实数据
                success = worker.execute_task(task, self.trajectory_data,
                                              task.true_data, self.dataset_bounds)
                if success:
                    current_participants.append(worker)
                    task.add_worker(worker)

            # 检查是否需要招募
            if len(current_participants) < self.min_workers_per_task:
                task_results['recruitment_needed'] = True
                print(f"\n  任务{task.id}只有{len(current_participants)}个工人参与，需要招募额外工人")

                # 招募额外工人
                recruited_workers = self._recruit_additional_workers(task, current_participants)
                total_recruited += len(recruited_workers)

                # 将招募的工人添加到任务中
                for worker in recruited_workers:
                    current_participants.append(worker)
                    task.add_worker(worker)
                    task_results['recruited_workers'].append(worker.id)

            # 整理最终结果
            for worker in current_participants:
                task_results['participating_workers'].append({
                    'worker_id': worker.id,
                    'error_probability': worker.error_probability,
                    'trajectory_id': worker.trajectory[0] if worker.trajectory else None,
                    'submitted_data': worker.submitted_data,
                    'total_submissions': worker.submission_count,
                    'is_recruited': worker.is_recruited
                })
                task_results['all_submitted_data'].append(worker.submitted_data)
                task_results['workers_who_submitted'].append(worker.id)
                task_results['submission_count'] += 1

            results[task.id] = task_results

        print(f"\n总共招募了 {total_recruited} 人次工人")
        return results

    def save_results(self, results: Dict, filename: str = None):
        """保存results结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crowdsensing_results_{timestamp}.json"

        try:
            # 创建保存的数据结构
            save_data = {
                'system_config': {
                    'dataset_choice': self.dataset_choice,
                    'dataset_name': self.dataset_info[self.dataset_choice]['name'],
                    'task_range_threshold': self.task_range_threshold,
                    'min_workers_per_task': self.min_workers_per_task,
                    'total_workers': len(self.workers),
                    'total_tasks': len(self.tasks)
                },
                'execution_timestamp': datetime.now().isoformat(),
                'results': results
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            print(f"\n结果已保存到文件: {filename}")
            return filename

        except Exception as e:
            print(f"保存结果时出错: {e}")
            return None

    def analyze_results(self, results: Dict):
        """分析结果"""
        print("\n=== 系统执行结果分析 ===")
        print(f"使用数据集: {self.dataset_info[self.dataset_choice]['name']}")
        print(f"任务范围大小阈值: {self.task_range_threshold}")
        print(f"每任务最少工人要求: {self.min_workers_per_task}人")

        if self.dataset_choice in self.real_datasets and not self.real_datasets[self.dataset_choice].empty:
            df = self.real_datasets[self.dataset_choice]
            print(f"数据集形状: {df.shape}")
            print(f"数据列: {', '.join(df.columns)}")

        total_submissions = sum(result['submission_count'] for result in results.values())
        total_tasks_with_submissions = sum(1 for result in results.values()
                                           if result['submission_count'] > 0)
        tasks_needed_recruitment = sum(1 for result in results.values()
                                       if result['recruitment_needed'])
        total_recruited = sum(len(result['recruited_workers']) for result in results.values())

        print(f"总提交次数: {total_submissions}")
        print(f"有提交数据的任务数: {total_tasks_with_submissions}/{len(results)}")
        print(f"需要招募的任务数: {tasks_needed_recruitment}")
        print(f"总招募人次: {total_recruited}")

        # 检查每个任务的参与人数
        tasks_below_min = 0
        for result in results.values():
            if result['submission_count'] < self.min_workers_per_task:
                tasks_below_min += 1

        print(f"仍低于最低人数要求的任务: {tasks_below_min}")

        # 计算数据准确性
        print("\n=== 数据质量分析 ===")
        groups = {
            "1-50 (无错误)": range(1, 51),
            "51-60 (10%错误)": range(51, 61),
            "61-70 (20%错误)": range(61, 71),
            "71-80 (30%错误)": range(71, 81),
            "81-90 (40%错误)": range(81, 91),
            "91-100 (50%错误)": range(91, 101)
        }

        for group_name, worker_range in groups.items():
            group_data = []
            group_submissions = 0
            recruited_submissions = 0

            for task_result in results.values():
                for worker_info in task_result['participating_workers']:
                    worker_id = worker_info['worker_id']
                    if worker_id in worker_range:
                        submitted = worker_info['submitted_data']
                        group_data.append(submitted)
                        group_submissions += 1
                        if worker_info.get('is_recruited', False):
                            recruited_submissions += 1

            if group_data:
                print(f"{group_name}: {group_submissions}次提交 (其中{recruited_submissions}次为招募)")
                if group_data:
                    sample_data = group_data[0]
                    print(f"  数据样本: {sample_data}")
            else:
                print(f"{group_name}: 无提交记录")

        # 分析真值与提交数据的差异
        print("\n=== 数据准确性分析 ===")

        # 随机选择几个任务进行详细分析
        sample_tasks = random.sample(list(results.values()), min(3, len(results)))

        for task_result in sample_tasks:
            task_id = task_result['task_id']
            true_data = task_result['task_true_data']
            all_submitted = task_result['all_submitted_data']

            print(f"\n任务{task_id}详细分析:")
            print(f"  真实数据: {true_data}")
            print(f"  参与工人数: {len(all_submitted)}")

            if all_submitted:
                # 计算每维数据的平均值
                avg_submitted = []
                for dim in range(len(true_data)):
                    dim_values = [data[dim] for data in all_submitted if dim < len(data)]
                    if dim_values:
                        avg_submitted.append(round(np.mean(dim_values), 2))
                        print(f"  提交数据平均值: {avg_submitted}")

                        # 计算误差
                        if len(avg_submitted) == len(true_data):
                            errors = []
                            for i in range(len(true_data)):
                                error = abs(true_data[i] - avg_submitted[i])
                                relative_error = (error / abs(true_data[i])) * 100 if true_data[i] != 0 else 0
                                errors.append(round(relative_error, 2))
                            print(f"  相对误差(%): {errors}")
                            print(f"  平均相对误差: {round(np.mean(errors), 2)}%")


def main():
    """主函数"""
    print("=== MCS系统启动 ===")

    # 轨迹文件路径
    trajectory_file = r"D:\py\IRPP\AlgorithmModule\Trace.pkl"

    # 创建系统实例
    print(f"执行MCS过程 (数据集选择: {DataSelect})")
    system = CrowdSensingSystem(
        trajectory_file_path=trajectory_file,
        choice=DataSelect,
        task_range_threshold=1650
    )

    # 创建任务
    print(f"创建 {Number_of_tasks} 个任务...")
    system.create_tasks(Number_of_tasks)

    # 执行任务
    print("开始执行所有任务...")
    results = system.execute_all_tasks()

    # 分析结果
    system.analyze_results(results)

    # 保存结果
    saved_file = system.save_results(results)

    print(f"\n=== 系统执行完成 ===")
    if saved_file:
        print(f"结果文件: {saved_file}")

    return results


if __name__ == "__main__":
    results = main()