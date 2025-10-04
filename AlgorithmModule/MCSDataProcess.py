import json
from datetime import datetime


def reorganize_task_worker_data(input_file_path, output_file_path):
    """
    整理任务-工人-数据对应关系
    """

    # 读取原始数据
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"文件未找到: {input_file_path}")
        return
    except json.JSONDecodeError:
        print("JSON文件格式错误")
        return

    # 创建简化的数据结构
    reorganized_data = {
        "reorganization_info": {
            "original_timestamp": original_data.get("execution_timestamp", ""),
            "reorganization_timestamp": datetime.now().isoformat(),
            "total_tasks": len(original_data.get("results", {})),
            "description": "每个任务对应的工人ID和提交数据"
        },
        "task_worker_data": {}
    }

    # 处理每个任务的数据
    results = original_data.get("results", {})

    for task_key, task_data in results.items():
        task_id = task_data.get("task_id", int(task_key))

        # 创建任务数据结构
        task_info = {
            "task_id": task_id,
            "task_true_data": task_data.get("task_true_data", []),
            "worker_submissions": []
        }

        # 提取每个工人的ID和对应数据
        participating_workers = task_data.get("participating_workers", [])

        for worker in participating_workers:
            worker_submission = {
                "worker_id": worker.get("worker_id"),
                "submitted_data": worker.get("submitted_data", [])
            }
            task_info["worker_submissions"].append(worker_submission)

        # 按工人ID排序，便于查看
        task_info["worker_submissions"].sort(key=lambda x: x["worker_id"])

        reorganized_data["task_worker_data"][str(task_id)] = task_info

    # 保存整理后的数据
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(reorganized_data, f, ensure_ascii=False, indent=2)
        print(f"任务-工人-数据对应关系整理完成，保存到: {output_file_path}")

        # 打印简单统计
        total_tasks = len(reorganized_data["task_worker_data"])
        total_submissions = sum(
            len(task["worker_submissions"]) for task in reorganized_data["task_worker_data"].values())
        print(f"共整理了 {total_tasks} 个任务，{total_submissions} 个工人提交记录")

    except Exception as e:
        print(f"保存文件时出错: {e}")


def create_simple_csv_export(json_file_path, csv_file_path):
    """
    可选：导出为CSV格式，便于在Excel中查看
    """
    try:
        import pandas as pd

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 准备CSV数据
        csv_data = []

        for task_id, task_info in data["task_worker_data"].items():
            task_true_data = task_info["task_true_data"]

            for submission in task_info["worker_submissions"]:
                worker_id = submission["worker_id"]
                submitted_data = submission["submitted_data"]

                # 创建一行数据
                row = {
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "task_true_data": str(task_true_data),
                    "submitted_data": str(submitted_data)
                }

                # 如果数据是数值列表，可以展开为单独的列
                if isinstance(submitted_data, list) and len(submitted_data) > 0:
                    for i, value in enumerate(submitted_data):
                        row[f"data_point_{i + 1}"] = value

                csv_data.append(row)

        # 保存为CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        print(f"CSV格式数据保存到: {csv_file_path}")

    except ImportError:
        print("未安装pandas库，跳过CSV导出")
    except Exception as e:
        print(f"导出CSV时出错: {e}")


# 主程序执行
if __name__ == "__main__":
    input_file = r"D:\py\IRPP\AlgorithmModule\crowdsensing_results_20250922_135114.json"
    output_file = r"D:\py\IRPP\AlgorithmModule\task_worker_data_simplified.json"
    csv_file = r"D:\py\IRPP\AlgorithmModule\task_worker_data.csv"

    print("开始整理任务-工人-数据对应关系...")
    reorganize_task_worker_data(input_file, output_file)

    print("\n导出CSV格式（可选）...")
    create_simple_csv_export(output_file, csv_file)

    print("\n整理完成！")
