from abod_worker_assessment import WorkerQualityAssessor, create_assessor, print_performance_details


def example_with_timing():
    """带运行时间的使用示例"""

    assessor = create_assessor(u1_threshold=1e-7, u2_threshold=1e-9)

    print("=" * 80)
    print("ABOD工人数据质量评估系统 - 带运行时间统计")
    print("=" * 80)

    # ===================== 功能一：多工人评估 =====================
    worker_data = {
        9: [996.44, 280.13, 79.98, 1240.16, 205.58],
        16: [995.92, 279.68, 80.06, 1234.21, 206.16],
        19: [995.91, 279.66, 79.27, 1233.27, 204.96],
        21: [994.35, 279.04, 79.95, 1228.54, 206.65],
        23: [991.26, 279.48, 79.95, 1238.37, 205.97],
        65: [819.52, 278.32, 80.18, 1238.07, 206.25],
        74: [1092.37, 240.06, 86.14, 1238.42, 226.78],
        85: [1186.66, 313.1, 80.07, 1399.89, 231.91],
    }

    # 功能一：评估多个工人
    multi_assessment = assessor.assess_multiple_workers(
        worker_data=worker_data,
        include_classification=True
    )

    print("\n【功能一】多工人评估结果:")
    print(f"执行时间: {multi_assessment.execution_time:.4f} 秒")
    print("-" * 60)
    for result in multi_assessment.results:
        print(f"工人{result.worker_id:>3}: 角度方差={result.angle_variance:>12.2e}, "
              f"质量={result.data_quality.value}")

    # 详细性能信息
    print_performance_details(multi_assessment, "多工人评估详细性能")

    # ===================== 功能二：单工人评估 =====================
    target_worker_data = [819.52, 278.32, 80.18, 1238.07, 206.25]
    other_workers_data = [
        [996.44, 280.13, 79.98, 1240.16, 205.58],
        [995.92, 279.68, 80.06, 1234.21, 206.16],
        [995.91, 279.66, 79.27, 1233.27, 204.96],
        [994.35, 279.04, 79.95, 1228.54, 206.65],
        [991.26, 279.48, 79.95, 1238.37, 205.97]
    ]

    # 功能二：评估单个工人
    single_assessment = assessor.assess_single_worker(
        target_worker_data=target_worker_data,
        other_workers_data=other_workers_data,
        include_classification=True
    )

    print(f"\n【功能二】单工人评估结果:")
    print(f"执行时间: {single_assessment.execution_time:.4f} 秒")
    print("-" * 60)
    result = single_assessment.results
    print(f"目标工人: 角度方差={result.angle_variance:.2e}, 质量={result.data_quality.value}")

    # 详细性能信息
    print_performance_details(single_assessment, "单工人评估详细性能")

    # ===================== 性能对比 =====================
    print(f"\n【性能对比】")
    print("-" * 40)
    print(f"多工人评估 ({len(worker_data)}个工人): {multi_assessment.execution_time:.4f} 秒")
    print(f"单工人评估 (1个目标+{len(other_workers_data)}个对比): {single_assessment.execution_time:.4f} 秒")
    print(f"多工人评估平均每工人: {multi_assessment.execution_time / len(worker_data):.6f} 秒")


if __name__ == "__main__":
    example_with_timing()
