#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAG 模型超参数自动调优工具

使用贝叶斯优化 (Optuna TPE) 自动搜索最优超参数组合。

功能:
  - 贝叶斯优化搜索超参数空间
  - 支持并行执行（多终端共享 SQLite 数据库）
  - 自动解析训练结果
  - 完整日志记录（JSONL 格式）
  - 输出最优参数、调参历史和推荐运行命令

使用方式:
  # 单进程调参
  python tools/auto_tune.py \\
      --config-path config/business_day_ahead_config.json \\
      --data-name merged_result_ninghe.csv \\
      --n-trials 30

  # 并行调参（在多个终端同时运行相同命令）
  python tools/auto_tune.py \\
      --config-path config/business_day_ahead_config.json \\
      --data-name merged_result_ninghe.csv \\
      --n-trials 30 \\
      --storage sqlite:///tune.db \\
      --study-name dag_tune

  # 自定义搜索空间
  python tools/auto_tune.py \\
      --config-path config/business_day_ahead_config.json \\
      --data-name merged_result_ninghe.csv \\
      --search-space my_search_space.json

依赖:
  pip install optuna
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("错误: 需要安装 optuna")
    print("  pip install optuna")
    sys.exit(1)

import pandas as pd


# ============================================================
# 默认搜索空间
# ============================================================

DEFAULT_SEARCH_SPACE = {
    "lr": {
        "type": "logfloat",
        "low": 1e-4,
        "high": 3e-3,
        "comment": "学习率，对数尺度搜索",
    },
    "alpha": {
        "type": "float",
        "low": 0.1,
        "high": 0.7,
        "step": 0.05,
        "comment": "TC/CC 融合权重",
    },
    "d_model": {
        "type": "categorical",
        "choices": [128, 256, 512],
        "comment": "模型隐藏维度",
    },
    "d_ff": {
        "type": "categorical",
        "choices": [128, 256, 512, 1024],
        "comment": "前馈网络维度",
    },
    "dropout": {
        "type": "float",
        "low": 0.1,
        "high": 0.5,
        "step": 0.05,
        "comment": "Dropout 率",
    },
    "batch_size": {
        "type": "categorical",
        "choices": [32, 64, 128],
        "comment": "批大小",
    },
    "patch_len": {
        "type": "categorical",
        "choices": [48, 96],
        "comment": "Patch 长度",
    },
    "stride": {
        "type": "categorical",
        "choices": [24, 48],
        "comment": "Patch 步长",
    },
    "e_layers": {
        "type": "int",
        "low": 1,
        "high": 3,
        "comment": "编码器层数",
    },
    "patience": {
        "type": "int",
        "low": 3,
        "high": 10,
        "comment": "早停耐心值",
    },
}


# ============================================================
# 搜索空间工具
# ============================================================


def suggest_hyperparams(trial, search_space):
    """
    根据搜索空间定义为当前 trial 建议超参数组合。

    Parameters
    ----------
    trial : optuna.Trial
        当前 Optuna trial 对象
    search_space : dict
        搜索空间定义，每个参数包含 type/low/high/choices 等字段

    Returns
    -------
    dict
        建议的超参数字典
    """
    params = {}
    for name, config in search_space.items():
        ptype = config["type"]
        if ptype == "logfloat":
            params[name] = trial.suggest_float(
                name, config["low"], config["high"], log=True
            )
        elif ptype == "float":
            step = config.get("step", None)
            params[name] = trial.suggest_float(
                name, config["low"], config["high"], step=step
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(name, config["low"], config["high"])
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, config["choices"])
    return params


def load_search_space(path=None):
    """
    加载搜索空间配置。

    Parameters
    ----------
    path : str, optional
        搜索空间 JSON 文件路径。为 None 时使用默认搜索空间。

    Returns
    -------
    dict
        搜索空间定义
    """
    if path is None:
        return DEFAULT_SEARCH_SPACE.copy()

    with open(path, "r", encoding="utf-8") as f:
        space = json.load(f)

    # 移除 comment 字段（仅用于文档，不应传给 optuna）
    for config in space.values():
        config.pop("comment", None)

    return space


# ============================================================
# 试验执行与结果解析
# ============================================================


def run_trial_subprocess(params, base_args, trial_id, timeout_per_trial):
    """
    通过子进程运行单次训练试验。

    Parameters
    ----------
    params : dict
        本次试验的超参数
    base_args : dict
        基础运行参数（config_path, data_name 等）
    trial_id : int
        试验编号
    timeout_per_trial : int
        单次试验超时时间（秒）

    Returns
    -------
    dict
        试验结果，包含 status, metric, elapsed, error 等字段
    """
    save_path = os.path.join(base_args["output_dir"], f"trial_{trial_id}")
    os.makedirs(save_path, exist_ok=True)

    # 构建命令行
    cmd = [
        sys.executable,
        os.path.join(base_args["project_root"], "scripts", "run_benchmark.py"),
        "--config-path",
        base_args["config_path"],
        "--data-name-list",
        base_args["data_name"],
        "--model-name",
        base_args["model_name"],
        "--model-hyper-params",
        json.dumps(params),
        "--gpus",
        str(base_args["gpus"]),
        "--num-workers",
        str(base_args.get("num_workers", 1)),
        "--timeout",
        str(timeout_per_trial * 1000),
        "--save-path",
        save_path,
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_per_trial,
            cwd=base_args["project_root"],
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else "unknown"
            return {
                "status": "failed",
                "metric": 0.0,
                "elapsed": elapsed,
                "error": stderr_tail,
            }

        # 解析结果
        metric = parse_trial_result(save_path, base_args["metric_name"])
        if metric is None:
            return {
                "status": "no_result",
                "metric": 0.0,
                "elapsed": elapsed,
                "error": "无法从输出目录解析结果",
            }

        return {"status": "success", "metric": float(metric), "elapsed": elapsed}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "status": "timeout",
            "metric": 0.0,
            "elapsed": elapsed,
            "error": f"试验超时 ({timeout_per_trial}s)",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "metric": 0.0,
            "elapsed": elapsed,
            "error": str(e),
        }


def parse_trial_result(save_path, metric_name="march_accuracy_mean"):
    """
    从试验输出目录解析评估指标。

    依次尝试以下方式:
    1. 直接读取 CSV 文件
    2. 解压 tar.gz 后读取 CSV

    Parameters
    ----------
    save_path : str
        试验输出目录路径
    metric_name : str
        目标指标名称

    Returns
    -------
    float or None
        解析到的指标值，解析失败返回 None
    """
    save_dir = Path(save_path)

    # 方式1: 直接查找 CSV 文件
    csv_files = list(save_dir.glob("*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if metric_name in df.columns:
                val = df[metric_name].iloc[0]
                if pd.notna(val):
                    return float(val)
        except Exception:
            continue

    # 方式2: 查找 tar.gz 并解压读取
    tar_files = list(save_dir.glob("*.csv.tar.gz"))
    if tar_files:
        latest_tar = max(tar_files, key=lambda p: p.stat().st_mtime)
        try:
            with tarfile.open(latest_tar, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".csv"):
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        try:
                            df = pd.read_csv(f)
                            if metric_name in df.columns:
                                val = df[metric_name].iloc[0]
                                if pd.notna(val):
                                    return float(val)
                        except Exception:
                            continue
        except Exception as e:
            print(f"  [警告] 解压结果文件失败: {e}")

    # 方式3: 递归搜索子目录
    for csv_file in save_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if metric_name in df.columns:
                val = df[metric_name].iloc[0]
                if pd.notna(val):
                    return float(val)
        except Exception:
            continue

    return None


# ============================================================
# Optuna 目标函数
# ============================================================


def create_objective(search_space, base_args, timeout_per_trial, log_file):
    """
    创建 Optuna 目标函数。

    Parameters
    ----------
    search_space : dict
        搜索空间定义
    base_args : dict
        基础运行参数
    timeout_per_trial : int
        单次试验超时时间（秒）
    log_file : str
        日志文件路径

    Returns
    -------
    callable
        Optuna 目标函数
    """

    def objective(trial):
        # 建议超参数
        params = suggest_hyperparams(trial, search_space)

        # 添加固定参数
        for key, val in base_args.get("fixed_params", {}).items():
            params[key] = val

        trial_id = trial.number
        params_str = json.dumps(params, ensure_ascii=False, indent=2)
        print(f"\n{'=' * 60}")
        print(f"Trial {trial_id}")
        print(f"参数: {params_str}")
        print(f"{'=' * 60}")

        # 运行试验
        result = run_trial_subprocess(params, base_args, trial_id, timeout_per_trial)

        # 记录日志
        log_entry = {
            "trial": trial_id,
            "params": params,
            "status": result["status"],
            "metric": result["metric"],
            "elapsed": round(result["elapsed"], 1),
            "error": result.get("error", ""),
            "timestamp": datetime.now().isoformat(),
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # 打印结果摘要
        if result["status"] == "success":
            print(
                f"  >> Trial {trial_id}: "
                f"{base_args['metric_name']}={result['metric']:.4f}, "
                f"耗时={result['elapsed']:.1f}s"
            )
        else:
            print(
                f"  >> Trial {trial_id}: 失败 ({result['status']}), "
                f"错误={result.get('error', '')[:100]}"
            )

        return result["metric"]

    return objective


# ============================================================
# 结果输出
# ============================================================


def print_final_report(study, args, output_dir):
    """
    打印最终调参报告。

    Parameters
    ----------
    study : optuna.Study
        完成的 Optuna study
    args : argparse.Namespace
        命令行参数
    output_dir : str
        输出目录
    """
    print(f"\n{'=' * 60}")
    print("调参完成!")
    print(f"{'=' * 60}")

    # 基本信息
    total = len(study.trials)
    success = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"总试验次数: {total}")
    print(f"成功: {success}, 失败/超时: {total - success}")

    if study.best_trial is None:
        print("没有成功的试验，无法确定最优参数。")
        return

    # 最优结果
    print(f"\n最优试验: Trial {study.best_trial.number}")
    print(f"最优指标: {args.metric} = {study.best_value:.6f}")
    print(f"最优参数:")
    for key, val in sorted(study.best_params.items()):
        print(f"  {key}: {val}")

    # 保存最优参数
    best_file = os.path.join(output_dir, "best_params.json")
    best_result = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "metric": args.metric,
        "direction": args.direction,
        "n_trials": total,
        "n_success": success,
        "timestamp": datetime.now().isoformat(),
    }
    with open(best_file, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)
    print(f"\n最优参数已保存: {best_file}")

    # 保存试验历史
    history_file = os.path.join(output_dir, "trial_history.csv")
    df = study.trials_dataframe()
    df.to_csv(history_file, index=False, encoding="utf-8-sig")
    print(f"试验历史已保存: {history_file}")

    # 生成推荐运行命令
    best_params_str = json.dumps(study.best_params)
    print(f"\n推荐运行命令:")
    print(f"python ./scripts/run_benchmark.py \\")
    print(f"  --config-path {args.config_path} \\")
    print(f"  --data-name-list {args.data_name} \\")
    print(f"  --model-name {args.model_name} \\")
    print(f"  --model-hyper-params '{best_params_str}' \\")
    print(f"  --gpus {args.gpus} \\")
    print(f"  --save-path best_model_result")

    # Top-5 试验
    print(f"\nTop-5 试验:")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=(args.direction == "maximize"))
    for i, t in enumerate(sorted_trials[:5]):
        print(f"  #{i + 1}: Trial {t.number}, {args.metric}={t.value:.6f}")


# ============================================================
# 自定义搜索空间生成器
# ============================================================


def generate_search_space_file(output_path):
    """
    生成默认搜索空间的 JSON 配置文件模板。

    Parameters
    ----------
    output_path : str
        输出文件路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_SEARCH_SPACE, f, indent=2, ensure_ascii=False)
    print(f"搜索空间模板已保存: {output_path}")
    print("修改后通过 --search-space 参数指定即可使用自定义搜索空间")


# ============================================================
# 主函数
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="DAG 模型超参数自动调优工具（贝叶斯优化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv

  # 指定 GPU 和试验次数
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv --gpus 0 --n-trials 50

  # 并行调参（在多个终端运行相同命令）
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv --storage sqlite:///tune.db

  # 生成搜索空间模板后自定义
  python tools/auto_tune.py --gen-search-space my_space.json
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv --search-space my_space.json

  # 固定部分参数只调其余参数
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv --fixed-params '{"seq_len": 576, "patch_len": 96}'
        """,
    )
    parser.add_argument("--config-path", required=False, help="评估策略配置文件路径")
    parser.add_argument("--data-name", required=False, help="数据集文件名")
    parser.add_argument(
        "--model-name", default="dag.DAG", help="模型名称 (默认: dag.DAG)"
    )
    parser.add_argument("--gpus", type=int, default=0, help="GPU 编号 (默认: 0)")
    parser.add_argument(
        "--n-trials", type=int, default=30, help="总试验次数 (默认: 30)"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="单次试验超时时间/秒 (默认: 3600)"
    )
    parser.add_argument(
        "--metric",
        default="march_accuracy_mean",
        help="优化目标指标 (默认: march_accuracy_mean)",
    )
    parser.add_argument(
        "--direction",
        default="maximize",
        choices=["maximize", "minimize"],
        help="优化方向 (默认: maximize)",
    )
    parser.add_argument(
        "--output-dir", default="tune_results", help="调参结果输出目录 (默认: tune_results)"
    )
    parser.add_argument("--search-space", default=None, help="自定义搜索空间 JSON 文件")
    parser.add_argument(
        "--gen-search-space",
        default=None,
        help="生成默认搜索空间模板到指定路径并退出",
    )
    parser.add_argument(
        "--study-name", default="dag_tune", help="Optuna study 名称 (默认: dag_tune)"
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna 存储 URL，用于并行调参 (如 sqlite:///tune.db)",
    )
    parser.add_argument(
        "--fixed-params",
        default=None,
        help="固定参数 JSON 字符串 (如 '{\"seq_len\": 576}')",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="项目根目录 (默认: 自动检测)",
    )

    args = parser.parse_args()

    # 生成搜索空间模板模式
    if args.gen_search_space:
        generate_search_space_file(args.gen_search_space)
        return

    # 检查必要参数
    if not args.config_path or not args.data_name:
        parser.error("调参模式需要 --config-path 和 --data-name 参数")

    # 确定项目根目录
    if args.project_root:
        project_root = args.project_root
    else:
        # 默认为 tools/ 的上级目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 加载搜索空间
    search_space = load_search_space(args.search_space)

    # 固定参数
    fixed_params = {}
    if args.fixed_params:
        fixed_params = json.loads(args.fixed_params)

    # 基础参数
    base_args = {
        "project_root": project_root,
        "config_path": args.config_path,
        "data_name": args.data_name,
        "model_name": args.model_name,
        "gpus": args.gpus,
        "num_workers": 1,
        "metric_name": args.metric,
        "output_dir": args.output_dir,
        "fixed_params": fixed_params,
    }

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 日志文件
    log_file = os.path.join(
        args.output_dir, f"tune_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # 创建 Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        sampler=TPESampler(seed=42),
        load_if_exists=True,
    )

    # 打印调参配置
    print(f"\n{'=' * 60}")
    print("DAG 模型超参数自动调优")
    print(f"{'=' * 60}")
    print(f"优化目标: {args.direction} {args.metric}")
    print(f"搜索空间: {len(search_space)} 个参数")
    print(f"  {', '.join(search_space.keys())}")
    print(f"总试验次数: {args.n_trials}")
    print(f"单次超时: {args.timeout}s")
    print(f"模型: {args.model_name}")
    print(f"数据: {args.data_name}")
    print(f"结果目录: {args.output_dir}")
    print(f"日志文件: {log_file}")
    if fixed_params:
        print(f"固定参数: {json.dumps(fixed_params)}")
    if args.storage:
        print(f"并行模式: 已启用 (storage={args.storage})")
        print(f"  可在另一个终端运行相同命令来并行调参")
    print(f"{'=' * 60}")

    # 运行优化
    objective = create_objective(search_space, base_args, args.timeout, log_file)
    study.optimize(objective, n_trials=args.n_trials)

    # 输出最终报告
    print_final_report(study, args, args.output_dir)


if __name__ == "__main__":
    main()
