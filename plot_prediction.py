#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型预测效果可视化分析工具

读取模型训练后生成的 prediction_detail CSV 文件，
生成真实值与预测值的对比曲线图，支持按天分组、误差分析等可视化。

使用方式:
    # 基础用法：生成全月对比图
    python tools/plot_prediction.py --input prediction_detail_xxx.csv

    # 指定输出目录和容量线
    python tools/plot_prediction.py --input prediction_detail_xxx.csv --output ./plots --capacity 80.04

    # 仅生成指定日期范围
    python tools/plot_prediction.py --input prediction_detail_xxx.csv --start-date 2026-03-10 --end-date 2026-03-15

    # 生成单天详细图
    python tools/plot_prediction.py --input prediction_detail_xxx.csv --daily-detail 2026-03-10

依赖: pip install matplotlib pandas numpy
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def load_prediction_data(csv_path: str) -> pd.DataFrame:
    """加载预测详情CSV文件。

    Parameters
    ----------
    csv_path : str
        prediction_detail_{series_name}.csv 的路径

    Returns
    -------
    pd.DataFrame
        包含 date, time, actual_power, predicted_power, error, abs_error, daily_accuracy 列

    Raises
    ------
    FileNotFoundError
        文件不存在
    ValueError
        文件格式不正确，缺少必要列
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"date", "time", "actual_power", "predicted_power"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV文件缺少必要列: {missing}")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def plot_full_comparison(
    df: pd.DataFrame,
    capacity: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: str = ".",
    title_prefix: str = "",
    dpi: int = 150,
) -> str:
    """生成全时段真实值与预测值对比曲线图。

    Parameters
    ----------
    df : pd.DataFrame
        预测详情数据
    capacity : float, optional
        场站装机容量，若提供则绘制容量参考线
    start_date : str, optional
        起始日期过滤（如 "2026-03-10"）
    end_date : str, optional
        结束日期过滤
    output_dir : str
        输出目录
    title_prefix : str
        标题前缀
    dpi : int
        图片分辨率

    Returns
    -------
    str
        保存的图片路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # 日期过滤
    plot_df = df.copy()
    if start_date:
        plot_df = plot_df[plot_df["date"] >= start_date]
    if end_date:
        plot_df = plot_df[plot_df["date"] <= end_date]
    if plot_df.empty:
        print("过滤后无数据，跳过绘图")
        return ""

    fig, axes = plt.subplots(3, 1, figsize=(20, 14), gridspec_kw={"height_ratios": [4, 2, 2]})
    fig.suptitle(
        f"{title_prefix}Power Prediction vs Actual"
        f"  ({plot_df['date'].iloc[0]} ~ {plot_df['date'].iloc[-1]})",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ---- 子图1: 真实值 vs 预测值对比 ----
    ax1 = axes[0]
    ax1.plot(plot_df["time"], plot_df["actual_power"],
             color="#1f77b4", linewidth=0.8, alpha=0.9, label="Actual Power")
    ax1.plot(plot_df["time"], plot_df["predicted_power"],
             color="#ff7f0e", linewidth=0.8, alpha=0.7, label="Predicted Power")

    # 装机容量参考线
    if capacity is not None and capacity > 0:
        ax1.axhline(y=capacity, color="red", linestyle="--", linewidth=0.8,
                     alpha=0.5, label=f"Capacity ({capacity} MW)")

    # 每天分界线
    dates = plot_df["date"].unique()
    for d in dates:
        day_start = plot_df[plot_df["date"] == d]["time"].iloc[0]
        ax1.axvline(x=day_start, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    ax1.set_ylabel("Power (MW)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())

    # ---- 子图2: 误差分布 ----
    ax2 = axes[1]
    errors = plot_df["predicted_power"] - plot_df["actual_power"]
    ax2.fill_between(plot_df["time"], 0, errors,
                      where=errors >= 0, color="#ff7f0e", alpha=0.4, label="Over-prediction")
    ax2.fill_between(plot_df["time"], 0, errors,
                      where=errors < 0, color="#1f77b4", alpha=0.4, label="Under-prediction")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    for d in dates:
        day_start = plot_df[plot_df["date"] == d]["time"].iloc[0]
        ax2.axvline(x=day_start, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)
    ax2.set_ylabel("Error (MW)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())

    # ---- 子图3: 每日准确率 ----
    ax3 = axes[2]
    if "daily_accuracy" in plot_df.columns:
        daily_acc = plot_df.groupby("date")["daily_accuracy"].first()
        colors = ["#2ca02c" if v >= 0.9 else "#ff7f0e" if v >= 0.8 else "#d62728"
                  for v in daily_acc.values]
        ax3.bar(range(len(daily_acc)), daily_acc.values, color=colors, alpha=0.8, width=0.8)
        ax3.set_xticks(range(len(daily_acc)))
        ax3.set_xticklabels([d[-5:] for d in daily_acc.index], rotation=45, fontsize=7)
        ax3.set_ylabel("Daily Accuracy", fontsize=12)
        ax3.axhline(y=0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.5, label="0.9 threshold")
        ax3.axhline(y=0.8, color="orange", linestyle="--", linewidth=0.8, alpha=0.5, label="0.8 threshold")
        ax3.set_ylim(0, 1.05)
        ax3.legend(loc="lower right", fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "No daily_accuracy data", transform=ax3.transAxes,
                  ha="center", fontsize=14, color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    suffix = ""
    if start_date or end_date:
        s = start_date or "start"
        e = end_date or "end"
        suffix = f"_{s}_to_{e}"
    path = os.path.join(output_dir, f"full_comparison{suffix}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"全月对比图已保存: {path}")
    return path


def plot_daily_detail(
    df: pd.DataFrame,
    date: str,
    capacity: Optional[float] = None,
    output_dir: str = ".",
    title_prefix: str = "",
    dpi: int = 150,
) -> str:
    """生成单天的详细对比图。

    Parameters
    ----------
    df : pd.DataFrame
        预测详情数据
    date : str
        指定日期（如 "2026-03-10"）
    capacity : float, optional
        装机容量
    output_dir : str
        输出目录
    title_prefix : str
        标题前缀
    dpi : int
        图片分辨率

    Returns
    -------
    str
        保存的图片路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    day_df = df[df["date"] == date].copy()
    if day_df.empty:
        print(f"日期 {date} 无数据，跳过")
        return ""

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # 准确率信息
    acc = day_df["daily_accuracy"].iloc[0] if "daily_accuracy" in day_df.columns else None
    acc_str = f" | Accuracy: {acc:.4f}" if acc is not None and not np.isnan(acc) else ""

    fig.suptitle(
        f"{title_prefix}{date} Power Prediction{acc_str}",
        fontsize=14, fontweight="bold",
    )

    # ---- 子图1: 对比曲线 ----
    ax1 = axes[0]
    ax1.plot(day_df["time"], day_df["actual_power"],
             "o-", color="#1f77b4", markersize=3, linewidth=1.2,
             label="Actual Power", zorder=3)
    ax1.plot(day_df["time"], day_df["predicted_power"],
             "s-", color="#ff7f0e", markersize=3, linewidth=1.2,
             label="Predicted Power", zorder=3)

    # 填充误差区域
    ax1.fill_between(day_df["time"],
                      day_df["actual_power"], day_df["predicted_power"],
                      alpha=0.15, color="gray", label="Error region")

    if capacity is not None and capacity > 0:
        ax1.axhline(y=capacity, color="red", linestyle="--", linewidth=0.8,
                     alpha=0.5, label=f"Capacity ({capacity} MW)")

    # 标注峰值时刻
    peak_idx = day_df["actual_power"].idxmax()
    peak_time = day_df.loc[peak_idx, "time"]
    peak_val = day_df.loc[peak_idx, "actual_power"]
    pred_at_peak = day_df.loc[peak_idx, "predicted_power"]
    ax1.annotate(
        f"Peak: {peak_val:.1f} MW\nPred: {pred_at_peak:.1f} MW",
        xy=(peak_time, peak_val), xytext=(20, 10),
        textcoords="offset points", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    ax1.set_ylabel("Power (MW)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    # ---- 子图2: 逐点误差 ----
    ax2 = axes[1]
    errors = day_df["predicted_power"] - day_df["actual_power"]
    colors = ["#ff7f0e" if e >= 0 else "#1f77b4" for e in errors]
    ax2.bar(day_df["time"], errors, color=colors, alpha=0.7, width=0.01)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    mae = day_df["abs_error"].mean() if "abs_error" in day_df.columns else np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    ax2.set_ylabel("Error (MW)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_title(f"MAE: {mae:.3f} MW | RMSE: {rmse:.3f} MW", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"daily_detail_{date}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"单天详细图已保存: {path}")
    return path


def plot_scatter_analysis(
    df: pd.DataFrame,
    capacity: Optional[float] = None,
    output_dir: str = ".",
    title_prefix: str = "",
    dpi: int = 150,
) -> str:
    """生成散点图分析（预测值 vs 真实值）。

    Parameters
    ----------
    df : pd.DataFrame
        预测详情数据
    capacity : float, optional
        装机容量，用于设置坐标轴范围
    output_dir : str
        输出目录
    title_prefix : str
        标题前缀
    dpi : int
        图片分辨率

    Returns
    -------
    str
        保存的图片路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    actual = df["actual_power"].values
    predicted = df["predicted_power"].values

    # 散点图，按日期着色
    dates = df["date"].unique()
    cmap = plt.cm.get_cmap("tab20", len(dates))
    for i, d in enumerate(dates):
        mask = df["date"] == d
        ax.scatter(df.loc[mask, "actual_power"], df.loc[mask, "predicted_power"],
                    s=8, alpha=0.5, color=cmap(i), label=d[-5:])

    # 对角线（完美预测线）
    max_val = max(actual.max(), predicted.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="Perfect prediction")

    # 统计信息
    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    corr = np.corrcoef(actual, predicted)[0, 1]
    ax.text(0.05, 0.95,
            f"MAE: {mae:.3f} MW\nRMSE: {rmse:.3f} MW\nCorr: {corr:.4f}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Actual Power (MW)", fontsize=12)
    ax.set_ylabel("Predicted Power (MW)", fontsize=12)
    ax.set_title(f"{title_prefix}Scatter: Predicted vs Actual", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7, ncol=4)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "scatter_analysis.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"散点分析图已保存: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="模型预测效果可视化分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成全月对比图
  python tools/plot_prediction.py --input prediction_detail_xxx.csv

  # 指定容量和输出目录
  python tools/plot_prediction.py --input prediction_detail_xxx.csv --capacity 80.04 --output ./plots

  # 仅生成指定日期范围
  python tools/plot_prediction.py --input prediction_detail_xxx.csv --start-date 2026-03-10 --end-date 2026-03-15

  # 生成单天详细图
  python tools/plot_prediction.py --input prediction_detail_xxx.csv --daily-detail 2026-03-10

  # 生成所有图（全月+每天+散点）
  python tools/plot_prediction.py --input prediction_detail_xxx.csv --all --capacity 80.04
        """,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="prediction_detail CSV 文件路径")
    parser.add_argument("--capacity", type=float, default=None,
                        help="场站装机容量(MW)，用于绘制容量参考线")
    parser.add_argument("--start-date", type=str, default=None,
                        help="起始日期过滤(如 2026-03-10)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="结束日期过滤(如 2026-03-15)")
    parser.add_argument("--daily-detail", type=str, default=None,
                        help="生成指定日期的单天详细图(如 2026-03-10)")
    parser.add_argument("--all", action="store_true",
                        help="生成所有图（全月对比+每天详细+散点分析）")
    parser.add_argument("--output", type=str, default="./prediction_plots",
                        help="输出目录，默认 ./prediction_plots")
    parser.add_argument("--title-prefix", type=str, default="",
                        help="标题前缀（如场站名称）")
    parser.add_argument("--dpi", type=int, default=150,
                        help="图片分辨率，默认150")

    args = parser.parse_args()

    # 加载数据
    print(f"加载数据: {args.input}")
    df = load_prediction_data(args.input)
    print(f"数据量: {len(df)} 行, 日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f"天数: {df['date'].nunique()}")

    prefix = args.title_prefix
    if prefix and not prefix.endswith(" "):
        prefix += " "

    generated = []

    # 全月对比图
    if args.all or (not args.daily_detail):
        p = plot_full_comparison(
            df, capacity=args.capacity,
            start_date=args.start_date, end_date=args.end_date,
            output_dir=args.output, title_prefix=prefix, dpi=args.dpi,
        )
        if p:
            generated.append(p)

    # 单天详细图
    if args.daily_detail:
        p = plot_daily_detail(
            df, date=args.daily_detail, capacity=args.capacity,
            output_dir=args.output, title_prefix=prefix, dpi=args.dpi,
        )
        if p:
            generated.append(p)

    # 所有天的详细图
    if args.all:
        for d in sorted(df["date"].unique()):
            p = plot_daily_detail(
                df, date=d, capacity=args.capacity,
                output_dir=args.output, title_prefix=prefix, dpi=args.dpi,
            )
            if p:
                generated.append(p)

        # 散点分析
        p = plot_scatter_analysis(
            df, capacity=args.capacity,
            output_dir=args.output, title_prefix=prefix, dpi=args.dpi,
        )
        if p:
            generated.append(p)

    print(f"\n共生成 {len(generated)} 张图，保存在: {args.output}")


if __name__ == "__main__":
    main()
