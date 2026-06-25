#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV文件时间对齐与数据合并工具

功能：实现两个CSV文件的精确时间对齐与数据合并，支持缺失值处理和特殊情况检测。

用法示例：
    python merge_csv.py --file-a data_a.csv --file-b data_b.csv --columns-a col1,col2,col3
    python merge_csv.py  # 交互式模式
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 数据读取模块
# =============================================================================

def read_csv_safe(filepath: str, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    安全读取CSV文件，支持大文件分块读取。

    Parameters
    ----------
    filepath : str
        CSV文件路径。
    chunksize : int, optional
        分块读取的行数，None表示一次性读取。

    Returns
    -------
    pd.DataFrame
        读取的数据。

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出。
    pd.errors.EmptyDataError
        文件为空时抛出。
    ValueError
        文件格式异常时抛出。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"读取文件: {filepath} (大小: {file_size_mb:.2f} MB)")

    try:
        if chunksize and file_size_mb > 100:
            logger.info(f"大文件模式，分块读取 (chunksize={chunksize})")
            chunks = []
            for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(filepath, low_memory=False)
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"文件为空: {filepath}")
    except Exception as e:
        raise ValueError(f"读取文件失败 [{filepath}]: {e}")

    if df.empty:
        raise ValueError(f"文件无有效数据: {filepath}")

    logger.info(f"读取完成: {len(df)} 行, {len(df.columns)} 列")
    return df


# =============================================================================
# 时间处理模块
# =============================================================================

def parse_time_format_a(series: pd.Series) -> pd.Series:
    """
    解析文件a的时间格式 "YYYYMMDDHHMM"（如202401030000）。

    Parameters
    ----------
    series : pd.Series
        原始时间列数据。

    Returns
    -------
    pd.Series
        解析后的datetime序列。
    """
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M", errors="coerce")


def parse_time_format_b(series: pd.Series) -> pd.Series:
    """
    解析文件b的时间格式 "YYYY/M/D HH:MM:SS"（如2024/1/3 00:00:00）。

    Parameters
    ----------
    series : pd.Series
        原始时间列数据。

    Returns
    -------
    pd.Series
        解析后的datetime序列。
    """
    return pd.to_datetime(series, errors="coerce")


def auto_detect_time_column(df: pd.DataFrame) -> str:
    """
    自动检测DataFrame中的时间列。

    Parameters
    ----------
    df : pd.DataFrame
        输入数据。

    Returns
    -------
    str
        时间列名。

    Raises
    ------
    ValueError
        未找到时间列时抛出。
    """
    time_keywords = ["time", "date", "datetime", "timestamp", "时间", "日期"]
    for col in df.columns:
        if col.strip().lower() in time_keywords:
            return col
    raise ValueError(f"未找到时间列，当前列名: {list(df.columns)}")


def standardize_time(df: pd.DataFrame, time_col: str, fmt: str) -> pd.DataFrame:
    """
    将时间列标准化为datetime格式，并重命名为"time"。

    Parameters
    ----------
    df : pd.DataFrame
        输入数据。
    time_col : str
        时间列名。
    fmt : str
        时间格式标识，"a" 表示 YYYYMMDDHHMM，"b" 表示 YYYY/M/D HH:MM:SS。

    Returns
    -------
    pd.DataFrame
        时间列标准化后的数据。

    Raises
    ------
    ValueError
        时间解析失败率过高时抛出。
    """
    df = df.copy()

    if fmt == "a":
        df["time"] = parse_time_format_a(df[time_col])
    elif fmt == "b":
        df["time"] = parse_time_format_b(df[time_col])
    else:
        df["time"] = pd.to_datetime(df[time_col], errors="coerce")

    na_count = df["time"].isna().sum()
    if na_count > len(df) * 0.1:
        raise ValueError(f"时间解析失败率过高: {na_count}/{len(df)} ({na_count/len(df)*100:.1f}%)")

    if na_count > 0:
        logger.warning(f"时间解析失败 {na_count} 行，已剔除")
        df = df.dropna(subset=["time"])

    df = df.drop(columns=[time_col])
    return df


# =============================================================================
# 合并模块
# =============================================================================

def merge_on_time(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    selected_columns_a: List[str],
) -> pd.DataFrame:
    """
    基于time列合并两个DataFrame，以文件b的时间为基准。

    文件b的时间记录为最终结果：a缺失b的时间补NaN，a多余的时间丢弃。
    文件a的选定列追加在文件b最后一列之后。

    Parameters
    ----------
    df_a : pd.DataFrame
        文件a数据，需包含time列和selected_columns_a中的列。
    df_b : pd.DataFrame
        文件b数据，需包含time列。
    selected_columns_a : List[str]
        从文件a中选择的属性列。

    Returns
    -------
    pd.DataFrame
        合并后的数据。

    Raises
    ------
    ValueError
        选择的列不存在时抛出。
    """
    for col in selected_columns_a:
        if col not in df_a.columns:
            raise ValueError(f"文件a中不存在列: '{col}'，可用列: {list(df_a.columns)}")

    # 精确到分钟对齐
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a["time"] = df_a["time"].dt.floor("min")
    df_b["time"] = df_b["time"].dt.floor("min")

    # 去重，保留最后一条
    df_a = df_a.drop_duplicates(subset="time", keep="last")
    df_b = df_b.drop_duplicates(subset="time", keep="last")

    # 只保留文件a中需要的列
    df_a_subset = df_a[["time"] + selected_columns_a].copy()

    # 以文件b的time为基准左连接
    merged = df_b.merge(df_a_subset, on="time", how="left")

    # 按文件b的时间排序
    merged = merged.sort_values("time").reset_index(drop=True)

    logger.info(
        f"合并完成: 文件b {len(df_b)} 行, 文件a {len(df_a)} 行 → 合并后 {len(merged)} 行"
    )

    return merged


# =============================================================================
# 数据清洗模块
# =============================================================================

def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    全面检测缺失值，生成统计报告。

    Parameters
    ----------
    df : pd.DataFrame
        输入数据。

    Returns
    -------
    pd.DataFrame
        缺失值统计表，包含每列的缺失数量和缺失率。
    """
    total = len(df)
    missing_count = df.isna().sum()
    missing_ratio = missing_count / total

    report = pd.DataFrame({
        "列名": df.columns,
        "缺失数量": missing_count.values,
        "缺失率": missing_ratio.values,
        "非空数量": (total - missing_count).values,
    })

    total_missing = missing_count.sum()
    logger.info(f"缺失值检测: 共 {total_missing} 个缺失值")
    for _, row in report.iterrows():
        if row["缺失数量"] > 0:
            logger.info(f"  列 '{row['列名']}': {row['缺失数量']} 缺失 ({row['缺失率']:.2%})")

    return report


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "interpolate",
    fill_value: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    处理缺失值。

    Parameters
    ----------
    df : pd.DataFrame
        含缺失值的数据。
    strategy : str
        处理策略:
        - "drop": 删除含缺失值的行
        - "fill_zero": 用0填充
        - "fill_mean": 用均值填充
        - "fill_median": 用中位数填充
        - "interpolate": 线性插值（默认）
        - "ffill": 前向填充
        - "bfill": 后向填充
        - "custom": 用fill_value填充
    fill_value : float, optional
        strategy为"custom"时使用的填充值。

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        处理后的数据和处理记录。
    """
    df = df.copy()
    missing_before = df.isna().sum().sum()
    record = {"strategy": strategy, "missing_before": int(missing_before), "details": {}}

    if missing_before == 0:
        logger.info("无缺失值，跳过处理")
        record["missing_after"] = 0
        return df, record

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_per_col = df[numeric_cols].isna().sum()
    missing_per_col = missing_per_col[missing_per_col > 0]

    if strategy == "drop":
        before_len = len(df)
        df = df.dropna()
        record["details"]["dropped_rows"] = before_len - len(df)
        logger.info(f"删除缺失行: {before_len} → {len(df)}")

    elif strategy == "fill_zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
        logger.info("用0填充缺失值")

    elif strategy == "fill_mean":
        for col in missing_per_col.index:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            record["details"][col] = f"mean={mean_val:.4f}"
        logger.info("用均值填充缺失值")

    elif strategy == "fill_median":
        for col in missing_per_col.index:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            record["details"][col] = f"median={median_val:.4f}"
        logger.info("用中位数填充缺失值")

    elif strategy == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        # 插值后仍有残留NaN（首尾），用前向/后向填充
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        logger.info("线性插值填充缺失值")

    elif strategy == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        logger.info("前向填充缺失值")

    elif strategy == "bfill":
        df[numeric_cols] = df[numeric_cols].bfill().ffill()
        logger.info("后向填充缺失值")

    elif strategy == "custom":
        if fill_value is None:
            raise ValueError("strategy为'custom'时必须指定fill_value")
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)
        logger.info(f"用 {fill_value} 填充缺失值")

    else:
        raise ValueError(f"未知策略: {strategy}")

    missing_after = df.isna().sum().sum()
    record["missing_after"] = int(missing_after)
    logger.info(f"缺失值处理: {missing_before} → {missing_after}")

    return df, record


def detect_special_cases(
    df: pd.DataFrame,
    target_columns: List[str],
    window_hours: int = 24,
    zero_threshold: float = 0.0,
    time_col: str = "time",
) -> List[Dict]:
    """
    检测"全天限电"等特殊情况：特定列在连续时间窗口内持续为0或接近0。

    Parameters
    ----------
    df : pd.DataFrame
        输入数据。
    target_columns : List[str]
        需要检测的列名列表。
    window_hours : int
        连续时间窗口阈值（小时），默认24。
    zero_threshold : float
        判定为"持续为0"的阈值，默认0.0（严格为0）。
    time_col : str
        时间列名。

    Returns
    -------
    List[Dict]
        检测到的特殊情况列表，每项包含列名、起止时间、持续时长。
    """
    if not target_columns:
        return []

    # 推断时间间隔（分钟）
    time_diffs = df[time_col].diff().dropna()
    median_interval = time_diffs.median()
    interval_minutes = median_interval.total_seconds() / 60
    window_points = int(window_hours * 60 / interval_minutes)

    if window_points < 1:
        window_points = 1

    results = []

    for col in target_columns:
        if col not in df.columns:
            logger.warning(f"检测列不存在: '{col}'，跳过")
            continue

        is_zero = df[col].abs() <= zero_threshold

        # 找连续为0的区间
        groups = (~is_zero).cumsum()
        zero_segments = is_zero.groupby(groups).apply(
            lambda x: (x.sum(), x.index[0], x.index[-1]) if x.sum() > 0 else None
        ).dropna()

        for _, seg_info in zero_segments.items():
            if seg_info is None:
                continue
            count, start_idx, end_idx = seg_info
            if count >= window_points:
                start_time = df.loc[start_idx, time_col]
                end_time = df.loc[end_idx, time_col]
                duration_hours = (end_time - start_time).total_seconds() / 3600
                results.append({
                    "column": col,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_hours": round(duration_hours, 2),
                    "row_count": int(count),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                })

    if results:
        logger.info(f"检测到 {len(results)} 个特殊情况:")
        for r in results:
            logger.info(
                f"  列 '{r['column']}': {r['start_time']} ~ {r['end_time']}, "
                f"持续 {r['duration_hours']} 小时, {r['row_count']} 行"
            )
    else:
        logger.info("未检测到特殊情况")

    return results


def handle_special_cases(
    df: pd.DataFrame,
    special_cases: List[Dict],
    strategy: str = "drop",
    time_col: str = "time",
) -> Tuple[pd.DataFrame, Dict]:
    """
    处理检测到的特殊情况。

    Parameters
    ----------
    df : pd.DataFrame
        输入数据。
    special_cases : List[Dict]
        detect_special_cases的输出。
    strategy : str
        处理策略:
        - "drop": 删除整段记录
        - "fill_mean": 用该列非特殊值的均值替换
        - "fill_median": 用中位数替换
        - "interpolate": 线性插值
        - "keep": 保留不处理
    time_col : str
        时间列名。

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        处理后的数据和处理记录。
    """
    df = df.copy()
    record = {"strategy": strategy, "cases_handled": 0, "details": []}

    if not special_cases or strategy == "keep":
        logger.info("特殊情况: 无需处理" if not special_cases else "特殊情况: 保留不处理")
        return df, record

    if strategy == "drop":
        drop_indices = set()
        for case in special_cases:
            drop_indices.update(range(case["start_idx"], case["end_idx"] + 1))
        existing_indices = set(df.index)
        drop_indices = drop_indices & existing_indices
        before_len = len(df)
        df = df.drop(index=drop_indices).reset_index(drop=True)
        record["cases_handled"] = len(special_cases)
        record["details"].append(f"删除 {len(drop_indices)} 行")
        logger.info(f"特殊情况处理(删除): {before_len} → {len(df)} 行")

    elif strategy in ("fill_mean", "fill_median"):
        for case in special_cases:
            col = case["column"]
            mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
            non_special = df.loc[~mask, col]
            if strategy == "fill_mean":
                fill_val = non_special.mean()
            else:
                fill_val = non_special.median()
            df.loc[mask, col] = fill_val
            record["cases_handled"] += 1
            record["details"].append(
                f"列 '{col}' {case['start_time']}~{case['end_time']}: "
                f"{strategy}={fill_val:.4f}"
            )
        logger.info(f"特殊情况处理({strategy}): 处理 {record['cases_handled']} 个区间")

    elif strategy == "interpolate":
        for case in special_cases:
            col = case["column"]
            mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
            df.loc[mask, col] = np.nan
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        record["cases_handled"] = len(special_cases)
        logger.info(f"特殊情况处理(插值): 处理 {len(special_cases)} 个区间")

    else:
        raise ValueError(f"未知策略: {strategy}")

    return df, record


# =============================================================================
# 输出模块
# =============================================================================

def preview_data(
    df: pd.DataFrame,
    label: str = "数据",
    n: int = 5,
) -> None:
    """
    数据预览，展示前后n行和基本统计。

    Parameters
    ----------
    df : pd.DataFrame
        数据。
    label : str
        数据标签。
    n : int
        预览行数。
    """
    print(f"\n{'='*60}")
    print(f" {label} 预览")
    print(f"{'='*60}")
    print(f"记录数: {len(df)}, 列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    print(f"\n前 {n} 行:")
    print(df.head(n).to_string())
    print(f"\n后 {n} 行:")
    print(df.tail(n).to_string())
    print(f"{'='*60}\n")


def save_merged_data(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    保存合并后的数据到CSV文件。

    Parameters
    ----------
    df : pd.DataFrame
        合并后的数据。
    output_path : str
        输出文件路径。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"已保存合并结果: {output_path} ({len(df)} 行)")


class MergeLogger:
    """合并操作日志记录器，同时输出到控制台和日志文件。"""

    def __init__(self, log_dir: str = "."):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"merge_log_{timestamp}.txt")

        self.logger = logging.getLogger("merge_csv")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # 文件handler
        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        # 控制台handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.log_data = {
            "start_time": datetime.now(),
            "end_time": None,
            "file_a": None,
            "file_b": None,
            "selected_columns_a": None,
            "missing_value_record": None,
            "special_case_record": None,
            "errors": [],
        }

    def record_file_info(self, file_a: str, file_b: str, columns_a: List[str]):
        self.log_data["file_a"] = file_a
        self.log_data["file_b"] = file_b
        self.log_data["selected_columns_a"] = columns_a

    def record_missing(self, record: Dict):
        self.log_data["missing_value_record"] = record

    def record_special(self, record: Dict):
        self.log_data["special_case_record"] = record

    def record_error(self, msg: str):
        self.log_data["errors"].append(msg)

    def finalize(self):
        self.log_data["end_time"] = datetime.now()
        duration = (self.log_data["end_time"] - self.log_data["start_time"]).total_seconds()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(" 合并日志摘要")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"开始时间: {self.log_data['start_time']}")
        self.logger.info(f"结束时间: {self.log_data['end_time']}")
        self.logger.info(f"总耗时: {duration:.2f} 秒")
        self.logger.info(f"文件a: {self.log_data['file_a']}")
        self.logger.info(f"文件b: {self.log_data['file_b']}")
        self.logger.info(f"选择列: {self.log_data['selected_columns_a']}")
        if self.log_data["missing_value_record"]:
            self.logger.info(f"缺失值处理: {self.log_data['missing_value_record']}")
        if self.log_data["special_case_record"]:
            self.logger.info(f"特殊情况处理: {self.log_data['special_case_record']}")
        if self.log_data["errors"]:
            self.logger.info(f"异常记录: {self.log_data['errors']}")
        self.logger.info(f"日志文件: {self.log_path}")
        self.logger.info(f"{'='*60}")


# 全局logger，在main中初始化
logger: logging.Logger = logging.getLogger("merge_csv")


# =============================================================================
# 主流程
# =============================================================================

def interactive_input(prompt: str, choices: Optional[List[str]] = None) -> str:
    """交互式输入，支持选项校验。"""
    while True:
        value = input(prompt).strip()
        if not value:
            print("输入不能为空，请重新输入")
            continue
        if choices and value not in choices:
            print(f"请输入有效选项: {choices}")
            continue
        return value


def main():
    """主入口函数。"""
    parser = argparse.ArgumentParser(description="CSV文件时间对齐与数据合并工具")
    parser.add_argument("--file-a", type=str, help="文件a路径")
    parser.add_argument("--file-b", type=str, help="文件b路径")
    parser.add_argument("--time-col-a", type=str, default=None, help="文件a的时间列名（自动检测则不填）")
    parser.add_argument("--time-col-b", type=str, default=None, help="文件b的时间列名（自动检测则不填）")
    parser.add_argument("--columns-a", type=str, default=None, help="从文件a选择的列，逗号分隔")
    parser.add_argument("--missing-strategy", type=str, default="interpolate",
                        choices=["drop", "fill_zero", "fill_mean", "fill_median", "interpolate", "ffill", "bfill"],
                        help="缺失值处理策略")
    parser.add_argument("--special-strategy", type=str, default="keep",
                        choices=["drop", "fill_mean", "fill_median", "interpolate", "keep"],
                        help="特殊情况处理策略")
    parser.add_argument("--special-columns", type=str, default=None,
                        help="需要检测特殊情况的列，逗号分隔")
    parser.add_argument("--window-hours", type=int, default=24, help="特殊情况检测窗口（小时）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--log-dir", type=str, default=".", help="日志目录")

    args = parser.parse_args()

    # 初始化日志
    global logger
    merge_log = MergeLogger(log_dir=args.log_dir)
    logger = merge_log.logger

    logger.info(f"处理开始: {datetime.now()}")

    try:
        # ---- 文件路径 ----
        file_a = args.file_a or interactive_input("请输入文件a路径: ")
        file_b = args.file_b or interactive_input("请输入文件b路径: ")

        merge_log.record_file_info(file_a, file_b, None)

        # ---- 读取文件 ----
        logger.info("=" * 40 + " 读取文件 " + "=" * 40)
        df_a = read_csv_safe(file_a, chunksize=50000)
        df_b = read_csv_safe(file_b, chunksize=50000)

        preview_data(df_a, "文件a（原始）")
        preview_data(df_b, "文件b（原始）")

        # ---- 时间列检测与标准化 ----
        logger.info("=" * 40 + " 时间处理 " + "=" * 40)
        time_col_a = args.time_col_a or auto_detect_time_column(df_a)
        time_col_b = args.time_col_b or auto_detect_time_column(df_b)
        logger.info(f"文件a时间列: '{time_col_a}', 文件b时间列: '{time_col_b}'")

        df_a = standardize_time(df_a, time_col_a, fmt="a")
        df_b = standardize_time(df_b, time_col_b, fmt="b")

        logger.info(f"文件a时间范围: {df_a['time'].min()} ~ {df_a['time'].max()}")
        logger.info(f"文件b时间范围: {df_b['time'].min()} ~ {df_b['time'].max()}")

        # ---- 选择文件a的列 ----
        available_cols_a = [c for c in df_a.columns if c != "time"]
        if args.columns_a:
            selected_cols_a = [c.strip() for c in args.columns_a.split(",")]
        else:
            print(f"\n文件a可用列: {available_cols_a}")
            cols_input = interactive_input("请输入要从文件a选择的列（逗号分隔）: ")
            selected_cols_a = [c.strip() for c in cols_input.split(",")]

        merge_log.record_file_info(file_a, file_b, selected_cols_a)
        logger.info(f"从文件a选择的列: {selected_cols_a}")

        # ---- 合并 ----
        logger.info("=" * 40 + " 数据合并 " + "=" * 40)
        merged = merge_on_time(df_a, df_b, selected_cols_a)
        preview_data(merged, "合并后（缺失值处理前）")

        # ---- 缺失值检测与处理 ----
        logger.info("=" * 40 + " 缺失值处理 " + "=" * 40)
        missing_report = detect_missing_values(merged)
        print("\n缺失值统计:")
        print(missing_report.to_string(index=False))

        missing_strategy = args.missing_strategy
        merged, missing_record = handle_missing_values(merged, strategy=missing_strategy)
        merge_log.record_missing(missing_record)

        # ---- 特殊情况检测与处理 ----
        logger.info("=" * 40 + " 特殊情况检测 " + "=" * 40)
        if args.special_columns:
            special_cols = [c.strip() for c in args.special_columns.split(",")]
        else:
            special_cols_input = input("请输入需要检测特殊情况的列（逗号分隔，直接回车跳过）: ").strip()
            special_cols = [c.strip() for c in special_cols_input.split(",") if c.strip()] if special_cols_input else []

        special_cases = detect_special_cases(
            merged, target_columns=special_cols,
            window_hours=args.window_hours, time_col="time"
        )

        if special_cases:
            print(f"\n检测到 {len(special_cases)} 个特殊情况:")
            for i, case in enumerate(special_cases, 1):
                print(f"  {i}. 列 '{case['column']}': {case['start_time']} ~ {case['end_time']}, "
                      f"持续 {case['duration_hours']}h")

            special_strategy = args.special_strategy
            if special_strategy == "keep":
                print("当前策略: 保留不处理")
            merged, special_record = handle_special_cases(merged, special_cases, strategy=special_strategy)
            merge_log.record_special(special_record)
        else:
            merge_log.record_special({"strategy": "none", "cases_handled": 0})

        # ---- 数据校验 ----
        logger.info("=" * 40 + " 数据校验 " + "=" * 40)
        remaining_nan = merged.isna().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"合并后仍有 {remaining_nan} 个缺失值")
        else:
            logger.info("校验通过: 无残留缺失值")

        # ---- 输出 ----
        logger.info("=" * 40 + " 输出结果 " + "=" * 40)
        preview_data(merged, "最终结果")

        if args.output:
            output_path = args.output
        else:
            output_path = input("请输入输出文件路径（直接回车使用默认）: ").strip()
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"merged_result_{timestamp}.csv"

        save_merged_data(merged, output_path)

    except FileNotFoundError as e:
        logger.error(f"文件错误: {e}")
        merge_log.record_error(str(e))
    except ValueError as e:
        logger.error(f"数据错误: {e}")
        merge_log.record_error(str(e))
    except Exception as e:
        logger.error(f"未预期错误: {e}")
        merge_log.record_error(str(e))
    finally:
        merge_log.finalize()


if __name__ == "__main__":
    main()
