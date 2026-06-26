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
# 属性分类配置
# =============================================================================
# 夜间合法为0的属性：夜间无光照/无发电，值为0是正常的
NIGHTTIME_ZERO_COLS = {"power", "sr"}

# 太阳能相关属性：白天应有非零值，夜间为0正常；列名含这些关键词的自动识别
SOLAR_KEYWORDS = ["solar", "radiation", "日照", "辐射"]

# 理论上全天不为0的属性：值为0一定是数据异常
# 大气长波辐射、相对湿度等物理量不可能严格为0
NEVER_ZERO_KEYWORDS = ["thermal_radiation", "humidity", "热辐射", "湿度"]


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
    """
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M", errors="coerce")


def parse_time_format_b(series: pd.Series) -> pd.Series:
    """
    解析文件b的时间格式 "YYYY/M/D HH:MM:SS"（如2024/1/3 00:00:00）。
    """
    return pd.to_datetime(series, errors="coerce")


def auto_detect_time_column(df: pd.DataFrame) -> str:
    """自动检测DataFrame中的时间列。"""
    time_keywords = ["time", "date", "datetime", "timestamp", "时间", "日期"]
    for col in df.columns:
        if col.strip().lower() in time_keywords:
            return col
    raise ValueError(f"未找到时间列，当前列名: {list(df.columns)}")


def standardize_time(df: pd.DataFrame, time_col: str, fmt: str) -> pd.DataFrame:
    """将时间列标准化为datetime格式，并重命名为"time"。"""
    df = df.copy()

    parsed = None
    if fmt == "a":
        parsed = parse_time_format_a(df[time_col])
    elif fmt == "b":
        parsed = parse_time_format_b(df[time_col])
    else:
        parsed = pd.to_datetime(df[time_col], errors="coerce")

    if time_col == "time":
        df["time"] = parsed
    else:
        df = df.drop(columns=[time_col])
        df["time"] = parsed

    na_count = df["time"].isna().sum()
    if na_count > len(df) * 0.1:
        raise ValueError(f"时间解析失败率过高: {na_count}/{len(df)} ({na_count/len(df)*100:.1f}%)")

    if na_count > 0:
        logger.warning(f"时间解析失败 {na_count} 行，已剔除")
        df = df.dropna(subset=["time"])

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
    """
    for col in selected_columns_a:
        if col not in df_a.columns:
            raise ValueError(f"文件a中不存在列: '{col}'，可用列: {list(df_a.columns)}")

    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a["time"] = df_a["time"].dt.floor("min")
    df_b["time"] = df_b["time"].dt.floor("min")

    df_a = df_a.drop_duplicates(subset="time", keep="last")
    df_b = df_b.drop_duplicates(subset="time", keep="last")

    df_a_subset = df_a[["time"] + selected_columns_a].copy()

    merged = df_b.merge(df_a_subset, on="time", how="left")
    merged = merged.sort_values("time").reset_index(drop=True)

    logger.info(
        f"合并完成: 文件b {len(df_b)} 行, 文件a {len(df_a)} 行 → 合并后 {len(merged)} 行"
    )

    return merged


# =============================================================================
# 数据清洗模块
# =============================================================================

def classify_column(col: str) -> str:
    """
    根据列名自动分类属性的昼夜分布类型。

    Returns
    -------
    str
        "nighttime_zero": 夜间合法为0（power, sr）
        "solar": 太阳能相关，白天不应为0，夜间为0正常
        "never_zero": 理论上全天不为0，为0一定是异常
        "other": 其他（如温度，0可能是有效物理值）
    """
    col_lower = col.lower()
    if col in NIGHTTIME_ZERO_COLS:
        return "nighttime_zero"
    if any(k in col_lower for k in SOLAR_KEYWORDS):
        return "solar"
    if any(k in col_lower for k in NEVER_ZERO_KEYWORDS):
        return "never_zero"
    return "other"


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """全面检测缺失值，生成统计报告。"""
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
    test_cutoff: Optional[str] = None,
    window_hours: int = 24,
    zero_threshold: float = 0.0,
    time_col: str = "time",
    daytime_start: int = 7,
    daytime_end: int = 19,
) -> List[Dict]:
    """
    检测异常数据，按列类型和昼夜规律分类处理：

    - nighttime_zero 类（power, sr）：全天为0才判定异常（限电），
      夜间为0是正常的。检测范围排除测试期数据。
    - solar 类：只检测白天时段是否全为0。夜间为0正常。
      检测范围排除测试期数据。
    - never_zero 类（thermal_radiation, humidity）：任何时刻为0即为异常。
      全时段检测，不区分昼夜，不排除测试期（因为是外生变量，修正不影响评估公平性）。
    - other 类：不检测。

    Parameters
    ----------
    df : pd.DataFrame
        输入数据。
    target_columns : List[str]
        需要检测的列名列表。
    test_cutoff : str, optional
        测试期起始日期（如 "2026-03-01"），此日期之后的内生变量数据不参与检测，
        避免修改真实值影响评估。外生变量不受此限制。None 表示不排除。
    window_hours : int
        保留参数，按天检测模式下不直接使用。
    zero_threshold : float
        判定为"为0"的阈值，默认0.0。
    time_col : str
        时间列名。
    daytime_start : int
        白天开始小时（含），默认7。
    daytime_end : int
        白天结束小时（不含），默认19。

    Returns
    -------
    List[Dict]
        检测到的特殊情况列表。
    """
    if not target_columns:
        return []

    df = df.copy()
    df["_hour"] = df[time_col].dt.hour
    df["_date"] = df[time_col].dt.date

    # 测试期截止：内生变量不修改测试期数据
    test_cutoff_date = None
    if test_cutoff:
        test_cutoff_date = pd.Timestamp(test_cutoff).date()
        logger.info(f"测试期起始: {test_cutoff}，内生变量不检测此日期之后的数据")

    results = []

    for col in target_columns:
        if col not in df.columns:
            logger.warning(f"检测列不存在: '{col}'，跳过")
            continue

        col_type = classify_column(col)

        # other 类不检测
        if col_type == "other":
            logger.info(f"列 '{col}' 分类为 'other'，跳过特殊检测")
            continue

        # 判断该列是否受测试期限制（内生变量受限制，外生变量不受）
        is_endogenous = col_type == "nighttime_zero"
        # solar 和 never_zero 属于外生变量，不受测试期限制

        for date, day_df in df.groupby("_date"):
            # 内生变量：跳过测试期
            if is_endogenous and test_cutoff_date and date >= test_cutoff_date:
                continue

            is_abnormal = False

            if col_type == "nighttime_zero":
                # power, sr: 全天为0才算异常（夜间为0正常但白天也0说明限电）
                is_abnormal = (day_df[col].abs() <= zero_threshold).all()

            elif col_type == "solar":
                # 太阳能类：白天时段全为0才算异常
                day_hours = day_df[
                    (day_df["_hour"] >= daytime_start) & (day_df["_hour"] < daytime_end)
                ]
                if day_hours.empty:
                    continue
                is_abnormal = (day_hours[col].abs() <= zero_threshold).all()

            elif col_type == "never_zero":
                # 全天不应为0的属性：任何时刻为0都算异常
                zero_mask = day_df[col].abs() <= zero_threshold
                if zero_mask.any():
                    is_abnormal = True

            if is_abnormal:
                if col_type == "never_zero":
                    # never_zero: 只标记为0的那些行，不是整天
                    zero_rows = day_df[day_df[col].abs() <= zero_threshold]
                    start_idx = zero_rows.index[0]
                    end_idx = zero_rows.index[-1]
                    start_time = df.loc[start_idx, time_col]
                    end_time = df.loc[end_idx, time_col]
                    results.append({
                        "column": col,
                        "col_type": col_type,
                        "date": str(date),
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_hours": round(
                            (end_time - start_time).total_seconds() / 3600, 2
                        ),
                        "row_count": len(zero_rows),
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                    })
                else:
                    # nighttime_zero / solar: 标记整天
                    start_time = day_df[time_col].iloc[0]
                    end_time = day_df[time_col].iloc[-1]
                    start_idx = day_df.index[0]
                    end_idx = day_df.index[-1]
                    results.append({
                        "column": col,
                        "col_type": col_type,
                        "date": str(date),
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_hours": round(
                            (end_time - start_time).total_seconds() / 3600, 2
                        ),
                        "row_count": len(day_df),
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                    })

    df = df.drop(columns=["_hour", "_date"])

    if results:
        logger.info(f"检测到 {len(results)} 个特殊情况:")
        for r in results:
            logger.info(
                f"  列 '{r['column']}' [{r['col_type']}]: "
                f"{r['date']}, {r['start_time']} ~ {r['end_time']}, "
                f"{r['row_count']} 行"
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

    根据 col_type 区分处理策略：
    - nighttime_zero (power, sr): 限电天，推荐 drop（删除整行）或 keep
    - solar: 白天全0，推荐 drop 或 keep
    - never_zero: 异常0值，用前后正常值插值替换（不用 drop 或 fill_mean）

    Parameters
    ----------
    strategy : str
        对 nighttime_zero 和 solar 类的处理策略:
        - "drop": 删除整段记录
        - "fill_mean": 用非异常值均值替换
        - "fill_median": 用中位数替换
        - "interpolate": 线性插值
        - "keep": 保留不处理
        never_zero 类始终使用插值替换，忽略此参数。
    """
    df = df.copy()
    record = {"strategy": strategy, "cases_handled": 0, "details": []}

    if not special_cases:
        logger.info("特殊情况: 无需处理")
        return df, record

    # 分类处理
    never_zero_cases = [c for c in special_cases if c.get("col_type") == "never_zero"]
    other_cases = [c for c in special_cases if c.get("col_type") != "never_zero"]

    # ---- never_zero 类：强制插值替换 ----
    if never_zero_cases:
        for case in never_zero_cases:
            col = case["column"]
            mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
            # 找到该列中值为0的位置
            zero_mask = mask & (df[col].abs() <= 1e-10)
            if zero_mask.any():
                # 将异常0值设为NaN，然后插值
                df.loc[zero_mask, col] = np.nan
                record["cases_handled"] += 1
                count = zero_mask.sum()
                record["details"].append(
                    f"列 '{col}' {case['date']}: {count} 个异常0值 → 插值替换"
                )

        # 对 never_zero 列做插值
        never_zero_cols = list(set(c["column"] for c in never_zero_cases))
        for col in never_zero_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
                df[col] = df[col].ffill().bfill()

        logger.info(f"never_zero 类处理: {len(never_zero_cases)} 个区间，插值替换")

    # ---- nighttime_zero / solar 类：按用户策略处理 ----
    if not other_cases or strategy == "keep":
        if other_cases:
            logger.info("nighttime_zero/solar 类: 保留不处理")
        return df, record

    if strategy == "drop":
        drop_indices = set()
        for case in other_cases:
            drop_indices.update(range(case["start_idx"], case["end_idx"] + 1))
        existing_indices = set(df.index)
        drop_indices = drop_indices & existing_indices
        before_len = len(df)
        df = df.drop(index=drop_indices).reset_index(drop=True)
        record["cases_handled"] += len(other_cases)
        record["details"].append(f"删除 {len(drop_indices)} 行 (限电/太阳能异常)")
        logger.info(f"nighttime_zero/solar 处理(删除): {before_len} → {len(df)} 行")

    elif strategy in ("fill_mean", "fill_median"):
        for case in other_cases:
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
        logger.info(f"nighttime_zero/solar 处理({strategy}): 处理 {len(other_cases)} 个区间")

    elif strategy == "interpolate":
        for case in other_cases:
            col = case["column"]
            mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
            df.loc[mask, col] = np.nan
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        record["cases_handled"] += len(other_cases)
        logger.info(f"nighttime_zero/solar 处理(插值): 处理 {len(other_cases)} 个区间")

    else:
        raise ValueError(f"未知策略: {strategy}")

    return df, record


def validate_data(df: pd.DataFrame, time_col: str = "time") -> Dict:
    """
    数据校验：检查各属性是否符合昼夜分布规律。

    Returns
    -------
    Dict
        校验结果，包含各列的违规统计。
    """
    df = df.copy()
    df["_hour"] = df[time_col].dt.hour
    issues = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        col_type = classify_column(col)
        col_issues = []

        if col_type == "never_zero":
            # 全天不应为0
            zero_count = (df[col].abs() <= 1e-10).sum()
            if zero_count > 0:
                col_issues.append(f"存在 {zero_count} 个0值（应为非零）")

        elif col_type == "nighttime_zero":
            # 白天（7-19点）不应全为0（但个别点为0可能正常，如早晚过渡）
            day_df = df[(df["_hour"] >= 7) & (df["_hour"] < 19)]
            day_zero_count = (day_df[col].abs() <= 1e-10).sum()
            day_total = len(day_df)
            if day_total > 0 and day_zero_count == day_total:
                col_issues.append("白天时段全部为0（可能限电）")

        elif col_type == "solar":
            # 白天不应全为0
            day_df = df[(df["_hour"] >= 7) & (df["_hour"] < 19)]
            day_zero_count = (day_df[col].abs() <= 1e-10).sum()
            day_total = len(day_df)
            if day_total > 0 and day_zero_count == day_total:
                col_issues.append("白天时段全部为0（异常）")

        if col_issues:
            issues[col] = col_issues

    df = df.drop(columns=["_hour"])

    if issues:
        logger.warning(f"数据校验发现 {len(issues)} 列存在潜在问题:")
        for col, col_issues in issues.items():
            for issue in col_issues:
                logger.warning(f"  列 '{col}': {issue}")
    else:
        logger.info("数据校验通过: 各属性昼夜分布符合预期")

    return issues


# =============================================================================
# 输出模块
# =============================================================================

def preview_data(df: pd.DataFrame, label: str = "数据", n: int = 5) -> None:
    """数据预览，展示前后n行和基本统计。"""
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


def save_merged_data(df: pd.DataFrame, output_path: str) -> None:
    """保存合并后的数据到CSV文件。"""
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

        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

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
    parser.add_argument("--time-col-a", type=str, default=None, help="文件a的时间列名")
    parser.add_argument("--time-col-b", type=str, default=None, help="文件b的时间列名")
    parser.add_argument("--columns-a", type=str, default=None, help="从文件a选择的列，逗号分隔")
    parser.add_argument("--missing-strategy", type=str, default="interpolate",
                        choices=["drop", "fill_zero", "fill_mean", "fill_median", "interpolate", "ffill", "bfill"],
                        help="缺失值处理策略")
    parser.add_argument("--special-strategy", type=str, default="keep",
                        choices=["drop", "fill_mean", "fill_median", "interpolate", "keep"],
                        help="特殊情况处理策略（仅对 nighttime_zero/solar 类有效）")
    parser.add_argument("--special-columns", type=str, default=None,
                        help="需要检测特殊情况的列，逗号分隔")
    parser.add_argument("--test-cutoff", type=str, default=None,
                        help="测试期起始日期(如2026-03-01)，内生变量不检测此日期之后的数据")
    parser.add_argument("--window-hours", type=int, default=24, help="保留参数")
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

        # 打印属性分类信息
        all_data_cols = [c for c in merged.columns if c != "time"]
        logger.info("属性昼夜分类:")
        for col in all_data_cols:
            col_type = classify_column(col)
            desc = {
                "nighttime_zero": "夜间合法为0（内生变量，检测排除测试期）",
                "solar": "太阳能相关，白天不应为0（检测排除测试期）",
                "never_zero": "全天不应为0（外生变量，全时段检测）",
                "other": "其他（不检测）",
            }
            logger.info(f"  {col}: {col_type} — {desc[col_type]}")

        if args.special_columns:
            special_cols = [c.strip() for c in args.special_columns.split(",")]
        else:
            special_cols_input = input("请输入需要检测特殊情况的列（逗号分隔，直接回车跳过）: ").strip()
            special_cols = [c.strip() for c in special_cols_input.split(",") if c.strip()] if special_cols_input else []

        special_cases = detect_special_cases(
            merged, target_columns=special_cols,
            test_cutoff=args.test_cutoff,
            window_hours=args.window_hours, time_col="time"
        )

        if special_cases:
            print(f"\n检测到 {len(special_cases)} 个特殊情况:")
            for i, case in enumerate(special_cases, 1):
                print(f"  {i}. 列 '{case['column']}' [{case['col_type']}]: "
                      f"{case['date']}, {case['start_time']} ~ {case['end_time']}, "
                      f"{case['row_count']} 行")

            merged, special_record = handle_special_cases(merged, special_cases, strategy=args.special_strategy)
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

        validation_issues = validate_data(merged, time_col="time")

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
