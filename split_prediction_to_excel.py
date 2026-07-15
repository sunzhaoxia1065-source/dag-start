#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 prediction_detail CSV 文件按日期拆分为 Excel 多工作表文件。

用法:
    python tools/split_prediction_to_excel.py --input prediction_detail_xxx.csv [--output result.xlsx]

输入 CSV 需包含 time, actual_power, predicted_power 列。
输出 Excel 中每个工作表对应一天，以 YYYY-MM-DD 命名。
"""

import argparse
import os
import sys

import pandas as pd


def load_and_validate(input_path: str) -> pd.DataFrame:
    """读取 CSV 并校验必需列是否存在。

    Args:
        input_path: CSV 文件路径。

    Returns:
        加载后的 DataFrame。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺少必需列。
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")

    df = pd.read_csv(input_path)

    required = {"time", "actual_power", "predicted_power"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必需列: {missing}，当前列: {list(df.columns)}")

    return df


def split_and_write(df: pd.DataFrame, output_path: str) -> None:
    """按日期拆分 DataFrame 并写入 Excel 多工作表。

    Args:
        df: 包含 time, actual_power, predicted_power 的 DataFrame。
        output_path: 输出 Excel 文件路径。
    """
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.strftime("%Y-%m-%d")

    dates = sorted(df["date"].unique())
    total_rows = len(df)
    assigned_rows = 0

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for date_str in dates:
            day_df = df[df["date"] == date_str][["time", "actual_power", "predicted_power"]].copy()
            day_df.reset_index(drop=True, inplace=True)
            day_df.to_excel(writer, sheet_name=date_str, index=False)
            assigned_rows += len(day_df)

    # 数据完整性校验
    if assigned_rows != total_rows:
        print(f"[警告] 数据分配不一致: 总行数={total_rows}, 分配行数={assigned_rows}")
    else:
        print(f"[通过] 数据完整性校验: {total_rows} 条记录全部正确分配到 {len(dates)} 个工作表")

    print(f"已输出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="将 prediction_detail CSV 按日期拆分为 Excel 多工作表")
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", default=None, help="输出 Excel 文件路径（默认: 输入文件名.xlsx）")
    args = parser.parse_args()

    df = load_and_validate(args.input)

    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        output = base + "_daily.xlsx"

    split_and_write(df, output)


if __name__ == "__main__":
    main()
