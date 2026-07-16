#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算空气密度属性列，生成仅含 air_density 的新 CSV 文件。

公式: air_density = surface_pressure / (temperature + 273.15)
其中温度单位为摄氏度，需转为开尔文。

用法:
    python tools/compute_air_density.py --input data.csv --output air_density.csv
    python tools/compute_air_density.py --input data.csv --pressure-col surface_pressure_surface --temp-col temperature_surface_2_metre
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def load_and_validate(input_path: str, pressure_col: str, temp_col: str) -> pd.DataFrame:
    """读取 CSV 并校验必需列。

    Args:
        input_path: CSV 文件路径。
        pressure_col: 气压列名。
        temp_col: 温度列名。

    Returns:
        DataFrame。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺少必需列或数据含非法值。
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")

    df = pd.read_csv(input_path)

    missing = []
    if pressure_col not in df.columns:
        missing.append(pressure_col)
    if temp_col not in df.columns:
        missing.append(temp_col)
    if missing:
        raise ValueError(f"CSV 缺少必需列: {missing}，当前列: {list(df.columns)}")

    # 检查数值类型
    for col in [pressure_col, temp_col]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"列 '{col}' 不是数值类型: {df[col].dtype}")

    # 检查 NaN
    nan_counts = {pressure_col: df[pressure_col].isna().sum(),
                  temp_col: df[temp_col].isna().sum()}
    for col, cnt in nan_counts.items():
        if cnt > 0:
            print(f"[警告] 列 '{col}' 含 {cnt} 个 NaN 值，将跳过对应行")

    return df


def compute_air_density(df: pd.DataFrame, pressure_col: str, temp_col: str) -> pd.DataFrame:
    """计算空气密度。

    Args:
        df: 原始 DataFrame。
        pressure_col: 气压列名（Pa）。
        temp_col: 温度列名（°C）。

    Returns:
        新 DataFrame，仅含 air_density 列。
    """
    pressure = df[pressure_col].to_numpy(dtype=np.float64)
    temperature = df[temp_col].to_numpy(dtype=np.float64)

    # 摄氏度 → 开尔文
    temp_k = temperature + 273.15

    # air_density = P / T  (理想气体近似)
    air_density = np.empty_like(pressure)
    air_density[:] = np.nan

    valid_mask = (~np.isnan(pressure)) & (~np.isnan(temp_k)) & (temp_k > 0)
    air_density[valid_mask] = pressure[valid_mask] / temp_k[valid_mask]

    invalid_count = int(np.sum(~valid_mask))
    if invalid_count > 0:
        print(f"[警告] {invalid_count} 行因 NaN 或温度≤0K 无法计算，设为 NaN")

    # 保存 time 列（如果存在）用于索引
    result = pd.DataFrame()
    if "time" in df.columns:
        result["time"] = df["time"]
    result["air_density"] = air_density

    return result


def main():
    parser = argparse.ArgumentParser(description="计算 air_density 并输出新 CSV")
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", default=None, help="输出 CSV 文件路径（默认: 输入文件名_air_density.csv）")
    parser.add_argument("--pressure-col", default="surface_pressure_surface",
                        help="气压列名（默认: surface_pressure_surface）")
    parser.add_argument("--temp-col", default="temperature_surface_2_metre",
                        help="温度列名（默认: temperature_surface_2_metre）")
    args = parser.parse_args()

    df = load_and_validate(args.input, args.pressure_col, args.temp_col)
    print(f"已读取 {len(df)} 行，列: {list(df.columns)}")

    result = compute_air_density(df, args.pressure_col, args.temp_col)

    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        output = base + "_air_density.csv"

    result.to_csv(output, index=False)
    print(f"已输出: {output}（{len(result)} 行）")


if __name__ == "__main__":
    main()
