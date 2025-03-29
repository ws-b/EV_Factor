import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------
# (A) 사용자 설정
# ------------------------------------------------
directory = r"D:\SamsungSTF\Processed_Data\Merged_period_final\Ioniq5"

color_palette = [
    "#0073c2",  # 파랑: 방전
    "#efc000",  # 노랑: 일반 충전
    "#cd534c",  # 빨강: 고속 충전
    "#20854e",  # 녹색 (추가 용도)
    "#925e9f",  # 보라
    "#e18727",  # 주황
    "#4dbbd5",  # 하늘
    "#ee4c97",  # 분홍
    "#7e6148",  # 갈색
    "#747678"   # 회색
]

# 필요한 컬럼 전체 (추가: chrg_cable_conn, fast_chrg_port_conn)
needed_cols = [
    'time', 'pack_volt', 'pack_current', 'pack_power_kW',
    'cum_input_kWh', 'cum_output_kWh', 'storage_kWh',
    'estimated_capacity_kWh', 'estimated_capacity_Ah', 'init_soc', 'final_soc',
    'net_kWh', 'chrg_cable_conn', 'fast_chrg_port_conn'
]

directory = r"D:\SamsungSTF\Processed_Data\Merged_period_final\Ioniq5"
pattern = r"(?:bms|bms_altitude)_(\d+)_d(\d+)\.csv"
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

for file in tqdm(files[50:54]):
    match = re.match(pattern, file)
    if not match:
        continue
    device_id = match.group(1)
    week_idx = int(match.group(2))

    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)

    # (1) 필요한 컬럼 체크
    missing_cols = [col for col in needed_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARN] 필수 컬럼 부족 -> 스킵: {file} / 부족 컬럼={missing_cols}")
        continue

    if 'soc_cc' not in df.columns or 'pack_volt' not in df.columns:
        continue

    # 시간순 정렬
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 준비: SOC(%) 및 상태분류
    df['SOC_%'] = df['soc_cc'] * 100
    # 충전/방전 판별
    df['state'] = 'discharging'
    df.loc[(df['chrg_cable_conn'] == 1) & (df['fast_chrg_port_conn'] == 0), 'state'] = 'charging_normal'
    df.loc[(df['chrg_cable_conn'] == 1) & (df['fast_chrg_port_conn'] == 1), 'state'] = 'charging_fast'

    # Plot (scatter 방식)
    plt.figure(figsize=(10, 6))

    # 방전
    df_dis = df[df['state'] == 'discharging']
    plt.scatter(df_dis['SOC_%'], df_dis['pack_volt'], s=4, color=color_palette[0], label='discharging')

    # 일반 충전
    df_chg_normal = df[df['state'] == 'charging_normal']
    plt.scatter(df_chg_normal['SOC_%'], df_chg_normal['pack_volt'], s=4, color=color_palette[1], label='slow charging')

    # 급속 충전
    df_chg_fast = df[df['state'] == 'charging_fast']
    plt.scatter(df_chg_fast['SOC_%'], df_chg_fast['pack_volt'], s=4, color=color_palette[2], label='fast charging')

    plt.xlabel('SOC (%)')
    plt.ylabel('Pack Voltage [V]')
    plt.title(f"[Device={device_id}, Week={week_idx}] Pack Voltage vs SOC (Scatter)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
