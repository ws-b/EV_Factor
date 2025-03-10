import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------
# (A) 사용자 설정
# ------------------------------------------------
directory = r"E:\SamsungSTF\Processed_Data\Merged_period\Ioniq5"

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

# ------------------------------------------------
# (B) 메인: 폴더 내 전처리 CSV 파일 불러와 Plot (SOC vs Pack Voltage)
# ------------------------------------------------
pattern = r"(?:bms|bms_altitude)_(\d+)_d(\d+)\.csv"
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 필요한 컬럼 전체 (추가: chrg_cable_conn, fast_chrg_port_conn)
needed_cols = [
    'time', 'pack_volt', 'pack_current', 'pack_power_kW',
    'cum_input_kWh', 'cum_output_kWh', 'storage_kWh',
    'estimated_capacity_kWh', 'estimated_capacity_Ah', 'init_soc', 'final_soc',
    'net_kWh', 'chrg_cable_conn', 'fast_chrg_port_conn'
]

for file in tqdm(files[50:54]):
    match = re.match(pattern, file)
    if not match:
        print(f"[WARN] 파일명 형식 불일치(전처리 파일 아님): {file}")
        continue

    device_id = match.group(1)
    week_idx = int(match.group(2))
    file_path = os.path.join(directory, file)

    # 전처리된 CSV 로드
    df = pd.read_csv(file_path)

    # (1) 필요한 컬럼 체크
    missing_cols = [col for col in needed_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARN] 필수 컬럼 부족 -> 스킵: {file} / 부족 컬럼={missing_cols}")
        continue

    # 시간 컬럼 파싱 및 정렬
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # (초기 값들)
    estimated_capacity_kWh = df['estimated_capacity_kWh'].iloc[0]
    init_soc = df['init_soc'].iloc[0]
    final_soc = df['final_soc'].iloc[0]

    # 마지막 몇개 샘플(노이즈 등) 제외해서 사용
    if len(df) > 15:
        df_plot = df.iloc[:-15].copy()
    else:
        df_plot = df.copy()

    # 충전/방전 상태에 따른 DataFrame 분리
    df_discharging = df_plot[df_plot['chrg_cable_conn'] == 0]
    df_charging_normal = df_plot[(df_plot['chrg_cable_conn'] == 1) & (df_plot['fast_chrg_port_conn'] == 0)]
    df_charging_fast = df_plot[(df_plot['chrg_cable_conn'] == 1) & (df_plot['fast_chrg_port_conn'] == 1)]

    # 각 상태별 SOC 계산 (초기 estimated_capacity_kWh 사용)
    soc_discharging = (df_discharging['soc_cc']) * 100
    soc_charging_normal = (df_charging_normal['soc_cc']) * 100
    soc_charging_fast = (df_charging_fast['soc_cc']) * 100

    # 플롯 생성
    plt.figure(figsize=(10, 6))
    if not df_discharging.empty:
        plt.plot(soc_discharging, df_discharging['pack_volt'], color=color_palette[0], label='discharging')
    if not df_charging_normal.empty:
        plt.plot(soc_charging_normal, df_charging_normal['pack_volt'], color=color_palette[1], label='slow charging')
    if not df_charging_fast.empty:
        plt.plot(soc_charging_fast, df_charging_fast['pack_volt'], color=color_palette[2], label='fast charging')

    plt.xlabel('SOC (%)')
    plt.ylabel('Pack Voltage [V]')
    plt.title(f"[Device={device_id}, Week={week_idx}] Pack Voltage vs SOC\n"
              f"Init SOC={init_soc:.3f}, Final SOC={final_soc:.3f}")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\n시각화 작업 완료.")
