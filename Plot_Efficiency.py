import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.dates import DateFormatter

# ------------------------------------------------
# (A) 사용자 설정
# ------------------------------------------------
directory = r"E:\SamsungSTF\Processed_Data\Merged_period\Ioniq5"

color_palette = [
    "#0073c2",  # 파랑
    "#efc000",  # 노랑
    "#cd534c",  # 빨강
    "#20854e",  # 녹색
    "#925e9f",  # 보라
    "#e18727",  # 주황
    "#4dbbd5",  # 하늘
    "#ee4c97",  # 분홍
    "#7e6148",  # 갈색
    "#747678"   # 회색
]
marker_palette = ['#FF5733', '#33FF57']  # 첫 번째: 초기, 두 번째: 최종

# ------------------------------------------------
# (B) 메인: 폴더 내 전처리 CSV 파일 불러와 Plot
# ------------------------------------------------
pattern = r"(?:bms|bms_altitude)_(\d+)_d(\d+)\.csv"
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 필요한 컬럼 전체
needed_cols = [
    'time', 'pack_volt', 'pack_current', 'pack_power_kW',
    'cum_input_kWh', 'cum_output_kWh', 'storage_kWh',
    'estimated_capacity_kWh', 'init_soc', 'final_soc',
    'net_kWh'
]

for file in tqdm(files[:3]):
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

    # 나머지 로직
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    estimated_capacity_kWh = df['estimated_capacity_kWh'].iloc[0]  # 동일값
    init_soc = df['init_soc'].iloc[0]
    final_soc = df['final_soc'].iloc[0]

    # 마지막 몇개 샘플(노이즈 등) 제외해서 Plot할 메인 데이터
    if len(df) > 15:
        df_plot = df.iloc[:-15].copy()
    else:
        df_plot = df.copy()

    # x축 범위 설정
    time_margin = pd.Timedelta(hours=5)
    if len(df_plot) > 0:
        x_min = df_plot['time'].min() - time_margin
        x_max = df_plot['time'].max() + time_margin
    else:
        x_min = pd.NaT
        x_max = pd.NaT

    # ----------------------------
    # Plot 1: 여러 항목 세로 분할
    # ----------------------------
    fig, axes = plt.subplots(6, 1, figsize=(12, 22), sharex=True)
    fig.suptitle(f"[Device={device_id}, Week={week_idx}]  "
                 f"Estimated Capacity(Avg)={estimated_capacity_kWh:.2f} kWh\n"
                 f"Init SOC={init_soc:.3f}, Final SOC={final_soc:.3f}",
                 fontsize=16)

    axes[0].plot(df_plot['time'], df_plot['pack_volt'],
                 color=color_palette[0], label='Pack Volt [V]')
    axes[0].set_ylabel("Pack Volt [V]")
    axes[0].legend(loc='upper left')

    axes[1].plot(df_plot['time'], df_plot['pack_current'],
                 color=color_palette[1], label='Pack Current [A]')
    axes[1].set_ylabel("Pack Current [A]")
    axes[1].legend(loc='upper left')

    axes[2].plot(df_plot['time'], df_plot['cum_input_kWh'],
                 color=color_palette[2], label='Input [kWh]')
    axes[2].set_ylabel("Input [kWh]")
    axes[2].legend(loc='upper left')

    axes[3].plot(df_plot['time'], df_plot['cum_output_kWh'],
                 color=color_palette[3], label='Output [kWh]')
    axes[3].set_ylabel("Output [kWh]")
    axes[3].legend(loc='upper left')

    axes[4].plot(df_plot['time'], df_plot['storage_kWh'],
                 color=color_palette[4], label='Storage [kWh]')
    # 초기/최종 시점 표시
    if len(df_plot) > 0:
        axes[4].scatter(df_plot['time'].iloc[0],
                        df_plot['storage_kWh'].iloc[0],
                        color=marker_palette[0], marker='o', s=25, zorder=5, label='Init SOC')
        axes[4].scatter(df_plot['time'].iloc[-1],
                        df_plot['storage_kWh'].iloc[-1],
                        color=marker_palette[1], marker='o', s=25, zorder=5, label='Final SOC')
    axes[4].set_ylabel("Storage [kWh]")
    axes[4].legend(loc='upper left')

    axes[5].plot(df_plot['time'], df_plot['pack_power_kW'],
                 color=color_palette[5], label='Pack Power [kW]')
    axes[5].set_ylabel("Pack Power [kW]")
    axes[5].set_xlabel("Time")
    axes[5].legend(loc='upper left')

    date_formatter = DateFormatter('%m-%d %H:%M')
    axes[-1].xaxis.set_major_formatter(date_formatter)

    if len(df_plot) > 0:
        axes[-1].set_xlim(x_min, x_max)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # ----------------------------
    # Plot 2: Storage / Net / Input / Output
    # ----------------------------
    if len(df_plot) > 0:
        time_init = df_plot['time'].iloc[0]
        time_final_plot = df_plot['time'].iloc[-1]
        init_storage_kWh_plot = df_plot['storage_kWh'].iloc[0]
        final_storage_kWh_plot = df_plot['storage_kWh'].iloc[-1]
    else:
        time_init = pd.NaT
        time_final_plot = pd.NaT
        init_storage_kWh_plot = np.nan
        final_storage_kWh_plot = np.nan

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.xaxis.set_major_formatter(date_formatter)

    # (a) storage_kWh
    ax2.plot(df_plot['time'], df_plot['storage_kWh'],
             color=color_palette[4], alpha=0.5,label='Storage [kWh]')

    # (b) net_kWh
    ax2.plot(df_plot['time'], df_plot['net_kWh'],
             color=color_palette[6], alpha=0.5,label='Net kWh')

    # (c) input_kWh / output_kWh
    ax2.plot(df_plot['time'], df_plot['cum_input_kWh'],
             color=color_palette[2], alpha=0.5, label='Input [kWh]')
    ax2.plot(df_plot['time'], df_plot['cum_output_kWh'],
             color=color_palette[3], alpha=0.5, label='Output [kWh]')

    # 초기/최종 SOC 표시 (storage_kWh 기준)
    ax2.scatter(time_init, init_storage_kWh_plot,
                color=marker_palette[0], marker='o', s=25, zorder=5, label='Init SOC')
    ax2.scatter(time_final_plot, final_storage_kWh_plot,
                color=marker_palette[1], marker='o', s=25, zorder=5, label='Final SOC')

    if len(df_plot) > 0:
        ax2.set_xlim(time_init - time_margin, time_final_plot + time_margin)

    ax2.set_title(f"Storage / Net / Input / Output\n"
                  f"Init SOC={init_soc:.3f}, Final SOC={final_soc:.3f}, "
                  f"Estimated Capacity={estimated_capacity_kWh:.2f} kWh")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Energy [kWh]")
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

print("\n시각화 작업 완료.")
