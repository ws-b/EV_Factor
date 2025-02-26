import os
import glob
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -------------------------------------------------------------------------------
# 0. 환경 세팅
# -------------------------------------------------------------------------------
random.seed(42)

root_dir = r"E:\SamsungSTF\Processed_Data\TripByTrip_soc\Ioniq5"
ocv_table_path = r"D:\SamsungSTF\Data\GSmbiz\NE_SOC_OCV.csv"

# 셀 1개 용량 [Ah]
CELL_CAPACITY = 55.6

# -------------------------------------------------------------------------------
# 1. OCV 테이블 불러오기 및 보간 함수 준비
# -------------------------------------------------------------------------------
df_ocv = pd.read_csv(ocv_table_path)

# OCV -> SOC 보간 함수 (Pack 관점만 사용)
f_soc_from_ocv = interp1d(
    df_ocv["OCV"].values,
    df_ocv["SOC"].values,
    kind='linear',
    fill_value="extrapolate"
)

# -------------------------------------------------------------------------------
# 2. CSV 파일 리스트에서 랜덤하게 10개 선택 (지정 파일은 반드시 포함)
# -------------------------------------------------------------------------------
required_file_names = [
    "bms_01241227999-2023-09-trip-5", "bms_01241227999-2023-09-trip-7", "bms_01241227999-2023-09-trip-6",
    "01241364543-2024-06-trip-8", "01241364543-2024-06-trip-10", "01241364543-2024-06-trip-9"
]

all_csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)

selected_required_files = [
    f for f in all_csv_files
    if any(req in os.path.splitext(os.path.basename(f))[0] for req in required_file_names)
]

remaining_files = [f for f in all_csv_files if f not in selected_required_files]

total_files_to_select = 10
num_random_files_needed = max(0, total_files_to_select - len(selected_required_files))
selected_random_files = random.sample(remaining_files, num_random_files_needed) if num_random_files_needed > 0 else []

selected_files = selected_required_files + selected_random_files

print("최종 선택된 파일들:")
for file in selected_files:
    print(" -", file)

# -------------------------------------------------------------------------------
# 3. 보조 함수 ( cell_volt_list 문자열 → 평균 / 개수 )
# -------------------------------------------------------------------------------
def parse_cell_volt_list(cell_volt_str):
    """예: '1,3,1,3,2,3' → float 변환 후 평균."""
    if pd.isna(cell_volt_str):
        return np.nan
    volt_list = [float(x) for x in cell_volt_str.split(',')]
    return np.mean(volt_list)

def count_cell_volt(cell_volt_str):
    """cell_volt_list의 항목 수를 세기."""
    if pd.isna(cell_volt_str):
        return np.nan
    return len(cell_volt_str.split(','))

# -------------------------------------------------------------------------------
# 4. 파일별 처리 및 플롯
# -------------------------------------------------------------------------------
# 색상 팔레트 정의 (순서대로 사용)
color_palette = [
    "#0073c2",  # 0
    "#efc000",  # 1
    "#cd534c",  # 2
    "#20854e",  # 3
    "#925e9f",  # 4
    "#e18727",  # 5
    "#4dbbd5",  # 6
    "#ee4c97",  # 7
    "#7e6148",  # 8 (Pack 관점 coulomb counting 표시)
    "#747678"   # 9
]

for csv_file in selected_files:
    print(f"[처리 중] {csv_file}")

    df = pd.read_csv(csv_file)

    # 필요한 컬럼 체크
    required_cols = {"time", "cell_volt_list", "pack_volt", "soc", "pack_current"}
    if not required_cols.issubset(df.columns):
        print(f"  필요한 컬럼({required_cols})이 존재하지 않아 스킵합니다.")
        continue

    # (1) time 컬럼을 datetime으로 변환
    df["time_dt"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")

    # (2) cell_volt_avg / pack_volt_normalized 계산
    df["cell_volt_avg"] = df["cell_volt_list"].apply(parse_cell_volt_list)
    df["cell_count"] = df["cell_volt_list"].apply(count_cell_volt)
    df["pack_volt_normalized"] = df["pack_volt"] / df["cell_count"]

    # ---------------------------------------------------------------------------
    # 기존 SOC(Cell, OCV) 계산 부분 제거
    # (원래는 아래와 같이 계산되었음)
    # df["interp_soc"] = f_soc_from_ocv(df["cell_volt_avg"])            # Cell
    # ---------------------------------------------------------------------------
    # Pack 관점 OCV 기반 SOC 계산만 수행
    df["interp_soc_pack"] = f_soc_from_ocv(df["pack_volt_normalized"])

    # (4) SOH 기반 Q Cell
    if "soh" not in df.columns:
        df["soh"] = 100.0

    Q_cell = CELL_CAPACITY * df["soh"] / 100

    # (A) 앞쪽 30행 제거
    if len(df) < 30:
        print("  데이터가 30행 미만이므로 제거 불가. 스킵합니다.")
        continue
    df = df.iloc[30:].reset_index(drop=True)

    n_after_first_cut = len(df)
    # (B) 마지막 31~35번째 행에서 SOC(Pack, OCV)를 확인하여, 0~100 범위 내 유효 값 선택
    if n_after_first_cut < 35:
        continue

    target_idx = n_after_first_cut - 31

    found_valid = False
    for offset in range(29, 24, -1):  # 29, 28, 27, 26, 25
        source_idx = n_after_first_cut - offset
        pack_val = df.at[source_idx, "interp_soc_pack"]

        if 0 <= pack_val <= 100:
            df.at[target_idx, "interp_soc_pack"] = pack_val
            found_valid = True
            break

    if not found_valid:
        continue

    # (C) 마지막 30행 제거
    df = df.iloc[:(n_after_first_cut - 30)].reset_index(drop=True)

    # (D) elapsed time 재설정
    df["elapsed"] = (df["time_dt"] - df["time_dt"].iloc[0]).dt.total_seconds()

    if len(df) < 2:
        print("  뒤 30행 제거 후 남은 데이터가 2행 미만. 스킵합니다.")
        continue

    # ---------------------------------------------------------------------------
    # (E) Coulomb Counting
    # ---------------------------------------------------------------------------
    # 1) 전류 부호 반전
    df["pack_current_corrected"] = -df["pack_current"]

    # -------------------
    # [Cell 관점] 계산
    # -------------------
    # - 2평행(parallel)이므로, 셀 전류 = pack_current_corrected / 2
    df["current_cell"] = df["pack_current_corrected"] / 2.0
    df["dt_s"] = df["elapsed"].diff().fillna(0)
    df["dAh_cell"] = df["current_cell"] * df["dt_s"] / 3600.0
    df["cumAh_cell"] = df["dAh_cell"].cumsum()
    # 기존: start_soc_cell_ocv = df["interp_soc"].iloc[0]
    # -> BMS에서 측정한 SOC를 시작값으로 사용 (SOC(Cell, OCV) 제거)
    start_soc_cell = df["soc"].iloc[0]
    df["soc_cc_cell"] = (
        start_soc_cell
        + (df["cumAh_cell"] - df["cumAh_cell"].iloc[0]) / Q_cell * 100.0
    )

    # -------------------
    # [Pack 관점] 계산
    # -------------------
    PACK_CAPACITY = 2 * CELL_CAPACITY
    Q_pack = PACK_CAPACITY * df["soh"] / 100

    df["dAh_pack"] = df["pack_current_corrected"] * df["dt_s"] / 3600.0
    df["cumAh_pack"] = df["dAh_pack"].cumsum()
    start_soc_pack_ocv = df["interp_soc_pack"].iloc[0]
    df["soc_cc_pack"] = (
        start_soc_pack_ocv
        + (df["cumAh_pack"] - df["cumAh_pack"].iloc[0]) / Q_pack * 100.0
    )

    # ---------------------------------------------------------------------------
    # 그래프 그리기 (4개의 Subplot)
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    file_name = os.path.splitext(os.path.basename(csv_file))[0]
    fig.suptitle(f"File: {file_name}", fontsize=16)

    # (1) subplot: pack_volt_normalized만 표시 (Cell Volt Avg 제거)
    axes[0].plot(df["elapsed"], df["pack_volt_normalized"],
                 label="Cell Volt from Pack Volt", color=color_palette[1], alpha=0.7)
    axes[0].set_ylabel("Voltage [V]")
    axes[0].legend(loc="upper right")

    # (2) Pack Voltage
    axes[1].plot(df["elapsed"], df["pack_volt"],
                 label="Pack Voltage", color=color_palette[2])
    axes[1].set_ylabel("Pack Voltage [V]")
    axes[1].legend(loc="upper right")

    # (3) Pack Current
    axes[2].plot(df["elapsed"], df["pack_current"],
                 label="Pack Current", color=color_palette[3])
    axes[2].set_ylabel("Pack Current [A]")
    axes[2].legend(loc="upper right")

    # (4) SOC 관련 Subplot (SOC(Cell, OCV) 관련 부분 제거)
    # - (a) BMS에서 측정한 SOC 라인
    axes[3].plot(df["elapsed"], df["soc"],
                 label="SOC(BMS)", color=color_palette[4], linestyle='-')
    # - (b) SOC(Pack, OCV)의 처음/마지막 점
    axes[3].scatter(df["elapsed"].iloc[0],
                    df["interp_soc_pack"].iloc[0],
                    color=color_palette[6],
                    marker='x', s=70,
                    label="SOC(Pack, OCV)")
    axes[3].scatter(df["elapsed"].iloc[-1],
                    df["interp_soc_pack"].iloc[-1],
                    color=color_palette[6],
                    marker='x', s=70)
    # # - (c) SOC(Coulomb Counting, Cell) 라인
    # axes[3].plot(df["elapsed"], df["soc_cc_cell"],
    #              label="SOC(CC, Cell)",
    #              color=color_palette[7], linestyle='--')
    # - (d) SOC(Coulomb Counting, Pack) 라인
    axes[3].plot(df["elapsed"], df["soc_cc_pack"],
                 label="SOC(CC, Pack)",
                 color=color_palette[8], linestyle='--')

    axes[3].set_ylabel("SOC [%]")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("Elapsed Time [s]")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------------
    # 최종 누적값 간단 확인 (Cell 관점)
    # ---------------------------------------------------------------------------
    netAh_cell = df["cumAh_cell"].iloc[-1] - df["cumAh_cell"].iloc[0]
    net_coulomb_soc_delta_cell = (netAh_cell / CELL_CAPACITY) * 100.0
    print(f"  >> [Cell] netAh (start~end) = {netAh_cell:.3f} Ah")
    print(f"  >> [Cell] ΔSOC(by coulomb) = {net_coulomb_soc_delta_cell:+.2f}%")
    print(f"  >> [Cell] Start SOC(BMS) = {start_soc_cell:.2f}%")
    print(f"  >> [Cell] Final SOC(CC) = {df['soc_cc_cell'].iloc[-1]:.2f}%")

    # ---------------------------------------------------------------------------
    # 최종 누적값 간단 확인 (Pack 관점)
    # ---------------------------------------------------------------------------
    netAh_pack = df["cumAh_pack"].iloc[-1] - df["cumAh_pack"].iloc[0]
    net_coulomb_soc_delta_pack = (netAh_pack / (PACK_CAPACITY)) * 100.0
    print(f"  >> [Pack] netAh (start~end) = {netAh_pack:.3f} Ah")
    print(f"  >> [Pack] ΔSOC(by coulomb) = {net_coulomb_soc_delta_pack:+.2f}%")
    print(f"  >> [Pack] Start SOC(Pack, OCV) = {start_soc_pack_ocv:.2f}%")
    print(f"  >> [Pack] Final SOC(CC) = {df['soc_cc_pack'].iloc[-1]:.2f}%")
