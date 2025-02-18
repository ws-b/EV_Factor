import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------
# (0) 파라미터 설정 / 색상 팔레트
# ------------------------------------------------
base_dir = r"E:\SamsungSTF\Processed_Data\Merged_period\Ioniq5"
ocv_file_path = r"D:\SamsungSTF\Data\GSmbiz\NE_SOC_OCV.csv"

# ------------------------------------------------
# (1) OCV-SOC 테이블 로드 및 인터폴레이터 준비
# ------------------------------------------------
ocv_df = pd.read_csv(ocv_file_path)  # 컬럼: ['SOC','OCV'] (예: SOC=0~100, OCV=단위 V)
ocv_soc_array = ocv_df['SOC'].values / 100.0  # 0~1 범위로 변환
ocv_volt_array = ocv_df['OCV'].values

def get_soc_from_ocv(cell_voltage):
    """
    OCV-SOC 커브를 기반으로 SOC 추정 (선형 보간)
    cell_voltage: 셀 단위 평균 전압
    return: SOC (0~1 범위)
    """
    return np.interp(cell_voltage, ocv_volt_array, ocv_soc_array)

# ------------------------------------------------
# (2) 파일 목록 읽기
# ------------------------------------------------
pattern = r"(?:bms|bms_altitude)_(\d+)_d(\d+)\.csv"
files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]

eta_list = []  # 계산된 eta를 저장할 리스트

for file in tqdm(files):
    match = re.match(pattern, file)
    if not match:
        # 파일명 규칙이 맞지 않는 경우
        continue

    device_id = match.group(1)
    week_idx = int(match.group(2))

    file_path = os.path.join(base_dir, file)

    # CSV 로드
    try:
        df = pd.read_csv(file_path)
    except:
        continue

    # 필수 컬럼 체크
    needed_cols = ['time', 'pack_volt', 'pack_current', 'cell_volt_list']
    if any(col not in df.columns for col in needed_cols):
        continue

    # time 정렬 및 dt 계산
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df['time'].isna().all():
        # time이 전부 NaN이면 스킵
        continue

    df['dt_s'] = df['time'].diff().dt.total_seconds().fillna(0)
    threshold_sec = 10
    max_sec_for_integration = 2
    df.loc[df['dt_s'] > threshold_sec, 'dt_s'] = max_sec_for_integration
    df['dt_h'] = df['dt_s'] / 3600.0

    # 배터리 용량 추정
    cell_volt_str = df['cell_volt_list'].iloc[0]
    cell_list = cell_volt_str.split(',')
    cell_count = len(cell_list)

    soh = df['soh'].iloc[0] if 'soh' in df.columns else 100.0
    if cell_count == 180:
        battery_capacity_kWh = 72.6 * (soh / 100.0)
    else:
        battery_capacity_kWh = 77.4 * (soh / 100.0)

    # Power, Input, Output
    df['pack_power_kW'] = (df['pack_volt'] * df['pack_current']) / 1000.0
    df['delta_energy_kWh'] = df['pack_power_kW'] * df['dt_h']

    df['input_kWh'] = np.where(df['pack_power_kW'] < 0,
                               -df['delta_energy_kWh'],
                               0)
    df['output_kWh'] = np.where(df['pack_power_kW'] > 0,
                                df['delta_energy_kWh'],
                                0)

    # 총 입력량, 총 출력량
    total_input_kWh = df['input_kWh'].sum()
    total_output_kWh = df['output_kWh'].sum()

    # 초기 SOC 계산 (CSV 첫 행)
    first_pack_volt = df['pack_volt'].iloc[0]
    first_cell_volt = first_pack_volt / cell_count
    init_soc = get_soc_from_ocv(first_cell_volt)
    init_storage_kWh = init_soc * battery_capacity_kWh

    # 최종 SOC 계산 (마지막 15행 시작 또는 부족 시 마지막 행)
    if len(df) <= 15:
        final_pack_volt = df['pack_volt'].iloc[-1]
    else:
        final_pack_volt = df['pack_volt'].iloc[-15]
    final_cell_volt = final_pack_volt / cell_count
    final_soc = get_soc_from_ocv(final_cell_volt)
    final_storage_kWh = final_soc * battery_capacity_kWh

    # eta = (initial storage + input) / (output + final storage)
    denominator = init_storage_kWh + total_input_kWh
    if denominator <= 0:
        # 0 또는 음수면 계산 불가(이상치) → 스킵
        continue

    eta = (total_output_kWh + final_storage_kWh) / denominator

    # 계산된 eta를 리스트에 저장
    eta_list.append(eta)

# 모든 파일 처리 후 히스토그램
if len(eta_list) == 0:
    print("유효한 eta가 없습니다. 작업 종료.")
else:
    plt.figure(figsize=(8,6))
    plt.hist(eta_list, bins=50, color='skyblue', edgecolor='k')
    plt.title("Eta Histogram")
    plt.xlabel("eta")
    plt.ylabel("File Count")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"총 {len(eta_list)}개 파일에 대한 eta 히스토그램을 표시했습니다.")
