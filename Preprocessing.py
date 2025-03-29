import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

# ------------------------------------------------
# (A) 사용자 설정
# ------------------------------------------------
base_dir = r"D:\SamsungSTF\Processed_Data\Merged_period_final\Ioniq5"
ocv_file_path = r"D:\SamsungSTF\Data\NE_SOC_OCV.csv"

REST_GAP_SEC = 7200  # 2시간 (REST 구간 판단 기준)
THRESHOLD_SEC = 10
MAX_SEC_FOR_INTEGRATION = 2
SOC_USAGE_MIN = 0.30  # ΔSOC 30% 기준

# 스킵된 파일에서 최종적으로 "유지할 컬럼" 목록:
cols_to_keep = [
    "time", "speed", "acceleration", "ext_temp", "int_temp",
    "chrg_cnt", "chrg_cnt_q", "cumul_energy_chrgd", "cumul_energy_chrgd_q",
    "mod_temp_list", "odometer", "op_time", "soc", "soh",
    "chrg_cable_conn", "fast_chrg_port_conn", "pack_volt", "pack_current", "cell_volt_list",
    "min_deter", "Power_data"
]

columns_order = [
    "time", "speed", "acceleration", "ext_temp", "int_temp", "chrg_cnt",
    "chrg_cnt_q", "cumul_energy_chrgd", "cumul_energy_chrgd_q", "mod_temp_list",
    "odometer", "op_time", "soc", "soh", "chrg_cable_conn", "fast_chrg_port_conn",
    "pack_volt", "pack_current", "cell_volt_list", "min_deter", "Power_data",
    "dt_s", "dt_h", "pack_power_kW", "delta_energy_kWh", "input_kWh", "output_kWh",
    "cum_input_kWh", "cum_output_kWh", "soc_cc", "init_soc", "final_soc",
    "estimated_capacity_kWh", "estimated_capacity_Ah", "storage_kWh", "net_kWh",
    # 새로 추가할 효율 컬럼
    "battery_eta"
]

# ------------------------------------------------
# (B) OCV-SOC 테이블 로드
# ------------------------------------------------
ocv_df = pd.read_csv(ocv_file_path)  # 컬럼: ['SOC', 'OCV']
ocv_soc_array = ocv_df['SOC'].values / 100
ocv_volt_array = ocv_df['OCV'].values

def get_soc_from_ocv(cell_voltage):
    return np.interp(cell_voltage, ocv_volt_array, ocv_soc_array)

# ------------------------------------------------
# (C) 세그먼트 분할 함수
# ------------------------------------------------
def segment_data(df):
    segs = []
    n = len(df)
    if n == 0:
        return segs

    start_i = 0
    for i in range(n - 1):
        time_gap = (df['time'].iloc[i + 1] - df['time'].iloc[i]).total_seconds()
        if time_gap >= REST_GAP_SEC:
            segs.append({
                'start_idx': start_i,
                'end_idx': i + 1,
                'reason': 'rest'
            })
            start_i = i + 1

    if start_i < n:
        segs.append({
            'start_idx': start_i,
            'end_idx': n,
            'reason': 'end'
        })
    return segs

# ------------------------------------------------
# (D) 세그먼트별 정보 계산
# ------------------------------------------------
def compute_segment_info(df, seg):
    s = seg['start_idx']
    e = seg['end_idx']
    n = len(df)

    if e >= n or (e - s) < 2:
        return None

    sub = df.iloc[s:e].copy()

    # cell_volt_list 값 가져오기
    cell_volt_val = sub['cell_volt_list'].iloc[0]

    # cell_volt_list가 비어있거나 NaN이면 이 세그먼트는 skip
    if pd.isna(cell_volt_val) or str(cell_volt_val).strip() == "":
        print(f"[WARN] 세그먼트 시작 인덱스 {s}의 cell_volt_list 값이 비어있어 스킵합니다.")
        return None

    cell_volt_str = str(cell_volt_val)
    cell_count = len(cell_volt_str.split(','))

    first_volt = sub['pack_volt'].iloc[0]
    init_soc = get_soc_from_ocv(first_volt / cell_count)

    # 추가로 'time' 차이 체크
    final_volt = df['pack_volt'].iloc[e]

    # --- 추가 검증 ---
    time_e = df['time'].iloc[e]
    time_e_minus1 = df['time'].iloc[e - 1]
    dt_rest = (time_e - time_e_minus1).total_seconds()

    if dt_rest >= REST_GAP_SEC:
        # row e는 2시간 이상 지난 샘플 -> OCV 간주
        final_soc = get_soc_from_ocv(final_volt / cell_count)
        delta_soc = abs(init_soc - final_soc)
    else:
        print("[INFO] final_volt 시점이 REST가 아님 -> 세그먼트 계산 스킵")
        return None

    sub['current_ah'] = sub['pack_current'] * sub['dt_h']
    net_ah = sub['current_ah'].sum()

    sub['voltamp_ah'] = sub['pack_volt'] * sub['current_ah']
    voltamp_ah = sub['voltamp_ah'].sum()

    return {
        'start_idx': s,
        'end_idx': e,
        'init_soc': init_soc,
        'final_soc': final_soc,
        'delta_soc': delta_soc,
        'net_ah': net_ah,
        'voltamp_ah': voltamp_ah,
        'cell_count': cell_count
    }

# ------------------------------------------------
# (E) 세그먼트 병합 & 용량 추정
# ------------------------------------------------
def merge_segments_if_needed(seg_info_list, soc_min=0.30):
    """
    세그먼트별로 추정한 net_ah, delta_soc 등을 병합하여,
    ΔSOC ≥ soc_min 인 구간이 있으면 용량(Ah, kWh)과 Q_battery (Ah)를 추정.
    """
    merged_results = []
    if not seg_info_list:
        return merged_results

    def finalize_buffer(buf):
        net_ah = buf['net_ah']
        voltamp_ah = buf['voltamp_ah']
        dsoc = buf['delta_soc']

        if dsoc > 0 and abs(net_ah) > 1e-9:
            # 배터리 용량 (Ah 단위)
            cap_ah = abs(net_ah) / dsoc
            # 평균 전압 (V)
            mean_volt = voltamp_ah / net_ah
            # 배터리 용량 (kWh 단위)
            cap_kwh = cap_ah * mean_volt / 1000.0
        else:
            cap_ah = 0.0
            cap_kwh = 0.0

        buf['estimated_capacity_Ah'] = cap_ah
        buf['estimated_capacity_kWh'] = cap_kwh

        return buf

    buffer = seg_info_list[0].copy()
    i = 1
    n = len(seg_info_list)

    while i < n:
        current = seg_info_list[i]
        buffer['end_idx'] = current['end_idx']
        buffer['final_soc'] = current['final_soc']
        buffer['delta_soc'] = abs(buffer['init_soc'] - buffer['final_soc'])
        buffer['net_ah'] += current['net_ah']
        buffer['voltamp_ah'] += current['voltamp_ah']

        if buffer['delta_soc'] >= soc_min:
            merged_results.append(finalize_buffer(buffer))
            if i + 1 < n:
                buffer = seg_info_list[i + 1].copy()
                i += 2
            else:
                buffer = None
                break
        else:
            i += 1

    if buffer is not None:
        if buffer['delta_soc'] >= soc_min:
            merged_results.append(finalize_buffer(buffer))

    return merged_results


# ------------------------------------------------
# (F) 메인 로직 + ΔSOC≥30% 구간 부족한 파일 추적
# ------------------------------------------------
pattern = r"(?:bms|bms_altitude)_(\d+)_d(\d+)\.csv"
files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]

skipped_files = []  # ΔSOC≥30% 구간이 없어 스킵된 파일들

for file in tqdm(files):
    match = re.match(pattern, file)
    if not match:
        print(f"[WARN] 파일명 형식 불일치: {file}")
        continue

    device_id = match.group(1)
    week_idx = int(match.group(2))
    file_path = os.path.join(base_dir, file)

    df = pd.read_csv(file_path)
    needed_cols = ['time', 'pack_volt', 'pack_current', 'cell_volt_list']
    if any(col not in df.columns for col in needed_cols):
        print(f"[WARN] 필수 컬럼({needed_cols})이 없어 스킵: {file}")
        continue

    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # dt_s, dt_h 계산
    df['dt_s'] = df['time'].diff().dt.total_seconds().fillna(0)
    df.loc[df['dt_s'] > THRESHOLD_SEC, 'dt_s'] = MAX_SEC_FOR_INTEGRATION
    df['dt_h'] = df['dt_s'] / 3600.0

    # 세그먼트 분할
    segments = segment_data(df)
    if not segments:
        print(f"[WARN] 세그먼트 분할 결과가 없습니다: {file}")
        continue

    # 세그먼트별 정보
    seg_info_list = []
    for seg in segments:
        info = compute_segment_info(df, seg)
        if info is not None:
            seg_info_list.append(info)

    if not seg_info_list:
        print(f"[INFO] {file}: 모든 세그먼트가 유효 계산 없이 스킵됨.")
        continue

    # ΔSOC≥30% 구간 병합 & 용량 추정
    merged_results = merge_segments_if_needed(seg_info_list, SOC_USAGE_MIN)
    if not merged_results:
        # ΔSOC≥30% 구간이 없어서 스킵
        print(f"[INFO] {file}: ΔSOC≥30% 구간 없어 용량 추정 불가.")
        skipped_files.append(file)
        continue
    else:
        # 여러 세그먼트의 평균용량
        cap_kwh_list = [mr['estimated_capacity_kWh'] for mr in merged_results]
        estimated_capacity_kWh = np.mean(cap_kwh_list)
        cap_ah_list = [mr['estimated_capacity_Ah'] for mr in merged_results]
        estimated_capacity_Ah = np.mean(cap_ah_list)

    # pack_power_kW 및 에너지 흐름 계산
    df['pack_power_kW'] = (df['pack_volt'] * df['pack_current']) / 1000
    df['delta_energy_kWh'] = df['pack_power_kW'] * df['dt_h']
    df['input_kWh'] = np.where(df['pack_power_kW'] < 0,
                               -df['delta_energy_kWh'], 0)
    df['output_kWh'] = np.where(df['pack_power_kW'] > 0,
                                df['delta_energy_kWh'], 0)
    df['cum_input_kWh'] = df['input_kWh'].cumsum()
    df['cum_output_kWh'] = df['output_kWh'].cumsum()

    # 최빈 cell_count
    valid_counts = df['cell_volt_list'].dropna().apply(lambda s: len(s.split(',')))
    if not valid_counts.empty:
        cell_count = valid_counts.mode()[0]
    else:
        raise ValueError("유효한 cell_volt_list 값이 없습니다.")

    # 전체 구간 초기/최종 SOC (OCV)
    n_df = len(df)
    if n_df < 15:
        final_pack_volt = df['pack_volt'].iloc[-1]
    else:
        final_pack_volt = df['pack_volt'].iloc[-15]

    init_soc = get_soc_from_ocv(df['pack_volt'].iloc[0] / cell_count)
    final_soc = get_soc_from_ocv(final_pack_volt / cell_count)

    # net_kWh: (초기 저장량 + 입력 - 출력)
    init_storage_kWh = init_soc * estimated_capacity_kWh
    df['net_kWh'] = init_storage_kWh + (df['cum_input_kWh'] - df['cum_output_kWh'])

    # (G2) storage_kWh 계산 (SOC × 추정용량), Coulomb counting 기반 SOC 시계열
    df['soc_cc'] = np.nan
    df.loc[0, 'soc_cc'] = init_soc

    for i in range(1, len(df)):
        dAh = df.loc[i, 'pack_current'] * df.loc[i, 'dt_h']
        # pack_current>0 ⇒ 방전 ⇒ SOC 감소 / pack_current<0 ⇒ 충전 ⇒ SOC 증가
        df.loc[i, 'soc_cc'] = df.loc[i - 1, 'soc_cc'] - (dAh / estimated_capacity_Ah)

    df['soc_cc'] = df['soc_cc'].clip(lower=0.0, upper=1.0)
    df['storage_kWh'] = df['soc_cc'] * estimated_capacity_kWh

    # 파일 전체 메타 정보
    df['init_soc'] = init_soc
    df['final_soc'] = final_soc
    df['estimated_capacity_kWh'] = estimated_capacity_kWh
    df['estimated_capacity_Ah'] = estimated_capacity_Ah

    # --- 아래가 **새로 추가된 부분** (η_battery 정의) ---
    # 그림의 정의: η_battery ⋅ (Storage_initial + Q_charged) = Q_discharged + Storage_final
    # 여기서는 kWh 단위로 계산 (모든 시계열이 kWh 기준)

    final_storage_kWh = df['storage_kWh'].iloc[-1]       # Storage_final
    final_input_kWh = df['cum_input_kWh'].iloc[-1]       # Q_charged
    final_output_kWh = df['cum_output_kWh'].iloc[-1]     # Q_discharged

    denominator = init_storage_kWh + final_input_kWh
    if denominator == 0:
        battery_eta = np.nan
    else:
        battery_eta = (final_output_kWh + final_storage_kWh) / denominator

    # 전 구간에 대해 단일 값으로 계산된 효율이므로, 모든 행에 동일하게 기록
    df['battery_eta'] = battery_eta

    # 컬럼 순서 재배치 후 저장
    df = df[columns_order]

    out_path = os.path.join(base_dir, file)
    df.to_csv(out_path, index=False)
    print(f"[OK] 전처리 완료(η_battery 포함), 저장: {file}")

# ------------------------------------------------
# (G) 스킵된 파일들 후처리 (ΔSOC≥30% 구간 없음)
# ------------------------------------------------
if skipped_files:
    print("\n[INFO] ΔSOC≥30% 구간이 없어 스킵된 파일 목록(컬럼 축소 대상):")
    for sf in skipped_files:
        print("  -", sf)

    print("\n[INFO] 스킵 파일에 대해 불필요 컬럼 제거 후 재저장합니다.")
    for sf in skipped_files:
        sf_path = os.path.join(base_dir, sf)
        df_skip = pd.read_csv(sf_path)

        # df_skip.columns 와 cols_to_keep 의 교집합
        actual_cols = df_skip.columns.intersection(cols_to_keep)
        df_skip = df_skip[actual_cols]

        df_skip.to_csv(sf_path, index=False)
        print(f"[OK] {sf} -> {list(actual_cols)} 컬럼만 남기고 저장 완료.")
else:
    print("\n[INFO] ΔSOC≥30% 미만으로 스킵된 파일 없음.")
