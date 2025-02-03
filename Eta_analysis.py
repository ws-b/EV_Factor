import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.dates import DateFormatter

# ------------------------------------------------
# (0) 파라미터 설정 / 색상 팔레트
# ------------------------------------------------
base_dir = r"D:\SamsungSTF\Processed_Data\Merged_period\Ioniq5"

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

# ------------------------------------------------
# (1) 파일 목록 읽기
# ------------------------------------------------
pattern = r"(?:bms|bms_altitude)_(\d+)_d(\d+)\.csv"
files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]

eta_records = []  # (device_id, week, ext_temp_avg, eta)를 저장할 리스트

# ------------------------------------------------
# (2) 주 파일 순회하여 eta 계산
# ------------------------------------------------
for file in tqdm(files[:30]):
    match = re.match(pattern, file)
    if not match:
        print(f"[WARN] 파일명 형식 불일치: {file}")
        continue

    device_id = match.group(1)
    week_idx = int(match.group(2))

    file_path = os.path.join(base_dir, file)

    # CSV 로드
    df = pd.read_csv(file_path)

    # (2-1) 외기온도 컬럼이 없으면 스킵(eta 계산이 무의미하진 않지만, 여기서는 예시로 제외)
    if 'ext_temp' not in df.columns:
        print(f"[INFO] 'ext_temp' 컬럼이 없어 스킵: {file}")
        continue

    # ------------------------------------------------
    # (3) time 처리 (타임스탬프 간격이 너무 큰 경우 2초로 제한)
    # ------------------------------------------------
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    df['dt_s'] = df['time'].diff().dt.total_seconds().fillna(0)
    threshold_sec = 10
    max_sec_for_integration = 2
    df.loc[df['dt_s'] > threshold_sec, 'dt_s'] = max_sec_for_integration
    df['dt_h'] = df['dt_s'] / 3600.0

    # ------------------------------------------------
    # (4) cell_volt_list 개수로부터 배터리 총용량(Ah) 추정 (예시)
    # ------------------------------------------------
    # 만약 컬럼이 비어있거나 NaN이 있을 수 있으므로 예외처리 필요할 수 있음
    if 'cell_volt_list' not in df.columns or df['cell_volt_list'].isna().all():
        print(f"[WARN] cell_volt_list가 비어있음. 파일 스킵: {file}")
        continue

    cell_volt_str = df['cell_volt_list'].iloc[0]
    cell_list = cell_volt_str.split(',')
    cell_count = len(cell_list)

    cell_capacity = 55.6  # 셀 1개 당 용량(Ah) (예시)
    battery_capacity_Ah = cell_capacity * 2  # 2P 구성 가정 (예시)

    # ------------------------------------------------
    # (5) SOC 계산
    #   - 시작 SOC: df['soc'].iloc[0]
    #   - 최종 SOC: "마지막 15행" 중 가장 먼저 만나는 유효 SOC(>0) or 마지막행
    # ------------------------------------------------
    if 'soc' not in df.columns:
        print(f"[WARN] soc 컬럼이 없어 스킵: {file}")
        continue

    # 시작 SOC
    soc_start = df['soc'].iloc[0]

    # (5-1) 마지막 15행 중 유효 SOC 찾기
    last_15 = df.iloc[-15:].copy()  # 끝에서 15행
    # 0이 아니고, NaN이 아닌 SOC
    valid_soc_rows = last_15[(last_15['soc'].notna()) & (last_15['soc'] != 0)]
    if not valid_soc_rows.empty:
        # 가장 앞(상위) 행의 SOC
        soc_end = valid_soc_rows['soc'].iloc[0]
    else:
        # 15행 모두가 0 또는 NaN이면, 기존 방식대로 마지막행 SOC
        soc_end = df['soc'].iloc[-1]

    # SOC 범위 체크 (0~100 => 0~1 변환)
    if df['soc'].max() > 1.0:
        soc_start /= 100.0
        soc_end /= 100.0

    init_residual_Ah = soc_start * battery_capacity_Ah
    final_residual_Ah = soc_end * battery_capacity_Ah

    # ------------------------------------------------
    # (6) Charged Q, Discharged Q
    # ------------------------------------------------
    if 'pack_current' not in df.columns:
        print(f"[WARN] pack_current 컬럼이 없어 스킵: {file}")
        continue

    df['dQ'] = df['pack_current'] * df['dt_h']
    charged_Q_Ah = df.loc[df['dQ'] < 0, 'dQ'].sum() * (-1)
    discharged_Q_Ah = df.loc[df['dQ'] > 0, 'dQ'].sum()

    # ------------------------------------------------
    # (7) eta 계산
    # ------------------------------------------------
    numerator = charged_Q_Ah + init_residual_Ah
    denominator = discharged_Q_Ah + final_residual_Ah

    if denominator == 0:
        eta = np.nan
    else:
        eta = numerator / denominator

    # ------------------------------------------------
    # (8) ext_temp 평균
    # ------------------------------------------------
    ext_temp_avg = df['ext_temp'].mean()

    eta_records.append((device_id, week_idx, ext_temp_avg, eta))

# ------------------------------------------------
# (9) eta_df 생성
# ------------------------------------------------
eta_df = pd.DataFrame(eta_records, columns=['device_id', 'week', 'ext_temp_avg', 'eta'])

# ------------------------------------------------
# (10) eta > 1 인 행만 샘플링
# ------------------------------------------------
df_over_1 = eta_df[eta_df['eta'] > 1].copy()
if len(df_over_1) == 0:
    print("[INFO] eta > 1 인 파일이 없습니다. 종료합니다.")
else:
    sample_count = min(5, len(df_over_1))  # 만약 5개보다 적으면 전체 사용
    df_over_1_sample = df_over_1.sample(sample_count, random_state=42)

    # ------------------------------------------------
    # (11) 샘플링된 파일들에 대해 서브플롯 시각화
    # ------------------------------------------------
    for idx, row in df_over_1_sample.iterrows():
        device_id = str(row['device_id'])
        week = int(row['week'])
        eta_val = row['eta']

        # 파일명 (예: bms_0000_d1.csv)
        file_name_bms = f"bms_{device_id}_d{week}.csv"
        file_path_bms = os.path.join(base_dir, file_name_bms)

        if not os.path.exists(file_path_bms):
            print(f"[WARN] 해당 파일이 존재하지 않음: {file_name_bms}")
            continue

        # CSV 로드
        df_bms = pd.read_csv(file_path_bms)
        if 'time' not in df_bms.columns:
            print(f"[WARN] time 컬럼이 없어 스킵: {file_name_bms}")
            continue

        # time 변환
        df_bms['time'] = pd.to_datetime(df_bms['time'])
        df_bms.sort_values('time', inplace=True)
        df_bms.reset_index(drop=True, inplace=True)

        # dt_s, dt_h 재계산
        df_bms['dt_s'] = df_bms['time'].diff().dt.total_seconds().fillna(0)
        threshold_sec = 10
        max_sec_for_integration = 2
        df_bms.loc[df_bms['dt_s'] > threshold_sec, 'dt_s'] = max_sec_for_integration
        df_bms['dt_h'] = df_bms['dt_s'] / 3600.0

        # dQ = pack_current * dt_h
        if 'pack_current' not in df_bms.columns:
            print(f"[WARN] pack_current 컬럼이 없어 스킵: {file_name_bms}")
            continue

        df_bms['dQ'] = df_bms['pack_current'] * df_bms['dt_h']

        # 누적 충/방전량
        df_bms['charged_Q'] = np.where(df_bms['dQ'] < 0, -df_bms['dQ'], 0)
        df_bms['discharged_Q'] = np.where(df_bms['dQ'] > 0, df_bms['dQ'], 0)
        df_bms['cum_charged_Q'] = df_bms['charged_Q'].cumsum()
        df_bms['cum_discharged_Q'] = df_bms['discharged_Q'].cumsum()

        # 5개 서브플롯
        fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f"{device_id}, Week={week}, eta={eta_val:.3f}", fontsize=16)

        # (1) pack_volt
        if 'pack_volt' in df_bms.columns:
            axes[0].plot(df_bms['time'], df_bms['pack_volt'],
                         color=color_palette[0], label='Pack Volt')
            axes[0].set_ylabel("Pack Volt [V]")
            axes[0].legend(loc='upper left')
        else:
            axes[0].text(0.5, 0.5, 'pack_volt 없음', ha='center', va='center')

        # (2) pack_current
        axes[1].plot(df_bms['time'], df_bms['pack_current'],
                     color=color_palette[1], label='Pack Current')
        axes[1].set_ylabel("Pack Current [A]")
        axes[1].legend(loc='upper left')

        # (3) soc
        if 'soc' in df_bms.columns:
            axes[2].plot(df_bms['time'], df_bms['soc'],
                         color=color_palette[2], label='SOC')
            axes[2].set_ylabel("SOC")
            axes[2].legend(loc='upper left')
        else:
            axes[2].text(0.5, 0.5, 'soc 없음', ha='center', va='center')

        # (4) 누적 충전량
        axes[3].plot(df_bms['time'], df_bms['cum_charged_Q'],
                     color=color_palette[3], label='Cumulative Charged [Ah]')
        axes[3].set_ylabel("Charged [Ah]")
        axes[3].legend(loc='upper left')

        # (5) 누적 방전량
        axes[4].plot(df_bms['time'], df_bms['cum_discharged_Q'],
                     color=color_palette[4], label='Cumulative Discharged [Ah]')
        axes[4].set_ylabel("Discharged [Ah]")
        axes[4].set_xlabel("Time")
        axes[4].legend(loc='upper left')

        # x축 시간 포맷
        date_formatter = DateFormatter('%m-%d %H:%M')
        axes[4].xaxis.set_major_formatter(date_formatter)

        plt.tight_layout()
        plt.show()

print("\n작업 완료.")
