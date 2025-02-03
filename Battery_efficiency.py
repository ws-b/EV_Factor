import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
#     bms_단말기번호_d주차.csv
#     bms_altitude_단말기번호_d주차.csv
# ------------------------------------------------
pattern = r"(?:bms|bms_altitude)_(\d+)_d(\d+)\.csv"
files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]

# device_id, 주차(week), eta, 평균 ext_temp
eta_records = []  # (device_id, week, ext_temp_avg, eta)를 저장할 리스트

# ------------------------------------------------
# (2) 주 파일 순회
# ------------------------------------------------
for file in tqdm(files):
    match = re.match(pattern, file)
    if not match:
        print(f"[WARN] 파일명 형식 불일치: {file}")
        continue

    device_id = int(match.group(1))  # 예: "123"
    week_idx = int(match.group(2))   # 예: "4"

    file_path = os.path.join(base_dir, file)

    # CSV 로드
    df = pd.read_csv(file_path)

    # (2-1) 외기온도 컬럼이 없으면 무시하고 넘어가는 경우 처리
    if 'ext_temp' not in df.columns:
        print(f"[INFO] 'ext_temp' 컬럼이 없어 스킵: {file}")
        continue

    # ------------------------------------------------
    # (3) time 처리 (타임스탬프 간격이 너무 큰 경우 2초로 제한)
    # ------------------------------------------------
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # 3-1. 시간 간격(초) 계산
    df['dt_s'] = df['time'].diff().dt.total_seconds().fillna(0)

    # 3-2. 기준 이상의 긴 간격은 2초로 클램핑
    threshold_sec = 10
    max_sec_for_integration = 2
    df.loc[df['dt_s'] > threshold_sec, 'dt_s'] = max_sec_for_integration

    # 3-3. Ah 계산을 위해 시(hour) 단위로 변환
    df['dt_h'] = df['dt_s'] / 3600.0

    # ------------------------------------------------
    # (4) cell_volt_list 개수로부터 배터리 총용량(Ah) 추정
    # ------------------------------------------------
    cell_volt_str = df['cell_volt_list'].iloc[0]  # 첫 행 기준
    cell_list = cell_volt_str.split(',')
    cell_count = len(cell_list)

    # 여기서는 예시 값을 직접 지정
    cell_capacity = 55.6  # 셀 1개 당 용량(Ah) (예시)
    battery_capacity_Ah = cell_capacity * 2       # 2P(병렬) 구성 가정 (예시)

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
    # (6) Charged Q, Discharged Q 계산
    # ------------------------------------------------
    # dQ = I(A) * dt(h)
    df['dQ'] = df['pack_current'] * df['dt_h']

    # 전류가 음수 => 충전 / 양수 => 방전
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

    if eta > 1:
        print(f"[경고] eta > 1 ({eta:.3f}) : {file}")

    # ------------------------------------------------
    # (8) ext_temp 평균
    # ------------------------------------------------
    ext_temp_avg = df['ext_temp'].mean()

    eta_records.append((device_id, week_idx, ext_temp_avg, eta))

# ------------------------------------------------
# (9) DataFrame화
# ------------------------------------------------
eta_df = pd.DataFrame(eta_records, columns=['device_id', 'week', 'ext_temp_avg', 'eta'])

# device_id를 순차 번호로 매핑 (시각화 편의를 위해)
unique_devices = sorted(eta_df['device_id'].unique())
device_mapping = {dev_id: idx+1 for idx, dev_id in enumerate(unique_devices)}
eta_df['device_seq'] = eta_df['device_id'].map(device_mapping)

# ------------------------------------------------
# (10) 히스토그램 (전체 eta 분포)
# ------------------------------------------------
plt.figure(figsize=(8, 4))
plt.hist(eta_df['eta'].dropna(), bins=20, color='cornflowerblue', edgecolor='k', alpha=0.7)
plt.title('Battery Efficiency (eta) Distribution')
plt.xlabel('eta')
plt.ylabel('Frequency')
plt.show()

# ------------------------------------------------
# (11) x축=단말기번호, y축=eta 산점도
# ------------------------------------------------
plt.figure(figsize=(20, 10))
max_devices = 20
selected_devices = unique_devices[:max_devices]
selected_device_seq = [device_mapping[dev_id] for dev_id in selected_devices]

for dev_seq, dev_id in zip(selected_device_seq, selected_devices):
    dev_mask = (eta_df['device_id'] == dev_id)
    dev_eta_vals = eta_df.loc[dev_mask, 'eta']

    color = color_palette[(dev_seq - 1) % len(color_palette)]

    plt.scatter(
        [dev_seq] * len(dev_eta_vals),
        dev_eta_vals,
        color=color,
        alpha=0.7,
        edgecolor='k',
        label=f'Device {dev_seq}'
    )

plt.title('Device vs. Eta (Scatter Plot)')
plt.xlabel('Device Sequence ID')
plt.ylabel('Eta')
plt.xticks(ticks=range(1, max_devices + 1))
plt.xlim(0.5, max_devices + 0.5)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small', ncol=2)

plt.tight_layout()
plt.show()

# ------------------------------------------------
# (12) x축=ext_temp 평균, y축=eta 산점도
# ------------------------------------------------
plt.figure(figsize=(8, 6))

# NaN 제거 (ext_temp_avg, eta 둘 중 하나라도 결측이면 제외)
valid_mask = eta_df[['ext_temp_avg', 'eta']].notnull().all(axis=1)
plot_df = eta_df[valid_mask]

plt.scatter(
    plot_df['ext_temp_avg'],
    plot_df['eta'],
    color='cornflowerblue',
    alpha=0.7,
    edgecolor='k'
)

plt.title('Average External Temperature vs. Eta')
plt.xlabel('Average External Temperature (°C)')  # 예시 단위
plt.ylabel('Eta')
plt.grid(True, alpha=0.3)
plt.show()
