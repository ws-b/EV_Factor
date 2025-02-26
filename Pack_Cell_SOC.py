import os
import glob
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm

# -------------------------------------------------------------------------------
# 0. 환경 세팅
# -------------------------------------------------------------------------------
random.seed(42)

root_dir = r"E:\SamsungSTF\Processed_Data\TripByTrip_soc\Ioniq5"
ocv_table_path = r"D:\SamsungSTF\Data\GSmbiz\NE_SOC_OCV.csv"

# 셀 1개 용량 [Ah]
CELL_CAPACITY = 55.6

# -------------------------------------------------------------------------------
# [추가] 저장 폴더 설정
# -------------------------------------------------------------------------------
save_dir = r"C:\Users\BSL\Desktop\EV_Efficiency\Figure"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------------------------------------------------------
# 1. OCV 테이블 불러오기 및 보간 함수 준비
# -------------------------------------------------------------------------------
df_ocv = pd.read_csv(ocv_table_path)

# OCV -> SOC 보간 함수
f_soc_from_ocv = interp1d(
    df_ocv["OCV"].values,
    df_ocv["SOC"].values,
    kind='linear',
    fill_value="extrapolate"
)

# -------------------------------------------------------------------------------
# 2. CSV 파일 리스트
# -------------------------------------------------------------------------------
all_csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
print(f"발견된 CSV 파일 수: {len(all_csv_files)}")

# -------------------------------------------------------------------------------
# 3. 보조 함수들
# -------------------------------------------------------------------------------
def parse_cell_volt_list(cell_volt_str):
    """'1,3,1,3,2,3' → [float, float, ...]로 변환 후 평균."""
    if pd.isna(cell_volt_str):
        return np.nan
    volt_list = [float(x) for x in cell_volt_str.split(',')]
    return np.mean(volt_list)

def count_cell_volt(cell_volt_str):
    """cell_volt_list의 항목 수."""
    if pd.isna(cell_volt_str):
        return np.nan
    return len(cell_volt_str.split(','))

# -------------------------------------------------------------------------------
# 4. 전역 points 리스트
# -------------------------------------------------------------------------------
# points: 각 항목은 {"file": csv_file, "x": BMS_SOC, "y": (Cell 또는 Pack) SOC,
#          "row_type": "first"/"last", "soc_type": "cell"/"pack"} 의 딕셔너리
points = []

# -------------------------------------------------------------------------------
# 5. 파일별 처리
# -------------------------------------------------------------------------------
for csv_file in tqdm(all_csv_files):
    local_points = []

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[에러] CSV 읽기 실패 (파일 스킵): {csv_file} -> {e}")
        continue

    # (A) 필수 컬럼 체크
    required_cols = {"time", "cell_volt_list", "pack_volt", "soc", "pack_current"}
    if not required_cols.issubset(df.columns):
        print(f"[스킵] 필요한 컬럼({required_cols}) 누락: {csv_file}")
        continue

    # (B) time 컬럼 → datetime 변환
    df["time_dt"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    if df["time_dt"].isna().sum() > len(df) * 0.5:
        print(f"[스킵] time 변환 실패 많음: {csv_file}")
        continue

    # (C) cell_volt_avg 계산 및 cell_count 측정
    df["cell_volt_avg"] = df["cell_volt_list"].apply(parse_cell_volt_list)
    df["cell_count"] = df["cell_volt_list"].apply(count_cell_volt)

    # 전체 행에서 유효한 cell_count 값만 선택 (NaN 제외)
    valid_cell_counts = df["cell_count"].dropna().astype(int)

    # 만약 유효한 cell_count 값이 하나도 없다면 파일 스킵
    if valid_cell_counts.empty:
        print(f"[스킵] 유효한 cell_count가 없습니다: {csv_file}")
        continue

    # 최빈값(가장 많이 등장한 cell 개수) 계산
    majority_cell_count = valid_cell_counts.mode()[0]

    # majority_cell_count를 이용하여 pack_volt_normalized 계산
    df["pack_volt_normalized"] = df["pack_volt"] / majority_cell_count

    # (D) OCV 기반 SOC (Cell / Pack)
    df["interp_soc"] = f_soc_from_ocv(df["cell_volt_avg"])          # Cell
    df["interp_soc_pack"] = f_soc_from_ocv(df["pack_volt_normalized"])  # Pack

    # (E) SOH 기반 Q Cell (여기서는 사용하지 않지만 코드상 존재)
    if "soh" in df.columns:
        Q_cell = CELL_CAPACITY * df["soh"] / 100
    else:
        Q_cell = CELL_CAPACITY

    # -------------------------------------------------------------------------------
    # 앞쪽 30행 제거
    # -------------------------------------------------------------------------------
    if len(df) < 30:
        print(f"[스킵] 앞 30행 미만: {csv_file}")
        continue
    df = df.iloc[30:].reset_index(drop=True)

    n_after_cut = len(df)
    if n_after_cut < 2:
        print(f"[스킵] 전처리 후 길이 < 2: {csv_file}")
        continue

    # -------------------------------------------------------------------------------
    # (1) 'Start SOC' 찾기: 앞쪽 30행 내 cell_volt_avg가 2.5~4.4 범위인 첫 행
    # -------------------------------------------------------------------------------
    start_idx = None
    max_check_for_start = min(30, n_after_cut)
    for i in range(max_check_for_start):
        avg_v = df["cell_volt_avg"].iloc[i]
        if 2.5 <= avg_v <= 4.4:
            start_idx = i
            break

    if start_idx is None:
        print(f"[스킵] 첫 30행 내 cell_volt_avg(3~4.4) 만족 행 없음: {csv_file}")
        continue

    # -------------------------------------------------------------------------------
    # (2) 'Final SOC' 찾기: 마지막 31~45행 중 cell/pack SOC 모두 0~100 범위인 행
    # -------------------------------------------------------------------------------
    if n_after_cut < 45:
        print(f"[스킵] 뒤에서 45행도 안 됨(데이터 부족): {csv_file}")
        continue

    target_idx = n_after_cut - 31  # 기준 인덱스 (원래 마지막 행의 31번째 전)
    found_valid = False
    for offset in range(29, 14, -1):  # offset 29부터 15까지 (총 15행)
        source_idx = n_after_cut - offset
        cell_val = df.at[source_idx, "interp_soc"]
        pack_val = df.at[source_idx, "interp_soc_pack"]
        if 0 <= cell_val <= 100 and 0 <= pack_val <= 100:
            df.at[target_idx, "interp_soc"] = cell_val
            df.at[target_idx, "interp_soc_pack"] = pack_val
            found_valid = True
            break

    if not found_valid:
        print(f"[스킵] 마지막 31~45행 모두 Cell/Pack SOC OOR(0~100): {csv_file}")
        continue

    # -------------------------------------------------------------------------------
    # (3) 마지막 30행 제거
    # -------------------------------------------------------------------------------
    df = df.iloc[:(n_after_cut - 30)].reset_index(drop=True)
    if len(df) < 2:
        print(f"[스킵] 뒤 30행 제거 후 2행 미만: {csv_file}")
        continue

    # -------------------------------------------------------------------------------
    # elapsed 시간 재설정 (필요시)
    # -------------------------------------------------------------------------------
    df["elapsed"] = (df["time_dt"] - df["time_dt"].iloc[0]).dt.total_seconds()

    # -------------------------------------------------------------------------------
    # (4) 최종: 'Start SOC' (start_idx행)과 'Final SOC' (마지막 행) 사용
    # -------------------------------------------------------------------------------
    first_row = df.iloc[start_idx]
    last_row = df.iloc[-1]

    local_points.append({
        "file": csv_file,
        "x": first_row["soc"],         # BMS SOC
        "y": first_row["interp_soc"],  # Cell SOC
        "row_type": "first",
        "soc_type": "cell"
    })
    local_points.append({
        "file": csv_file,
        "x": first_row["soc"],
        "y": first_row["interp_soc_pack"],
        "row_type": "first",
        "soc_type": "pack"
    })
    local_points.append({
        "file": csv_file,
        "x": last_row["soc"],
        "y": last_row["interp_soc"],
        "row_type": "last",
        "soc_type": "cell"
    })
    local_points.append({
        "file": csv_file,
        "x": last_row["soc"],
        "y": last_row["interp_soc_pack"],
        "row_type": "last",
        "soc_type": "pack"
    })

    # 4개 중 NaN이 있는지 체크
    if any(pd.isna(p["x"]) or pd.isna(p["y"]) for p in local_points):
        print(f"[스킵] Start/Final SOC 중 NaN 존재: {csv_file}")
        continue

    points.extend(local_points)

# -------------------------------------------------------------------------------
# 6. RMSE 계산
# -------------------------------------------------------------------------------
print(f"\n[INFO] 최종 수집된 점(Points) 개수: {len(points)}")

cell_sq_errors = []
pack_sq_errors = []

for p in points:
    error = p["y"] - p["x"]  # (예측값 - 참값)
    if p["soc_type"] == "cell":
        cell_sq_errors.append(error**2)
    else:
        pack_sq_errors.append(error**2)

rmse_cell = np.sqrt(np.mean(cell_sq_errors)) if cell_sq_errors else np.nan
rmse_pack = np.sqrt(np.mean(pack_sq_errors)) if pack_sq_errors else np.nan

print(f"[INFO] RMSE(Cell) = {rmse_cell:.4f},  RMSE(Pack) = {rmse_pack:.4f}")

# -------------------------------------------------------------------------------
# [추가] 45도 선에서 많이 벗어난 샘플 파일명 출력
# -------------------------------------------------------------------------------
# (BMS SOC와 보간된 SOC 간 차이가 임계값(여기서는 10%) 이상이면 파일명을 출력)
deviation_threshold = 10  # 임계값 설정 (예: 10% 이상 차이)
deviated_files = set()
for p in points:
    if abs(p["x"] - p["y"]) > deviation_threshold:
        deviated_files.add(p["file"])

if deviated_files:
    print("\n[INFO] 45도 선에서 많이 벗어난 샘플 파일들:")
    for f in deviated_files:
        print(f)
else:
    print("\n[INFO] 45도 선에서 많이 벗어난 샘플 파일이 없습니다.")

# -------------------------------------------------------------------------------
# 7. Scatter Plot (BMS SOC vs Cell/Pack SOC) + 45도 라인
# -------------------------------------------------------------------------------
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

x_first_cell = [p["x"] for p in points if p["row_type"] == "first" and p["soc_type"] == "cell"]
y_first_cell = [p["y"] for p in points if p["row_type"] == "first" and p["soc_type"] == "cell"]

x_first_pack = [p["x"] for p in points if p["row_type"] == "first" and p["soc_type"] == "pack"]
y_first_pack = [p["y"] for p in points if p["row_type"] == "first" and p["soc_type"] == "pack"]

x_last_cell = [p["x"] for p in points if p["row_type"] == "last" and p["soc_type"] == "cell"]
y_last_cell = [p["y"] for p in points if p["row_type"] == "last" and p["soc_type"] == "cell"]

x_last_pack = [p["x"] for p in points if p["row_type"] == "last" and p["soc_type"] == "pack"]
y_last_pack = [p["y"] for p in points if p["row_type"] == "last" and p["soc_type"] == "pack"]

fig, ax = plt.subplots(figsize=(8, 6))

# 첫 행 (Start)
ax.scatter(x_first_cell, y_first_cell,
           color=color_palette[0], marker='o', alpha=0.5,
           label='Start SOC - Cell')
ax.scatter(x_first_pack, y_first_pack,
           color=color_palette[0], marker='x', alpha=0.5,
           label='Start SOC - Pack')

# 마지막 행 (Final)
ax.scatter(x_last_cell, y_last_cell,
           color=color_palette[1], marker='o', alpha=0.5,
           label='Final SOC - Cell')
ax.scatter(x_last_pack, y_last_pack,
           color=color_palette[1], marker='x', alpha=0.5,
           label='Final SOC - Pack')

# 45도 기준선
line_min, line_max = 0, 100
xx = np.linspace(line_min, line_max, 101)
ax.plot(xx, xx, 'k--')

ax.set_xlim(line_min, line_max)
ax.set_ylim(line_min, line_max)
ax.set_xlabel("BMS SOC")
ax.set_ylabel("Cell/Pack SOC")
ax.set_title("Scatter of Start/Final SOCs")

# 오른쪽 하단 RMSE 표시
textstr = f"RMSE(Cell) = {rmse_cell:.2f}%\nRMSE(Pack) = {rmse_pack:.2f}%"
ax.text(0.95, 0.05, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='none'))

ax.legend()
plt.tight_layout()

# [저장] 첫 번째 플롯을 PNG로 저장
fig_path1 = os.path.join(save_dir, "Scatter_of_Start_Final_SOCs.png")
plt.savefig(fig_path1, dpi=300)
print(f"[INFO] Scatter plot (Start/Final SOCs) 저장: {fig_path1}")

plt.show()

# -------------------------------------------------------------------------------
# [추가] X축: Cell SOC, y축: Pack SOC scatter plot
# -------------------------------------------------------------------------------
# 각 파일의 동일 row_type("first" 또는 "last")에 대해 cell SOC와 pack SOC를 한 쌍으로 묶습니다.
paired_points = {}
for p in points:
    key = (p["file"], p["row_type"])
    if key not in paired_points:
        paired_points[key] = {}
    paired_points[key][p["soc_type"]] = p["y"]

start_cell_soc = []
start_pack_soc = []
final_cell_soc = []
final_pack_soc = []

for (file, row_type), soc_dict in paired_points.items():
    if "cell" in soc_dict and "pack" in soc_dict:
        if row_type == "first":
            start_cell_soc.append(soc_dict["cell"])
            start_pack_soc.append(soc_dict["pack"])
        elif row_type == "last":
            final_cell_soc.append(soc_dict["cell"])
            final_pack_soc.append(soc_dict["pack"])

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(start_cell_soc, start_pack_soc, color=color_palette[4], marker='o', alpha=0.5, label='Start (Cell vs Pack)')
ax.scatter(final_cell_soc, final_pack_soc, color=color_palette[5], marker='x', alpha=0.5, label='Final (Cell vs Pack)')
ax.plot([0, 100], [0, 100], 'k--')  # 45도 기준선
ax.set_xlabel("Cell SOC")
ax.set_ylabel("Pack SOC")
ax.set_title("Scatter Plot: Cell SOC vs Pack SOC")
ax.legend()
plt.tight_layout()

# [저장] 두 번째 플롯을 PNG로 저장
fig_path2 = os.path.join(save_dir, "Scatter_Cell_vs_Pack_SOCs.png")
plt.savefig(fig_path2, dpi=300)
print(f"[INFO] Scatter plot (Cell vs Pack SOCs) 저장: {fig_path2}")

plt.show()
