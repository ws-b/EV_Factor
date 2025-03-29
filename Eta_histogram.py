import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

base_dir = r"D:\SamsungSTF\Processed_Data\Merged_period_final\Ioniq5"
pattern = r"(?:bms|bms_altitude)_(\d+)_d(\d+)\.csv"

battery_eta_list = []

files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
for file in tqdm(files):
    if not re.match(pattern, file):
        # 파일명 형식이 맞지 않으면 스킵
        continue

    csv_path = os.path.join(base_dir, file)
    df = pd.read_csv(csv_path)

    # 전처리에서 battery_eta를 컬럼에 넣어 저장해 두었다고 가정
    if 'battery_eta' in df.columns:
        # 값이 NaN이 아니면 리스트에 추가
        battery_eta_list.extend(df['battery_eta'].dropna().tolist())

# 히스토그램 그리기
if len(battery_eta_list) == 0:
    print("battery_eta 데이터가 없습니다.")
else:
    plt.hist(battery_eta_list, bins=50, color='skyblue', edgecolor='black')
    plt.title('Battery Eta Histogram')
    plt.xlabel('Battery Eta')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.show()
