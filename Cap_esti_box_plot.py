import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 데이터가 저장된 디렉토리 경로
base_dir = r"E:\SamsungSTF\Processed_Data\Merged_period\Ioniq5"

# 단말기 번호별 데이터를 저장할 딕셔너리
terminal_data = {}

# base_dir 내의 모든 CSV 파일 순회
for filename in os.listdir(base_dir):
    if not filename.endswith('.csv'):
        continue
    filepath = os.path.join(base_dir, filename)

    # CSV 파일 읽기
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"파일 {filename} 읽기 실패: {e}")
        continue

    # 'estimated_capacity_Ah' 컬럼이 존재하는지 확인
    if 'estimated_capacity_Ah' not in df.columns:
        print(f"'{filename}' 파일에 'estimated_capacity_Ah' 컬럼이 없습니다. 건너뜁니다.")
        continue

    # 파일명에서 단말기 번호 추출 (예: bms_01241228003_d44.csv -> "01241228003")
    match = re.search(r"bms_(.*?)_d", filename)
    if match:
        terminal = match.group(1)
    else:
        print(f"터미널 번호 추출 실패: {filename}")
        continue

    # 단말기 번호에 해당하는 리스트에 값을 추가
    if terminal not in terminal_data:
        terminal_data[terminal] = []
    terminal_data[terminal].extend(df['estimated_capacity_Ah'].tolist())

# 시각화를 위한 데이터가 있는지 확인
if not terminal_data:
    print("시각화할 데이터가 없습니다.")
else:
    # 단말기 번호를 정렬하여 일관된 순서로 표시
    terminals = sorted(terminal_data.keys())

    # 대표 단말기 번호 5개만 선택 (예: 가장 앞의 5개)
    represent_num = 25
    if len(terminals) > represent_num:
        terminals = terminals[:represent_num]

    data_to_plot = [terminal_data[t] for t in terminals]

    # Box Plot으로 시각화
    plt.figure(figsize=(12, 6))
    plt.boxplot(data_to_plot, tick_labels=terminals, showfliers=False)
    plt.xlabel("Device ID")
    plt.ylabel("Estimated Capacity (Ah)")
    plt.title(f"Estimated Capacity (Ah), {represent_num} cars")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
