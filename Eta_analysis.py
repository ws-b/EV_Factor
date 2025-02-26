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
base_dir = r"E:\SamsungSTF\Processed_Data\Merged_period\Ioniq5"
ocv_file_path = r"D:\SamsungSTF\Data\GSmbiz\NE_SOC_OCV.csv"

REST_GAP_SEC = 7200      # 2시간 (REST 구간 판단 기준)
THRESHOLD_SEC = 10       # 샘플 간격이 10초보다 크면 max_sec_for_integration 적용
MAX_SEC_FOR_INTEGRATION = 2
SOC_USAGE_MIN = 0.30     # ΔSOC 30% 기준

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
# (1) OCV-SOC 테이블 로드 및 인터폴레이터 준비
# ------------------------------------------------
ocv_df = pd.read_csv(ocv_file_path)  # 컬럼: ['SOC', 'OCV'] (SOC=0~100 가정)
ocv_soc_array = ocv_df['SOC'].values / 100  # 0~1 범위
ocv_volt_array = ocv_df['OCV'].values

def get_soc_from_ocv(cell_voltage):
    """OCV-SOC 커브를 기반으로 SOC 추정 (선형 보간)"""
    return np.interp(cell_voltage, ocv_volt_array, ocv_soc_array)

# ------------------------------------------------
# (A) 세그먼트 분할
# ------------------------------------------------
def segment_data(df):
    """
    시간 간격이 2시간 이상이면 끊음
    seg = { 'start_idx': s, 'end_idx': e, 'reason': ... }
    """
    segs = []
    n = len(df)
    if n == 0:
        return segs

    start_i = 0
    for i in range(n - 1):
        time_gap = (df['time'].iloc[i+1] - df['time'].iloc[i]).total_seconds()
        if time_gap >= REST_GAP_SEC:
            segs.append({
                'start_idx': start_i,
                'end_idx': i+1,
                'reason': 'rest'
            })
            start_i = i+1

    # 마지막 구간
    if start_i < n:
        segs.append({
            'start_idx': start_i,
            'end_idx': n,
            'reason': 'end'
        })
    return segs


# ------------------------------------------------
# (B) 세그먼트별 SOC/전류적산(Ah) 계산
#     "마지막 SOC"은 다음 행(df.iloc[e]) 전압 사용 (가능하면)
#     다음 행이 없으면 => 스킵(return None)
# ------------------------------------------------
def compute_segment_info(df, seg):
    """
    seg: {'start_idx': s, 'end_idx': e}
    전류적산은 df.iloc[s:e] 범위
    최종 SOC 전압 = df.iloc[e]['pack_volt'] (단, e<len(df) 일때만)
      그렇지 않으면 스킵(return None)
    """
    s = seg['start_idx']
    e = seg['end_idx']
    n = len(df)

    # 1) e == n 이면 '다음 행'이 존재하지 않으므로 스킵
    if e >= n:
        return None

    # 2) 구간 길이도 최소 2는 되어야 함
    if (e - s) < 2:
        return None

    # 전류적산용 데이터
    sub = df.iloc[s:e].copy()

    # cell_count 추정
    cell_volt_str = sub['cell_volt_list'].iloc[0]
    cell_list = cell_volt_str.split(',')
    cell_count = len(cell_list)

    # 초기 SOC => 구간 첫 행
    first_pack_volt = sub['pack_volt'].iloc[0]
    init_soc = get_soc_from_ocv(first_pack_volt / cell_count)

    # 최종 SOC => 다음 행 (df.iloc[e])
    final_pack_volt = df['pack_volt'].iloc[e]
    final_soc = get_soc_from_ocv(final_pack_volt / cell_count)
    delta_soc = abs(init_soc - final_soc)

    # 전류적산(Ah)
    sub['current_ah'] = sub['pack_current'] * sub['dt_h']
    net_ah = sub['current_ah'].sum()

    # 전류가중 전압
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
# (C) 세그먼트 병합 & 용량 추정
# ------------------------------------------------
def merge_segments_if_needed(seg_info_list, soc_min=0.30):
    """
    seg_info_list = [{...}, ...]
    ΔSOC >= 30% 넘어가면 Capacity 계산, merged_results에 추가
    """
    merged_results = []
    if not seg_info_list:
        return merged_results

    def finalize_buffer(buf):
        net_ah = buf['net_ah']
        voltamp_ah = buf['voltamp_ah']
        dsoc = buf['delta_soc']

        if dsoc > 0 and abs(net_ah) > 1e-9:
            cap_ah = abs(net_ah) / dsoc
            mean_volt = (voltamp_ah / net_ah)
            cap_kwh = cap_ah * mean_volt / 1000
        else:
            cap_ah = 0
            cap_kwh = 0

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
            if i+1 < n:
                buffer = seg_info_list[i+1].copy()
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
# (D) 메인 루프
# ------------------------------------------------
pattern = r"(?:bms|bms_altitude)_(\d+)_d(\d+)\.csv"
files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]

for file in tqdm(files[:10]):
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

    df['dt_s'] = df['time'].diff().dt.total_seconds().fillna(0)
    df.loc[df['dt_s'] > THRESHOLD_SEC, 'dt_s'] = MAX_SEC_FOR_INTEGRATION
    df['dt_h'] = df['dt_s'] / 3600.0

    # 세그먼트 분할
    segments = segment_data(df)
    if not segments:
        print(f"[WARN] 세그먼트 분할 결과가 없습니다: {file}")
        continue

    # 세그먼트별 계산
    seg_info_list = []
    for seg in segments:
        info = compute_segment_info(df, seg)
        if info is not None:
            seg_info_list.append(info)
        else:
            # e == len(df) 등으로 인해 None이면 pass
            pass

    if not seg_info_list:
        print(f"[INFO] {file}: 모든 세그먼트가 다음행이 없어 스킵됨.")
        continue

    merged_results = merge_segments_if_needed(seg_info_list, SOC_USAGE_MIN)
    if not merged_results:
        print(f"[INFO] {file}: ΔSOC≥30% 구간 없어 용량 추정 불가, 스킵.")
        continue

    cap_ah_list = [mr['estimated_capacity_Ah'] for mr in merged_results]
    cap_kwh_list = [mr['estimated_capacity_kWh'] for mr in merged_results]
    avg_cap_ah  = np.mean(cap_ah_list)
    avg_cap_kwh = np.mean(cap_kwh_list)

    print(f"\n[INFO] {file}: 세그먼트별 용량 추정 결과 {len(merged_results)}건")
    print(f"       [Ah]  평균={avg_cap_ah:.1f}Ah")
    print(f"       [kWh] 평균={avg_cap_kwh:.2f}kWh")
    for m in merged_results:
        print("     - seg({}~{}), ΔSOC={:.1%}, "
              "Cap={:.1f}Ah ({:.2f}kWh)".format(
                    m['start_idx'], m['end_idx'],
                    m['delta_soc'],
                    m['estimated_capacity_Ah'],
                    m['estimated_capacity_kWh']))

    estimated_capacity_kWh = avg_cap_kwh

    # ------------------------------------------------
    # (E) 누적 Input/Output 및 Storage 계산
    # ------------------------------------------------
    df['pack_power_kW'] = (df['pack_volt'] * df['pack_current']) / 1000.0
    df['delta_energy_kWh'] = df['pack_power_kW'] * df['dt_h']

    df['input_kWh'] = np.where(df['pack_power_kW'] < 0,
                               -df['delta_energy_kWh'], 0)
    df['output_kWh'] = np.where(df['pack_power_kW'] > 0,
                                df['delta_energy_kWh'], 0)

    df['cum_input_kWh'] = df['input_kWh'].cumsum()
    df['cum_output_kWh'] = df['output_kWh'].cumsum()

    # 전체 구간 초기/최종 SOC (OCV)
    n_df = len(df)
    if n_df < 15:
        final_pack_volt = df['pack_volt'].iloc[-1]
    else:
        final_pack_volt = df['pack_volt'].iloc[-15]

    # cell_count 간단 추정
    cell_volt_str = df['cell_volt_list'].iloc[0]
    cell_count = len(cell_volt_str.split(','))
    init_soc = get_soc_from_ocv(df['pack_volt'].iloc[0] / cell_count)
    final_soc = get_soc_from_ocv(final_pack_volt / cell_count)

    init_storage_kWh = init_soc * estimated_capacity_kWh
    final_storage_kWh = final_soc * estimated_capacity_kWh
    df['storage_kWh'] = init_storage_kWh + (df['cum_input_kWh'] - df['cum_output_kWh'])

    # ------------------------------------------------
    # (F) 시각화 (생략 가능)
    # ------------------------------------------------
    if len(df) > 15:
        df_plot1 = df.iloc[:-15].copy()
    else:
        df_plot1 = df.copy()

    time_margin = pd.Timedelta(hours=5)
    x_min = df_plot1['time'].min() - time_margin
    x_max = df_plot1['time'].max() + time_margin

    fig, axes = plt.subplots(6, 1, figsize=(12, 22), sharex=True)
    fig.suptitle(f"[Device={device_id}, Week={week_idx}]  "
                 f"Estimated Capacity(Avg)={estimated_capacity_kWh:.2f} kWh\n"
                 f"Init SOC={init_soc:.3f}, Final SOC={final_soc:.3f}",
                 fontsize=16)

    axes[0].plot(df_plot1['time'], df_plot1['pack_volt'],
                 color=color_palette[0], label='Pack Volt [V]')
    axes[0].set_ylabel("Pack Volt [V]")
    axes[0].legend(loc='upper left')

    axes[1].plot(df_plot1['time'], df_plot1['pack_current'],
                 color=color_palette[1], label='Pack Current [A]')
    axes[1].set_ylabel("Pack Current [A]")
    axes[1].legend(loc='upper left')

    axes[2].plot(df_plot1['time'], df_plot1['cum_input_kWh'],
                 color=color_palette[2], label='Input [kWh]')
    axes[2].set_ylabel("Input [kWh]")
    axes[2].legend(loc='upper left')

    axes[3].plot(df_plot1['time'], df_plot1['cum_output_kWh'],
                 color=color_palette[3], label='Output [kWh]')
    axes[3].set_ylabel("Output [kWh]")
    axes[3].legend(loc='upper left')

    axes[4].plot(df_plot1['time'], df_plot1['storage_kWh'],
                 color=color_palette[4], label='Storage [kWh]')
    axes[4].scatter(df_plot1['time'].iloc[0], df_plot1['storage_kWh'].iloc[0],
                    color=marker_palette[0], marker='o', s=25, zorder=5, label='Init SOC')
    axes[4].scatter(df_plot1['time'].iloc[-1], df_plot1['storage_kWh'].iloc[-1],
                    color=marker_palette[1], marker='o', s=25, zorder=5, label='Final SOC')
    axes[4].set_ylabel("Storage [kWh]")
    axes[4].legend(loc='upper left')

    axes[5].plot(df_plot1['time'], df_plot1['pack_power_kW'],
                 color=color_palette[5], label='Pack Power [kW]')
    axes[5].set_ylabel("Pack Power [kW]")
    axes[5].set_xlabel("Time")
    axes[5].legend(loc='upper left')

    date_formatter = DateFormatter('%m-%d %H:%M')
    axes[-1].xaxis.set_major_formatter(date_formatter)
    axes[-1].set_xlim(x_min, x_max)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # 2번째 플롯
    if len(df) > 15:
        df_plot2 = df.iloc[:-15].copy()
    else:
        df_plot2 = df.copy()

    time_init = df_plot2['time'].iloc[0]
    time_final_plot = df_plot2['time'].iloc[-1]
    init_storage_kWh_plot = df_plot2['storage_kWh'].iloc[0]
    final_storage_kWh_plot = df_plot2['storage_kWh'].iloc[-1]

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.xaxis.set_major_formatter(date_formatter)

    ax2.plot(df_plot2['time'], df_plot2['storage_kWh'],
             color=color_palette[4], label='Storage [kWh]')
    ax2.plot(df_plot2['time'], df_plot2['cum_input_kWh'],
             color=color_palette[2], label='Input [kWh]')
    ax2.plot(df_plot2['time'], df_plot2['cum_output_kWh'],
             color=color_palette[3], label='Output [kWh]')

    ax2.scatter(time_init, init_storage_kWh_plot,
                color=marker_palette[0], marker='o', s=25, zorder=5, label='Init SOC')
    ax2.scatter(time_final_plot, final_storage_kWh_plot,
                color=marker_palette[1], marker='o', s=25, zorder=5, label='Final SOC')

    all_times = pd.concat([df_plot2['time'], pd.Series([time_final_plot])])
    ax2.set_xlim(all_times.min() - time_margin, all_times.max() + time_margin)

    ax2.set_title(f"Storage / Input / Output\n"
                  f"Init SOC={init_soc:.3f}, Final SOC={final_soc:.3f}, "
                  f"Estimated Capacity={estimated_capacity_kWh:.2f} kWh")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Energy [kWh]")
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

print("\n작업 완료.")
