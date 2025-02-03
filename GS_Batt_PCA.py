import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from GS_vehicle_dict import vehicle_dict

################################################################################
# 1) CONFIGURATION
################################################################################

directory = r'D:\SamsungSTF\Processed_Data\TripByTrip'
selected_car = 'KonaEV'

# 예: vehicle_dict에서 Device ID 추출 (실제 프로젝트 환경에 맞게 수정)
device_ids = vehicle_dict.get(selected_car, [])


if not device_ids:
    print(f"No device IDs found for the selected vehicle: {selected_car}")
    raise SystemExit

all_files = glob.glob(os.path.join(directory, '*.csv'))
files = [file for file in all_files if any(device_id in os.path.basename(file) for device_id in device_ids)]

# 결과 저장 리스트
data_list = []

################################################################################
# 2) MAIN LOOP
################################################################################

for filename in tqdm(files):
    try:
        df = pd.read_csv(filename)

        #-----------------------------------------------------------------------
        # (A) 필수 컬럼 체크
        #-----------------------------------------------------------------------
        required_cols = [
            'time',
            'speed',
            'acceleration',
            'Power_data'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in {filename}. Skipping file.")
            continue

        #-----------------------------------------------------------------------
        # (B) 시간 정렬 및 delta_time
        #-----------------------------------------------------------------------
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.sort_values('time').reset_index(drop=True)

        # time_seconds
        df['time_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
        df['delta_time'] = df['time_seconds'].diff().fillna(0)

        #-----------------------------------------------------------------------
        # (C) 속도 / 거리 계산
        #-----------------------------------------------------------------------
        # speed(기존 단위 m/s 가정)
        df['speed_kmh'] = df['speed'] * 3.6
        df['speed_m_s'] = df['speed']

        # 거리 적분 (누적)
        distance_m = cumulative_trapezoid(df['speed_m_s'], df['time_seconds'], initial=0)
        df['distance'] = distance_m

        total_distance_m = df['distance'].iloc[-1]
        total_distance_km = total_distance_m / 1000.0
        total_duration_s = df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]

        # 주행중(속도>=2km/h) 시간
        driving_duration_s = df.loc[df['speed_kmh'] >= 2, 'delta_time'].sum()

        #-----------------------------------------------------------------------
        # (D) **Power_data를 이용한 사용 에너지(kWh) 적분**
        #-----------------------------------------------------------------------
        # 1) 만약 Power_data가 W 단위라 가정
        #    -> 적분 결과(J) -> kWh 변환(/ 3.6e6)
        used_energy_joules = cumulative_trapezoid(
            df['Power_data'],  # W
            df['time_seconds'],  # s
            initial=0
        )
        used_energy_total_joules = used_energy_joules[-1]  # 마지막 값이 총합
        used_energy_kwh = used_energy_total_joules / 3.6e6  # J -> kWh

        # 2) Trip 전비(kWh/100km)
        trip_consumption_kwh_per_100km = np.nan
        if total_distance_km > 0:
            trip_consumption_kwh_per_100km = (used_energy_kwh / total_distance_km) * 100.0

        #-----------------------------------------------------------------------
        # (E) Driving Pattern 파라미터
        #-----------------------------------------------------------------------
        average_speed = (total_distance_km / (total_duration_s/3600.0)) if total_duration_s>0 else 0
        average_squared_speed = (df['speed_kmh']**2).mean()

        driving_speeds = df.loc[df['speed_kmh']>=2, 'speed_kmh']
        average_driving_speed = driving_speeds.mean() if not driving_speeds.empty else 0

        sd_speed = df['speed_kmh'].std()
        max_speed = df['speed_kmh'].max()

        standstill_time = df.loc[df['speed_kmh']<2, 'delta_time'].sum()
        percentage_standstill_time = (standstill_time/total_duration_s * 100) if total_duration_s>0 else 0

        # 속도 구간 비율
        speed_bins = [0,15,30,50,70,90,110, np.inf]
        speed_labels = ['3–15 km/h','16–30 km/h','31–50 km/h','51–70 km/h','71–90 km/h','91–110 km/h','>110 km/h']
        df['speed_bin'] = pd.cut(df['speed_kmh'], bins=speed_bins, labels=speed_labels, right=False)
        time_in_speed_bins = df.groupby('speed_bin', observed=False)['delta_time'].sum()
        percentage_time_in_speed_bins = (time_in_speed_bins/total_duration_s)*100 if total_duration_s>0 else 0

        # Aerodynamic Work
        df['delta_distance'] = df['distance'].diff().fillna(0)
        df['aw_component'] = (df['speed_m_s']**2)*df['delta_distance']
        aerodynamic_work = df['aw_component'].sum()/total_distance_m if total_distance_m>0 else 0

        # 가속도 통계
        rms_acceleration = np.sqrt((df['acceleration']**2).mean())
        average_acceleration = df.loc[df['acceleration']>0, 'acceleration'].mean()
        sd_acceleration = df.loc[df['acceleration']>0, 'acceleration'].std()
        max_acceleration = df['acceleration'].max()

        average_deceleration = df.loc[df['acceleration']<0, 'acceleration'].mean()
        sd_deceleration = df.loc[df['acceleration']<0, 'acceleration'].std()
        max_deceleration = df['acceleration'].min()

        acceleration_time = df.loc[df['acceleration']>0, 'delta_time'].sum()
        deceleration_time = df.loc[df['acceleration']<0, 'delta_time'].sum()
        percentage_acceleration_time = (acceleration_time/total_duration_s * 100) if total_duration_s>0 else 0
        percentage_deceleration_time = (deceleration_time/total_duration_s * 100) if total_duration_s>0 else 0

        # 가속도 구간
        accel_bins = [0,0.5,1.0,1.5, np.inf]
        accel_labels = ['0.0–0.5 m/s^2','0.5–1.0 m/s^2','1.0–1.5 m/s^2','>1.5 m/s^2']
        df_accel = df.loc[df['acceleration']>0].copy()
        df_accel['accel_bin'] = pd.cut(df_accel['acceleration'], bins=accel_bins, labels=accel_labels, right=False)
        time_in_accel_bins = df_accel.groupby('accel_bin', observed=False)['delta_time'].sum()
        percentage_time_in_accel_bins = (time_in_accel_bins/total_duration_s)*100 if total_duration_s>0 else 0

        decel_bins = [0,0.5,1.0,1.5, np.inf]
        decel_labels = ['0.0–0.5 m/s^2','0.5–1.0 m/s^2','1.0–1.5 m/s^2','>1.5 m/s^2']
        df_decel = df.loc[df['acceleration']<0].copy()
        df_decel['decel_bin'] = pd.cut(abs(df_decel['acceleration']), bins=decel_bins, labels=decel_labels, right=False)
        time_in_decel_bins = df_decel.groupby('decel_bin', observed=False)['delta_time'].sum()
        percentage_time_in_decel_bins = (time_in_decel_bins/total_duration_s)*100 if total_duration_s>0 else 0

        # Speed change 지표
        df['delta_speed'] = df['speed_m_s'].diff().fillna(0)
        positive_speed_changes = df.loc[df['delta_speed']>0, 'delta_speed']
        negative_speed_changes = df.loc[df['delta_speed']<0, 'delta_speed']

        positive_speed_change_total = (positive_speed_changes**2).sum()
        negative_speed_change_total = (negative_speed_changes**2).sum()

        positive_speed_change_per_km = (positive_speed_change_total/total_distance_km) if total_distance_km>0 else 0
        negative_speed_change_per_km = (negative_speed_change_total/total_distance_km) if total_distance_km>0 else 0

        # PKE / NKE
        df['v_i_squared'] = df['speed_m_s']**2
        df['v_i+1_squared'] = df['v_i_squared'].shift(-1).fillna(0)
        df['delta_v_squared'] = df['v_i+1_squared'] - df['v_i_squared']

        positive_delta_v_squared = df.loc[df['delta_v_squared']>0, 'delta_v_squared'].sum()
        negative_delta_v_squared = df.loc[df['delta_v_squared']<0, 'delta_v_squared'].sum()

        pke = (positive_delta_v_squared/total_distance_m) if total_distance_m>0 else 0
        nke = (negative_delta_v_squared/total_distance_m) if total_distance_m>0 else 0

        # Oscillation / Stop
        df['speed_change'] = (df['speed_kmh']>2)
        df['oscillation'] = df['speed_change']!=df['speed_change'].shift(1)
        num_oscillations_2kmh = df['oscillation'].sum()

        df['speed_change_10'] = (df['speed_kmh']>10)
        df['oscillation_10'] = df['speed_change_10']!=df['speed_change_10'].shift(1)
        num_oscillations_10kmh = df['oscillation_10'].sum()

        total_minutes = total_duration_s/60.0 if total_duration_s>0 else 1e-6
        num_oscillations_2kmh_per_km  = (num_oscillations_2kmh / total_distance_km) if total_distance_km>0 else 0
        num_oscillations_10kmh_per_km = (num_oscillations_10kmh / total_distance_km) if total_distance_km>0 else 0
        num_oscillations_2kmh_per_min  = (num_oscillations_2kmh / total_minutes) if total_minutes>0 else 0
        num_oscillations_10kmh_per_min = (num_oscillations_10kmh / total_minutes) if total_minutes>0 else 0

        df['stopped'] = (df['speed_kmh']<2)
        df['start_moving'] = (~df['stopped']) & (df['stopped'].shift(1)==True)
        num_stops = df['start_moving'].sum()
        num_stops_per_km = (num_stops/total_distance_km) if total_distance_km>0 else 0
        num_stops_per_min = (num_stops/total_minutes) if total_minutes>0 else 0

        df['stop_group'] = (df['stopped'] != df['stopped'].shift()).cumsum()
        stop_durations = df.loc[df['stopped']].groupby('stop_group')['delta_time'].sum()
        average_stop_duration = stop_durations.mean() if not stop_durations.empty else 0

        #-----------------------------------------------------------------------
        # (F) Battery Parameters
        #-----------------------------------------------------------------------
        # 1) mod_temp_list 파싱 -> 평균/최소/최대 모듈 온도
        df['mod_temp_list_parsed'] = df['mod_temp_list'].apply(
            lambda x: [float(v) for v in str(x).split(',') if v.strip() != '']
        )
        df['mod_temp_avg'] = df['mod_temp_list_parsed'].apply(np.mean)
        df['mod_temp_min'] = df['mod_temp_list_parsed'].apply(np.min)
        df['mod_temp_max'] = df['mod_temp_list_parsed'].apply(np.max)

        # 2) cell_volt_list 파싱 -> 평균, 분산, 최대, 최소 전압
        df['cell_volt_list_parsed'] = df['cell_volt_list'].apply(
            lambda x: [float(v) for v in str(x).split(',') if v.strip() != '']
        )
        df['cell_volt_mean'] = df['cell_volt_list_parsed'].apply(np.mean)
        df['cell_volt_var']  = df['cell_volt_list_parsed'].apply(np.var)
        df['cell_volt_max']  = df['cell_volt_list_parsed'].apply(np.max)
        df['cell_volt_min']  = df['cell_volt_list_parsed'].apply(np.min)

        # 3) Initial, Final SOC of Data
        initial_soc = df['soc'].iloc[0]
        final_soc   = df['soc'].iloc[-1]
        final_soh   = df['soh'].iloc[-1]

        # 4) Vehicle Status: odometer, op_time
        final_odometer = df['odometer'].iloc[-1]
        final_op_time  = df['op_time'].iloc[-1]

        #-----------------------------------------------------------------------
        # (G) data_entry
        #-----------------------------------------------------------------------
        data_entry = {
            # (1) Power_data 적분 기반 Used Energy
            # 'Used Energy (kWh)': used_energy_kwh,
            # 'Trip Consumption (kWh/100km)': trip_consumption_kwh_per_100km,

            # (2) Driving Pattern
            'Total length (km)': total_distance_km,
            'Total duration (s)': total_duration_s,
            'Driving duration (s)': driving_duration_s,
            'Average speed (km/h)': average_speed,
            'Average squared speed': average_squared_speed,
            'Average driving speed (>=2km/h)': average_driving_speed,
            'SD of speed (km/h)': sd_speed,
            'Maximum speed (km/h)': max_speed,
            'Percentage of standstill time': percentage_standstill_time,
            'Aerodynamic work (AW)': aerodynamic_work,
            'RMS acceleration (m/s^2)': rms_acceleration,
            'Average acceleration (m/s^2)': average_acceleration,
            'SD of acceleration (m/s^2)': sd_acceleration,
            'Maximum acceleration (m/s^2)': max_acceleration,
            'Average deceleration (m/s^2)': average_deceleration,
            'SD of deceleration (m/s^2)': sd_deceleration,
            'Maximum deceleration (m/s^2)': max_deceleration,
            'Percentage of acceleration time': percentage_acceleration_time,
            'Percentage of deceleration time': percentage_deceleration_time,
            'Positive change of speed per km': positive_speed_change_per_km,
            'Negative change of speed per km': negative_speed_change_per_km,
            'Positive kinetic energy (PKE)': pke,
            'Negative kinetic energy (NKE)': nke,
            'Number of oscillations (>2km/h) per km': num_oscillations_2kmh_per_km,
            'Number of oscillations (>10km/h) per km': num_oscillations_10kmh_per_km,
            'Number of oscillations (>2km/h) per min': num_oscillations_2kmh_per_min,
            'Number of oscillations (>10km/h) per min': num_oscillations_10kmh_per_min,
            'Number of stops per km': num_stops_per_km,
            'Number of stops per min': num_stops_per_min,
            'Average stop duration (s)': average_stop_duration,
        }

        # 속도구간별 시간 비율 추가
        for label in speed_labels:
            data_entry[f'Pct time in speed {label}'] = percentage_time_in_speed_bins.get(label, 0)

        # 가속/감속 구간별 시간 비율 추가
        for label in accel_labels:
            data_entry[f'Pct time in accel {label}'] = percentage_time_in_accel_bins.get(label, 0)
        for label in decel_labels:
            data_entry[f'Pct time in decel {label}'] = percentage_time_in_decel_bins.get(label, 0)

        # (3) Battery Parameter && Vehicle Status
        # --- Battery SOC/SOH ---
        data_entry['Initial SOC (%)'] = initial_soc
        data_entry['Final SOC (%)'] = final_soc
        data_entry['Final SOH (%)'] = final_soh

        # --- 모듈 온도 통계(파일 전체 평균/최소/최대) ---
        data_entry['Avg module temp (°C)'] = df['mod_temp_avg'].mean()
        data_entry['Min module temp (°C)'] = df['mod_temp_min'].min()
        data_entry['Max module temp (°C)'] = df['mod_temp_max'].max()

        # --- 셀 전압 통계(파일 전체 mean/var/max/min의 평균들) ---
        data_entry['Mean of cell_volt_mean (V)'] = df['cell_volt_mean'].mean()
        data_entry['Mean of cell_volt_var (V^2)'] = df['cell_volt_var'].mean()
        data_entry['Max of cell_volt_max (V)'] = df['cell_volt_max'].max()
        data_entry['Min of cell_volt_min (V)'] = df['cell_volt_min'].min()

        # --- Vehicle Status(마지막 값) ---
        data_entry['Final odometer'] = final_odometer
        data_entry['Final op_time'] = final_op_time

        data_list.append(data_entry)

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

################################################################################
# 3) RESULT DATAFRAME
################################################################################

df_factors = pd.DataFrame(data_list).dropna()

print("\n=== df_factors Preview ===")
print(df_factors.head())

################################################################################
# 4) CORRELATION (예: 전비와의 상관관계)
################################################################################

if 'Trip Consumption (kWh/100km)' in df_factors.columns:
    cor_target = df_factors.corr()['Trip Consumption (kWh/100km)'].sort_values(ascending=False)
    print("\n=== Correlation with Trip Consumption (kWh/100km) ===")
    print(cor_target)

# 상관행렬 시각화
corr_matrix = df_factors.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(20,15))
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',
    vmax=1.0,
    vmin=-1.0,
    center=0,
    annot=False,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .8}
)
plt.title('Correlation Matrix of All Parameters')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

################################################################################
# 5) PCA (선택)
################################################################################

X = df_factors.copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=7)
principal_components = pca.fit_transform(X_scaled)
principal_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(7)])

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(
    loadings,
    index=X.columns,
    columns=[f'PC{i+1}' for i in range(7)]
)

print("\n=== PCA Loadings ===")
print(loadings_df)

# threshold 적용 예
loadings_display = loadings_df.copy()
loadings_display[loadings_display.abs()<0.4] = np.nan

plt.figure(figsize=(20,12))
sns.heatmap(
    loadings_display,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=.5
)
plt.title('PCA Loadings (|loading| ≥ 0.4 shown)')
plt.tight_layout()
plt.show()

print("\n=== Explained Variance Ratio ===")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio*100:.2f}%")

plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Scree Plot')
plt.show()
