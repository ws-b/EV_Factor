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

# CSV 파일들이 있는 디렉토리 경로를 설정합니다.
directory = r'D:\SamsungSTF\Processed_Data\TripByTrip'
selected_car = 'Ioniq5'

device_ids = vehicle_dict.get(selected_car, [])

if not device_ids:
    print(f"No device IDs found for the selected vehicle: {selected_car}")
    exit()

# 디렉토리에서 CSV 파일을 가져옵니다.
all_files = glob.glob(os.path.join(directory, '*.csv'))

# 단말기 번호가 파일명에 포함된 파일만 선택합니다.
files = [file for file in all_files if any(device_id in os.path.basename(file) for device_id in device_ids)]

# Initialize a list to store the computed data
data_list = []

for filename in tqdm(files):
    try:
        # Read the CSV file
        df = pd.read_csv(filename)

        # Parse 'time' column to datetime
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

        # Sort by time in case it's not sorted
        df = df.sort_values('time').reset_index(drop=True)

        # Calculate time differences in seconds
        df['time_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

        # Calculate delta time between consecutive records
        df['delta_time'] = df['time_seconds'].diff().fillna(0)

        # Convert speed to km/h (assuming original speed is in m/s)
        df['speed_kmh'] = df['speed'] * 3.6  # Convert m/s to km/h

        # Convert speed to m/s (in case it's not already)
        df['speed_m_s'] = df['speed']  # Assuming 'speed' is in m/s

        # Calculate distance using cumulative trapezoidal integration
        distance_m = cumulative_trapezoid(df['speed_m_s'], df['time_seconds'], initial=0)
        df['distance'] = distance_m

        # Compute total distance and duration
        total_distance_m = df['distance'].iloc[-1]
        total_distance_km = total_distance_m / 1000  # Convert to km
        total_duration_s = df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]

        # Driving duration (excluding standstill)
        driving_duration_s = df.loc[df['speed_kmh'] >= 2, 'delta_time'].sum()

        # Average speed (total distance / total duration)
        average_speed = (total_distance_km / (total_duration_s / 3600)) if total_duration_s > 0 else 0

        # Average squared speed
        average_squared_speed = (df['speed_kmh'] ** 2).mean()

        # Average driving speed (excluding v < 2 km/h)
        driving_speeds = df.loc[df['speed_kmh'] >= 2, 'speed_kmh']
        average_driving_speed = driving_speeds.mean() if not driving_speeds.empty else 0

        # Standard deviation (SD) of speed
        sd_speed = df['speed_kmh'].std()

        # Maximum speed
        max_speed = df['speed_kmh'].max()

        # Percentage of standstill time (v < 2 km/h)
        standstill_time = df.loc[df['speed_kmh'] < 2, 'delta_time'].sum()
        percentage_standstill_time = (standstill_time / total_duration_s) * 100 if total_duration_s > 0 else 0

        # Percentage of time in speed intervals
        speed_bins = [0, 15, 30, 50, 70, 90, 110, np.inf]
        speed_labels = ['3–15 km/h', '16–30 km/h', '31–50 km/h', '51–70 km/h', '71–90 km/h', '91–110 km/h', '>110 km/h']
        df['speed_bin'] = pd.cut(df['speed_kmh'], bins=speed_bins, labels=speed_labels, right=False)
        time_in_speed_bins = df.groupby('speed_bin', observed=False)['delta_time'].sum()
        percentage_time_in_speed_bins = (time_in_speed_bins / total_duration_s) * 100 if total_duration_s > 0 else 0

        # Aerodynamic work (AW)
        # AW = Σ(v_i^2 * s_i) / Total Distance
        df['delta_distance'] = df['distance'].diff().fillna(0)  # s_i
        df['aw_component'] = (df['speed_m_s'] ** 2) * df['delta_distance']
        aerodynamic_work = df['aw_component'].sum() / total_distance_m if total_distance_m > 0 else 0

        # Root mean square acceleration/deceleration
        rms_acceleration = np.sqrt((df['acceleration'] ** 2).mean())

        # Average acceleration and deceleration
        average_acceleration = df.loc[df['acceleration'] > 0, 'acceleration'].mean()
        sd_acceleration = df.loc[df['acceleration'] > 0, 'acceleration'].std()
        max_acceleration = df['acceleration'].max()

        average_deceleration = df.loc[df['acceleration'] < 0, 'acceleration'].mean()
        sd_deceleration = df.loc[df['acceleration'] < 0, 'acceleration'].std()
        max_deceleration = df['acceleration'].min()  # Since deceleration is negative

        # Percentage of acceleration time
        acceleration_time = df.loc[df['acceleration'] > 0, 'delta_time'].sum()
        percentage_acceleration_time = (acceleration_time / total_duration_s) * 100 if total_duration_s > 0 else 0

        # Percentage of deceleration time
        deceleration_time = df.loc[df['acceleration'] < 0, 'delta_time'].sum()
        percentage_deceleration_time = (deceleration_time / total_duration_s) * 100 if total_duration_s > 0 else 0

        # Percentage of time in acceleration intervals
        accel_bins = [0, 0.5, 1.0, 1.5, np.inf]
        accel_labels = ['0.0–0.5 m/s^2', '0.5–1.0 m/s^2', '1.0–1.5 m/s^2', '>1.5 m/s^2']
        df_accel = df.loc[df['acceleration'] > 0].copy()
        df_accel['accel_bin'] = pd.cut(df_accel['acceleration'], bins=accel_bins, labels=accel_labels, right=False)
        time_in_accel_bins = df_accel.groupby('accel_bin', observed=False)['delta_time'].sum()
        percentage_time_in_accel_bins = (time_in_accel_bins / total_duration_s) * 100 if total_duration_s > 0 else 0

        # Percentage of time in deceleration intervals
        decel_bins = [0, 0.5, 1.0, 1.5, np.inf]
        decel_labels = ['0.0–0.5 m/s^2', '0.5–1.0 m/s^2', '1.0–1.5 m/s^2', '>1.5 m/s^2']
        df_decel = df.loc[df['acceleration'] < 0].copy()
        df_decel['decel_bin'] = pd.cut(abs(df_decel['acceleration']), bins=decel_bins, labels=decel_labels, right=False)
        time_in_decel_bins = df_decel.groupby('decel_bin', observed=False)['delta_time'].sum()
        percentage_time_in_decel_bins = (time_in_decel_bins / total_duration_s) * 100 if total_duration_s > 0 else 0

        # Positive change of speed per km
        df['delta_speed'] = df['speed_m_s'].diff().fillna(0)
        positive_speed_changes = df.loc[df['delta_speed'] > 0, 'delta_speed']
        positive_speed_change_total = (positive_speed_changes ** 2).sum()
        positive_speed_change_per_km = positive_speed_change_total / total_distance_km if total_distance_km > 0 else 0

        # Negative change of speed per km
        negative_speed_changes = df.loc[df['delta_speed'] < 0, 'delta_speed']
        negative_speed_change_total = (negative_speed_changes ** 2).sum()
        negative_speed_change_per_km = negative_speed_change_total / total_distance_km if total_distance_km > 0 else 0

        # PKE (Positive Kinetic Energy)
        # PKE = Σ(v_{i+1}^2 - v_i^2) / Total Distance, where v_{i+1} > v_i
        df['v_i_squared'] = df['speed_m_s'] ** 2
        df['v_i+1_squared'] = df['v_i_squared'].shift(-1).fillna(0)
        df['delta_v_squared'] = df['v_i+1_squared'] - df['v_i_squared']
        positive_delta_v_squared = df.loc[df['delta_v_squared'] > 0, 'delta_v_squared'].sum()
        pke = positive_delta_v_squared / total_distance_m if total_distance_m > 0 else 0

        # NKE (Negative Kinetic Energy)
        # NKE = Σ(v_{i+1}^2 - v_i^2) / Total Distance, where v_{i+1} < v_i
        negative_delta_v_squared = df.loc[df['delta_v_squared'] < 0, 'delta_v_squared'].sum()
        nke = negative_delta_v_squared / total_distance_m if total_distance_m > 0 else 0

        # Number of oscillations (>2 km/h) per km and per min
        df['speed_change'] = df['speed_kmh'] > 2
        df['oscillation'] = df['speed_change'] != df['speed_change'].shift(1)
        num_oscillations_2kmh = df['oscillation'].sum()
        num_oscillations_2kmh_per_km = num_oscillations_2kmh / total_distance_km if total_distance_km > 0 else 0
        total_minutes = total_duration_s / 60 if total_duration_s > 0 else 0.0001
        num_oscillations_2kmh_per_min = num_oscillations_2kmh / total_minutes if total_minutes > 0 else 0

        # Number of oscillations (>10 km/h) per km and per min
        df['speed_change_10'] = df['speed_kmh'] > 10
        df['oscillation_10'] = df['speed_change_10'] != df['speed_change_10'].shift(1)
        num_oscillations_10kmh = df['oscillation_10'].sum()
        num_oscillations_10kmh_per_km = num_oscillations_10kmh / total_distance_km if total_distance_km > 0 else 0
        num_oscillations_10kmh_per_min = num_oscillations_10kmh / total_minutes if total_minutes > 0 else 0

        # Number of stops per km and per min
        df['stopped'] = df['speed_kmh'] < 2
        df['start_moving'] = (~df['stopped']) & (df['stopped'].shift(1) == True)
        num_stops = df['start_moving'].sum()
        num_stops_per_km = num_stops / total_distance_km if total_distance_km > 0 else 0
        num_stops_per_min = num_stops / total_minutes if total_minutes > 0 else 0

        # Average stop duration
        df['stop_group'] = (df['stopped'] != df['stopped'].shift()).cumsum()
        stop_durations = df.loc[df['stopped']].groupby('stop_group')['delta_time'].sum()
        average_stop_duration = stop_durations.mean() if not stop_durations.empty else 0

        # Prepare data entry
        data_entry = {
            'Total length (km)': total_distance_km,
            'Total duration (s)': total_duration_s,
            'Driving duration (s)': driving_duration_s,
            'Average speed (km/h)': average_speed,
            'Average squared speed': average_squared_speed,
            'Average driving speed (excluding v<2km/h)': average_driving_speed,
            'SD of speed (km/h)': sd_speed,
            'Maximum speed (km/h)': max_speed,
            'Percentage of standstill time (v<2km/h)': percentage_standstill_time,
            'Aerodynamic work (AW)': aerodynamic_work,
            'RMS acceleration/deceleration (m/s^2)': rms_acceleration,
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
            'Average stop duration (s)': average_stop_duration
        }

        # Add percentage of time in speed intervals
        for label in speed_labels:
            percentage = percentage_time_in_speed_bins.get(label, 0)
            data_entry[f'Percentage of time in speed interval {label}'] = percentage

        # Add percentage of time in acceleration intervals
        for label in accel_labels:
            percentage = percentage_time_in_accel_bins.get(label, 0)
            data_entry[f'Percentage of time in acceleration interval {label}'] = percentage

        # Add percentage of time in deceleration intervals
        for label in decel_labels:
            percentage = percentage_time_in_decel_bins.get(label, 0)
            data_entry[f'Percentage of time in deceleration interval {label}'] = percentage

        # Append the data entry to the list
        data_list.append(data_entry)

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Create DataFrame from the data list
df_factors = pd.DataFrame(data_list)

# Drop rows with NaN values
df_factors = df_factors.dropna()

# Prepare data for PCA
X = df_factors.copy()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA to reduce to 6 components
pca = PCA(n_components=6)
principal_components = pca.fit_transform(X_scaled)

# Create DataFrame for principal components
principal_df = pd.DataFrame(data=principal_components,
                            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])


# Compute the correlation matrix between principal components and original variables
# Reconstruct the PCA components to get the loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a DataFrame for loadings
loadings_df = pd.DataFrame(
    loadings,
    index=X.columns,
    columns=[f'PC{i+1}' for i in range(loadings.shape[1])]
)

# Set loadings with absolute value less than 0.4 to NaN
loadings_display = loadings_df.copy()
loadings_display[loadings_display.abs() < 0.4] = np.nan

# Plot the heatmap of correlations between original variables and principal components
plt.figure(figsize=(20, 10))
sns.heatmap(
    loadings_display,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=.5,
    annot_kws={"size": 8}
)
plt.title('Correlation between Principal Components and Original Variables (|loading| ≥ 0.4)')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Alternatively, you can print the loadings
print("Loadings (Correlation between Principal Components and Original Variables):")
print(loadings_df)

# You can also check the explained variance ratio
print("\nExplained Variance Ratio of each Principal Component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio*100:.2f}%")

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Compute the correlation matrix
corr_matrix = df_factors.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(30, 20))

# Draw the heatmap with the mask and correct aspect ratio
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
    cbar_kws={"shrink": .5}
)
plt.title('Correlation Matrix of Original Parameters')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
