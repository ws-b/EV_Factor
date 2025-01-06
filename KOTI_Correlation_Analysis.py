import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid

# Define the folder path containing the CSV files
folder_path = r'D:\SamsungSTF\Processed_Data\KOTI'

# Get all CSV files in the folder
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Initialize a list to store the computed data
data_list = []

# for filename in tqdm(all_files):
#     try:
#         # Read the CSV file
#         df = pd.read_csv(filename)
#
#         # Parse 'time' column to datetime
#         df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
#
#         # Sort by time in case it's not sorted
#         df = df.sort_values('time').reset_index(drop=True)
#
#         # Calculate time differences in seconds
#         df['time_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
#
#         # Integrate speed over time to get distance (in meters)
#         df['speed'] = df['speed'] * 3.6
#         distance_m = cumulative_trapezoid(df['speed'], df['time_seconds'], initial=0)
#         total_distance_m = distance_m[-1]  # Total distance in meters
#         total_distance_km = total_distance_m / 1000  # Convert to kilometers
#
#         # Compute total time in seconds and minutes
#         total_time = df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]
#         total_minutes = total_time / 60 if total_time > 0 else 0.0001  # Avoid division by zero
#
#         # Compute Idle Time (%)
#         # Calculate time differences between consecutive rows
#         df['delta_time'] = df['time_seconds'].diff().fillna(0)
#         idle_time_total = df.loc[df['speed'] == 0, 'delta_time'].sum()
#         idle_time_percentage = (idle_time_total / total_time) * 100 if total_time > 0 else 0
#
#         # Average Speed (in km/h)
#         average_speed = df['speed'].mean()
#
#         # Variance of Speed
#         SD_speed = df['speed'].std()
#
#         # Average Acceleration
#         average_acceleration = df['acceleration'].mean()
#
#         # Variance of Acceleration
#         SD_acceleration = df['acceleration'].std()
#
#         # Average External Temperature
#         average_ext_temp = df['ext_temp'].mean()
#
#         # Sudden Decelerations per Minute (acceleration < -3 m/s^2)
#         num_sudden_decel = df[df['acceleration'] < -3].shape[0] / total_minutes if total_minutes > 0 else 0
#
#         # Sudden Accelerations per Minute (acceleration > 3 m/s^2)
#         num_sudden_accel = df[df['acceleration'] > 3].shape[0] / total_minutes if total_minutes > 0 else 0
#
#         # 1. Total Number of Stops per Minute
#         df['stopped'] = df['speed'] == 0
#         df['start_moving'] = (~df['stopped']) & (df['stopped'].shift(1) == True)
#         total_stops = df['start_moving'].sum()
#         stops_per_minute = total_stops / total_minutes if total_minutes > 0 else 0
#
#         # 2. Average Positive and Negative Acceleration
#         positive_accelerations = df[df['acceleration'] > 0]['acceleration']
#         average_positive_acceleration = positive_accelerations.mean() if not positive_accelerations.empty else 0
#
#         negative_accelerations = df[df['acceleration'] < 0]['acceleration']
#         average_negative_acceleration = negative_accelerations.mean() if not negative_accelerations.empty else 0
#
#         # 3. Time Spent in Speed Bins
#         bins = [0, 20, 40, 60, 80, 100, np.inf]
#         labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
#         df['speed_bin'] = pd.cut(df['speed'], bins=bins, labels=labels, right=False)
#         time_in_bins = df.groupby('speed_bin', observed=False)['delta_time'].sum()
#         total_time = df['delta_time'].sum()
#         proportion_time_in_bins = time_in_bins / total_time if total_time > 0 else 0
#         proportion_time_in_bins = proportion_time_in_bins.to_dict()
#
#         # Total Energy Consumption in kWh
#         total_energy_kWh = df['cumul_energy'].iloc[-1]
#
#         # Energy per Unit Distance (Wh/km)
#         if total_distance_km > 0:
#             energy_per_km = (total_energy_kWh * 1000) / total_distance_km  # Convert kWh to Wh
#         else:
#             energy_per_km = np.nan  # Handle cases where distance is zero
#         # Proportion of Time during Acceleration
#         time_accelerating = df.loc[df['acceleration'] > 0, 'delta_time'].sum()
#         proportion_time_accelerating = (time_accelerating / total_time) * 100 if total_time > 0 else 0
#
#         # Proportions of Time with Acceleration in Different Bins
#         accel_bins = [0, 0.5, 1.0, 1.5, np.inf]
#         accel_labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '>1.5']
#
#         df_accel = df[df['acceleration'] > 0].copy()
#         df_accel['accel_bin'] = pd.cut(df_accel['acceleration'], bins=accel_bins, labels=accel_labels, right=False)
#         time_in_accel_bins = df_accel.groupby('accel_bin', observed=False)['delta_time'].sum()
#         proportion_time_in_accel_bins = (time_in_accel_bins / total_time) * 100 if total_time > 0 else 0
#
#         # Proportion of Time during Deceleration
#         time_decelerating = df.loc[df['acceleration'] < 0, 'delta_time'].sum()
#         proportion_time_decelerating = (time_decelerating / total_time) * 100 if total_time > 0 else 0
#
#         # Proportions of Time with Deceleration in Different Bins
#         decel_bins = [0, 0.5, 1.0, 1.5, np.inf]
#         decel_labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '>1.5']
#
#         df_decel = df[df['acceleration'] < 0].copy()
#         df_decel['decel_bin'] = pd.cut(abs(df_decel['acceleration']), bins=decel_bins, labels=decel_labels, right=False)
#         time_in_decel_bins = df_decel.groupby('decel_bin', observed=False)['delta_time'].sum()
#         proportion_time_in_decel_bins = (time_in_decel_bins / total_time) * 100 if total_time > 0 else 0
#
#         # Update data_entry with new parameters
#         data_entry = {
#             'Idle Time (%)': idle_time_percentage,
#             'Average Speed': average_speed,
#             'SD Speed': SD_speed,
#             'Average Acceleration': average_acceleration,
#             'SD Acceleration': SD_acceleration,
#             'Average Ext Temp': average_ext_temp,
#             'Sudden Decelerations per Minute': num_sudden_decel,
#             'Sudden Accelerations per Minute': num_sudden_accel,
#             'Stops per Minute': stops_per_minute,
#             'Average Positive Acceleration': average_positive_acceleration,
#             'Average Negative Acceleration': average_negative_acceleration,
#             'Total Duration (s)': total_time,
#             'Total Distance (km)': total_distance_km,
#             'Proportion Time Accelerating (%)': proportion_time_accelerating,
#             'Proportion Time Decelerating (%)': proportion_time_decelerating,
#             'ECR': energy_per_km
#         }
#
#         # Add acceleration bin proportions
#         for label in accel_labels:
#             proportion = proportion_time_in_accel_bins.get(label, 0)
#             data_entry[f'Time in Acceleration {label} (%)'] = proportion
#
#         # Add deceleration bin proportions
#         for label in decel_labels:
#             proportion = proportion_time_in_decel_bins.get(label, 0)
#             data_entry[f'Time in Deceleration {label} (%)'] = proportion
#
#         # Existing code to add time proportions for each speed bin
#         for bin_label in labels:
#             proportion = proportion_time_in_bins.get(bin_label, 0)
#             data_entry[f'Time in Speed {bin_label} (%)'] = proportion * 100  # Convert to percentage
#
#         # Append the data entry to the list
#         data_list.append(data_entry)
#     except Exception as e:
#         print(f"Error processing file {filename}: {e}")
for filename in tqdm(all_files):
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

        # Total Energy Consumption in kWh
        total_energy_kWh = df['cumul_energy'].iloc[-1]

        # Energy per Unit Distance (Wh/km)
        if total_distance_km > 0:
            energy_per_km = (total_energy_kWh * 1000) / total_distance_km  # Convert kWh to Wh
        else:
            energy_per_km = np.nan  # Handle cases where distance is zero

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
            'Average stop duration (s)': average_stop_duration,
            'ECR': energy_per_km
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
# Create a DataFrame from the data list
df_factors = pd.DataFrame(data_list)

# Drop rows with NaN values (e.g., where total distance was zero)
df_factors = df_factors.dropna()

# Compute Spearman's rank correlation coefficients
correlation_matrix = df_factors.corr(method='spearman')

# Get correlations between 'ECR' and other factors
correlations = correlation_matrix['ECR'].drop('ECR')

# Sort correlations in descending order
correlations = correlations.sort_values(ascending=False)

# Plot the correlations
plt.figure(figsize=(15, 10))
correlations.plot(kind='barh', color='skyblue', edgecolor='black')
plt.xlabel('Spearman Correlation Coefficient')
plt.title('Spearman Correlation between ECR and Driving Factors')
plt.gca().invert_yaxis()  # To have the highest correlation on top
plt.tight_layout()
plt.show()