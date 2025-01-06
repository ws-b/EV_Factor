import pandas as pd
import numpy as np
import glob
import os
import statsmodels.api as sm
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid

# Define the folder path containing the CSV files
folder_path = r'D:\SamsungSTF\Processed_Data\KOTI'

# Get all CSV files in the folder
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Initialize a list to store the computed data
data_list = []

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

        # Integrate speed over time to get distance (in meters)
        df['speed'] = df['speed'] * 3.6
        distance_m = cumulative_trapezoid(df['speed'], df['time_seconds'], initial=0)
        total_distance_m = distance_m[-1]  # Total distance in meters
        total_distance_km = total_distance_m / 1000  # Convert to kilometers

        # Compute total time in seconds and minutes
        total_time = df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]
        total_minutes = total_time / 60 if total_time > 0 else 0.0001  # Avoid division by zero

        # Compute Idle Time (%)
        # Calculate time differences between consecutive rows
        df['delta_time'] = df['time_seconds'].diff().fillna(0)
        idle_time_total = df.loc[df['speed'] == 0, 'delta_time'].sum()
        idle_time_percentage = (idle_time_total / total_time) * 100 if total_time > 0 else 0

        # Average Speed (in km/h)
        average_speed = df['speed'].mean()

        # Variance of Speed
        SD_speed = df['speed'].std()

        # Average Acceleration
        average_acceleration = df['acceleration'].mean()

        # Variance of Acceleration
        SD_acceleration = df['acceleration'].std()

        # Average External Temperature
        average_ext_temp = df['ext_temp'].mean()

        # Sudden Decelerations per Minute (acceleration < -3 m/s^2)
        num_sudden_decel = df[df['acceleration'] < -3].shape[0] / total_minutes if total_minutes > 0 else 0

        # Sudden Accelerations per Minute (acceleration > 3 m/s^2)
        num_sudden_accel = df[df['acceleration'] > 3].shape[0] / total_minutes if total_minutes > 0 else 0

        # 1. Total Number of Stops per Minute
        df['stopped'] = df['speed'] == 0
        df['start_moving'] = (~df['stopped']) & (df['stopped'].shift(1) == True)
        total_stops = df['start_moving'].sum()
        stops_per_minute = total_stops / total_minutes if total_minutes > 0 else 0

        # 2. Average Positive and Negative Acceleration
        positive_accelerations = df[df['acceleration'] > 0]['acceleration']
        average_positive_acceleration = positive_accelerations.mean() if not positive_accelerations.empty else 0

        negative_accelerations = df[df['acceleration'] < 0]['acceleration']
        average_negative_acceleration = negative_accelerations.mean() if not negative_accelerations.empty else 0

        # 3. Time Spent in Speed Bins
        bins = [0, 20, 40, 60, 80, 100, np.inf]
        labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
        df['speed_bin'] = pd.cut(df['speed'], bins=bins, labels=labels, right=False)
        time_in_bins = df.groupby('speed_bin', observed=False)['delta_time'].sum()
        total_time = df['delta_time'].sum()
        proportion_time_in_bins = time_in_bins / total_time if total_time > 0 else 0
        proportion_time_in_bins = proportion_time_in_bins.to_dict()

        # Total Energy Consumption in kWh
        total_energy_kWh = df['cumul_energy'].iloc[-1]

        # Energy per Unit Distance (Wh/km)
        if total_distance_km > 0:
            energy_per_km = (total_energy_kWh * 1000) / total_distance_km  # Convert kWh to Wh
        else:
            energy_per_km = np.nan  # Handle cases where distance is zero
        # Proportion of Time during Acceleration
        time_accelerating = df.loc[df['acceleration'] > 0, 'delta_time'].sum()
        proportion_time_accelerating = (time_accelerating / total_time) * 100 if total_time > 0 else 0

        # Proportions of Time with Acceleration in Different Bins
        accel_bins = [0, 0.5, 1.0, 1.5, np.inf]
        accel_labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '>1.5']

        df_accel = df[df['acceleration'] > 0].copy()
        df_accel['accel_bin'] = pd.cut(df_accel['acceleration'], bins=accel_bins, labels=accel_labels, right=False)
        time_in_accel_bins = df_accel.groupby('accel_bin', observed=False)['delta_time'].sum()
        proportion_time_in_accel_bins = (time_in_accel_bins / total_time) * 100 if total_time > 0 else 0

        # Proportion of Time during Deceleration
        time_decelerating = df.loc[df['acceleration'] < 0, 'delta_time'].sum()
        proportion_time_decelerating = (time_decelerating / total_time) * 100 if total_time > 0 else 0

        # Proportions of Time with Deceleration in Different Bins
        decel_bins = [0, 0.5, 1.0, 1.5, np.inf]
        decel_labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '>1.5']

        df_decel = df[df['acceleration'] < 0].copy()
        df_decel['decel_bin'] = pd.cut(abs(df_decel['acceleration']), bins=decel_bins, labels=decel_labels, right=False)
        time_in_decel_bins = df_decel.groupby('decel_bin', observed=False)['delta_time'].sum()
        proportion_time_in_decel_bins = (time_in_decel_bins / total_time) * 100 if total_time > 0 else 0

        # Update data_entry with new parameters
        data_entry = {
            'Idle Time (%)': idle_time_percentage,
            'Average Speed': average_speed,
            'SD Speed': SD_speed,
            'Average Acceleration': average_acceleration,
            'SD Acceleration': SD_acceleration,
            'Average Ext Temp': average_ext_temp,
            'Sudden Decelerations per Minute': num_sudden_decel,
            'Sudden Accelerations per Minute': num_sudden_accel,
            'Stops per Minute': stops_per_minute,
            'Average Positive Acceleration': average_positive_acceleration,
            'Average Negative Acceleration': average_negative_acceleration,
            'Total Duration (s)': total_time,
            'Total Distance (km)': total_distance_km,
            'Proportion Time Accelerating (%)': proportion_time_accelerating,
            'Proportion Time Decelerating (%)': proportion_time_decelerating,
            'ECR': energy_per_km
        }

        # Add acceleration bin proportions
        for label in accel_labels:
            proportion = proportion_time_in_accel_bins.get(label, 0)
            data_entry[f'Time in Acceleration {label} (%)'] = proportion

        # Add deceleration bin proportions
        for label in decel_labels:
            proportion = proportion_time_in_decel_bins.get(label, 0)
            data_entry[f'Time in Deceleration {label} (%)'] = proportion

        # Existing code to add time proportions for each speed bin
        for bin_label in labels:
            proportion = proportion_time_in_bins.get(bin_label, 0)
            data_entry[f'Time in Speed {bin_label} (%)'] = proportion * 100  # Convert to percentage

        # Append the data entry to the list
        data_list.append(data_entry)
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Create DataFrame from the data list
df_factors = pd.DataFrame(data_list)

# Drop rows with NaN values (e.g., where total distance was zero)
df_factors = df_factors.dropna()

# Prepare data for regression
X = df_factors.drop('ECR', axis=1)
y = df_factors['ECR']

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the multiple regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(model.summary())
