import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility

# Correctly generate random start points within 1-50 range for each coordinate
start_lat = np.random.rand() * 49 + 1  # Generates a value between 1 and 50
start_lon = np.random.rand() * 49 + 1
start_alt = np.random.rand() * 49 + 1

# Correctly generate random end points within 100-200 range for each coordinate
end_lat = np.random.rand() * 100 + 100  # Generates a value between 100 and 200
end_lon = np.random.rand() * 100 + 100
end_alt = np.random.rand() * 100 + 100

# Generate linear paths for simplicity
time_steps = 200
latitudes = np.linspace(start_lat, end_lat, time_steps)
longitudes = np.linspace(start_lon, end_lon, time_steps)
altitudes = np.linspace(start_alt, end_alt, time_steps) + np.random.normal(0, 1, time_steps)  # Adding slight noise to altitude

# Stack the real coordinate components
real_coords = np.column_stack((latitudes, longitudes, altitudes))

# Add random noise to generate noisy GPS data
noise_std = 2
gps_coords_noisy = real_coords + np.random.normal(0, noise_std, real_coords.shape)


# EKF Initialization
n = 6  # State vector size: [lat, lon, alt, v_lat, v_lon, v_alt]
m = 3  # Measurement vector size: [lat, lon, alt]
X = np.zeros(n)  # Initial state estimate
P = np.eye(n) * 0.1  # Initial estimate error covariance
F = np.eye(n)  # State transition matrix
F[:3, 3:] = np.eye(3)  # Position to velocity transition
Q = np.eye(n) * 0.001  # Process noise covariance matrix
H = np.hstack((np.eye(3), np.zeros((3, 3))))  # Observation matrix
R = np.eye(m) * 0.1  # Measurement noise covariance matrix
ekf_coords = np.zeros((time_steps, 3))  # EKF estimated coordinates

# EKF Algorithm
for t in range(time_steps):
    # Prediction step
    X = F @ X
    P = F @ P @ F.T + Q

    # Update step
    Z = gps_coords_noisy[t]
    Y = Z - H @ X
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    X += K @ Y
    P -= K @ H @ P

    ekf_coords[t, :] = X[:3]  # Only save position information

# Plotting
time = np.arange(time_steps)
fig, axs = plt.subplots(3, 1, figsize=(15, 10))

labels = ['Latitude', 'Longitude', 'Altitude']
colors = ['g', 'r', 'b']  # Colors: green for real, red for noisy GPS, blue for EKF estimate
data_sets = [real_coords, gps_coords_noisy, ekf_coords]
data_labels = ['Real Value', 'Noisy GPS Value', 'EKF Estimated']

for i, label in enumerate(labels):
    for j, data_set in enumerate(data_sets):
        axs[i].plot(time, data_set[:, i], color=colors[j], label=data_labels[j])
    axs[i].set_title(f"{label} Over Time")
    axs[i].set_xlabel('Time Step')
    axs[i].set_ylabel(label)
    axs[i].legend()

plt.tight_layout()
plt.show()
