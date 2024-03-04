import numpy as np
import matplotlib.pyplot as plt

# Reading noisy GPS data and real coordinates
gps_coords_noisy = np.loadtxt('/Kalman_Filter/gps_coordinates_noisy.txt', delimiter=' ', skiprows=1)
real_coords = np.loadtxt('/Kalman_Filter/real_coordinates.txt', delimiter=' ', skiprows=1)

# Initial parameters for EKF
n = 6  # Size of the state vector: [x, y, z, vx, vy, vz]
m = 3  # Size of the measurement vector: [x, y, z]
X = np.zeros(n)  # Initial state estimate
P = np.eye(n)  # Initial estimate error covariance
F = np.eye(n)  # State transition matrix
Q = np.eye(n) * 0.1  # Process noise covariance matrix
H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])  # Observation matrix
R = np.eye(m) * 0.01  # Measurement noise covariance matrix
ekf_coords = np.zeros_like(gps_coords_noisy)  # Coordinates estimated with EKF

# EKF algorithm
for t in range(len(gps_coords_noisy)):
    # Prediction step
    X_pred = F @ X
    P_pred = F @ P @ F.T + Q

    # Update step
    Z = gps_coords_noisy[t]
    Y = Z - H @ X_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    X = X_pred + K @ Y
    P = (np.eye(n) - K @ H) @ P_pred

    ekf_coords[t] = X[:3]  # Only keep the position information

# Plotting
time = np.arange(len(gps_coords_noisy))
fig, axs = plt.subplots(3, 1, figsize=(15, 10))  # Create 3 rows, 1 column subplot

labels = ['Latitude', 'Longitude', 'Altitude']
colors = ['g', 'r', 'b']  # Colors: green, red, blue
data_sets = [real_coords, gps_coords_noisy, ekf_coords]  # Data sets: Real, Noisy GPS, EKF
data_labels = ['Real Value', 'Noisy GPS Value', 'EKF Estimate']

for i in range(3):  # Loop for each coordinate (lat, lon, alt)
    for j, data_set in enumerate(data_sets):
        axs[i].plot(time, data_set[:, i], color=colors[j], label=data_labels[j])
    axs[i].set_title(f"{labels[i]} Time Series")
    axs[i].set_xlabel('Time Step')
    axs[i].set_ylabel(labels[i])
    axs[i].legend()

plt.tight_layout()
plt.show()
