import numpy as np
import matplotlib.pyplot as plt

class ExtendedKalmanFilter3D:
    def __init__(self, dt, initial_state, initial_covariance, process_noise_var, meas_noise_var):
        self.dt = dt
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_var = process_noise_var
        self.meas_noise_var = meas_noise_var
        self.Q = np.eye(6) * process_noise_var  # Dynamic process noise matrix
        self.R = np.eye(3) * meas_noise_var  # Dynamic measurement noise matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])  # Measurement matrix

    def predict(self, acceleration):
        A = np.array([[1, self.dt, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, self.dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, self.dt],
                      [0, 0, 0, 0, 0, 1]])  # State transition matrix

        acc_vector = np.array([0.5 * self.dt ** 2 * acceleration[0], self.dt * acceleration[0],
                               0.5 * self.dt ** 2 * acceleration[1], self.dt * acceleration[1],
                               0.5 * self.dt ** 2 * acceleration[2], self.dt * acceleration[2]])  # Acceleration vector

        self.state = np.dot(A, self.state) + acc_vector
        self.covariance = np.dot(A, np.dot(self.covariance, A.T)) + self.Q

    def update(self, measurement):
        # Measurement preprocessing: A simple low-pass filter could be applied (directly used here as an example)
        processed_measurement = measurement  # More complex preprocessing steps can be added

        S = np.dot(self.H, np.dot(self.covariance, self.H.T)) + self.R
        K = np.dot(self.covariance, np.dot(self.H.T, np.linalg.inv(S)))
        y = processed_measurement - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)
        self.covariance = self.covariance - np.dot(K, np.dot(self.H, self.covariance))

    def adjust_noise_matrices(self, new_process_noise_var, new_meas_noise_var):
        # Dynamically adjusting process and measurement noise matrices
        self.Q = np.eye(6) * new_process_noise_var
        self.R = np.eye(3) * new_meas_noise_var

# Creating simulation and EKF object
dt = 1.0
initial_state = np.array([0, 0, 0, 0, 0, 0])
initial_covariance = np.eye(6) * 500
process_noise_var = 0.1
meas_noise_var = 5.0

ekf = ExtendedKalmanFilter3D(dt, initial_state, initial_covariance, process_noise_var, meas_noise_var)

# Simulation loop and data collection
total_time = 120
acceleration = np.array([0.0001, 0.0001, 0.0002])

latitudes, longitudes, altitudes = [], [], []
est_latitudes, est_longitudes, est_altitudes = [], [], []

for t in range(1, total_time + 1):
    # Simulating real motion and measurement
    measurement = np.array([acceleration[0] * t ** 2, acceleration[1] * t ** 2, acceleration[2] * t ** 2]) + np.random.normal(0, np.sqrt(meas_noise_var), 3)

    latitudes.append(measurement[0])
    longitudes.append(measurement[1])
    altitudes.append(measurement[2])

    # EKF prediction loop
    ekf.predict(acceleration)
    ekf.update(measurement)

    est_latitudes.append(ekf.state[0])
    est_longitudes.append(ekf.state[2])
    est_altitudes.append(ekf.state[4])

# Visualizing the data
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(latitudes, label='Real Latitude')
axs[0].plot(est_latitudes, label='EKF Latitude Estimation')
axs[0].set_title('Latitude')
axs[0].legend()

axs[1].plot(longitudes, label='Real Longitude')
axs[1].plot(est_longitudes, label='EKF Longitude Estimation')
axs[1].set_title('Longitude')
axs[1].legend()

axs[2].plot(altitudes, label='Real Altitude')
axs[2].plot(est_altitudes, label='EKF Altitude Estimation')
axs[2].set_title('Altitude')
axs[2].legend()

plt.tight_layout()
plt.show()
