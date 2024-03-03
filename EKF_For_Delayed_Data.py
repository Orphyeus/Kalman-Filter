import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class ExtendedKalmanFilter3D:
    def __init__(self, dt, process_noise_var, meas_noise_var, initial_state, initial_covariance):
        self.dt = dt
        self.state = initial_state
        self.covariance = initial_covariance
        self.Q = np.eye(len(initial_state)) * process_noise_var  # Dynamic process noise matrix
        self.R = np.eye(len(meas_noise_var)) * meas_noise_var  # Dynamic measurement noise matrix
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0], 
                           [0, 0, 1, 0, 0, 0], 
                           [0, 0, 0, 0, 1, 0]])  

    def predict(self, acceleration):
        A = np.eye(len(self.state))  # State transition matrix
        for i in range(3):  # For each axis
            A[i*2, i*2+1] = self.dt
        acc_vector = np.array([0.5 * self.dt**2 * acc for acc in acceleration] + [self.dt * acc for acc in acceleration])
        
        self.state = A @ self.state + acc_vector
        self.covariance = A @ self.covariance @ A.T + self.Q

    def update(self, measurement):
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        self.covariance = (np.eye(len(self.state)) - K @ self.H) @ self.covariance

# Simulation parameters
dt = 1.0
delay = 1  # Delay in seconds
process_noise_var = 0.1
meas_noise_var = np.array([5.0, 5.0, 5.0])
initial_state = np.zeros(6)  # [latitude, latitude_speed, longitude, longitude_speed, altitude, altitude_speed]
initial_covariance = np.eye(6) * 100

ekf = ExtendedKalmanFilter3D(dt, process_noise_var, meas_noise_var, initial_state, initial_covariance)

# Queue to manage delayed measurements
measurement_queue = deque(maxlen=delay+1)  # Include current measurement

# Data collection for plotting
latitudes, longitudes, altitudes = [], [], []
est_latitudes, est_longitudes, est_altitudes = [], [], []

total_time = 120  # Total simulation time in seconds
acceleration = [0.0001, 0.0001, 0.0002]  # Simulated constant acceleration for each axis

for t in range(total_time):
    # Simulate real motion and measurement
    true_position = np.array([0.5 * acc * (t+1)**2 for acc in acceleration])
    measurement = true_position + np.random.normal(0, np.sqrt(meas_noise_var))
    measurement_queue.append((t, measurement))  # Store time and measurement
    
    if t >= delay:
        _, delayed_measurement = measurement_queue.popleft()
        ekf.predict(acceleration)
        ekf.update(delayed_measurement)
    else:
        ekf.predict(acceleration)  # Only predict until we have enough measurements to update
    
    # Collect data for plotting
    latitudes.append(measurement[0])
    longitudes.append(measurement[1])
    altitudes.append(measurement[2])
    est_latitudes.append(ekf.state[0])
    est_longitudes.append(ekf.state[2])
    est_altitudes.append(ekf.state[4])

# Plotting results
time_steps = np.arange(total_time)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(time_steps, latitudes, label='Real Latitude')
plt.plot(time_steps, est_latitudes, label='Estimated Latitude')
plt.title('Latitude Tracking')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(time_steps, longitudes, label='Real Longitude')
plt.plot(time_steps, est_longitudes, label='Estimated Longitude')
plt.title('Longitude Tracking')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(time_steps, altitudes, label='Real Altitude')
plt.plot(time_steps, est_altitudes, label='Estimated Altitude')
plt.title('Altitude Tracking')
plt.legend()

plt.tight_layout()
plt.show()
