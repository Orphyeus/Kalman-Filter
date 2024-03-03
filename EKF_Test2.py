import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

def ekf_update(X, P, Z, Q, R, dt):
    F = np.eye(6) + np.array([[0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]) * dt
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])
    X_pred = F @ X
    P_pred = F @ P @ F.T + Q
    Y = Z - (H @ X_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    X_updated = X_pred + K @ Y
    P_updated = (np.eye(len(X)) - K @ H) @ P_pred
    return X_updated, P_updated

# Parametreler
X = np.zeros(6)  # Başlangıç durumu
P = np.eye(6) * 0.1  # Hata kovaryansı
Q = block_diag(np.eye(3) * 0.02, np.eye(3) * 0.01)  # Süreç gürültüsü
R = np.eye(3) * 0.05  # Gözlem gürültüsü
dt = 0.1  # Zaman adımı

# Simülasyon ve veri toplama
n_steps = 100
true_positions = np.zeros((n_steps, 3))  # Gerçek pozisyonlar
estimated_positions = np.zeros((n_steps, 3))  # Tahmin edilen pozisyonlar

for i in range(n_steps):
    # Gerçek pozisyon (simülasyon için)
    true_position = np.array([np.sin(i * dt), np.cos(i * dt), 0.1 * i])
    true_positions[i, :] = true_position
    
    # GPS ölçümü (gerçek pozisyona gürültü ekleyerek)
    Z = true_position + np.random.normal(0, 0.1, 3)
    
    # EKF güncellemesi
    X, P = ekf_update(X, P, Z, Q, R, dt)
    estimated_positions[i, :] = X[:3]

### 2. Grafik Çizimi

plt.figure(figsize=(12, 8))

# X konumu için grafik
plt.subplot(3, 1, 1)
plt.plot(true_positions[:, 0], label='Gerçek X')
plt.plot(estimated_positions[:, 0], label='Tahmini X')
plt.title('X Pozisyonu')
plt.legend()

# Y konumu için grafik
plt.subplot(3, 1, 2)
plt.plot(true_positions[:, 1], label='Gerçek Y')
plt.plot(estimated_positions[:, 1], label='Tahmini Y')
plt.title('Y Pozisyonu')
plt.legend()

# Z konumu için grafik
plt.subplot(3, 1, 3)
plt.plot(true_positions[:, 2], label='Gerçek Z')
plt.plot(estimated_positions[:, 2], label='Tahmini Z')
plt.title('Z Pozisyonu')
plt.legend()

plt.tight_layout()
plt.show()
