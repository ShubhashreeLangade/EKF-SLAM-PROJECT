import numpy as np

def update_landmarks_3d(mu, Sigma, measurements, R):
    """
    EKF Update for landmarks
    mu: current state
    Sigma: current covariance
    measurements: Nx3 LiDAR measurements
    R: measurement noise (3x3)
    """
    M = len(measurements)
    for i in range(M):
        lm_idx = 6 + 3*i
        z = measurements[i]
        lm_pred = mu[lm_idx:lm_idx+3]
        y = z - lm_pred
        S = Sigma[lm_idx:lm_idx+3,lm_idx:lm_idx+3] + R
        K = Sigma[:, lm_idx:lm_idx+3] @ np.linalg.inv(S)
        mu += K @ y
        Sigma -= K @ Sigma[lm_idx:lm_idx+3, :]
    return mu, Sigma
