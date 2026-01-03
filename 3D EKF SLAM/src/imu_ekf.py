import numpy as np

def predict_state_3d(mu, Sigma, v, omega, Q, dt):
    """
    3D IMU EKF Prediction Step
    mu: current state [x,y,z,roll,pitch,yaw, ...landmarks]
    Sigma: current covariance
    v: linear velocity (3,)
    omega: angular velocity (3,)
    Q: process noise (6x6)
    dt: timestep
    """
    # Ensure correct shapes
    v = np.asarray(v).flatten()
    omega = np.asarray(omega).flatten()
    if v.shape[0] != 3 or omega.shape[0] != 3:
        raise ValueError(f"v and omega must be 3-element vectors, got {v.shape}, {omega.shape}")

    # Predict position & orientation
    mu_pred = mu.copy()
    mu_pred[:3] += v * dt
    mu_pred[3:6] += omega * dt

    # Update covariance
    Sigma_pred = Sigma.copy()
    Sigma_pred[:6,:6] += Q

    return mu_pred, Sigma_pred
