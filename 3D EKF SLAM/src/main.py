# main.py
import os
import numpy as np
from imu_ekf import predict_state_3d
from landmark_ekf import update_landmarks_3d
from visualize_3d import Visualizer3D
from video_utils import images_to_video

# ---------------- Dataset Setup ----------------
dataset_number =1
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
data_file = os.path.join(data_dir, f'{dataset_number:02d}.npz')

# Generate fake dataset if missing
if not os.path.exists(data_file):
    N, M = 50, 5  # steps, landmarks
    linear_velocity = np.random.rand(N,3)*0.1
    angular_velocity = np.random.rand(N,3)*0.01
    landmarks = np.random.rand(M,3)*5
    features = np.tile(landmarks, (N,1,1)) + np.random.randn(N,M,3)*0.05
    np.savez(data_file, linear_velocity=linear_velocity,
             angular_velocity=angular_velocity,
             features=features)
    print(f"Fake dataset created: {data_file}")

# ---------------- Load Dataset ----------------
data = np.load(data_file)
linear_vel = data['linear_velocity']
angular_vel = data['angular_velocity']
features = data['features']
N, M, _ = features.shape

# ---------------- Simulate GPS (optional) ----------------
gps_available = True
if gps_available:
    # simple GPS: integrate velocities + small noise
    gps = np.cumsum(linear_vel*0.05, axis=0) + np.random.randn(N,3)*0.02
else:
    gps = None

# ---------------- EKF Initialization ----------------
state_dim = 6 + 3*M
mu = np.zeros(state_dim)
Sigma = np.eye(state_dim)*0.01
Q = np.eye(6)*0.0001
R = np.eye(3)*0.05
dt = 0.05

# ---------------- Outputs ----------------
output_dir = 'outputs'
images_dir = os.path.join(output_dir, 'images')
video_dir = os.path.join(output_dir, 'video')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Visualizer with max 20 images
viz = Visualizer3D(save_folder=images_dir, prefix=f'dataset_{dataset_number}', max_frames=20)

# ---------------- EKF-SLAM Loop ----------------
for t in range(N):
    # 1. IMU Prediction
    mu, Sigma = predict_state_3d(mu, Sigma, linear_vel[t], angular_vel[t], Q, dt)

    # 2. Simulated LiDAR Measurements
    lidar_measurements = features[t,:,:3] + np.random.randn(M,3)*0.05

    # 3. Landmark Update
    mu, Sigma = update_landmarks_3d(mu, Sigma, lidar_measurements, R)

    # 4. Update Visualizer (live) with GPS if available
    current_gps = gps[t] if gps is not None else None
    viz.update(mu, [mu[6+3*i:6+3*i+3] for i in range(M)],
               linear_vel=linear_vel, original_landmarks=features[0,:,:3],
               gps=current_gps)

# ---------------- Save final visualization ----------------
viz.save_final()

# ---------------- Generate video ----------------
images_to_video(images_dir, video_dir, video_name=f"ekf_slam_dataset_{dataset_number}.mp4", fps=2)

print(f"All images saved in '{images_dir}', video saved in '{video_dir}'")
