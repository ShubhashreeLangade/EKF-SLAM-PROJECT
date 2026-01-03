import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer3D:
    """
    Live 3D EKF-SLAM visualizer with 4 tabs
    - Trajectory + IMU (+ optional GPS)
    - Trajectory Only
    - SLAM vs Landmarks
    - 2D Top View (+ optional GPS)
    """

    def __init__(self, save_folder, prefix='frame', max_frames=20):
        self.save_folder = save_folder
        self.prefix = prefix
        self.max_frames = max_frames
        self.frame_count = 0

        os.makedirs(save_folder, exist_ok=True)

        self.mu_history = []
        self.landmarks_history = []
        self.gps_history = []

        plt.ion()
        self.fig, ((self.ax_traj_imu, self.ax_traj),
                   (self.ax_slam, self.ax_2d)) = plt.subplots(2,2, figsize=(16,12), subplot_kw={'projection':'3d'})
        self.ax_2d.remove()
        self.ax_2d = self.fig.add_subplot(2,2,4)

    def update(self, mu, landmarks=None, linear_vel=None, original_landmarks=None, gps=None):
        self.mu_history.append(mu.copy())
        if landmarks is not None:
            self.landmarks_history.append(landmarks)
        if gps is not None:
            self.gps_history.append(gps.copy())

        traj = np.array(self.mu_history)
        gps_arr = np.array(self.gps_history) if len(self.gps_history)>0 else None

        # 1️⃣ Trajectory + IMU
        self.ax_traj_imu.cla()
        self.ax_traj_imu.plot(traj[:,0], traj[:,1], traj[:,2], 'r-', label='EKF Trajectory')
        if gps_arr is not None:
            self.ax_traj_imu.plot(gps_arr[:,0], gps_arr[:,1], gps_arr[:,2], 'g--', label='GPS')
        if linear_vel is not None:
            for i in range(0,len(traj),3):
                self.ax_traj_imu.quiver(traj[i,0], traj[i,1], traj[i,2],
                                        linear_vel[i,0], linear_vel[i,1], linear_vel[i,2],
                                        color='b', length=0.4, normalize=True)
        self.ax_traj_imu.scatter(traj[0,0], traj[0,1], traj[0,2], c='g', s=60, label='Start')
        self.ax_traj_imu.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='k', s=60, label='End')
        self.ax_traj_imu.set_title("Trajectory + IMU + GPS")
        self.ax_traj_imu.set_xlabel("X"); self.ax_traj_imu.set_ylabel("Y"); self.ax_traj_imu.set_zlabel("Z")
        self.ax_traj_imu.legend()

        # 2️⃣ Trajectory Only
        self.ax_traj.cla()
        self.ax_traj.plot(traj[:,0], traj[:,1], traj[:,2], 'b-', label='Trajectory')
        self.ax_traj.scatter(traj[0,0], traj[0,1], traj[0,2], c='g', s=60)
        self.ax_traj.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='r', s=60)
        self.ax_traj.set_title("Trajectory Only")
        self.ax_traj.set_xlabel("X"); self.ax_traj.set_ylabel("Y"); self.ax_traj.set_zlabel("Z")
        self.ax_traj.legend()

        # 3️⃣ SLAM vs Landmarks
        self.ax_slam.cla()
        self.ax_slam.plot(traj[:,0], traj[:,1], traj[:,2], 'r-', label='EKF Trajectory')
        if original_landmarks is not None:
            self.ax_slam.scatter(original_landmarks[:,0], original_landmarks[:,1], original_landmarks[:,2],
                                 c='g', s=50, label='Original Landmarks')
        if landmarks is not None and len(landmarks)>0:
            lm_arr = np.array(landmarks)
            self.ax_slam.scatter(lm_arr[:,0], lm_arr[:,1], lm_arr[:,2], c='r', marker='x', s=50, label='SLAM Landmarks')
            for i in range(len(lm_arr)):
                self.ax_slam.plot([traj[-1,0], lm_arr[i,0]],
                                  [traj[-1,1], lm_arr[i,1]],
                                  [traj[-1,2], lm_arr[i,2]],
                                  'b--', alpha=0.4)
        self.ax_slam.set_title("SLAM vs Landmarks")
        self.ax_slam.set_xlabel("X"); self.ax_slam.set_ylabel("Y"); self.ax_slam.set_zlabel("Z")
        self.ax_slam.legend()

        # 4️⃣ 2D Top View
        self.ax_2d.cla()
        self.ax_2d.plot(traj[:,0], traj[:,1], 'r-', linewidth=2, label='EKF Trajectory')
        if gps_arr is not None:
            self.ax_2d.plot(gps_arr[:,0], gps_arr[:,1], 'g--', label='GPS')
        if landmarks is not None and len(landmarks)>0:
            lm_arr = np.array(landmarks)
            self.ax_2d.scatter(lm_arr[:,0], lm_arr[:,1], c='g', s=30, label='Landmarks')
        self.ax_2d.scatter(traj[0,0], traj[0,1], c='b', s=80, marker='s', label='Start')
        self.ax_2d.scatter(traj[-1,0], traj[-1,1], c='orange', s=80, marker='o', label='End')
        self.ax_2d.set_title("2D Top View")
        self.ax_2d.set_xlabel("X"); self.ax_2d.set_ylabel("Y"); self.ax_2d.grid(True); self.ax_2d.legend()

        plt.pause(0.1)

        if self.frame_count < self.max_frames:
            filename = f"{self.prefix}_{self.frame_count:03d}.png"
            self.fig.savefig(os.path.join(self.save_folder, filename))
        self.frame_count += 1

    def save_final(self):
        self.fig.savefig(os.path.join(self.save_folder, f"{self.prefix}_final.png"))
        plt.ioff()
        plt.show()
