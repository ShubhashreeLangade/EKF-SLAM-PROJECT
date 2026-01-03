import cv2
import os
from glob import glob

def images_to_video(images_folder, video_folder, video_name="output.mp4", fps=5):
    os.makedirs(video_folder, exist_ok=True)
    images = sorted(glob(os.path.join(images_folder, "*.png")))
    if not images:
        print("No images found to make video.")
        return

    # Get size from first image
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video_path = os.path.join(video_folder, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        out.write(img)
    out.release()
    print(f"Video saved: {video_path}")
