import cv2
import numpy as np
import json
import time
import os

# --- Settings ---
scale = 0.5  # 50% scaling
animation_delay = 60   # ms
alpha = 0.8
beta = -30
actual_traj_scale = 0.5  # Scaling factor for actual trajectory

# --- File paths ---
npofmap = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\top_down_map_last.npy"
json_list = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\lists.json"
image_folder = r"C:\Users\ofsaa\Desktop\photoso\rgb_output"  # Folder containing the images for the video
output_video_path = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\map_navigation\combined_animation_camera_left_fast.mp4"

# --- Load data ---
topdown_map = np.load(npofmap)
topdown_map = cv2.convertScaleAbs(topdown_map, alpha=alpha, beta=beta)

if topdown_map.dtype != np.uint8:  
    topdown_map = topdown_map.astype(np.uint8)

topdown_map = 255 - topdown_map

if len(topdown_map.shape) == 2:
    topdown_map = cv2.cvtColor(topdown_map, cv2.COLOR_GRAY2BGR)

new_size = (int(topdown_map.shape[1] * scale), int(topdown_map.shape[0] * scale))
topdown_map = cv2.resize(topdown_map, new_size, interpolation=cv2.INTER_AREA)
topdown_map = np.ascontiguousarray(topdown_map)

with open(json_list, "r") as f:
    data = json.load(f)

poses = data["poses"]
trajectory = data["trajectory"]
actual_trajectory = data["actual_trajectory"]

def scale_point(p):
    return (int((p[0] * scale)), int(p[1] * scale))

def scale_point_real_line(p):
    return (int(-(p[1] * scale)), int(p[0] * scale))

scaled_poses = [scale_point(p) for p in poses]
scaled_traj_xy = [scaled_poses[i] for i in trajectory]

def transform_actual_point(p, start_actual, start_plan, scale_factor):
    dy = p[0] - start_actual[0]
    dx = -(p[1] - start_actual[1])
    x_img = start_plan[0] + dx * scale_factor
    y_img = start_plan[1] + dy * scale_factor
    return (int(x_img), int(y_img))

start_actual = actual_trajectory[0]
start_plan = scale_point_real_line(start_actual)
scaled_actual_traj = [transform_actual_point(p, start_actual, start_plan, actual_traj_scale) for p in actual_trajectory]

def get_image_files(folder):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(extensions)]
    files.sort()
    return [os.path.join(folder, f) for f in files]

image_files = get_image_files(image_folder)
if not image_files:
    raise ValueError("No image files found in the specified folder")

sample_image = cv2.imread(image_files[0])
img_height, img_width = sample_image.shape[:2]

map_height, map_width = topdown_map.shape[:2]

fps = 20

output_height = min(img_height, map_height)

# --- Video writer ---
output_width = 0  # hesaplama aşağıda yapılacak
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

image_index = 0
max_image_index = len(image_files) - 1
trajectory_history = []

for i in range(1, len(scaled_actual_traj)):
    frame = topdown_map.copy()

    # Draw map points and trajectories
    for x, y in scaled_poses:
        cv2.circle(frame, (x, y), 3, (0, 0, 200), -1)
    for j in range(len(scaled_traj_xy) - 1):
        cv2.line(frame, scaled_traj_xy[j], scaled_traj_xy[j + 1], (50, 200, 30), 7)
    for x, y in scaled_traj_xy:
        cv2.circle(frame, (x, y), 4, (0, 200, 0), -1)

    cv2.rectangle(frame, (start_plan[0] - 5, start_plan[1] - 5),
                  (start_plan[0] + 5, start_plan[1] + 5), (144, 0, 3), 2)
    cv2.rectangle(frame, (scaled_traj_xy[-1][0] - 5, scaled_traj_xy[-1][1] - 5),
                  (scaled_traj_xy[-1][0] + 5, scaled_traj_xy[-1][1] + 5), (0, 200, 0), 2)

    pt1 = scaled_actual_traj[i - 1]
    pt2 = scaled_actual_traj[i]
    trajectory_history.append((pt1, pt2))

    for a, b in trajectory_history:
        cv2.line(frame, a, b, (200, 0, 0), 7)

    if image_index <= max_image_index:
        img_frame = cv2.imread(image_files[image_index])
        image_index += 1
    else:
        img_frame = cv2.imread(image_files[-1])

    # --- Resize both to same height (output_height) ---
    map_resized = cv2.resize(frame, (int(map_width * output_height / map_height), output_height))
    img_resized = cv2.resize(img_frame, (int(img_width * output_height / img_height), output_height))

    combined_width = map_resized.shape[1] + img_resized.shape[1]

    combined_frame = np.zeros((output_height, combined_width, 3), dtype=np.uint8)

    # Map solda, kamera sağda
    combined_frame[:, :map_resized.shape[1]] = map_resized
    combined_frame[:, map_resized.shape[1]:] = img_resized

    combined_frame[:, :img_resized.shape[1]] = img_resized
    combined_frame[:, img_resized.shape[1]:] = map_resized
    
    # Etiketleri de değiştirelim:
    cv2.putText(combined_frame, "Camera View", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_frame, "Map Animation", (img_resized.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if out is None:
        output_width = combined_width
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    out.write(combined_frame)

    cv2.imshow("Combined Animation Preview", combined_frame)
    if cv2.waitKey(animation_delay) & 0xFF == 27:
        break

out.release()
cv2.destroyAllWindows()

print(f"Combined animation saved to: {output_video_path}")
