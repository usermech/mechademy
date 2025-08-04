import cv2
import numpy as np
import json
import os
from tqdm import tqdm

# --- Settings ---
scale = 0.5  # Map scale
animation_delay = 50  # ms for preview
alpha = 0.8
beta = -30
actual_traj_scale = 0.5  # Actual traj scaling

# --- File Paths ---
npofmap = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\top_down_map_last.npy"
json_list = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\lists.json"
image_folder = r"C:\Users\ofsaa\Desktop\photoso\rgb_output"
output_video_path = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\map_navigation\combined_animation_map_left_fast.mp4"

# --- Load Map ---
topdown_map = np.load(npofmap)
topdown_map = cv2.convertScaleAbs(topdown_map, alpha=alpha, beta=beta)
topdown_map = 255 - topdown_map
if len(topdown_map.shape) == 2:
    topdown_map = cv2.cvtColor(topdown_map, cv2.COLOR_GRAY2BGR)
topdown_map = cv2.resize(topdown_map, (int(topdown_map.shape[1] * scale), int(topdown_map.shape[0] * scale)))

# --- Load JSON ---
with open(json_list, "r") as f:
    data = json.load(f)
poses = data["poses"]
trajectory = data["trajectory"]
actual_trajectory = data["actual_trajectory"]

# --- Helpers ---
def scale_point(p): return (int(p[0] * scale), int(p[1] * scale))
def scale_point_real_line(p): return (int(-p[1] * scale), int(p[0] * scale))
def transform_actual_point(p, start_actual, start_plan, scale_factor):
    dy = p[0] - start_actual[0]
    dx = -(p[1] - start_actual[1])
    x_img = start_plan[0] + dx * scale_factor
    y_img = start_plan[1] + dy * scale_factor
    return (int(x_img), int(y_img))

# --- Preprocess Trajectories ---
scaled_poses = [scale_point(p) for p in poses]
scaled_traj_xy = [scaled_poses[i] for i in trajectory]
start_actual = actual_trajectory[0]
start_plan = scale_point_real_line(start_actual)
scaled_actual_traj = [transform_actual_point(p, start_actual, start_plan, actual_traj_scale) 
                      for p in actual_trajectory]

# --- Load Image Files ---
def get_image_files(folder):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(extensions)]
    files.sort()
    return [os.path.join(folder, f) for f in files]

image_files = get_image_files(image_folder)
if not image_files:
    raise ValueError("No image files found")

# --- Init Video Writer ---
sample_image = cv2.imread(image_files[0])
img_height, img_width = sample_image.shape[:2]
map_height, map_width = topdown_map.shape[:2]
target_height = min(img_height, map_height)
fps = 20

out_writer = None

# --- Animation ---
image_index = 0
trajectory_history = []

for i in tqdm(range(1, len(scaled_actual_traj)), desc="Generating video"):
    # --- Map Frame ---
    map_frame = topdown_map.copy()
    for x, y in scaled_poses:
        cv2.circle(map_frame, (x, y), 3, (0, 0, 200), -1)
    for j in range(len(scaled_traj_xy) - 1):
        cv2.line(map_frame, scaled_traj_xy[j], scaled_traj_xy[j+1], (50, 200, 30), 7)
    for x, y in scaled_traj_xy:
        cv2.circle(map_frame, (x, y), 4, (0, 200, 0), -1)
    cv2.rectangle(map_frame, (start_plan[0]-5, start_plan[1]-5),
                  (start_plan[0]+5, start_plan[1]+5), (144, 0, 3), 2)
    cv2.rectangle(map_frame, (scaled_traj_xy[-1][0]-5, scaled_traj_xy[-1][1]-5),
                  (scaled_traj_xy[-1][0]+5, scaled_traj_xy[-1][1]+5), (0, 200, 0), 2)

    # Draw actual trajectory with trail
    pt1 = scaled_actual_traj[i-1]
    pt2 = scaled_actual_traj[i]
    trajectory_history.append((pt1, pt2))
    for a, b in trajectory_history:
        cv2.line(map_frame, a, b, (255, 0, 0), 7)

    # --- Camera Image Frame ---
    if image_index < len(image_files):
        img_frame = cv2.imread(image_files[image_index])
        image_index += 1
    else:
        img_frame = cv2.imread(image_files[-1])

    # --- Resize Both to Same Height ---
    img_resized = cv2.resize(img_frame, (int(img_width * target_height / img_height), target_height))
    map_resized = cv2.resize(map_frame, (int(map_width * target_height / map_height), target_height))

    # --- Combine Side-by-Side (Map Left, Camera Right) ---
    combined = np.hstack((map_resized, img_resized))

    # --- Add Labels ---
    cv2.putText(combined, "Map Animation", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(combined, "Camera View", (map_resized.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # --- Init Writer (once we know size) ---
    if out_writer is None:
        out_width = combined.shape[1]
        out_height = combined.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    out_writer.write(combined)

    # Optional preview
    cv2.imshow("Preview", combined)
    if cv2.waitKey(animation_delay) & 0xFF == 27:
        break

# --- Cleanup ---
if out_writer:
    out_writer.release()
cv2.destroyAllWindows()
print(f"Video saved to: {output_video_path}")
