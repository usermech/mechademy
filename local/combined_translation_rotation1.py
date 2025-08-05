from lightglue import SuperPoint, LightGlue
from itertools import product
import cv2
import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt

from feature_to_pose import ImagePairPoseEstimator
from feature_to_pose import rotate_360_image
from feature_to_pose import SIFTDetector
from feature_to_pose import SIFTMatcher
from feature_to_pose import ORBDetector
from feature_to_pose import ORBMatcher
from feature_to_pose import ORB_FLANN_Matcher
from feature_to_pose import print_progress_bar

start_time = time.time()
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Select one of the algortihm for ImagePairPoseEstimator object: LigthGlue, SIFT, ORB, ORB_FLANN
    pose_estimator_alg = "LightGlue" 
    if pose_estimator_alg == "LightGlue":
        detector = SuperPoint(max_num_keypoints=1024).eval().to(device)
        matcher  = LightGlue(features="superpoint").eval().to(device)
    elif pose_estimator_alg == "SIFT":
        detector = SIFTDetector()
        matcher  = SIFTMatcher()
    elif pose_estimator_alg == "ORB":
        detector = ORBDetector()
        matcher  = ORBMatcher()
    elif pose_estimator_alg == "ORB_FLANN":
        detector = ORBDetector()
        matcher  = ORB_FLANN_Matcher()
    else:
        raise ValueError("Select a valid image pair posing algorithm\nValid options are: LigthGlue, SIFT, ORB, ORB_FLANN")
    
    # Create an instance of ImagePairPoseEstimator with configured detector and matcher
    estimator = ImagePairPoseEstimator(detector, matcher, image_width=1024, image_height=512)

    # Dictionary containing error arrays with inputs as keys and results as values
    err_dict = dict()
    count = 0
    matricies = np.zeros((2*4*36*36,4,4))
    x0 = np.zeros((2*4*36*36,))
    y0 = np.zeros((2*4*36*36,))
    x1 = np.zeros((2*4*36*36,))
    y1 = np.zeros((2*4*36*36,))
    actl_yaws = np.zeros((2*4*36*36,))
    
    # Iterate over different center points
    for (cntr_x, cntr_y), dist in product([(610, 460), (720, 800), (830, 920), (1110, 420)], [r"30cm", r"50cm"]):

        # Set up file and folder path
        img_center = "img_" + str(cntr_x) + "_" + str(cntr_y) + ".jpg"
        center_folder_path = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\image_o\Test translation images\Center Points"
        img_dir_path = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\image_o\Test translation images"
        img_dir_path = os.path.join(img_dir_path, dist, str(cntr_x) + "_" + str(cntr_y))

        # Read center image and images around with cv2.imread()
        img0 = cv2.imread(os.path.join(center_folder_path, img_center))
        img_dict = {f:(cv2.imread(os.path.join(img_dir_path, f))) for f in os.listdir(img_dir_path)}

        # Iterate over images in specified directory
        for i, (img_name, img) in enumerate(img_dict.items()):
            print(f"{78 * '-'}\nReading {img_name} in comparison to {img_center}\n{78 * '-'}")
            coor_x, coor_y = map(int, img_name[4:-4].split('_')) # Get x, y coordinates from the file name: "img_xcoor_ycoor.jpg"
            img_unrotated = img_dict[img_name]

            # Rotate images
            for actl_yaw in range(-180,180,10):
                img_unrotated, img1 = rotate_360_image(img_unrotated, actl_yaw)

                # Calculate the R and t while measuring the runtime duration
                time_start = time.perf_counter()
                cam_pose, R, t = estimator.estimate(img0, img1)
                time_elapsed = time.perf_counter() - time_start

                # Calculate actual heading and calculated heading angle
                actl_heading = np.rad2deg(np.arctan2((coor_y-cntr_y),(coor_x-cntr_x)))
                calc_heading = np.rad2deg(np.arctan2(t[1],t[0]))
                heading_err = (calc_heading - actl_heading + 180) % 360 - 180
                
                # Calculate actual yaw and yaw heading angle
                calc_yaw = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
                yaw_err = (calc_yaw - actl_yaw + 180) % 360 - 180
                    
                # Print actual and calculated angles
                print(f"{'Calculated heading:':<27}{calc_heading:>10.4f} deg | {'Calculated yaw:':<20}{calc_yaw:>10.4f} deg")
                print(f"{'Actual heading:':<27}{actl_heading:>10.4f} deg | {'Actual yaw:':<20}{actl_yaw:>10.4f} deg")
                print(f"{'Error in heading angle:':<27}{heading_err:>10.4f} deg | {'Error in yaw angle:':<20}{yaw_err:>10.4f} deg")
                print(f"{'Elapsed calculation time:':<27}{time_elapsed:>10.4f} sec | \n{78 * '-'}")
                err_dict[(img_center,img_name,actl_yaw,actl_heading)] = (calc_yaw, yaw_err, calc_heading, heading_err, time_elapsed)
                matricies[count,:,:] = cam_pose
                x0[count] = cntr_x
                y0[count] = cntr_y
                x1[count] = coor_x
                y1[count] = coor_y
                actl_yaws[count] = actl_yaw
                count += 1
                print_progress_bar(len(err_dict),2*4*36*36)

            print(f"Reading {img_name} is done")
            
    # Save values to a csv file
    csv_file = 'combined_results_' + pose_estimator_alg + '.csv'
    csv_header = 'img0_name,img1_name,actl_yaw,actl_heading,calc_yaw,yaw_err,calc_heading,heading_err,time_elapsed\n'
    with open(csv_file, 'w') as f:
        f.write(csv_header)
        for err_in, err_out in err_dict.items():
            img0_name, img1_name, actl_yaw, actl_heading = err_in
            calc_yaw, yaw_err, calc_heading, heading_err, time_elapsed = err_out
            line = f"{img0_name},{img1_name},{actl_yaw:.6f},{actl_heading:.6f},{calc_yaw:.6f},{yaw_err:.6f},{calc_heading:.6f},{heading_err:.6f},{time_elapsed:.6f}\n"
            f.write(line)
    
    np.savez_compressed("combined_cam_pose_results_" + pose_estimator_alg + ".npz", x0=x0, y0=y0, x1=x1, y1=y1, actl_yaws=actl_yaws, matricies=matricies)

end_time = time.time()
real_time = end_time - start_time

#### PLOT ####
# Extract absolute errors
OUTLIER_THRESHOLD = 25.0

actual_yaws = []
yaw_errors = []
actual_headings = []
heading_errors = []

for key, value in err_dict.items():
    _, _, actl_yaw, actl_heading = key
    calc_yaw, yaw_err, calc_heading, heading_err, _ = value

    if abs(yaw_err) <= OUTLIER_THRESHOLD:
        actual_yaws.append(actl_yaw)
        yaw_errors.append(abs(yaw_err))
    if abs(heading_err) <= OUTLIER_THRESHOLD:
        actual_headings.append(actl_heading)
        heading_errors.append(abs(heading_err))

# ---------- Bar Plot (Distribution by threshold) ----------
# Define thresholds
thresholds = [5.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
threshold_labels = [f"error < {t:.1f} deg" for t in thresholds]

def compute_percentages(errors, thresholds):
    percentages = []
    total = len(errors)
    for t in thresholds:
        count = sum(e < t for e in errors)
        percentage = (count / total) * 100 if total > 0 else 0
        percentages.append(percentage)
    return percentages

yaw_percentages = compute_percentages(yaw_errors, thresholds)
heading_percentages = compute_percentages(heading_errors, thresholds)
x = np.arange(len(thresholds))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, yaw_percentages, width, label='Yaw Error (%)')
rects2 = ax.bar(x + width/2, heading_percentages, width, label='Heading Error (%)')

ax.set_ylabel('Percentage of samples (%)')
ax.set_xlabel('Absolute Error Threshold')
ax.set_title(f'Distribution of Absolute Errors (Filtered < {OUTLIER_THRESHOLD}°)')
ax.set_xticks(x)
ax.set_xticklabels(threshold_labels)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.tight_layout()
plt.show()

# ---------- Scatter Plot ----------
# --- Yaw Error Plot --- #
plt.figure(figsize=(10, 5))
plt.scatter(actual_yaws, yaw_errors, color='tomato', edgecolors='k', alpha=0.7)
plt.xlabel("Actual Yaw (°)")
plt.ylabel("Yaw Error (°)")
plt.title("Yaw Error vs. Actual Yaw")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --- Heading Error Plot --- #
plt.figure(figsize=(10, 5))
plt.scatter(actual_headings, heading_errors, color='royalblue', edgecolors='k', alpha=0.7)
plt.xlabel("Actual Heading (°)")
plt.ylabel("Heading Error (°)")
plt.title("Heading Error vs. Actual Heading")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ---------- Summary Statistics ----------
yaw_errors = np.array(yaw_errors)
heading_errors = np.array(heading_errors)

yaw_mean_abs_error = np.mean(yaw_errors)
yaw_max_abs_error = np.max(yaw_errors)
yaw_std_deviation = np.std(yaw_errors)

heading_mean_abs_error = np.mean(heading_errors)
heading_max_abs_error = np.max(heading_errors)
heading_std_deviation = np.std(heading_errors)

print("\n--- 🔍 Performance Summary (Errors < 25°) ---")
print(f"Yaw Mean Absolute Error      : {yaw_mean_abs_error:.3f}°")
print(f"Yaw Max Absolute Error       : {yaw_max_abs_error:.3f}°")
print(f"Yaw Standard Deviation (σ)   : {yaw_std_deviation:.3f}°")
print()
print(f"Heading Mean Absolute Error  : {heading_mean_abs_error:.3f}°")
print(f"Heading Max Absolute Error   : {heading_max_abs_error:.3f}°")
print(f"Heading Standard Deviation (σ): {heading_std_deviation:.3f}°")
print(f"Execution time is {real_time:.2f} seconds")