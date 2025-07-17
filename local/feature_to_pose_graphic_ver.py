from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional

import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from lightglue.utils import numpy_image_to_torch
from eightpa_solvers.camera_recovering import get_cam_pose_by_ransac_8pa

start_time = time.time()

def normalize_angle_deg(angle_deg):
    return (angle_deg + 360) % 360

@dataclass
class ImagePairPoseEstimator:
    detector:      torch.nn.Module
    matcher:       torch.nn.Module
    image_width:   int = 512
    image_height:  int = 256
    device:        str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # ---------- public API --------------------------------------------------
    def estimate(
        self,
        img_path0: str | np.ndarray,
        img_path1: str | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate relative pose between two equirectangular images.

        Parameters
        ----------
        img_path0, img_path1 : str | ndarray
            Filenames or already‑loaded RGB images (H×W×3, BGR or RGB – both OK).

        Returns
        -------
        cam_pose : (4,4) np.ndarray  – RT matrix with last row [0 0 0 1]
        R        : (3,3) np.ndarray  – rotation
        t        : (3,)  np.ndarray  – translation (unit scale)
        """
        img0 = self._load_rgb(img_path0)
        img1 = self._load_rgb(img_path1)

        feat0 = self._extract(img0)
        feat1 = self._extract(img1)
        matches = self._match(feat0, feat1)

        if matches.size == 0:
            raise RuntimeError("No matches returned by matcher.")

        # matched keypoints in image coords (pixel centres)
        kpts0 = feat0["keypoints"][0].cpu().numpy()[matches[:, 0]]
        kpts1 = feat1["keypoints"][0].cpu().numpy()[matches[:, 1]]

        # convert to 3‑D unit bearings on the unit sphere
        u0 = self._keypoints_to_unit_vectors(kpts0)
        u1 = self._keypoints_to_unit_vectors(kpts1)

        cam_pose = get_cam_pose_by_ransac_8pa(u0.T, u1.T)
        R, t = cam_pose[:3, :3], cam_pose[:3, 3]
        return cam_pose, R, t

    # ---------- helper methods ---------------------------------------------
    def _extract(self, rgb: np.ndarray) -> dict:
        tensor = numpy_image_to_torch(rgb).to(self.device)
        with torch.no_grad():
            return self.detector.extract(tensor)

    def _match(self, feat0: dict, feat1: dict) -> np.ndarray:
        with torch.no_grad():
            pred = self.matcher({"image0": feat0, "image1": feat1})
        return pred["matches"][0].cpu().numpy()

    def _keypoints_to_unit_vectors(self, kpts: np.ndarray) -> np.ndarray:
        az, el = self._pixel_to_angles(kpts)
        return self._angles_to_vectors(az, el)

    # ---------- static utility fns -----------------------------------------
    def _pixel_to_angles(self, kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Equirect. projection: pixel → (azimuth, elevation) in rad.
        0° azimuth = -π, 360° = π ;  0° elevation = +π/2 (north pole)
        """
        x, y = kpts[:, 0], kpts[:, 1]
        az  = (x / self.image_width) * 2 * np.pi - np.pi
        el  = (0.5 - y / self.image_height) * np.pi
        return az, el
    

    #Translation vektöründen heading açısını (derece cinsinden) hesapla.
    def compute_heading_angle(x1, y1, x2, y2):
        dx = x2 - x1  # East component
        dy = y2 - y1  # North component
        heading_rad = np.arctan2(dx, dy)  # Angle from NORTH (y-axis)
        heading_deg = np.degrees(heading_rad)
        return heading_deg
    
     #Normalize any angle (in degrees) to [0, 360).



    @staticmethod
    def _angles_to_vectors(az: np.ndarray, el: np.ndarray) -> np.ndarray:
        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)
        return np.stack((x, y, z), axis=1)

    @staticmethod
    def _load_rgb(src: str | np.ndarray) -> np.ndarray:
        if isinstance(src, np.ndarray):
            rgb = src
        else:
            bgr = cv2.imread(src, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(src)
            rgb = bgr[:, :, ::-1]
        return rgb  # H×W×3, RGB

# Rotating image function
def rotate_360_image(img, degrees, output_path=None):
    '''
    Take an 360 photo as a np.array and rotate it CCW in degrees 
    '''
    
    # Load the 360 image if a path is given instead of image
    if type(img) == type("str"):
        img = cv2.imread(img)

    # Raise error if image is not found
    if img is None:
        raise ValueError("Image could not be loaded. Check the path.")

    height, width, _ = img.shape

    # Convert degrees to pixels
    shift = int((degrees / 360.0) * width)

    # Perform horizontal rotation by shifting pixels (CCW is positive)
    rotated_img = np.roll(img, -shift, axis=1)

    # Save the rotated image if an output path is given
    if not (output_path is None):
        cv2.imwrite(output_path, rotated_img)
        print(f"Rotated image saved to {output_path}")

    # Return the rotated image and original image
    return img, rotated_img


# -----------------------------------------------------
### MIMIC LightGlue and SuperPoint for SIFT methods ###
########### I have no clue how this works #############
class SIFTDetector:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def extract(self, image_tensor: torch.Tensor) -> dict:
        # Convert from torch.Tensor (1, 3, H, W) → numpy RGB
        image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        image_gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(image_gray, None)

        kpts = np.array([kp.pt for kp in keypoints], dtype=np.float32)  # Nx2

        # Convert to torch.Tensor to keep interface consistent
        return {
            "keypoints": torch.from_numpy(kpts)[None],  # add batch dim
            "descriptors": torch.from_numpy(descriptors)[None]
        }

class SIFTMatcher:
    def __init__(self):
        index_params = dict(algorithm=1, trees=5)  # FLANN with KD-Tree
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def __call__(self, feats: dict) -> dict:
        desc0 = feats["image0"]["descriptors"].squeeze().cpu().numpy()
        desc1 = feats["image1"]["descriptors"].squeeze().cpu().numpy()

        # Match descriptors using KNN
        matches = self.matcher.knnMatch(desc0, desc1, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m.queryIdx, m.trainIdx])

        return {"matches": torch.tensor(good_matches, dtype=torch.long)[None]}  # batch dim
# -----------------------------------------------------
########## ChatGPT generated code ends here ###########

# -----------------------------------------------------
### MIMIC LightGlue and SuperPoint for ORB methods ####
########### I have no clue how this works #############
class ORBDetector:
    def __init__(self, nfeatures=1000):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

    def extract(self, image_tensor: torch.Tensor) -> dict:
        # Convert from torch.Tensor (1, 3, H, W) to uint8 RGB
        image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        image_gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(image_gray, None)

        if not keypoints or descriptors is None:
            return {"keypoints": torch.zeros((1, 0, 2)), "descriptors": torch.zeros((1, 0, 32), dtype=torch.uint8)}

        kpts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        return {
            "keypoints": torch.from_numpy(kpts)[None],  # (1, N, 2)
            "descriptors": torch.from_numpy(descriptors)[None]  # (1, N, 32)
        }

class ORBMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def __call__(self, feats: dict) -> dict:
        desc0 = feats["image0"]["descriptors"].squeeze().cpu().numpy()
        desc1 = feats["image1"]["descriptors"].squeeze().cpu().numpy()

        if desc0.shape[0] == 0 or desc1.shape[0] == 0:
            return {"matches": torch.zeros((1, 0, 2), dtype=torch.long)}

        matches = self.matcher.knnMatch(desc0, desc1, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m.queryIdx, m.trainIdx])

        return {"matches": torch.tensor(good_matches, dtype=torch.long)[None]}
    
class ORB_FLANN_Matcher:
    def __init__(self):
        index_params = dict(algorithm=6,  # LSH index
                            table_number=6,      # 12 is good for larger datasets
                            key_size=12,         # 20 is good for high precision
                            multi_probe_level=1) # 2 improves recall
        search_params = dict(checks=50)

        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def __call__(self, feats: dict) -> dict:
        desc0 = feats["image0"]["descriptors"].squeeze().cpu().numpy()
        desc1 = feats["image1"]["descriptors"].squeeze().cpu().numpy()

        # ORB descriptors must be type uint8 for LSH-based FLANN
        if desc0.dtype != np.uint8 or desc1.dtype != np.uint8:
            desc0 = desc0.astype(np.uint8)
            desc1 = desc1.astype(np.uint8)

        if desc0.shape[0] == 0 or desc1.shape[0] == 0:
            return {"matches": torch.zeros((1, 0, 2), dtype=torch.long)}

        matches = self.matcher.knnMatch(desc0, desc1, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) < 2:
                continue  # skip if not enough matches
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append([m.queryIdx, m.trainIdx])

        return {"matches": torch.tensor(good_matches, dtype=torch.long)[None]}
# -----------------------------------------------------
########## ChatGPT generated code ends here ###########


# ---------------- example usage ---------------------------------------------
if __name__ == "__main__":
    from lightglue import SuperPoint, LightGlue
    import os

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
    estimator = ImagePairPoseEstimator(detector, matcher, image_width=720, image_height=360)

    # Read images in a specified directory. Directory only contains image files.
    img_dir_path = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\local_sim_photo_720"
    img_dict = {f:(cv2.imread(os.path.join(img_dir_path, f))) for f in os.listdir(img_dir_path)}

    # Iterate over images in specified directory
    err_dict = dict() # Dict containing err_arr for each file name (image_name)
    for img_name, img in img_dict.items():
        print(f"-----------------------------\nReading {img_name}\n-----------------------------")
        err_arr = np.empty((73,2),np.float64) # Array containing actual orientation angle and error
        
        # Rotate 360 images in z-axis by 5 deg increments
        for i, deg in enumerate(range(-180,185,5)):
            img0, img1 = rotate_360_image(img, deg)
     
            cam_pose, R, t = estimator.estimate(img0, img1)

            yaw = np.arctan2(R[1, 0], R[0, 0])  #This is in radians
            print(f"Actual yaw angle {deg}")
            print(f"Calculated yaw angle {np.rad2deg(yaw)}")
            print("-----------------------------")
            print(f"{i} | Angle {deg} | Error {np.rad2deg(yaw) - deg}")
            err_arr[i,0] = deg # Actual rotation angle
            err_arr[i,1] = np.rad2deg(yaw) - deg # Error in degrees

            # bu kodu sor!!!!
            heading_angle = np.arctan2(t[1],t[0])   #This is in radians
            print(f"Translation vector" , t)
            print("this is heading angle:", heading_angle)
        print(f"-----------------------------\nReading {img_name} is done\n-----------------------------")
        err_dict[img_name] = err_arr


# Tüm hataları tek bir diziye topla
exclude_angles = {-180, -115, 115, 180}

all_errors = []
for err_arr in err_dict.values():
    for angle, error in err_arr:
        if int(angle) in exclude_angles:
            continue  # bu açıları atla
        all_errors.append(error)

all_errors = np.array(all_errors)
# Özet istatistikler
mean_abs_error = np.mean(np.abs(all_errors))       # Ortalama hata (mutlak)
max_abs_error = np.max(np.abs(all_errors))         # Maksimum hata (mutlak)
std_deviation = np.std(all_errors)                 # Standart sap
# Sonuçları yazdır
print("\n--- Performans Özeti ---")
print(f"Ortalama Hata (°): {mean_abs_error:.3f}")
print(f"Maksimum Hata (°): {max_abs_error:.3f}")
print(f"Standart Sapma (σ): {std_deviation:.3f}")

    
    # Plot image rotation vs. rotation error onto same scatter plot
angle_to_errors = defaultdict(list)

# Bütün fotoğrafların hata değerlerini açılara göre grupla
for err_arr in err_dict.values():
    for angle, error in err_arr:
        angle_to_errors[int(angle)].append(error)

# Tüm açılar için ortalama ve min/max hesapla
angles = sorted(angle_to_errors.keys())
mean_errors = []
min_errors = []
max_errors = []

for angle in angles:
    errors = angle_to_errors[angle]
    mean = np.mean(errors)
    min_err = np.min(errors)
    max_err = np.max(errors)

    mean_errors.append(mean)
    min_errors.append(min_err)
    max_errors.append(max_err)

# Errorbar grafiğini çiz
mean_errors = np.array(mean_errors)
min_errors = np.array(min_errors)
max_errors = np.array(max_errors)
lower_error = mean_errors - min_errors
upper_error = max_errors - mean_errors
asymmetric_error = [lower_error, upper_error]

end_time = time.time()
elapsed = end_time - start_time
print(f"\n✅ İşlem tamamlandı. Toplam süre: {elapsed:.2f} saniye")


plt.figure(figsize=(12, 6))
plt.errorbar(
    angles,
    mean_errors,
    yerr=asymmetric_error,
    fmt='o',
    ecolor='gray',
    capsize=3,
    markersize=5,
    label=pose_estimator_alg,
    color='green',
    elinewidth=1
)

# Eksen ayarları
plt.xlim((-185, 185))
plt.xticks(np.arange(-180, 181, 30))
plt.ylim((-0.1, 0.1))
plt.yticks(np.arange(-0.1, 0.1, 0.01))

# Yardımcı çizgi ve başlıklar
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Image rotation in degrees")
plt.ylabel("Rotation error (degrees)")
plt.title("Mean Rotation Error with Min-Max Range")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()



















#estimator = ImagePairPoseEstimator(detector, matcher, image_width=1024, image_height=512)
#
## --- 2 Görselin Yolu ---
#img_path0 = cv2.imread(r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\local_sim_photo\00800-TEEsavR23oF\img_700_850.png")
#img_path1 = cv2.imread(r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\local_sim_photo\00800-TEEsavR23oF\img_700_870.png")
#
## --- Pose Tahmini ---
#cam_pose, R, t = estimator.estimate(img_path0, img_path1)
#
#east = t[0]
#north = -t[1]
#
#rotated_east = north
#rotated_north = -east
#
## --- Heading angle hesaplama ---""
#heading_angle_rad = np.arctan2(t[0], -t[1])   # Y üzerinden X yönü
#heading_angle_deg = normalize_angle_deg(np.rad2deg(heading_angle_rad))
#print(f"Normalized Heading (translation) angle: {heading_angle_deg:.2f}°")
#
#
#def compute_heading_angle(x1, y1, x2, y2):
#    dx = x2 - x1  # East
#    dy = y2 - y1  # North
#    heading_rad = np.arctan2(dx, dy)  # North-referenced angle
#    heading_deg = normalize_angle_deg(np.rad2deg(heading_rad))
#    return normalize_angle_deg(heading_deg)
#
#
#x1, y1 = 700.0, 850.0   # İlk konum
#x2, y2 = 700.0, 870.0   # İkinci konum
#
#heading_angle = compute_heading_angle(x1, y1, x2, y2)
#print(f"Heading angle: {heading_angle:.2f} degrees")
#












    