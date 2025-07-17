
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional

import cv2
import numpy as np
import os
import torch


from lightglue.utils import numpy_image_to_torch         
from eightpa_solvers.camera_recovering import get_cam_pose_by_ransac_8pa


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
    



# ---------------- example usage ---------------------------------------------
if __name__ == "__main__":
    from lightglue import SuperPoint, LightGlue

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher  = LightGlue(features="superpoint").eval().to(device)

    estimator = ImagePairPoseEstimator(detector, matcher, image_width=1024, image_height=512)

    img1 = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\local_sim_photo\00800-TEEsavR23oF\img_800_800.png"
    img2 = r"C:\Users\ofsaa\Desktop\mechacademy\mechademy\local\local_sim_photo\00800-TEEsavR23oF\img_800_820.png"


  
    cam_pose, R, t = estimator.estimate(img1, img2)
    # print("Pose:\n", cam_pose)

    translation = cam_pose[:3,3]   
    print(translation)    
    rotation = cam_pose[:3,:3]


    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])  #This is in radians   
    print(f"Yaw angle {np.rad2deg(yaw)}")
        
    heading_angle = np.arctan2(translation[1],translation[0])   #This is in radians
    print(f"Translation {np.rad2deg(heading_angle)}")




    
    #def fix_translation_direction(t: np.ndarray, coord0: Tuple[float, float], coord1: Tuple[float, float]) -> np.ndarray:
    #
    #    t = t / np.linalg.norm(t)
    #
    #    gt_vec = np.array([coord1[0] - coord0[0], coord1[1] - coord0[1], 0.0])
    #
    #    if np.linalg.norm(gt_vec) == 0:
    #
    #        return t
    #
    #    gt_vec /= np.linalg.norm(gt_vec)
    #
    #    dot = np.dot(gt_vec[:2], t[:2])
    #
    #    if dot < 0:
    #
    #        return -t
    #
    #    return t
#
    #
    #cam_pose, R, t = estimator.estimate(img1, img2)
    #
    #t = fix_translation_direction(t, (800, 800), (800, 820))
#
    #
    #angle_rad = np.arctan2(t[1], t[0])
    #
    #angle_deg = np.degrees(angle_rad) % 360
    #
    #print(f"Normalized heading angle: {angle_deg:.2f}°")