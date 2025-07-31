#!/usr/bin/env python3
# navigation_main.py
import base64
import threading
import roslibpy

from scipy.optimize import curve_fit
from scipy.special import i0

from typing import List
from pathlib import Path
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from oop_map import PoseGraph,Node,Edge,MapObject,ChildMapObject
from lightglue import SuperPoint,LightGlue,viz2d
from lightglue.utils import numpy_image_to_torch

import heapq
import math
from typing import Dict, List, Tuple, Optional
from itertools import count
from scipy.spatial import KDTree
import os
import torch
from PIL import Image
import cv2
from collections import defaultdict

from itertools import combinations

import subprocess

from eightpa_solvers.camera_recovering import get_cam_pose_by_ransac_8pa

from object_projection import wraparound_centroid
from sinkhorn_matching import compute_sinkhorn,compute_low_cost_mass

### ONEFORMER IMPORTS
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from demo.defaults import DefaultPredictor
from oneformer import (
    add_oneformer_config, add_common_config,
    add_swin_config, add_dinat_config, add_convnext_config)

def equirectangular_to_perspective(eq_img,
                                   fov_rad,
                                   yaw_rad,
                                   out_hw=None,
                                   interp=cv2.INTER_NEAREST):
    """
    Convert an equirectangular panorama (H×W×C) to a perspective image.

    Parameters
    ----------
    eq_img   : np.ndarray
        Input panorama in equirectangular format, shape (H, W[, C]).
    fov_rad  : float
        Horizontal field of view of the virtual camera (in radians).
    yaw_rad  : float
        Yaw (heading) of the virtual camera (in radians). 0 = look forward,
        +ve turns camera to the right (clockwise from above).
    out_hw   : tuple(int, int) or None
        (height, width) of the output image. If None, the longer side is 640,
        the other is scaled to keep square pixels.
    interp   : int
        OpenCV interpolation flag (e.g. cv2.INTER_LINEAR, cv2.INTER_CUBIC).

    Returns
    -------
    persp_img : np.ndarray
        Output perspective image (out_h, out_w, C).
    """
    in_h, in_w = eq_img.shape[:2]
    if out_hw is None:
        long_side = 640
        out_w = long_side
        out_h = int(long_side * 1.0 / np.tan(fov_rad / 2) * np.tan(fov_rad / 2))
    else:
        out_h, out_w = out_hw

    f = 0.5 * out_w / np.tan(fov_rad / 2)

    jj, ii = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x_c = (jj - out_w / 2.0) / f
    y_c = (out_h / 2.0 - ii) / f
    z_c = np.ones_like(x_c)

    dirs = np.dstack((x_c, y_c, z_c))
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)

    R_yaw = np.array([[ np.cos(yaw_rad), 0,  np.sin(yaw_rad)],
                      [              0, 1,               0],
                      [-np.sin(yaw_rad), 0,  np.cos(yaw_rad)]])
    dirs = dirs @ R_yaw.T

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)

    u = (lon / (2 * np.pi) + 0.5) * (in_w - 1)
    v = (0.5 - lat / np.pi) * (in_h - 1)

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    persp_img = cv2.remap(eq_img, map_x, map_y, interp,
                          borderMode=cv2.BORDER_WRAP)

    return persp_img

def persp_pixels_to_angles(keypoints_xy,
                           out_hw,
                           fov_rad,
                           yaw_rad=0.0):
    """
    Convert 2D keypoints in a perspective image to (azimuth, elevation) angles.

    Parameters
    ----------
    keypoints_xy : (N, 2) ndarray
        Pixel coordinates (x, y) in the perspective image.
    out_hw       : tuple(int, int)
        Height and width of the perspective image.
    fov_rad      : float
        Horizontal field of view of the virtual camera in radians.
    yaw_rad      : float
        Yaw rotation of the virtual camera in radians.

    Returns
    -------
    azimuths   : (N,) ndarray
        Horizontal angles (lon) in radians.
    elevations : (N,) ndarray
        Vertical angles (lat) in radians.
    """
    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
    out_h, out_w = out_hw

    f = 0.5 * out_w / np.tan(fov_rad / 2.0)

    x = keypoints_xy[:, 0]
    y = keypoints_xy[:, 1]

    x_c = (x - out_w / 2.0) / f
    y_c = (out_h / 2.0 - y) / f
    z_c = np.ones_like(x_c)

    dirs = np.stack([x_c, y_c, z_c], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    R_yaw = np.array([[ np.cos(yaw_rad), 0,  np.sin(yaw_rad)],
                      [              0, 1,               0],
                      [-np.sin(yaw_rad), 0,  np.cos(yaw_rad)]])
    dirs = dirs @ R_yaw.T

    azimuths   = np.arctan2(dirs[:, 0], dirs[:, 2])
    elevations = np.arcsin (dirs[:, 1])

    return azimuths, elevations

def convert_pixel_to_angle(keypoints, image_width=1024, image_height=512):
    x = keypoints[:,0]
    y = keypoints[:,1]
    azimuth = (x / image_width) * 2 * np.pi - np.pi 
    elevation = (0.5 - y / image_height) * np.pi
    return azimuth, elevation

def detect_high_anomalies_z_score(data: np.ndarray, threshold: float = 3.0):
    """
    Detects high anomalies (outliers on the high end) in a 1D array using Z-score thresholding.

    Args:
        data (np.ndarray): 1D array of numerical values (e.g., similarities).
        threshold (float): Z-score threshold for identifying high anomalies.

    Returns:
        np.ndarray: Indices where the Z-score exceeds the threshold.
    """
    mean = np.nanmean(data)
    std = np.nanstd(data)

    if std == 0 or np.isnan(std):
        return np.array([], dtype=int)

    z_scores = (data - mean) / std
    return np.where(z_scores > threshold)[0], z_scores

def compute_disparity_from_pairwise_angles_1d(angles1, angles2, return_all_differences=False):
    """
    Compute average disparity between two sets of bearing angles (in radians)
    using pairwise angular differences.

    Args:
        angles1: array-like of N bearing angles in radians (image 1)
        angles2: array-like of N bearing angles in radians (image 2)
        return_all_differences: if True, also return list of all pairwise disparities (degrees)

    Returns:
        mean_disparity_deg: average pairwise angular disparity (degrees)
        (optional) delta_angles_deg: list of all pairwise disparities (degrees)
    """
    angles1 = np.array(angles1)
    angles2 = np.array(angles2)
    assert angles1.shape == angles2.shape
    n = len(angles1)

    delta_angles = []

    for i, j in combinations(range(n), 2):
        # Compute angular distance on unit circle (wrap-around aware)
        def angular_dist(a, b):
            diff = np.abs(a - b) % (2 * np.pi)
            return min(diff, 2 * np.pi - diff)

        dist1 = angular_dist(angles1[i], angles1[j])
        dist2 = angular_dist(angles2[i], angles2[j])

        delta = abs(dist1 - dist2)
        delta_angles.append(np.degrees(delta))

    mean_disparity = np.mean(delta_angles)
    if return_all_differences:
        return mean_disparity, delta_angles
    else:
        return mean_disparity
    
def get_keypoint_unit_vector(azimuth, elevation):
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    return np.stack((x, y, z), axis=1)
    

# ----------------------------------------------------------------------
#  PERCEPTION
# ----------------------------------------------------------------------
class VisualLocalizer:
    SWIN_CFG_DICT = {
        "cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
        "coco":       "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
        "ade20k":     "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
    }
    DINAT_CFG_DICT = {
        "cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
        "coco":       "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
        "ade20k":     "configs/ade20k/dinat/oneformer_dinat_large_bs16_160k.yaml",
    }

    def __init__(self, graph: PoseGraph,       
        dataset: str,
        model_path: str,
        use_swin: bool = False,
        device: str = "cuda",          
        repo_root: str = "/home/romer/umut/segmentation/OneFormer",  # path to config files
    ):
        self.graph = graph
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        self.dataset = dataset
        self.model_path = model_path
        self.use_swin = use_swin
        self.device_str = device
        self.repo_root = repo_root

        self._build_predictor()
        self.cpu_device = torch.device("cpu")
        self.prepare_objects()
        self.assignment_threshold = 0.15

    def prepare_objects(self):
        # object_list_raw = self.graph.objects
        # assigned_ids_raw = {obj_id: obj.get_id_values() for obj_id, obj in enumerate(object_list_raw)}
        object_list = self.graph.extended_objects
        self.object_collections = [obj.feature_collection for obj in object_list]
        self.assigned_ids = {obj.parent_id: obj.get_id_values() for obj in object_list}


    def _build_predictor(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)

        cfg_path = (
            self.SWIN_CFG_DICT[self.dataset]
            if self.use_swin else
            self.DINAT_CFG_DICT[self.dataset]
        )
        cfg.merge_from_file(os.path.join(self.repo_root, cfg_path))
        cfg.MODEL.DEVICE  = self.device_str
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)
        self.metadata  = MetadataCatalog.get(
            cfg.DATASETS.TEST_PANOPTIC[0]
            if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
        )

        if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
            from cityscapesscripts.helpers.labels import labels
            stuff_colors = [lab.color for lab in labels if lab.trainId != 255]
            self.metadata = self.metadata.set(stuff_colors=stuff_colors)

    def localize(self, rgb: np.ndarray, is_start = False):
        """
        Args:
            rgb : Current camera frame (H×W×3 RGB)

        Returns:
            node_id or None if ambiguous
        """
        tensor_img = numpy_image_to_torch(rgb).to(self.device)
        if is_start:
            node_id = -1
        else:
            node_id = -2

        self.graph.add_node(node_id,rgb_image=rgb)
        query_node = self.graph.nodes[node_id]

        output = self.detector.extract(tensor_img)        
        query_node.features = output
        query_node.semantic_masks = self._segment_image(query_node)
        query_node.centroids = self._calculate_centroids(query_node.semantic_masks)
        
        keypoints = output['keypoints'].squeeze(0).cpu().numpy()
        keypoints = np.round(keypoints).astype(int)
        descriptors = output['descriptors'].squeeze(0).cpu().numpy()
        node_scores = defaultdict(int)
        mask_collections = []
        for _, mask in query_node.semantic_masks.items():
            mask_values = mask[keypoints[:, 1], keypoints[:, 0]]
            keep = mask_values == 1
            # valid_keypoints = keypoints[keep]
            collection =  descriptors[keep].copy()
            if np.sum(keep) < 10:
                continue
            mask_collections.append(collection)
        n = len(mask_collections)
        if n < 3:
            print(f"[WARNING] Skipping node since since it has less than 3 objects.")
            return None
        m = len(self.object_collections)
        epsilon = 0.1 
        similarity_dict = np.zeros((n, m))
        for i, mask_collection in enumerate(mask_collections):
            for j, object_collection in enumerate(self.object_collections):
                P, C = compute_sinkhorn(mask_collection, object_collection, epsilon)
                score = compute_low_cost_mass(P, C)
                similarity_dict[i, j] = score


        assignment_matrix = similarity_dict
        argmax_indices = np.argmax(assignment_matrix, axis=1)
        max_values = np.max(assignment_matrix, axis=1)
        valid_mask = max_values >= self.assignment_threshold
        row_indices = np.arange(assignment_matrix.shape[0])[valid_mask]
        col_indices = argmax_indices[valid_mask]

        print(f"{len(row_indices)} objects are assigned")

        correspondances = defaultdict(dict)
        for row_index,col_index in zip(row_indices,col_indices):
            # object_id_pairs = assigned_ids_raw[origins[pose_graph.extended_objects[col_index].parent_id]]
            object_id_pairs = self.assigned_ids[self.graph.extended_objects[col_index].parent_id]
            for id_pair in object_id_pairs:
                correspondances[id_pair[0]][row_index] = id_pair[1]
                
        votes = np.zeros(len(self.graph.nodes))
        node_count = len(list(correspondances.keys()))
        print(f"There are {node_count} candidate nodes")
        for map_node_id, correspondance in correspondances.items():
            if len(correspondance) >= 3:
                # Extract indices
                query_object_ids = list(correspondance.keys())
                map_object_ids = list(correspondance.values())

                # Initialize bearing lists
                bearings0 = []
                bearings1 = []

                # Fetch the relevant nodes
                map_node = self.graph.nodes[map_node_id]

                query_keys = list(query_node.centroids.keys())
                map_keys = list(map_node.centroids.keys())

                for qid, mid in zip(query_object_ids, map_object_ids):
                    if qid > len(query_keys) and mid > len(map_keys):
                        raise IndexError(
                            f"Index out of bounds: qid={qid} (max {len(query_keys)-1}), "
                            f"mid={mid} (max {len(map_keys)-1})"
                        )
                    # qkey = query_keys[qid]
                    # mkey = map_keys[mid]
                    bearings0.append(query_node.centroids[qid])
                    bearings1.append(map_node.centroids[mid])


                # Convert to numpy arrays if needed
                bearings0 = np.array(bearings0)
                bearings1 = np.array(bearings1)
                disparity = compute_disparity_from_pairwise_angles_1d(bearings0,bearings1)
                votes[map_node_id] = len(correspondance)/disparity
            

        top_k = 3
        top_indices = np.argsort(votes)[-top_k:][::-1]
        top_scores = votes[top_indices]
        print(np.count_nonzero(votes))
        if top_scores[0] > 0:
            return top_indices[0]
        else:
            print(f"[WARNING] No valid node found, returning None")
            return None

    def resize(self,img,max_side=256):
        pil = Image.fromarray(img)
        w, h = pil.size                    # PIL: (width, height)
        if w <= max_side and h <= max_side:
            return img                     # nothing to do

        # 2. compute new size
        scale     = max_side / max(w, h)
        new_size  = (round(w * scale), round(h * scale))   # width, height

        pil_resized = pil.resize(new_size, Image.LANCZOS)

        # 4. back to NumPy (shares memory again)
        return np.asarray(pil_resized)
    
    def _segment_image(self, node):
        """
        Uses OneFormer to get panoptic mask and converts it into
        {segment_id: np.bool_ mask}
        """
        panoptic_seg, segments_info = self._infer_image(node.rgb_image, task="panoptic")
        mask = panoptic_seg.to(self.cpu_device).numpy()

        masks = {}
        height, width = mask.shape
        label_idx = 0
        area_threshold = 100
        for seg_info in segments_info:
            label = seg_info["id"]
            category = seg_info["category_id"]

            if category in [0, 2, 3, 5, 8, 11, 12, 13, 27]:
                continue
            binary_mask = (mask==label).astype(np.uint8)
            # Apply morphological operations to refine the mask
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            num_labels, labeled_mask = cv2.connectedComponents(binary_mask, connectivity=8)

            for row in range(height):
                left_pixel_label = labeled_mask[row, 0]
                right_pixel_label = labeled_mask[row, width - 1]

                if left_pixel_label != right_pixel_label and left_pixel_label != 0 and right_pixel_label != 0:
                    labeled_mask[labeled_mask == right_pixel_label] = left_pixel_label
            
            for i in range(1, num_labels):
                if np.sum(labeled_mask == i) < area_threshold:
                    continue
                # Create a new mask for each label
                refined_mask = np.zeros_like(binary_mask)
                refined_mask[labeled_mask == i] = 1
                # Initialize the dictionary for this index if not already done
                masks[label_idx] = refined_mask
                label_idx += 1
        ### HERE COMES THE MASK REFINEMENT
        return masks
    
    def _calculate_centroids(self, masks):
        centroids = {}
        for sid, mask in masks.items():
            if mask is None or mask.size == 0:
                continue
            xy = wraparound_centroid(mask)   # (x, y)
            angle, *_ = convert_pixel_to_angle(
                xy, mask.shape[1], mask.shape[0]
            )
            centroids[sid] = angle
        return centroids
    
    def _infer_image(self, img, task="semantic"):
        """Light wrapper around DefaultPredictor."""
        pred = self.predictor(img, task)
        panoptic_seg, segments_info = pred["panoptic_seg"]
        torch.cuda.empty_cache()
        return panoptic_seg, segments_info
    
# ----------------------------------------------------------------------
#  PLANNING
# ----------------------------------------------------------------------
class PathPlanner:
    """
    A* on a fixed, undirected pose-graph with pre-computed Euclidean-distance
    heuristic/cost lookup.  All nodes must have `x` and `y` defined.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------
    def __init__(self, graph: "PoseGraph"):
        self.graph = graph

        # 1. stable node ordering  id ↔ index
        self.id2idx: Dict[int, int] = {nid: k for k, nid in enumerate(graph.nodes)}
        self.idx2id: List[int]      = list(graph.nodes.keys())

        # 2. coordinate matrix  (N, 2)
        coords = np.array([(n.x, n.y) for n in graph.nodes.values()], dtype=np.float32)
        if np.isnan(coords).any():
            raise ValueError("All nodes must have finite x and y coordinates.")
        self._coords = coords

        # 3. full distance table  (N×N float32)
        diff = coords[:, None, :] - coords[None, :, :]
        self._dists = np.sqrt((diff ** 2).sum(-1), dtype=np.float32)

        # 4. adjacency list (deduplicated)
        self._adj = self._build_adj()

    # ------------------------------------------------------------------
    def _build_adj(self):
        """Return {node_id: [(neighbor_id, cost), …]} with no duplicates."""
        adj: Dict[int, List[Tuple[int, float]]] = {}
        seen: set[Tuple[int, int]] = set()

        for e in self.graph.edges:
            key = tuple(sorted((e.i, e.j)))   # unordered pair
            if key in seen:
                continue                      # skip reverse duplicate
            seen.add(key)

            cost = float(self._dists[self.id2idx[e.i], self.id2idx[e.j]])

            adj.setdefault(e.i, []).append((e.j, cost))
            adj.setdefault(e.j, []).append((e.i, cost))   # opposite direction

        return adj

    # ------------------------------------------------------------------
    def _h(self, a: int, b: int):
        """Heuristic: straight-line distance from table."""
        return float(self._dists[self.id2idx[a], self.id2idx[b]])

    # ------------------------------------------------------------------
    def a_star(self, start_id: int, goal_id: int):
        t0 = time.time()
        if start_id not in self.graph.nodes or goal_id not in self.graph.nodes:
            raise KeyError("Start or goal node not found in graph.")

        open_heap: List[Tuple[float, int, int]] = []
        seq = count()
        heapq.heappush(open_heap, (self._h(start_id, goal_id), next(seq), start_id))

        g_cost: Dict[int, float] = {start_id: 0.0}
        parent: Dict[int, int]   = {}
        closed: set[int]         = set()

        while open_heap:
            _, _, u = heapq.heappop(open_heap)
            if u in closed:
                continue
            if u == goal_id:
                # reconstruct path
                path = [u]
                while u in parent:
                    u = parent[u]
                    path.append(u)
                t1 = time.time()
                print(f"A-star took {t1-t0} seconds")
                return path[::-1]

            closed.add(u)

            for v, cost_uv in self._adj.get(u, []):
                if v in closed:
                    continue
                tentative = g_cost[u] + cost_uv
                if tentative < g_cost.get(v, math.inf):
                    g_cost[v] = tentative
                    parent[v] = u
                    f = tentative + self._h(v, goal_id)
                    heapq.heappush(open_heap, (f, next(seq), v))

        return None   # no path


# ----------------------------------------------------------------------
#  CONTROL
# ----------------------------------------------------------------------
class ActionCmd:
    """Thin container for whatever your robot API expects."""
    def __init__(self, linear: float, angular: float):
        self.linear  = linear
        self.angular = angular
class MotionController:
    V_FWD   = 0.30     # m/s
    W_TURN  = -0.25     # rad/s
    ANG_TOL = 0.10     # rad  (~6°)
    def __init__(self, graph: PoseGraph):
        self.graph = graph
        self._edge_xy: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for e in graph.edges:
            # R = e.R_ij
            # dtheta = np.arctan2(R[1, 0], R[0, 0])
            # dx, dy = e.t_ij[0], e.t_ij[1]          # use x, y components only
            x1, y1 = graph.nodes[e.i].x, graph.nodes[e.i].y
            x2, y2 = graph.nodes[e.j].x, graph.nodes[e.j].y
            dx = x2 - x1
            dy = y2 - y1
            dtheta1 = graph.nodes[e.j].theta
            dtheta2 = graph.nodes[e.i].theta
            self._edge_xy[(e.i, e.j)] = (dx, dy, dtheta1, dtheta2)
            # undirected graph: reverse is the negative translation
            self._edge_xy[(e.j, e.i)] = (-dx, -dy, dtheta2, dtheta1)

    def heading_angle(self, cur_id, next_id):
        dx, dy, dtheta1,dtheta2 = self._edge_xy[(cur_id, next_id)]
        global_heading = math.atan2(dy, dx)
        return global_heading
    
    def heading_error(self, cur_id, next_id, cur_theta):
        dx, dy, dtheta1,dtheta2 = self._edge_xy[(cur_id, next_id)]
        global_heading = math.atan2(dy, dx)
        local_heading = global_heading - dtheta2
        print(f"[MotionControl]Desired heading {np.rad2deg(local_heading)} degree")
        return self._wrap(local_heading - cur_theta)
    
    def arrival_angle(self, cur_id, next_id):
        """Return the angle to turn to face the next edge."""
        dx, dy, dtheta1, dtheta2 = self._edge_xy[(cur_id, next_id)]
        global_heading = math.atan2(dy, dx)
        local_heading = global_heading - dtheta1        
        return self._wrap(local_heading)
    
    def rotate_step(self, err):
        """One in-place rotation step toward zero heading error."""
        w = self.W_TURN if err > 0 else -self.W_TURN
        t = abs(err / w)
        return w,t
    
    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
'''
class MotionController:
    V_FWD   = 0.30     # m/s
    W_TURN  = -0.25     # rad/s
    ANG_TOL = 0.10     # rad  (~6°)
    def __init__(self, graph: PoseGraph):
        self.graph = graph
        self._edge_xy: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for e in graph.edges:
            R = e.R_ij
            dtheta = np.arctan2(R[1, 0], R[0, 0])
            dx, dy = e.t_ij[0], e.t_ij[1]          # use x, y components only
            self._edge_xy[(e.i, e.j)] = (dx, dy, dtheta)
            # undirected graph: reverse is the negative translation
            self._edge_xy[(e.j, e.i)] = (-dx, -dy, -dtheta)

    def heading_error(self, cur_id, next_id, cur_theta):
        dx, dy, _ = self._edge_xy[(cur_id, next_id)]
        desired = -math.atan2(dy, dx)
        print(f"[MotionControl]Desired heading {np.rad2deg(desired)} degree")
        return self._wrap(desired - cur_theta)
    
    def arrival_angle(self, cur_id, next_id):
        """Return the angle to turn to face the next edge."""
        dx, dy, dtheta = self._edge_xy[(cur_id, next_id)]
        desired = -math.atan2(dy, dx) - dtheta
        return self._wrap(desired)
    
    def rotate_step(self, err):
        """One in-place rotation step toward zero heading error."""
        w = self.W_TURN if err > 0 else -self.W_TURN
        t = abs(err / w)
        return w,t
           
    
    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
'''

# ----------------------------------------------------------------------
#  TIAGO-BASED ROBOT INTERFACE
# ----------------------------------------------------------------------
class Tiago:
    def __init__(self):
        self.conda_env = "habitat"
        self.script_path = "/home/romer/umut/GibsonData/habitat-lab/my_scripts/oneformer/explore_hm3d.py"
        self.frame = None
        self.position = None
        self.heading = None
        


    # ========== movement ===================================================
    def move(self, x=0, y=0, z=0, rx=0, ry=0, rz=0):
        msg = {'linear':  {'x': float(x),  'y': float(y),  'z': float(z)},
               'angular': {'x': float(rx), 'y': float(ry), 'z': float(rz)}}
        self.cmd_vel.publish(msg)

    
    def get_next_frame(self, x=800, y=800):
        conda_init = "source /home/romer/miniconda3/etc/profile.d/conda.sh"
        run_cmd = f"conda activate {self.conda_env} && python {self.script_path} --pos {y},{x} --env '00800-TEEsavR23oF'"
        full_cmd = f"bash -c '{conda_init} && {run_cmd}'"
        subprocess.run(full_cmd, shell=True, check=True,
               stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL)
        # Wait for the script to finish
        print(f"Executed command: {full_cmd}")
        try:
            img = np.load("/home/romer/umut/GibsonData/habitat-lab/my_scripts/oneformer/image.npy")
            img = img.astype(np.uint8)
            if img.shape[0] == 0 or img.shape[1] == 0:
                print("Image is empty, returning None")
                return None
            self.frame = img
            self.position = (x,y)
            return img
        except FileNotFoundError:
            print("Image file not found, returning None")    
        return None
    
    def rotate(self, angle: float):
        """
        Rotate the current frame by a given angle in radians. Image is 360 degrees, so rotation can be
        done by simply slicing the image.
        """
        if self.frame is None:
            print("No frame to rotate.")
            return None
        angle_deg = np.rad2deg(angle)
        if angle_deg < 0:
            angle_deg += 360
        elif angle_deg >= 360:
            angle_deg -= 360
        # Convert the angle to an index
        index = int(angle_deg / 360 * self.frame.shape[1])
        rotated_frame = np.roll(self.frame, -index, axis=1)
        self.frame = rotated_frame
        return rotated_frame

    def move(self, heading):
        x1,y1 = self.position

        dx = np.cos(heading)
        dy = np.sin(heading)

        # Update position
        x2 = x1 + dx*15
        y2 = y1 + dy*15

        x2 = int(round(x2))
        y2 = int(round(y2))

        return x2,y2



# ----------------------------------------------------------------------
#  HIGH-LEVEL ORCHESTRATOR
# ----------------------------------------------------------------------
class Navigator:
    """
    Glue everything together:
        1. Initial localization
        2. Target localization
        3. Plan path
        4. Follow path until target reached
    """
    def __init__(self,
                 graph: PoseGraph,
                 localizer: VisualLocalizer,
                 planner: PathPlanner,
                 controller: MotionController,
                 robot: Tiago):
        self.graph      = graph
        self.localizer  = localizer
        self.planner    = planner
        self.controller = controller
        self.robot      = robot
        self.calibration_start = False
        self.first_theta = None
        self.prev_theta = None
        self.prev_fov = None
        self.fov = None
        self.heading = None
        self.calib_dict = {}
        self.build_kd_tree()         

    # ------------------------------------------------------------
    def build_kd_tree(self):
        """Build KD-tree and lookup tables from the current graph."""
        # — coordinate and id look-ups —
        self._ids: List[int] = []
        coords: List[Tuple[float, float]] = []
        for nid, node in self.graph.nodes.items():
            self._ids.append(nid)
            coords.append((node.x, node.y))

        # — spatial index —
        self._kdtree = KDTree(coords)

        # quick map: node id -> (x, y)  (useful for distance helper)
        self._id2coord: Dict[int, Tuple[float, float]] = {
            nid: c for nid, c in zip(self._ids, coords)
        }

    def get_closest_nodes(self, target_id: int, k: int = 5):
        if target_id not in self.graph.nodes:
            raise KeyError(f"Node {target_id} not in graph")

        # query k+1 because the first hit is always the query point itself
        target_pt = self._id2coord[target_id]
        dists, idxs = self._kdtree.query(target_pt, k=k + 1)

        # map KD-tree indices back to node IDs, skip self
        return [self._ids[i] for i in idxs if self._ids[i] != target_id][:k]
    
    def find_similarity(self, arr0, arr1):
        # Zero-mean both sets
        # arr0_centered = arr0 - np.mean(arr0, axis=0)
        arr0_centered_x = arr0.copy()
        arr0_centered_x[:, 0] -= np.mean(arr0[:, 0])

        arr1_centered_x = arr1.copy()
        arr1_centered_x[:, 0] -= np.mean(arr1[:, 0])
        # arr1_centered = arr1 - np.mean(arr1, axis=0)

        diffs = np.abs(arr0_centered_x - arr1_centered_x)
        distances = np.linalg.norm(diffs, axis=1)
        similarity = 1 / np.mean(distances)

        return similarity

    def find_heading(self, node, current_features, current_observation):
        print(f"Shape of the current observation is {current_observation.shape}")
        print(f"Shape of the node is {node.rgb_image.shape}")
        with torch.no_grad():
            pred = self.localizer.matcher({"image0": current_features, "image1": node.features})
        matches = pred["matches"][0].cpu().numpy()
        keypoints = node.features['keypoints'][0].round().cpu().numpy()
        m_kpts0 = current_features['keypoints'][0].round().cpu().numpy()[matches[..., 0]]
        m_kpts1 = keypoints[matches[..., 1]]
        print(f"There are {len(matches)} matches")
        az0, ele0 = convert_pixel_to_angle(m_kpts0,1024,512)
        az1, ele1 = convert_pixel_to_angle(m_kpts1,512,256)
        u0 = get_keypoint_unit_vector(az0, ele0)
        u1 = get_keypoint_unit_vector(az1, ele1)
        bearing0 = u0.T
        bearing1 = u1.T  
        cam_pose = get_cam_pose_by_ransac_8pa(bearing0, bearing1)
        translation = cam_pose[:3,3]
        R = cam_pose[:3,:3]
        yaw_angle = np.arctan2(R[1, 0], R[0, 0])
        heading = math.atan2(translation[1], translation[0])
        similarity = self.find_similarity(np.stack((az0, ele0), axis=1),np.stack((az1, ele1), axis=1))

        return heading, yaw_angle, similarity

    
    def find_heading_perspective(self, img1, current_features, current_observation, prev_fov, prev_yaw,offset=0.0):
        persp_img = equirectangular_to_perspective(img1,
                                                    fov_rad=prev_fov + offset,
                                                    yaw_rad=prev_yaw,
                                                    out_hw=(480, 640))
        # print(f"Perspective image shape: {np.rad2deg(prev_fov + 0.45)}")
        tensor_img = numpy_image_to_torch(persp_img).to(self.localizer.device)
        output = self.localizer.detector.extract(tensor_img)    
        with torch.no_grad():
            pred = self.localizer.matcher({"image0": current_features, "image1":output})
        matches = pred["matches"][0].cpu().numpy()
        # feature_mask = np.zeros((480, 640),dtype=bool)

        m_kpts0 = current_features['keypoints'][0].round().cpu().numpy()[matches[..., 0]]
        m_kpts1 = output['keypoints'][0].round().cpu().numpy()[matches[..., 1]]

        # rows = m_kpts1[:,1].astype(int)
        # cols = m_kpts1[:,0].astype(int)
        h, w = current_observation.shape[:2]
        
        # feature_mask[rows, cols] = True
        min_x = np.min(m_kpts1[:,0])
        max_x = np.max(m_kpts1[:,0])
        mean_x = 0.5*(min_x + max_x)
        # print(f"Min:{min_x}, Max:{max_x}, mean:{mean_x}")
        # xy_centroid = wraparound_centroid(feature_mask)

        # angle_centroid, *_ = persp_pixels_to_angles(xy_centroid
        # )
        # print(f"Centroid calculated from mask {np.rad2deg(angle_centroid)}")
        angle_mean, *_ = persp_pixels_to_angles(
            np.array([[mean_x,0]]), out_hw=(480, 640),fov_rad=prev_fov + offset,yaw_rad=prev_yaw
        )
        angle_max, *_ = persp_pixels_to_angles(
            np.array([[max_x,0]]), out_hw=(480, 640),fov_rad=prev_fov + offset,yaw_rad=prev_yaw
        )
        angle_min, *_ = persp_pixels_to_angles(
            np.array([[min_x,0]]), out_hw=(480, 640),fov_rad=prev_fov + offset,yaw_rad=prev_yaw
        )
        # print(f"Max angle is {np.rad2deg(angle_max)}, min angle is {np.rad2deg(angle_min)}")
        fov = ((angle_max-angle_min) + math.pi) % (2 * math.pi) - math.pi
        # print(f"New FOV is {np.rad2deg(fov)}")
        # print(f"Initial centroid calculated from mean {np.rad2deg(angle_mean)}")
        # angle_diff = np.abs(angle_mean-angle_centroid)
        # if angle_diff >= np.pi:
        #     angle_mean = (angle_mean) % (2 * math.pi) - math.pi
        # print(f"Final centroid calculated from mean {np.rad2deg(angle_mean)}")
        # viz2d.plot_images([persp_img, current_observation])
        # viz2d.plot_keypoints([m_kpts1,m_kpts0], ps=10)
        # axes = viz2d.plot_images([current_observation, node.rgb_image])
        # viz2d.plot_matches(m_kpts0, m_kpts, color="lime", lw=0.2)
        # plt.show()
        
        return angle_mean[0], fov[0]
    
    def estimate_fov_and_heading(self,plot=False):
        amplitude_tolerance = 50.0
        x = np.array(list(self.calib_dict.keys()))
        y = np.array(list(self.calib_dict.values()))
        def von_mises_func(theta, A, mu, kappa, b):
            return A * np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa)) + b
        A_init = max(y) - min(y)
        mu_init = x[np.argmax(y)]
        kappa_init = 1.0
        b_init = min(y)

        p0 = [A_init, mu_init, kappa_init, b_init]
        bounds = ([0, -np.pi, 0.01, -np.inf], [np.inf, np.pi, 100, np.inf])
        popt, pcov = curve_fit(von_mises_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
        A_fit, mu_fit, kappa_fit, b_fit = popt
        if plot:
            print(f"Fitted parameters:\nA = {A_fit:.4f}\nmu = {np.rad2deg(mu_fit):.4f}\nkappa = {kappa_fit:.4f}\nb = {b_fit:.4f}")
            # Plot the result
            theta_fit = np.linspace(-np.pi, np.pi, 500)
            y_fit = von_mises_func(theta_fit, *popt)

            plt.figure(figsize=(8, 4))
            plt.scatter(x, y, label="Data", color="blue")
            plt.plot(theta_fit, y_fit, color='red', label="Von Mises Fit", linewidth=2)
            plt.xlabel("Angle (radians)")
            plt.ylabel("Value")
            plt.title("Von Mises Fit to Bell-Shaped Circular Data")
            plt.legend()
            plt.grid(True)
            plt.show()
        if A_fit <= amplitude_tolerance:
            self.localized = True
        else: 
            self.localized = False
            self.heading = mu_fit
            print(f"[Navigator] heading is {np.rad2deg(mu_fit)} degree")
        mean_fov = np.mean(von_mises_func(x, *popt))
        print(f"[Navigator] Mean FOV is {np.rad2deg(mean_fov)} degree")
        peak_fov = von_mises_func(popt[1], *popt)
        print(f"[Navigator] Peak FOV is {np.rad2deg(peak_fov)} degree")
        min_fov = np.min(von_mises_func(x, *popt))
        print(f"[Navigator] Min FOV is {np.rad2deg(min_fov)} degree")
        average_fov = (min_fov + peak_fov) / 2
        print(f"[Navigator] Average FOV is {np.rad2deg(average_fov)} degree")
        # average_fov = (mean_fov + peak_fov) / 2
        self.fov = average_fov
        self.fov = np.deg2rad(60.827363946)
        print(f"[Navigator] FOV is {np.rad2deg(self.fov)} degree")


    # -------------------- MAIN LOOP -----------------------------------
    def run(self, start_rgb, target_rgb: np.ndarray):

        # --- 1. Initial localization ----------------------------------
        # current_frame = self.robot.get_camera_frame()
        start_id = self.localizer.localize(start_rgb,is_start = True)
        if start_id is None:
            print("[Navigator] Initial localization failed.")
            return

        # --- 2. Localize target observation ---------------------------
        goal_id = self.localizer.localize(target_rgb)
        if goal_id is None:
            print("[Navigator] Target image not found in graph.")
            return

        print(f"[Navigator] start={start_id}, goal={goal_id}")

        # --- 3. Plan route (A*) ---------------------------------------
        path = self.planner.a_star(start_id, goal_id)
        if not path:
            print("[Navigator] No path found.")
            return
        print(f"[Navigator] Planned path: {path}")

        for node_id, node in self.graph.nodes.items():
            if node.x is not None and node.y is not None:
                if node_id in path:
                    plt.plot(node.x, node.y, 'go')
                    plt.text(node.x + 0.02, node.y + 0.02, str(node_id), fontsize=8)
                else:
                    plt.plot(node.x, node.y, 'ro')
                    plt.text(node.x + 0.02, node.y + 0.02, str(node_id), fontsize=8)
        # plt.savefig('found_path.png')

        current_observation = start_rgb
        self.robot.heading = 0
        self.image_counter = 0
        # current_features = self.graph.nodes[-1].features
        # add -1 to the top of the path
        # path.insert(0,-1)
        # print(f"[Navigator] Planned path: {path}")
        while True:
            cv2.imshow('Live Stream', current_observation)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(path) < 2:
                print("[Navigator] Reached final node.")
                break

            nxt_id  = path[0]
            
            tensor_img = numpy_image_to_torch(current_observation).to(self.localizer.device)
            current_features = self.localizer.detector.extract(tensor_img)
            node = self.graph.nodes[nxt_id]
            # theta= self.find_heading(node,current_features,current_observation)
            heading, yaw, similarity  = self.find_heading(node,current_features,current_observation)
            print(f"[Navigator] Heading in degrees: {np.rad2deg(heading)}, yaw in degrees: {np.rad2deg(yaw)}, similarity score: {similarity}")
            if similarity >= 30:
                print('[Navigator] Moving to next node')
                path.pop(0)         # remove current node from path
                cv2.imwrite('rotated.png',current_observation)
                if len(path) == 0:
                    break           # only one node left, we are done
            elif np.abs(heading) >= 0.09:
                print('[Navigator] Rotating to target node')
                current_observation = self.robot.rotate(heading,self.image_counter)
                self.robot.heading += heading
                # cv2.imwrite('rotated.png',current_observation)
            else:
                print(f'[Navigator] Heading angle is {np.rad2deg(self.robot.heading)}')
                x,y = self.robot.move(self.robot.heading)
                print(f'[Navigator] Stepping towards target node from {self.robot.position} to {(x,y)}')
                current_observation = self.robot.get_next_frame(x,y)
                current_observation = self.robot.rotate(self.robot.heading)
        cv2.destroyAllWindows()
        


# ----------------------------------------------------------------------
#  MAIN SCRIPT ENTRYPOINT
# ----------------------------------------------------------------------
def main():
    # 0. Load resources ------------------------------------------------
    graph_path   = Path("./data/pose_graph_simulation.pkl")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    # start_rgb = cv2.imread('./data/query_image.png')
    # start_rgb = cv2.cvtColor(start_rgb, cv2.COLOR_BGR2RGB)

    dest_str = input("Enter target node id: ").strip()

    try:
        dest_id = int(dest_str)
    except ValueError:
        print("[MAIN] Node id must be an integer.")
        return

    # sanity-check: does that node exist in the map?
    if dest_id not in graph.nodes:
        print(f"[MAIN] Node {dest_id} not found in graph.")
        return
    
    # we’ll feed the node’s stored key-frame to the visual localizer
    target_rgb = graph.nodes[dest_id].rgb_image

    # 1. Build modules -------------------------------------------------
    localizer    = VisualLocalizer(graph,    
                                   dataset="ade20k",
                                   model_path="/home/romer/umut/segmentation/OneFormer/250_16_dinat_l_oneformer_ade20k_160k.pth")
    planner      = PathPlanner(graph)
    controller   = MotionController(graph)
    robot        = Tiago()

    start_rgb = robot.get_next_frame()
    # 2. Start high-level navigation ----------------------------------
    nav = Navigator(graph, localizer, planner, controller, robot)

    nav.run(start_rgb,target_rgb)


if __name__ == "__main__":
    main()
