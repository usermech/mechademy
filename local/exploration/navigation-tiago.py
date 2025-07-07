#!/usr/bin/env python3
# navigation_main.py
import base64
import threading
import roslibpy

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
        rgb = self.resize(rgb)
        tensor_img = numpy_image_to_torch(rgb).to(self.device)
        if is_start:
            node_id = -1
        else:
            node_id = -2
        self.graph.add_node(node_id,rgb_image=rgb)
        node = self.graph.nodes[node_id]

        output = self.detector.extract(tensor_img)        
        node.features = output

        node.semantic_masks = self._segment_image(node)
        
        keypoints = output['keypoints'].squeeze(0).cpu().numpy()
        keypoints = np.round(keypoints).astype(int)
        descriptors = output['descriptors'].squeeze(0).cpu().numpy()
        node_scores = defaultdict(int)
        query_feature_collections = []
        for _, mask in node.semantic_masks.items():
            mask_values = mask[keypoints[:, 1], keypoints[:, 0]]
            keep = mask_values == 1
            # valid_keypoints = keypoints[keep]
            collection =  descriptors[keep].copy()
            if np.sum(keep) < 10:
                continue
            similarities = np.zeros(len(self.graph.extended_objects))

            for i,map_object in enumerate(self.graph.extended_objects):
                P, C = compute_sinkhorn(collection, map_object.feature_collection , epsilon=0.1)
                score = compute_low_cost_mass(P, C)
                similarities[i] = score
            candidate_objects, object_scores = detect_high_anomalies_z_score(similarities,2.5)
            print(candidate_objects, object_scores[candidate_objects]*similarities[candidate_objects])

            for i, candidate_object in enumerate(candidate_objects):
                for candidate_node_id,_ in self.graph.extended_objects[candidate_object].id_pairs.values():
                    node_scores[candidate_node_id] += object_scores[candidate_object]*similarities[candidate_object]
                    # candidate_node = self.graph.nodes[candidate_node_id]

        sorted_node_scores = sorted(node_scores, key=node_scores.get, reverse=True)
        top10_keys = sorted_node_scores[:10]
        print(f"Top 10 is {top10_keys}")
        best_match = 0
        for node_index in top10_keys:
            map_node = self.graph.nodes[node_index]
            with torch.no_grad():
                pred = self.matcher({"image0": output, "image1": map_node.features})
            matches = pred["matches"][0].cpu().numpy()
            if len(matches)>best_match:
                best_match=len(matches)
                best_node = node_index
        print(f"Best matching node is {best_node}, with {best_match} mathces")
        node.x,node.y = self.graph.nodes[best_node].x, self.graph.nodes[best_node].y
        return best_node
    
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
            dx, dy = e.t_ij[0], e.t_ij[1]          # use x, y components only
            self._edge_xy[(e.i, e.j)] = (dx, dy)
            # undirected graph: reverse is the negative translation
            self._edge_xy[(e.j, e.i)] = (-dx, -dy)

    def heading_error(self, cur_id, next_id, cur_theta):
        dx, dy = self._edge_xy[(cur_id, next_id)]
        desired = math.atan2(dy, dx)
        print(f"[MotionControl]Desired heading {np.rad2deg(desired)}")
        return self._wrap(desired - cur_theta)
    
    def rotate_step(self, err):
        """One in-place rotation step toward zero heading error."""
        w = self.W_TURN if err > 0 else -self.W_TURN
        t = abs(err / w)
        return w,t
           
    
    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

# ----------------------------------------------------------------------
#  TIAGO-BASED ROBOT INTERFACE
# ----------------------------------------------------------------------
class Tiago:
    def __init__(self,
                 host='localhost', port=9090,
                 vel_topic='/mobile_base_controller/cmd_vel',
                 img_topic='/xtion/rgb/image_raw/compressed'):
        # ───────────────────── connect ─────────────────────
        self.ros = roslibpy.Ros(host=host, port=port)
        self.ros.run()

        # Wait until the websocket handshake really finished
        while not self.ros.is_connected:
            time.sleep(0.05)

        # ─────────────── publishers / subscribers ───────────────
        self.cmd_vel = roslibpy.Topic(self.ros, vel_topic,
                                      'geometry_msgs/Twist')
        self.cmd_vel.advertise()

        self.image_sub = roslibpy.Topic(self.ros, img_topic,
                                        'sensor_msgs/CompressedImage')
        self.image_sub.subscribe(self.image_callback)

        self._latest_frame = None
        self._next_frame_event = threading.Event()
        self._stop_ros_time = None

            # ========== movement ===================================================
    def move(self, x=0, y=0, z=0, rx=0, ry=0, rz=0):
        msg = {'linear':  {'x': float(x),  'y': float(y),  'z': float(z)},
               'angular': {'x': float(rx), 'y': float(ry), 'z': float(rz)}}
        self.cmd_vel.publish(msg)

    def stop(self):
        self._stop_ros_time = self._now_ros_time() + 1
        self.move(0, 0, 0, 0, 0, 0)

    # ========== image callback ============================================
    @staticmethod
    def _decode_compressed(msg):
        """Return a BGR cv2 image from sensor_msgs/CompressedImage dict."""
        try:
            # The 'data' field is base‑64 ASCII, **not** a Python list
            buff = base64.b64decode(msg['data'])
            arr = np.frombuffer(buff, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError('cv2.imdecode failed.')
            return img[:,:,::-1]
        except Exception as exc:
            print(f'[WARN] image decode error: {exc}')
            return None

    def image_callback(self, message):
        img = self._decode_compressed(message)
        if img is None:
            return
        
        # cv2.imshow('TIAGo camera', img)
        # cv2.waitKey(1)
                # Extract ROS timestamp from header
        header = message.get('header', {})
        stamp = header.get('stamp', {})
        secs = stamp.get('secs', 0)
        nsecs = stamp.get('nsecs', 0)
        frame_time = secs + nsecs * 1e-9

        if (self._stop_ros_time is None) or (frame_time >= self._stop_ros_time):
            # store *every* incoming frame
            self._latest_frame = img
            # if somebody is waiting for “the next frame”, wake them up
            self._next_frame_event.set()                  
    
    def get_next_frame(self, timeout=2.0):
        """Block until a *new* frame arrives, then return it (or None on timeout)."""
        self._next_frame_event.clear()      # forget any previous frame
        arrived = self._next_frame_event.wait(timeout)
        return self._latest_frame if arrived else None
    
    def get_next_frame_after_stop(self, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.get_next_frame(timeout=deadline - time.time())
            if frame is None:
                return None
            # Here frame accepted already has timestamp >= stop time (from callback filtering)
            return frame
        return None
    
    def _now_ros_time(self):
        # rosbridge time is float seconds from epoch in header.stamp.secs + nsecs
        # but rosbridge usually encodes header.stamp as {'secs': int, 'nsecs': int}
        # or might just send a float string depending on config
        # We get system time here, but ideally use ROS time from /clock or header
        return time.time()
        
    # ========== clean shutdown ============================================
    def shutdown(self):
        self.image_sub.unsubscribe()
        self.cmd_vel.unadvertise()
        time.sleep(0.1)
        self.ros.terminate()
        cv2.destroyAllWindows()



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
    
    def find_heading(self, node, current_features,current_observation):
        with torch.no_grad():
            pred = self.localizer.matcher({"image0": current_features, "image1": node.features})
        matches = pred["matches"][0].cpu().numpy()
        feature_mask = np.zeros(node.rgb_image.shape[:2],dtype=bool)
        keypoints = node.features['keypoints'][0].round().cpu().numpy()
        m_kpts = keypoints[matches[..., 1]]
        m_kpts0 = current_features['keypoints'][0].round().cpu().numpy()[matches[..., 0]]
        rows = m_kpts[:,1].astype(int)
        cols = m_kpts[:,0].astype(int)

        feature_mask[rows, cols] = True
        min_x = np.min(m_kpts[:,0])
        max_x = np.max(m_kpts[:,0])
        mean_x = 0.5*(min_x + max_x)
        mean_x = np.array([[mean_x,0]])
        print(f"Min:{min_x}, Max:{max_x}")
        xy_centroid = wraparound_centroid(feature_mask)

        angle_centroid, *_ = convert_pixel_to_angle(
            xy_centroid, feature_mask.shape[1], feature_mask.shape[0]
        )
        print(f"Centroid calculated from mask {np.rad2deg(angle_centroid)}")
        angle_mean, *_ = convert_pixel_to_angle(
            mean_x, feature_mask.shape[1], feature_mask.shape[0]
        )
        print(f"Initial centroid calculated from mean {np.rad2deg(angle_mean)}")
        angle_diff = np.abs(angle_mean-angle_centroid)
        if angle_diff >= np.pi:
            angle_mean = (angle_mean) % (2 * math.pi) - math.pi
        print(f"Final centroid calculated from mean {np.rad2deg(angle_mean)}")
        viz2d.plot_images([node.rgb_image, current_observation])
        viz2d.plot_keypoints([m_kpts,m_kpts0], ps=10)
        # axes = viz2d.plot_images([current_observation, node.rgb_image])
        # viz2d.plot_matches(m_kpts0, m_kpts, color="lime", lw=0.2)
        plt.show()
        
        return angle_centroid
    
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

        phase = "ROTATE"          # start by aligning to next waypoint
        
        current_observation = start_rgb
        # current_features = self.graph.nodes[-1].features
        
        while True:
            cur_id  = path[0]
            nxt_id  = path[1]

            if phase == "ROTATE":
                # --- a) keep rotating until within tolerance ------------------
                # current_observation = self.localizer.resize(current_observation)
                tensor_img = numpy_image_to_torch(current_observation).to(self.localizer.device)
                current_features = self.localizer.detector.extract(tensor_img)
                node = self.graph.nodes[cur_id]
                theta = self.find_heading(node,current_features,current_observation)
                h_err = self.controller.heading_error(cur_id, nxt_id, theta)
                print(f"Heading error is {np.rad2deg(h_err)} degree")

                if abs(h_err) > self.controller.ANG_TOL:
                    print("Rotating")
                    rot, exec_time = self.controller.rotate_step(h_err)
                    print(f'rotate for {exec_time} seconds in diretion {rot}')
                    t_end = time.time() + exec_time
                    while time.time() < t_end and self.robot.ros.is_connected:
                        self.robot.move(rz=rot)
                        time.sleep(0.1)
                    # self.robot.send_action(cmd)
                    # self.robot.wait_execution()
                    print("Stop")
                    self.robot.stop()
                    current_observation = self.robot.get_next_frame_after_stop(timeout=2.0)
                    continue
                else:
                    phase = "DRIVE"
                    continue                    # go back to top of loop

            elif phase == "DRIVE":
                
                # --- b) drive straight toward nxt_id --------------------------
                # dist = self.controller.dist_to_next(cur_id, nxt_id)

                # if dist > self.controller.DIST_TOL:
                    # cmd = self.r
                    # print("move forward")
                    # time.sleep(5)
                    # self.robot.send_action(cmd)
                    # self.robot.wait_execution()

                    # OPTIONAL: mid-edge localisation if you want
                    # frame = robot.get_camera_frame()
                    # ...

                    # continue                # still driving, do not check waypoint

                # --- c) reached the new node ----------------------------------
                # current_id = nxt_id         # *now* we can claim we are there
                # next_wp_idx += 1
                t_end = time.time() + 2.0
                while time.time() < t_end and self.robot.ros.is_connected:
                    self.robot.move(x=0.2)
                    time.sleep(0.1)
                # self.robot.send_action(cmd)
                # self.robot.wait_execution()
                self.robot.stop()

                phase = "ROTATE"            # next edge will start with a turn
                candidate_node_list = self.get_closest_nodes(cur_id,5)
                candidate_node_list.append(cur_id)

                current_observation = self.robot.get_next_frame_after_stop(timeout=2.0)
                # current_observation = self.localizer.resize(current_observation)
                tensor_img = numpy_image_to_torch(current_observation).to(self.localizer.device)
                current_features = self.localizer.detector.extract(tensor_img)
                best_match = 0 
                for candidate_node in candidate_node_list:
                    node = self.graph.nodes[candidate_node]
                    with torch.no_grad():
                        pred = self.localizer.matcher({"image0": current_features, "image1": node.features})
                    matches = pred["matches"][0].cpu().numpy()
                    if len(matches)>best_match:
                        best_match = len(matches)
                        best_node = candidate_node
                path = self.planner.a_star(best_node, goal_id)
                print(f"[Navigator] Planned path: {path}")
            user_input = input("Press Enter to continue or 'q' to quit: ")
            if user_input.lower() == 'q':
                print("Exiting...")
                break
            elif user_input == '':
                print("Continuing the process...")
                
                    

                # print(f"[Navigator] Arrived at node {current_id}")


# ----------------------------------------------------------------------
#  MAIN SCRIPT ENTRYPOINT
# ----------------------------------------------------------------------
def main():
    # 0. Load resources ------------------------------------------------
    graph_path   = Path("pose_graph_mech_latest.pkl")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    # start_rgb = cv2.imread('query-0002.jpeg')
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
