#!/usr/bin/env python3
# navigation_main.py

from typing import List
from pathlib import Path
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from oop_map import PoseGraph,Node,Edge,MapObject,ChildMapObject
from lightglue import SuperPoint,LightGlue
from lightglue.utils import numpy_image_to_torch

import heapq
import math
from typing import Dict, List, Tuple, Optional
from itertools import count

import os
import torch
from PIL import Image
import cv2
from collections import defaultdict

from sinkhorn_matching import compute_sinkhorn,compute_low_cost_mass

### ONEFORMER IMPORTS
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from demo.defaults import DefaultPredictor
from oneformer import (
    add_oneformer_config, add_common_config,
    add_swin_config, add_dinat_config, add_convnext_config)

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

        # 3. resize with high-quality filter
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
    def __init__(self, graph: PoseGraph):
        self.graph = graph

    def compute_action(self, current_id: int, next_id: int):
        """
        Returns a velocity / steering command that will move
        the robot from current node toward next node.
        """
        # TODO: implement pure-pursuit, waypoint follower, etc.
        raise NotImplementedError


# ----------------------------------------------------------------------
#  ROBOT I/O
# ----------------------------------------------------------------------
class RobotInterface:
    """Adapter around ROS2, custom socket, simulator API, etc."""
    def send_action(self, cmd: ActionCmd):
        # TODO: publish to /cmd_vel or other interface
        raise NotImplementedError

    def wait_execution(self, timeout: float = 2.0):
        # TODO: block until motion complete or timeout
        time.sleep(timeout)

    def get_camera_frame(self):
        # TODO: read from driver / subscription
        raise NotImplementedError


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
                 robot: RobotInterface):
        self.graph      = graph
        self.localizer  = localizer
        self.planner    = planner
        self.controller = controller
        self.robot      = robot

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
        plt.savefig('found_path.png')

        """
        next_wp_idx = 1  # index in path of the first *future* waypoint

        # --- 4. Follow path ------------------------------------------
        while next_wp_idx < len(path):
            current_id = path[next_wp_idx - 1]
            next_id    = path[next_wp_idx]

            # 4a. Compute and send action
            cmd = self.controller.compute_action(current_id, next_id)
            self.robot.send_action(cmd)
            self.robot.wait_execution()

            # 4b. Grab new observation & update localization
            frame = self.robot.get_camera_frame()
            maybe_id = self.localizer.localize(frame)
            if maybe_id is not None:
                current_id = maybe_id
                print(f"[Navigator] Re-localized to node {current_id}")

            # 4c. If we have arrived at next waypoint, advance pointer
            if current_id == next_id:
                next_wp_idx += 1
        """
        print("[Navigator] Target reached.")


# ----------------------------------------------------------------------
#  MAIN SCRIPT ENTRYPOINT
# ----------------------------------------------------------------------
def main():
    # 0. Load resources ------------------------------------------------
    graph_path   = Path("pose_graph_mechatronics_loop.pkl")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    start_rgb = cv2.imread('query-0002.jpeg')
    start_rgb = cv2.cvtColor(start_rgb, cv2.COLOR_BGR2RGB)

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
    robot        = RobotInterface()

    # 2. Start high-level navigation ----------------------------------
    nav = Navigator(graph, localizer, planner, controller, robot)
    nav.run(start_rgb,target_rgb)


if __name__ == "__main__":
    main()
