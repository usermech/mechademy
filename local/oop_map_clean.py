import numpy as np
import cv2
from scipy.optimize import least_squares
import threading
from queue import Queue
import time
import pickle
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch
from eightpa_solvers.camera_recovering import get_cam_pose_by_ransac_8pa
import matplotlib.pyplot as plt
from sinkhorn_matching import compute_sinkhorn,compute_low_cost_mass
import networkx as nx
import community
from itertools import combinations
from collections import defaultdict,deque
from object_projection import wraparound_centroid,ransac_intersection
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import MapClass

import copy

from scipy.spatial.transform import Rotation 

### ONEFORMER IMPORTS
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from demo.defaults import DefaultPredictor
from oneformer import (
    add_oneformer_config, add_common_config,
    add_swin_config, add_dinat_config, add_convnext_config)

# --- Map Object Representation ---

class MapObject:
    def __init__(self,id_pairs):
        self.id_pairs = id_pairs  # int -> (node_id, mask_id)

    def get_id_keys(self):
        return list(self.id_pairs.keys())

    def get_id_values(self):
        return list(self.id_pairs.values())

    def remove_id_pair(self, node_id, mask_id):
        target = (node_id, mask_id)
        for key, value in list(self.id_pairs.items()):  # Use list to allow safe removal
            if value == target:
                del self.id_pairs[key]
                return True  # Successfully removed
        return False  # Not found


class ChildMapObject(MapObject):
    def __init__(self, feature_collection=None, position=None, parent_object=None, parent_id = None, id_pairs=None):
        super().__init__(id_pairs=id_pairs)
        self._feature_collection = np.array(feature_collection) if feature_collection is not None else np.array([])
        self.position = position
        self.parent_object = parent_object  # Should be a MapObject
        self._parent_id = parent_id

    # Feature collection property
    @property
    def feature_collection(self):
        return self._feature_collection

    @feature_collection.setter
    def feature_collection(self, value):
        if isinstance(value, np.ndarray):
            self._feature_collection = value
        else:
            raise TypeError("feature_collection must be a numpy array")

    # Position property
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            self._position = value
        else:
            raise ValueError("position must be a tuple of two numbers (x, y)")
    @property
    def parent_id(self):
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value):
        self._parent_id = value

    # Parent id_pairs access
    def get_parent_id_pairs(self):
        if self.parent_object and isinstance(self.parent_object, MapObject):
            return self.parent_object.id_pairs
        return self.id_pairs
    

# --- Graph Representation ---

class Node:
    def __init__(self, id):
        self.id = id
        self.x = None
        self.y = None
        self.theta = None  # Yaw angle in radians
        self.rgb_image = None
        self.semantic_masks = None  # Dictionary of semantic masks indexed by ID  
        self.features = None
        self.keypoints = None
        self.descriptors = None
        self.feature_collections = None  # Dictionary of feature collections indexed by ID
        self.centroids = None

class Edge:
    def __init__(self, i, j, t_ij, R_ij):
        self.i = i  # From node
        self.j = j  # To node
        self.t_ij = t_ij  # Relative translation from i to j (3D)
        self.R_ij = R_ij  # Relative rotation from i to j (3x3)
        self.information = self.calculate_information_matrix()

    def calculate_information_matrix(self):
        r = Rotation.from_matrix(self.R_ij)
        roll, pitch, yaw = r.as_euler("xyz", degrees=False)

        z = self.t_ij[2]

        # Define penalties for deviations
        z_penalty = abs(z)
        roll_penalty = abs(roll)
        pitch_penalty = abs(pitch)

        deviation_score = 4*z_penalty + roll_penalty + pitch_penalty

        # Clamp or scale deviation to affect confidence
        penalty_scale = 1.0 + 10.0 * deviation_score  # Tune this factor

        # Invert to get information strength (higher penalty = lower info)
        info_matrix = np.diag([1.0 / penalty_scale, 1.0 / penalty_scale, 1.0 / penalty_scale])
        # print(f"Edge ({self.i}, {self.j}) info matrix: {info_matrix}")
        return info_matrix

class PoseGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.objects = []
        self.extended_objects = []

    def add_node(self, id, rgb_image=None):
        if id not in self.nodes:
            self.nodes[id] = Node(id)
            self.nodes[id].rgb_image = rgb_image

    def add_edge(self, i, j, t_ij, R_ij):
        self.edges.append(Edge(i, j, t_ij, R_ij))
        self.add_node(i)
        self.add_node(j)

# --- Utility Functions ---

def extract_yaw_from_R(R):
    return np.arctan2(R[1, 0], R[0, 0])

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def initialize_2d_poses(graph, anchor_id=None):
    if anchor_id is None:
        anchor_id = next(iter(graph.nodes))

    anchor = graph.nodes[anchor_id]
    anchor.x, anchor.y, anchor.theta = 0.0, 0.0, 0.0
    initialized = {anchor_id}

    # BFS-like propagation
    queue = [anchor_id]

    while queue:
        current_id = queue.pop(0)
        current_node = graph.nodes[current_id]

        for edge in graph.edges:
            if edge.i == current_id and edge.j not in initialized:
                neighbor_id = edge.j
                t = edge.t_ij
                R = edge.R_ij
                yaw_ij = extract_yaw_from_R(R)

            elif edge.j == current_id and edge.i not in initialized:
                neighbor_id = edge.i
                t = -edge.t_ij
                R = edge.R_ij.T
                yaw_ij = extract_yaw_from_R(R)

            else:
                continue

            dx, dy = t[0], t[1]
            cos_theta = np.cos(current_node.theta)
            sin_theta = np.sin(current_node.theta)

            x_new = current_node.x + cos_theta * dx - sin_theta * dy
            y_new = current_node.y + sin_theta * dx + cos_theta * dy
            theta_new = normalize_angle(current_node.theta + yaw_ij)

            graph.nodes[neighbor_id].x = x_new
            graph.nodes[neighbor_id].y = y_new
            graph.nodes[neighbor_id].theta = theta_new

            initialized.add(neighbor_id)
            queue.append(neighbor_id)

            # print(f"Initialized edge: ({current_id}, {neighbor_id}) => Node {neighbor_id} at ({x_new:.2f}, {y_new:.2f}, θ={np.degrees(theta_new):.1f}°)")
        
        
def build_residuals_vector(graph, variable_ids):
    def residuals(x):
        id_to_idx = {nid: i for i, nid in enumerate(variable_ids)}
        poses = {}

        for i, nid in enumerate(variable_ids):
            xi = x[3 * i]
            yi = x[3 * i + 1]
            ti = x[3 * i + 2]
            poses[nid] = (xi, yi, ti)

        residual_list = []

        for edge in graph.edges:
            if edge.i not in graph.nodes or edge.j not in graph.nodes:
                continue

            ni, nj = graph.nodes[edge.i], graph.nodes[edge.j]
            if None in (ni.x, ni.y, ni.theta, nj.x, nj.y, nj.theta):
                continue

            # Get pose of node i
            if edge.i in variable_ids:
                xi, yi, thetai = poses[edge.i]
            else:
                xi, yi, thetai = ni.x, ni.y, ni.theta

            # Get pose of node j
            if edge.j in variable_ids:
                xj, yj, thetaj = poses[edge.j]
            else:
                xj, yj, thetaj = nj.x, nj.y, nj.theta

            # Compute relative pose
            dx = xj - xi
            dy = yj - yi
            dtheta = normalize_angle(thetaj - thetai)

            # Rotate into i’s frame
            cos_theta = np.cos(-thetai)
            sin_theta = np.sin(-thetai)
            dx_local = cos_theta * dx - sin_theta * dy
            dy_local = sin_theta * dx + cos_theta * dy

            t_ij = edge.t_ij
            R_ij = edge.R_ij
            expected_dx = t_ij[0]
            expected_dy = t_ij[1]
            expected_dtheta = extract_yaw_from_R(R_ij)

            err_x = dx_local - expected_dx
            err_y = dy_local - expected_dy
            err_theta = normalize_angle(dtheta - expected_dtheta)

            error_vec = np.array([err_x, err_y, err_theta])
            weighted_error = edge.information @ error_vec  # Apply information matrix
            # weighted_error = error_vec
            residual_list.extend(weighted_error.tolist())

        return residual_list

    return residuals

def optimize_pose_graph(graph, anchor_id=None):
    if anchor_id is None:
        anchor_id = next(iter(graph.nodes))

    variable_ids = [nid for nid in graph.nodes if nid != anchor_id]
    x0 = []

    for nid in variable_ids:
        node = graph.nodes[nid]
        x0.extend([node.x, node.y, node.theta])

    residual_func = build_residuals_vector(graph, variable_ids)
    print(f"Starting optimization with {len(variable_ids)} variable nodes.")

    result = least_squares(residual_func, x0, verbose=2)

    for i, nid in enumerate(variable_ids):
        graph.nodes[nid].x = result.x[3 * i]
        graph.nodes[nid].y = result.x[3 * i + 1]
        graph.nodes[nid].theta = normalize_angle(result.x[3 * i + 2])

    print("Optimization complete.")

def clean_disconnected_nodes_and_edges(graph: PoseGraph, anchor_id=0):
    visited = set()
    queue = deque([anchor_id])

    # Breadth-first search to find all reachable nodes from anchor_id
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        for edge in graph.edges:
            if edge.i == current and edge.j not in visited:
                queue.append(edge.j)
            elif edge.j == current and edge.i not in visited:
                queue.append(edge.i)

    # Remove unreachable nodes
    all_node_ids = set(graph.nodes.keys())
    unreachable = all_node_ids - visited
    for node_id in unreachable:
        del graph.nodes[node_id]

    # Remove edges that reference removed nodes
    original_edge_count = len(graph.edges)
    graph.edges = [
        edge for edge in graph.edges
        if edge.i in visited and edge.j in visited
    ]
    removed_edges = original_edge_count - len(graph.edges)

def all_nodes_have_edges(pose_graph):
    node_ids = sorted(pose_graph.nodes.keys())
    connected_nodes = set()
    for edge in pose_graph.edges:
        connected_nodes.add(edge.i)
        connected_nodes.add(edge.j)
    return all(n in connected_nodes for n in node_ids[1:])

def draw_pose_graph(graph, title="Pose Graph", save_path=None):
    node_ids = list(graph.nodes.keys())
    positions = [(-graph.nodes[nid].x, graph.nodes[nid].y) for nid in node_ids if graph.nodes[nid].x is not None]

    if not positions:
        print("No valid positions found.")
        return

    positions = np.array(positions)
    x, y = positions[:, 0], positions[:, 1]
    frame_indices = np.arange(len(positions))  # color gradient source

    # Plot with color progression
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=frame_indices, cmap='viridis', s=40, edgecolors='k')

    # Add labels
    for idx, nid in enumerate(node_ids):
        node = graph.nodes[nid]
        if node.x is not None and node.y is not None:
            plt.text(-node.x + 0.02, node.y + 0.02, str(nid), fontsize=7)

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Frame Index (Traversal Order)")

    # Highlight start and end
    plt.scatter(x[0], y[0], color='green', s=80, marker='o', label='Start')
    plt.scatter(x[-1], y[-1], color='red', s=80, marker='X', label='End')
    plt.legend()

    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    if save_path:
        plt.savefig(save_path)
        print(f"Pose graph saved to {save_path}")

    plt.show()

    # # Draw edges
    # for edge in graph.edges:
    #     if edge.i in graph.nodes and edge.j in graph.nodes:
    #         xi, yi = graph.nodes[edge.i].x, graph.nodes[edge.i].y
    #         xj, yj = graph.nodes[edge.j].x, graph.nodes[edge.j].y
    #         if None not in (xi, yi, xj, yj):
    #             plt.plot([xi, xj], [yi, yj], 'k-', linewidth=0.5)

    # Draw nodes
    # for node_id, node in graph.nodes.items():
    #     if node.x is not None and node.y is not None:
    #         plt.plot(node.x, node.y, 'ro')
    #         plt.text(node.x + 0.02, node.y + 0.02, str(node_id), fontsize=8)

    # plt.title(title)
    # plt.axis('equal')
    # plt.grid(True)
    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")
    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"Pose graph saved to {save_path}")
    # plt.show()


def convert_pixel_to_angle(keypoints, image_width=512, image_height=256):
    x = keypoints[:,0]
    y = keypoints[:,1]
    azimuth = (x / image_width) * 2 * np.pi - np.pi 
    elevation = (0.5 - y / image_height) * np.pi
    return azimuth, elevation

def get_keypoint_unit_vector(azimuth, elevation):
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    return np.stack((x, y, z), axis=1)

def calculate_heading_angle_ransac(feats0,feats1,matches,image_width=512, image_height=256):
    mkpts0, mkpts1 = feats0[matches[...,0]], feats1[matches[...,1]]
    first_n_keypoints = 100
    az0, ele0 = convert_pixel_to_angle(mkpts0[:first_n_keypoints],1024,512)
    az1, ele1 = convert_pixel_to_angle(mkpts1[:first_n_keypoints],1024,512)
    u0 = get_keypoint_unit_vector(az0, ele0)
    u1 = get_keypoint_unit_vector(az1, ele1)
    bearing0 = u0.T
    bearing1 = u1.T  
    cam_pose = get_cam_pose_by_ransac_8pa(bearing0, bearing1)
    translation = cam_pose[:3,3]
    rotation = cam_pose[:3,:3]
    return translation, rotation

def average_edge_distance(pose_graph):
    total_dist = 0.0
    count = 0
    for edge in pose_graph.edges:
        n1 = pose_graph.nodes[edge.i]
        n2 = pose_graph.nodes[edge.j]
        dx = n1.x - n2.x
        dy = n1.y - n2.y
        dist = np.hypot(dx, dy)
        total_dist += dist
        count += 1
    return total_dist / count if count > 0 else 0.0

def find_candidate_neighbors(pose_graph, max_dist=5.0):
    cands = []
    ids = sorted(pose_graph.nodes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            n1, n2 = pose_graph.nodes[ids[i]], pose_graph.nodes[ids[j]]
            if any((e.i == ids[i] and e.j == ids[j]) or (e.j == ids[i] and e.i == ids[j])
                   for e in pose_graph.edges):
                continue
            dx, dy = n1.x - n2.x, n1.y - n2.y
            if np.hypot(dx, dy) < max_dist:
                cands.append((ids[i], ids[j]))
    return cands

def add_new_edges(pose_graph, candidates, edge_creator, lock):
    for id1, id2 in candidates:
        with lock:
            n1, n2 = pose_graph.nodes[id1], pose_graph.nodes[id2]
        t, R = edge_creator.match_and_estimate(n1, n2)
        if t is not None:
            with lock:
                pose_graph.add_edge(id1, id2, t, R)
            # print(f"[Main] Added new neighbor edge {id1} ↔ {id2}")
        else:
            pass
            # print(f"[Main] No edge for neighbor pair {id1} ↔ {id2}")

def detect_high_anomalies_z_score(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
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
    return np.where(z_scores > threshold)[0]


def extract_subgraphs(edges_list, split_threshold=30, min_size=3):
    G = nx.Graph()
    G.add_edges_from(edges_list)

    raw_subgraphs = [
        G.subgraph(c).copy()
        for c in nx.connected_components(G)
        if len(c) >= min_size
    ]

    partitioned_subgraphs = []
    adj_matrices = []
    subgraph_origins = []

    for raw_index, sg in enumerate(raw_subgraphs):
        if len(sg.nodes) <= split_threshold:
            # Keep small subgraphs as-is
            partitioned_subgraphs.append(sg)
            adj_matrices.append(nx.to_numpy_array(sg))
            subgraph_origins.append(raw_index)
        else:
            # Louvain partitioning for larger subgraphs
            partition = community.best_partition(sg)
            groups = defaultdict(list)

            for node, group_id in partition.items():
                groups[group_id].append(node)
            print(f"[PostProcessing] Cluster {raw_index} divided into {len(groups)} groups.")
            for nodes in groups.values():
                if len(nodes) >= min_size:
                    sub = sg.subgraph(nodes).copy()
                    partitioned_subgraphs.append(sub)
                    adj_matrices.append(nx.to_numpy_array(sub))
                    subgraph_origins.append(raw_index)

    return raw_subgraphs, partitioned_subgraphs, adj_matrices, subgraph_origins


def save_clustered_masks(partitioned_subgraphs, pose_graph, precomputed_masks, output_dir="clustered_masks"):
    """
    Save visualizations of masks with outlines for each cluster in separate folders.

    Args:
        partitioned_subgraphs (List[nx.Graph]): List of subgraphs, each containing (node_id, mask_id) tuples as nodes.
        precomputed_images (Dict[int, np.ndarray]): Dictionary mapping node_id to RGB image.
        precomputed_masks (Dict[int, Dict[int, np.ndarray]]): Nested dict mapping node_id → mask_id → mask.
        output_dir (str): Directory where output folders/images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for cluster_idx, subgraph in enumerate(partitioned_subgraphs):
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_idx}")
        os.makedirs(cluster_dir, exist_ok=True)

        for node_id, mask_id in subgraph:
            # Load image and mask
            rgb_img = pose_graph.nodes[node_id].rgb_image.copy()
            mask = pose_graph.nodes[node_id].semantic_masks[mask_id]

            # Dilate mask to find edges
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            edges = dilated_mask - mask

            # Overlay edges in red
            overlay_img = rgb_img.copy()
            overlay_img[edges == 1] = [255, 0, 0]  # Red edge

            # Save image
            save_path = os.path.join(cluster_dir, f"node_{node_id}_mask_{mask_id}.png")
            plt.imsave(save_path, overlay_img)
    


class SemanticSegmentationWorker(threading.Thread):
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

    def __init__(
        self,
        queue,
        pose_graph,
        lock,
        feature_collection_queue,
        *,
        dataset: str,
        model_path: str,
        use_swin: bool = False,
        device: str = "cuda",          
        filter_dynamic_objects: bool = False,
        repo_root: str = "/home/romer/umut/segmentation/OneFormer",  # path to config files
    ):
        super().__init__(daemon=True)          
        self.queue = queue
        self.pose_graph = pose_graph
        self.lock = lock
        self.feature_collection_queue = feature_collection_queue
        self.dataset = dataset
        self.model_path = model_path
        self.use_swin = use_swin
        self.device_str = device
        self.repo_root = repo_root
        self.filter_dynamic_objects = filter_dynamic_objects
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

    def run(self):
        while True:
            node_id = self.queue.get()
            try:
                with self.lock:
                    if node_id not in self.pose_graph.nodes:
                        continue
                    node = self.pose_graph.nodes[node_id]
                    node.semantic_masks = self._segment_image(node,self.filter_dynamic_objects)                    
                    print(f"[SegmentationWorker] Node {node_id} segmented into {len(node.semantic_masks)}.")

                    # Calculate centroids and store angles
                    node.centroids = self._calculate_centroids(node.semantic_masks)

                    # Check if keypoints already exist
                    if node.keypoints is not None:
                        self.feature_collection_queue.put(node_id)
                        # print(f"[SegmentationWorker] Node {node_id} pushed to feature collection queue.")
            finally:
                self.queue.task_done()

    def _segment_image(self, node, filter_dynamic_objects=True):
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

            if filter_dynamic_objects and category in [3, 12, 19, 75]:
                dynamics_object_mask = (mask == label).astype(bool)

                kp = node.features['keypoints'][0].round().to(torch.long).cpu()
                keypoints_drop = dynamics_object_mask[kp[:, 1], kp[:, 0]]
                keypoints_keep = ~keypoints_drop
                keypoints_keep = torch.tensor(keypoints_keep, dtype=torch.bool).to(node.features["keypoints"].device)

                node.features["keypoints"] = node.features["keypoints"][:, keypoints_keep, :]
                node.features["descriptors"] = node.features["descriptors"][:, keypoints_keep, :]
                node.features["keypoint_scores"] = node.features["keypoint_scores"][:, keypoints_keep]

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
        return masks
    
    def _infer_image(self, img, task="semantic"):
        """Light wrapper around DefaultPredictor."""
        pred = self.predictor(img, task)
        panoptic_seg, segments_info = pred["panoptic_seg"]
        torch.cuda.empty_cache()
        return panoptic_seg, segments_info

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
    
class FeatureExtractionWorker(threading.Thread):
    def __init__(self, queue, pose_graph, lock, feature_collection_queue,edge_queue, device=None):
        super().__init__(daemon=True)
        self.queue = queue
        self.pose_graph = pose_graph
        self.lock = lock
        self.feature_collection_queue = feature_collection_queue
        self.edge_queue = edge_queue
        self._device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = SuperPoint(max_num_keypoints=1024).eval().to(self._device)

    def run(self):
        while True:
            node_id = self.queue.get()
            node = self.pose_graph.nodes[node_id]
            tensor_img = numpy_image_to_torch(node.rgb_image.copy()).to(self._device)
            output = self.detector.extract(tensor_img)
            
            node.features = output
            node.keypoints = output['keypoints'].squeeze(0).cpu().numpy()
            node.descriptors = output['descriptors'].squeeze(0).cpu().numpy()
            print(f"[FeatureExtractionWorker] Node {node_id} features extracted.")

            self.edge_queue.put(node_id)
            self.queue.task_done()

class EdgeCreator(threading.Thread):
    def __init__(self, pose_graph, lock, edge_queue, window_size=5):
        super().__init__(daemon=True)
        self.pose_graph = pose_graph
        self.lock = lock
        self.edge_queue = edge_queue
        self.window_size = window_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def run(self):
        for item in iter(self.edge_queue.get, None):  
            try:
                if isinstance(item, int):
                    self._process_new_node(item)  
                elif isinstance(item, tuple) and len(item) == 2:
                    self._process_pair(*item)
            finally:
                self.edge_queue.task_done()          

        self.edge_queue.task_done()

    # ────────────────────────────────────────────────────────────
    # 1. Sliding-window logic 
    # ────────────────────────────────────────────────────────────
    def _process_new_node(self, node_id):
        with self.lock:
            node_ids = sorted(self.pose_graph.nodes.keys())
            try:
                idx_current = node_ids.index(node_id)
            except ValueError:                      
                return
            node_current = self.pose_graph.nodes[node_id]

        if node_current.keypoints is None or node_current.descriptors is None:
            return

        for offset in range(1, self.window_size + 1):
            idx_prev = idx_current - offset
            if idx_prev < 0:
                break

            with self.lock:
                id_prev = node_ids[idx_prev]
                node_prev = self.pose_graph.nodes[id_prev]

                if self.edge_exists(node_id, id_prev):
                    continue
                if (node_prev.keypoints is None or
                        node_prev.descriptors is None):
                    continue

            t_ij, R_ij = self.match_and_estimate(node_current, node_prev)
            if t_ij is not None and R_ij is not None:
                with self.lock:
                    self.pose_graph.add_edge(node_id, id_prev, t_ij, R_ij)
                    print(f"[EdgeWorker] edge {node_id} ↔ {id_prev} added")

    # ────────────────────────────────────────────────────────────
    # 2. Pair logic
    # ────────────────────────────────────────────────────────────
    def _process_pair(self, i, j):
        if i > j:
            i, j = j, i

        with self.lock:
            try:
                node_i = self.pose_graph.nodes[i]
                node_j = self.pose_graph.nodes[j]
            except KeyError:
                return

            if self.edge_exists(i, j):
                return
            if (node_i.keypoints is None or node_i.descriptors is None or
                    node_j.keypoints is None or node_j.descriptors is None):
                return

        t_ij, R_ij = self.match_and_estimate(node_i, node_j,400)
        if t_ij is not None and R_ij is not None:
            with self.lock:
                self.pose_graph.add_edge(i, j, t_ij, R_ij)
                print(f"[EdgeWorker] edge {i} ↔ {j} added from loop closure")


    def edge_exists(self, i, j):
        return any((e.i == i and e.j == j) or (e.i == j and e.j == i) for e in self.pose_graph.edges)
    
    def match_and_estimate(self, node1, node2,min_matches=200):
        with torch.no_grad():
            pred = self.matcher({"image0": node1.features, "image1": node2.features})
        matches = pred["matches"][0].cpu().numpy()
        if len(matches) < min_matches:
            return None, None
        
        feats0 = node1.features['keypoints'].squeeze(0).cpu().numpy()
        feats1 = node2.features['keypoints'].squeeze(0).cpu().numpy()
        # feats0 = node1.keypoints.copy()
        # feats1 = node2.keypoints.copy()
        t_ij, R_ij = calculate_heading_angle_ransac(feats0, feats1, matches)
        return t_ij, R_ij
    
class FeatureCollectionWorker(threading.Thread):
    def __init__(self, queue, pose_graph, lock, similarity_matrix_lock, similarity_matrix):
        super().__init__(daemon=True)
        self.queue = queue
        self.pose_graph = pose_graph
        self.lock = lock
        self.similarity_matrix_lock = similarity_matrix_lock
        self.similarity_matrix = similarity_matrix

    def run(self):
        print("[FeatureCollectionWorker] Started.")
        while True:
            

            node_id = self.queue.get()
            with self.lock:
                if node_id not in self.pose_graph.nodes:
                    self.queue.task_done()
                    continue

                node = self.pose_graph.nodes[node_id]

                if not node.semantic_masks:
                    self.queue.task_done()
                    continue
                if node.keypoints is None or node.descriptors is None:
                    self.queue.task_done()
                    continue

                keypoints = np.round(node.keypoints).astype(int)
                descriptors = node.descriptors

            if node.feature_collections is None:
                node.feature_collections = {}


            for mask_id, mask in node.semantic_masks.items():
                valid_mask = (keypoints[:, 1] < mask.shape[0]) & (keypoints[:, 0] < mask.shape[1])
                valid_points = keypoints[valid_mask]
                valid_desc = descriptors[valid_mask]

                mask_values = mask[valid_points[:, 1], valid_points[:, 0]]
                keep = mask_values == 1

                if np.sum(keep) < 10:
                    continue

                collection = valid_desc[keep].copy()
                node.feature_collections[mask_id] = collection

            print(f"[FeatureCollectionWorker] Node {node_id} - {len(node.feature_collections)} collections computed.")

            with self.similarity_matrix_lock:
                for mask_id, new_collection in node.feature_collections.items():
                    new_row = []

                    for existing in self.similarity_matrix["collections"]:
                        P, C = compute_sinkhorn(new_collection, existing)
                        score = compute_low_cost_mass(P, C)
                        new_row.append(score)

                    for i in range(len(self.similarity_matrix["matrix"])):
                        self.similarity_matrix["matrix"][i].append(new_row[i])

                    new_row.append(1.0)
                    self.similarity_matrix["matrix"].append(new_row)
                    self.similarity_matrix["index_mapping"].append((node_id, mask_id))
                    self.similarity_matrix["collections"].append(new_collection)

                    # print(f"[FeatureCollectionWorker] Added (Node {node_id}, Mask {mask_id}) to similarity matrix.")

            self.queue.task_done()

def load_and_prepare_semantic_map(filepath):
    with open(filepath, "rb") as f:
        semantic_map = pickle.load(f)
    print("Semantic map loaded.")
    return semantic_map


def start_all_workers(pose_graph, lock, similarity_matrix, similarity_matrix_lock):
    seg_queue = Queue()
    feat_queue = Queue()
    fc_queue = Queue()
    edge_queue = Queue()

    seg_worker = SemanticSegmentationWorker(
    queue=seg_queue,
    pose_graph=pose_graph,
    lock=lock,
    feature_collection_queue=fc_queue,
    dataset="ade20k",
    model_path="/home/romer/umut/segmentation/OneFormer/250_16_dinat_l_oneformer_ade20k_160k.pth",
    use_swin=False,              # or True for Swin
    device="cuda",               # "cpu" if no GPU
    filter_dynamic_objects=True,  # Set to True to filter dynamic objects
    )
    feat_worker = FeatureExtractionWorker(feat_queue, pose_graph, lock, fc_queue, edge_queue)
    edge_worker = EdgeCreator(pose_graph, lock, edge_queue, window_size=5)
    fc_worker = FeatureCollectionWorker(fc_queue, pose_graph, lock, similarity_matrix_lock, similarity_matrix)

    for worker in [seg_worker, feat_worker, edge_worker, fc_worker]:
        worker.start()

    print("Workers started.")
    return seg_queue, feat_queue, fc_queue, edge_queue, edge_worker

    # for worker in [seg_worker, feat_worker, fc_worker]:
    #     worker.start()

    # print("Workers started.")
    # return seg_queue, feat_queue, fc_queue, None

def add_images_to_graph(images, pose_graph, lock, seg_queue, feat_queue):
    for node_id, image in enumerate(images.values()):
        with lock:
            pose_graph.add_node(node_id, rgb_image=image)
        feat_queue.put(node_id)
        seg_queue.put(node_id)
        time.sleep(0.1)


def wait_for_all_queues(queues):
    print("[Main] Waiting for segmentation and feature extraction to complete...")
    for queue in queues:
        queue.join()


def postprocess_similarity_matrix(similarity_matrix):
    sim_mat_np = np.array(similarity_matrix["matrix"])
    index_mapping = similarity_matrix["index_mapping"]
    matched_pairs = []

    for i, row in enumerate(sim_mat_np):
        high_matches = detect_high_anomalies_z_score(row, threshold=4.0)
        for j in high_matches:
            matched_pairs.append((index_mapping[i], index_mapping[j]))

    print(f"[PostProcessing] {len(matched_pairs)} matched feature collection pairs.")
    return matched_pairs

def compute_row_score(similarity_matrix, row_index, selected_cols):
    
    if not selected_cols:
        return 0.0  # or np.nan, depending on your use case
    values = similarity_matrix[row_index, selected_cols]
    return np.sum(values)

def compute_set_scores(similarity_matrix, indices_set):

    scores = {}
    for idx in indices_set:
        others = [i for i in indices_set if i != idx]
        score = compute_row_score(similarity_matrix, idx, others)
        scores[idx] = score/len(indices_set)
        sorted_indices = sorted(scores, key=lambda x: scores[x],reverse=True)
    return sorted_indices

def extract_mutual_matches(P, threshold=0.0):
    """
    Extract mutual best matches from Sinkhorn transport matrix P.

    Args:
        P (np.ndarray): Transport matrix of shape (n0, n1).
        threshold (float): Minimum transport weight to consider a match.

    Returns:
        matches (list of tuples): List of (i, j) pairs of matched indices.
    """
    # For each vector i in set 0, find best j in set 1
    best_j_for_i = np.argmax(P, axis=1)

    # For each vector j in set 1, find best i in set 0
    best_i_for_j = np.argmax(P, axis=0)

    matches = []
    for i, j in enumerate(best_j_for_i):
        # Check mutual best match and threshold
        if best_i_for_j[j] == i and P[i, j] > threshold:
            matches.append((i, j))

    return matches

def get_merged_feature_collection(similarity_matrix, object_group,epsilon=0.1):
    feature_collections = similarity_matrix["collections"]
    sim_mat_np = np.array(similarity_matrix["matrix"])
    index_mapping = similarity_matrix["index_mapping"]
    tuple_to_index = {t: i for i, t in enumerate(index_mapping)}
    indices = [tuple_to_index[t] for t in object_group]
    sorted_set = compute_set_scores(sim_mat_np,indices)
    root_node = sorted_set[0]
    base_collection = list(feature_collections[root_node])
    for collection_id in sorted_set[1:]:
        new_collection = list(feature_collections[collection_id])
        P, C= compute_sinkhorn(base_collection, new_collection)
        matches = extract_mutual_matches(P)
        # Get set of indices in the new collection that were matched
        matched_new_indices = {i_new for _, i_new in matches}

        # Append only unmatched items from the new collection
        for i, item in enumerate(new_collection):
            if i not in matched_new_indices:
                base_collection.append(item)

    base_collection = np.array(base_collection)
    return base_collection

def cluster_semantic_objects(subs, pose_graph, similarity_matrix, angle_threshold=15):
    for cluster_index, group in enumerate(subs):
        # # parent_object = MapObject(dict(enumerate(obj)))
        # # pose_graph.objects.append(parent_object)
        # vectors = []
        # for node_id, mask_id in obj:
        #     if node_id not in pose_graph.nodes:
        #         continue
        #     node = pose_graph.nodes[node_id]
        #     mask = node.semantic_masks[mask_id]
        #     angle = node.centroids[mask_id] + node.theta
        #     direction = np.array([np.cos(angle[0]), np.sin(angle[0])])
        #     point = [node.x, node.y]
        #     vectors.append((point, direction, (node_id, mask_id)))

        # remaining = vectors.copy()
        # while len(remaining) >= 5:
        #     ransac_input = [(np.array(p), np.array(d)) for p, d, _ in remaining]
        #     intersection, inliers, _ = ransac_intersection(ransac_input, np.radians(angle_threshold), 100)
        #     if len(inliers) < 4:
        #         break

        #     inlier_set = set((tuple(p), tuple(d)) for p, d in inliers)
        #     group = []
        #     next_remaining = []
        #     for p, d, mask in remaining:
        #         if (tuple(p), tuple(d)) in inlier_set:
        #             group.append(mask)
        #         else:
        #             next_remaining.append((p, d, mask))
            
        #     merged_feature_collection = get_merged_feature_collection(similarity_matrix,group)
        #     pose_graph.extended_objects.append(ChildMapObject(feature_collection=merged_feature_collection,position=tuple(intersection),parent_object=None,parent_id = cluster_index,id_pairs=dict(enumerate(group))))
        #     remaining = next_remaining
        

        merged_feature_collection = get_merged_feature_collection(similarity_matrix,group)
        pose_graph.extended_objects.append(ChildMapObject(feature_collection=merged_feature_collection,position=(0,0),parent_object=None,parent_id = cluster_index,id_pairs=dict(enumerate(group))))

    print(f"[PostProcessing] Clustered {len(pose_graph.extended_objects)} directional object groups with {len(pose_graph.objects)} parents.")
    return pose_graph.extended_objects


def optimize_pose_graph_twice(pose_graph, lock, edge_worker, save_dir=None):
    print("Starting pose graph optimization...")
    initialize_2d_poses(pose_graph)
    clean_disconnected_nodes_and_edges(pose_graph,anchor_id=list(pose_graph.nodes.keys())[0])
    optimize_pose_graph(pose_graph)
    avg_dist = average_edge_distance(pose_graph)
    print(f"[Main] Average edge distance: {avg_dist:.2f} m")
    draw_pose_graph(pose_graph, title="Optimized Pose Graph", save_path=os.path.join(save_dir, "optimized_pose_graph.png") if save_dir else None)
    
    candidates = find_candidate_neighbors(pose_graph, max_dist=avg_dist)
    print(f"[Main] Found {len(candidates)} neighbor candidates.")
    add_new_edges(pose_graph, candidates, edge_worker, lock)

    optimize_pose_graph(pose_graph)
    print("Second optimization complete.")
    draw_pose_graph(pose_graph, title="Final Pose Graph", save_path=os.path.join(save_dir, "final_pose_graph.png") if save_dir else None)

def main():
    pose_graph = PoseGraph()
    lock = threading.Lock()
    data_name = "data_10"
    save_directory = os.path.join("./output", data_name)
    os.makedirs(save_directory, exist_ok=True)
    semantic_map = load_and_prepare_semantic_map(f"./processed_data/semantic_map_{data_name}.pkl")
    images =  semantic_map.rgb_observations

    # from itertools import islice

    # # Copy first 30 items from masks and images
    # masks = dict(islice(masks.items(), 10))
    # images = dict(islice(images.items(), 10))

    # with open('path.pkl','rb') as f:
    #     key_order = pickle.load(f)
    # ordered_images = {k: images[k] for k in key_order if k in images}
    # ordered_masks = {i: masks[k] for i,k in enumerate(key_order) if k in masks}
    # images = ordered_images
    # masks = ordered_masks
    similarity_matrix = {
        "matrix": [],
        "collections": [],
        "index_mapping": []
    }
    similarity_matrix_lock = threading.Lock()

    seg_q, feat_q, fc_q, edge_q, edge_worker = start_all_workers(
        pose_graph, lock, similarity_matrix, similarity_matrix_lock
    )

    add_images_to_graph(images, pose_graph, lock, seg_q, feat_q)
    wait_for_all_queues([seg_q, feat_q, fc_q])

    with open(f'{save_directory}/similarity_matrix_{data_name}.pkl','wb') as f:
        pickle.dump(similarity_matrix,f)
    matched_pairs = postprocess_similarity_matrix(similarity_matrix)

    raw_subs, part_subs, _, origins = extract_subgraphs(
        edges_list=matched_pairs,
        split_threshold=30,
        min_size=3
    )
    
    pose_graph.objects.clear()
    pose_graph.extended_objects.clear()
    for obj in raw_subs:
        parent_object = MapObject(dict(enumerate(obj)))
        pose_graph.objects.append(parent_object)
    objects = cluster_semantic_objects(part_subs, pose_graph, similarity_matrix, angle_threshold=15)
    # print(f"[LOADER] {pose_graph.objects} and {pose_graph.extended_objects}")

    # def enqueue_loop_closures(self):
    # """Find long-range node pairs & push them to edge_queue."""
    wait_for_all_queues([edge_q])
    naive_pose_graph = copy.deepcopy(pose_graph)
    print("Starting pose graph optimization...")
    initialize_2d_poses(naive_pose_graph)
    clean_disconnected_nodes_and_edges(naive_pose_graph,anchor_id=list(pose_graph.nodes.keys())[0])
    optimize_pose_graph(naive_pose_graph)
    draw_pose_graph(naive_pose_graph, title="Pose Graph", save_path=os.path.join(save_directory, "naive_pose_graph.png"))

    for map_object in pose_graph.extended_objects:
        nodes = list(map(lambda x: x[0], map_object.id_pairs.values()))

        for id_i, id_j in combinations(nodes, 2):
            if abs(id_i - id_j) <= 5:          
                continue
            if not any((e.i == id_i and e.j == id_j) or (e.i == id_j and e.j == id_i) for e in pose_graph.edges):
                edge_q.put((id_i, id_j))


    wait_for_all_queues([edge_q])

    with lock:
        print(f"Nodes: {len(pose_graph.nodes)}, Edges: {len(pose_graph.edges)}")

    
    # np.save('origins.npy',origins)
    print(f"[PostProcessing] Found {len(raw_subs)} raw subgraphs, partitioned into {len(part_subs)}.")

    optimize_pose_graph_twice(pose_graph, lock, edge_worker, save_directory)

    # for node in pose_graph.nodes.values():
    #     del node.features
    #     del node.descriptors
    #     if hasattr(node, 'keypoints') and isinstance(node.keypoints, torch.Tensor):
    #         node.keypoints = node.keypoints.cpu()
    
    ### LOAD THE POSE GRAPH OBJECT
    # with open('pose_graph.pkl','rb') as f:
    #     pose_graph = pickle.load(f)

    # plt.figure(figsize=(8, 8))
    # for node_id, node in pose_graph.nodes.items():
    #     if node.x is not None and node.y is not None:
    #         plt.plot(-node.x, node.y, 'ro')
    #         # plt.text(node.x + 0.02, node.y + 0.02, str(node_id), fontsize=8)
    # for object_instance in objects:
    #     x,y = -object_instance.position[0],object_instance.position[1]
    #     plt.plot(x, y, 'go')
    #     plt.text(x + 0.02, y + 0.02, str(object_instance.parent_id), fontsize=8)
        
    # plt.axis('equal')
    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")
    # plt.show()
    # plt.savefig('final_pose_graph.png')
    # plt.show()
    # ### SAVE THE POSE GRAPH OBJECT
    with open(f'{save_directory}/pose_graph_{data_name}.pkl','wb') as f:
        pickle.dump(pose_graph,f)
        
    masks = defaultdict(dict)
    for node_id , node in pose_graph.nodes.items():
        masks[node_id] = node.semantic_masks
    # save_clustered_masks(part_subs, pose_graph, masks)
if __name__ == "__main__":
    main()
