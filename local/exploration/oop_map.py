import numpy as np
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
from collections import defaultdict
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import MapClass

# --- Map Object Representation ---

class MapObject:
    def __init__(self, feature_collection=None, position=None, parent_object=None):
        # Initialize attributes
        self._feature_collection = np.array(feature_collection) if feature_collection is not None else np.array([])
        self.instances = {}  # int -> tuple
        self.position = position  # (x, y)
        self.parent_object = parent_object  # Reference to parent object

    # --- Property for feature_collection ---
    @property
    def feature_collection(self):
        return self._feature_collection

    @feature_collection.setter
    def feature_collection(self, value):
        if isinstance(value, np.ndarray):
            self._feature_collection = value
        else:
            raise TypeError("feature_collection must be a numpy array")
        
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            self._position = value
        else:
            raise ValueError("position must be a tuple of two numbers (x, y)")
        
    # --- Instance Access Methods ---
    def get_instance_keys(self):
        return list(self.instances.keys())

    def get_instance_values(self):
        return list(self.instances.values())

    # --- Parent Access Method ---
    def get_parent_instances(self):
        if self.parent_object and isinstance(self.parent_object, MapObject):
            return self.parent_object.instances
        return self.instances
    

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

class Edge:
    def __init__(self, i, j, t_ij, R_ij):
        self.i = i  # From node
        self.j = j  # To node
        self.t_ij = t_ij  # Relative translation from i to j (3D)
        self.R_ij = R_ij  # Relative rotation from i to j (3x3)

class PoseGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

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

            print(f"Initialized edge: ({current_id}, {neighbor_id}) => Node {neighbor_id} at ({x_new:.2f}, {y_new:.2f}, θ={np.degrees(theta_new):.1f}°)")
        
        
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

            residual_list.extend([err_x, err_y, err_theta])

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

def all_nodes_have_edges(pose_graph):
    node_ids = sorted(pose_graph.nodes.keys())
    connected_nodes = set()
    for edge in pose_graph.edges:
        connected_nodes.add(edge.i)
        connected_nodes.add(edge.j)
    return all(n in connected_nodes for n in node_ids[1:])

def draw_pose_graph(graph, title="Pose Graph"):
    plt.figure(figsize=(8, 8))

    # # Draw edges
    # for edge in graph.edges:
    #     if edge.i in graph.nodes and edge.j in graph.nodes:
    #         xi, yi = graph.nodes[edge.i].x, graph.nodes[edge.i].y
    #         xj, yj = graph.nodes[edge.j].x, graph.nodes[edge.j].y
    #         if None not in (xi, yi, xj, yj):
    #             plt.plot([xi, xj], [yi, yj], 'k-', linewidth=0.5)

    # Draw nodes
    for node_id, node in graph.nodes.items():
        if node.x is not None and node.y is not None:
            plt.plot(node.x, node.y, 'ro')
            plt.text(node.x + 0.02, node.y + 0.02, str(node_id), fontsize=8)

    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()

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
    az0, ele0 = convert_pixel_to_angle(mkpts0[:],1024,512)
    az1, ele1 = convert_pixel_to_angle(mkpts1[:],1024,512)
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


def extract_subgraphs(edges_list, split_threshold=10, min_size=3):
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

            for nodes in groups.values():
                if len(nodes) >= min_size:
                    sub = sg.subgraph(nodes).copy()
                    partitioned_subgraphs.append(sub)
                    adj_matrices.append(nx.to_numpy_array(sub))
                    subgraph_origins.append(raw_index)

    return raw_subgraphs, partitioned_subgraphs, adj_matrices, subgraph_origins

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def save_clustered_masks(partitioned_subgraphs, precomputed_images, precomputed_masks, output_dir="clustered_masks"):
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
            rgb_img = precomputed_images[node_id].copy()
            mask = precomputed_masks[node_id][mask_id]

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
    def __init__(self, queue, pose_graph, lock, precomputed_masks, feature_collection_queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.pose_graph = pose_graph
        self.lock = lock
        self.precomputed_masks = precomputed_masks
        self.feature_collection_queue = feature_collection_queue

    def run(self):
        while True:
            node_id = self.queue.get()
            with self.lock:
                if node_id in self.pose_graph.nodes:
                    node = self.pose_graph.nodes[node_id]
                    node.semantic_masks = self.segment_image(node_id)
                    # print(f"[SegmentationWorker] Node {node_id} segmented.")

                    # Check if keypoints already exist
                    if node.keypoints is not None:
                        self.feature_collection_queue.put(node_id)
                        # print(f"[SegmentationWorker] Node {node_id} pushed to feature collection queue.")

            self.queue.task_done()

    def segment_image(self, node_id):
        # Return precomputed mask if available
        return self.precomputed_masks.get(node_id, {})
    
class FeatureExtractionWorker(threading.Thread):
    def __init__(self, queue, pose_graph, lock, feature_collection_queue, device=None):
        super().__init__(daemon=True)
        self.queue = queue
        self.pose_graph = pose_graph
        self.lock = lock
        self.feature_collection_queue = feature_collection_queue
        self._device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = SuperPoint(max_num_keypoints=1024).eval().to(self._device)

    def run(self):
        while True:
            node_id = self.queue.get()
            with self.lock:
                node = self.pose_graph.nodes[node_id]
                tensor_img = numpy_image_to_torch(node.rgb_image.copy()).to(self._device)
            output = self.detector.extract(tensor_img)
            with self.lock:
                node.features = output
                node.keypoints = output['keypoints'].squeeze(0).cpu().numpy()
                node.descriptors = output['descriptors'].squeeze(0).cpu().numpy()
                # print(f"[FeatureExtractionWorker] Node {node_id} features extracted.")

                # Check if semantic masks are already available
                if node.semantic_masks:
                    self.feature_collection_queue.put(node_id)
                    # print(f"[FeatureExtractionWorker] Node {node_id} pushed to feature collection queue.")

            self.queue.task_done()

class EdgeCreator(threading.Thread):
    def __init__(self, pose_graph, lock, check_interval=1.0, window_size=5):
        super().__init__(daemon=True)
        self.pose_graph = pose_graph
        self.lock = lock
        self.check_interval = check_interval
        self.window_size = window_size
        # Configuration for SuperGlue
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def run(self):
        while True:
            time.sleep(self.check_interval)
            with self.lock:
                node_ids = sorted(self.pose_graph.nodes.keys())
                for idx_current in range(len(node_ids)):
                    id_current = node_ids[idx_current]
                    node_current = self.pose_graph.nodes[id_current]

                    if node_current.keypoints is None or node_current.descriptors is None:
                        continue

                    # Search up to `window_size` nodes behind
                    for offset in range(1, self.window_size + 1):
                        idx_prev = idx_current - offset
                        if idx_prev < 0:
                            break
                        id_prev = node_ids[idx_prev]
                        node_prev = self.pose_graph.nodes[id_prev]

                        if node_prev.keypoints is None or node_prev.descriptors is None:
                            continue

                        if self.edge_exists(id_current, id_prev):
                            continue

                        t_ij, R_ij = self.match_and_estimate(node_current, node_prev)
                        if t_ij is not None and R_ij is not None:
                            self.pose_graph.add_edge(id_current, id_prev, t_ij, R_ij)
                            print(f"[EdgeWorker] Edge added between Node {id_current} and Node {id_prev}")

    def edge_exists(self, i, j):
        return any((e.i == i and e.j == j) or (e.i == j and e.j == i) for e in self.pose_graph.edges)
    
    def match_and_estimate(self, node1, node2):
        MIN_MATCH_COUNT = 200 # Minimum number of matches to consider a valid pose estimation
        with torch.no_grad():
            pred = self.matcher({"image0": node1.features, "image1": node2.features})
        matches = pred["matches"][0].cpu().numpy()
        if len(matches) < MIN_MATCH_COUNT:
            return None, None
        feats0 = node1.keypoints.copy()
        feats1 = node2.keypoints.copy()
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

            # Process feature collections locally, not storing in node
            collections_for_node = []

            for mask_id, mask in node.semantic_masks.items():
                valid_mask = (keypoints[:, 1] < mask.shape[0]) & (keypoints[:, 0] < mask.shape[1])
                valid_points = keypoints[valid_mask]
                valid_desc = descriptors[valid_mask]

                mask_values = mask[valid_points[:, 1], valid_points[:, 0]]
                keep = mask_values == 1

                if np.sum(keep) < 10:
                    continue

                collection = valid_desc[keep].copy()
                collections_for_node.append((mask_id, collection))

            print(f"[FeatureCollectionWorker] Node {node_id} - {len(collections_for_node)} collections computed.")

            with self.similarity_matrix_lock:
                for mask_id, new_collection in collections_for_node:
                    new_row = []

                    for existing in self.similarity_matrix["collections"]:
                        P, C = compute_sinkhorn(new_collection, existing)
                        score = compute_low_cost_mass(P, C)
                        new_row.append(score)

                    for i in range(len(self.similarity_matrix["matrix"])):
                        self.similarity_matrix["matrix"][i].append(new_row[i])

                    new_row.append(1.0)
                    self.similarity_matrix["matrix"].append(new_row)

                    self.similarity_matrix["collections"].append(new_collection)
                    self.similarity_matrix["index_mapping"].append((node_id, mask_id))

                    print(f"[FeatureCollectionWorker] Added (Node {node_id}, Mask {mask_id}) to similarity matrix.")

            self.queue.task_done()



def main():
    pose_graph = PoseGraph()
    lock = threading.Lock()

    with open("semantic_map_mechatronics4.pkl", "rb") as f:
        semantic_map = pickle.load(f)

    print("Semantic map loaded.")
    if not hasattr(semantic_map,"refinde_prediction_masks"):
        print("Refining semantic predictions")
        semantic_map.refine_semantic_predictions()
    precomputed_masks = semantic_map.refined_prediction_masks
    precomputed_images = semantic_map.rgb_observations

    # Queues
    seg_queue = Queue()
    feat_queue = Queue()
    feature_collection_queue = Queue()

    # Shared similarity matrix and its lock
    similarity_matrix = {
        "matrix": [],            # 2D list (will become 2D numpy array if converted)
        "collections": [],       # List of feature collections
        "index_mapping": []      # List of (node_id, mask_id) tuples
    }
    similarity_matrix_lock = threading.Lock()

    # Start worker threads
    seg_worker = SemanticSegmentationWorker(
        queue=seg_queue,
        pose_graph=pose_graph,
        lock=lock,
        precomputed_masks=precomputed_masks,
        feature_collection_queue=feature_collection_queue
    )

    feat_worker = FeatureExtractionWorker(
        queue=feat_queue,
        pose_graph=pose_graph,
        lock=lock,
        feature_collection_queue=feature_collection_queue
    )

    edge_worker = EdgeCreator(
        pose_graph=pose_graph,
        lock=lock,
        window_size=5  # Optional: change the number of past nodes to connect
    )

    fc_worker = FeatureCollectionWorker(
        queue=feature_collection_queue,
        pose_graph=pose_graph,
        lock=lock,
        similarity_matrix_lock=similarity_matrix_lock,
        similarity_matrix=similarity_matrix
    )

    seg_worker.start()
    feat_worker.start()
    edge_worker.start()
    fc_worker.start()

    print("Workers started.")
    id_counter = 0

    try:
        for image in precomputed_images.values():
            with lock:
                pose_graph.add_node(id_counter, rgb_image=image)
            seg_queue.put(id_counter)
            feat_queue.put(id_counter)
            # print(f"[Main] Registered Node {id_counter}")
            id_counter += 1
            time.sleep(0.1)

        print("[Main] Waiting for segmentation and feature extraction to complete...")
 
        # Wait until all queues are empty
        seg_queue.join()
        feat_queue.join()
        feature_collection_queue.join()

        # while not all_nodes_have_edges(pose_graph):
        #     time.sleep(0.5)

        with lock:
            print(f"There are {len(pose_graph.nodes)} nodes in the graph.")
            print(f"There are {len(pose_graph.edges)} edges in the graph.")
        
        sim_mat_np = np.array(similarity_matrix["matrix"])
        print(f"The total number of feature clusters is {sim_mat_np.shape[0]}")
        index_mapping = similarity_matrix["index_mapping"]

        matched_pairs = []
        for i in range(len(sim_mat_np)):
            row = sim_mat_np[i, :]
            high_matches = detect_high_anomalies_z_score(row, threshold=4.0)
            for j in high_matches:
                matched_pairs.append((index_mapping[i], index_mapping[j]))
        print(f"[PostProcessing] {len(matched_pairs)} matched feature collection pairs.")

        raw_subs, part_subs, adj_mat, origins = extract_subgraphs(edges_list=matched_pairs,
        split_threshold=30,
        min_size=3
        )
        
        print(f"[PostProcessing] Found {len(raw_subs)} raw subgraphs.")
        print(f"[PostProcessing] Partitioned into {len(part_subs)} final subgraphs.")


        # save_clustered_masks(raw_subs, precomputed_images, precomputed_masks)

        print("Starting pose graph optimization...")
        initialize_2d_poses(pose_graph)
        optimize_pose_graph(pose_graph)
        print("Pose graph optimization complete.")
        avg_dist = average_edge_distance(pose_graph)
        print(f"[Main] Average distance between connected node pairs: {avg_dist:.2f} meters")
        draw_pose_graph(pose_graph, title="Optimized Pose Graph")


        # Search & add neighbor edges
        cands = find_candidate_neighbors(pose_graph, max_dist=avg_dist)
        print(f"[Main] Found {len(cands)} neighbor candidates.")
        add_new_edges(pose_graph, cands, edge_worker, lock)

        # Second optimization
        optimize_pose_graph(pose_graph)
        print("Second optimization complete. Final stats:")
        print(f"Nodes: {len(pose_graph.nodes)}, Edges: {len(pose_graph.edges)}")
        draw_pose_graph(pose_graph, title="Final Pose Graph")

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
if __name__ == "__main__":
    main()