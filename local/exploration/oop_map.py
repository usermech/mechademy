import numpy as np
from scipy.optimize import least_squares


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

    def add_edge(self, i, j, t_ij, R_ij):
        self.edges.append(Edge(i, j, t_ij, R_ij))
        self.add_node(i)
        self.add_node(j)

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
    


