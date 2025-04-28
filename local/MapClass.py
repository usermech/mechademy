import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle
import cv2
from collections import Counter
import networkx as nx
from lightglue import LightGlue
import torch

class SemanticMap():
    def __init__(self):

        # top_down_map is a numpy array with shape (height, width, 3) where each pixel represents the occupancy of that cell for the environment
        self.top_down_map = None
        self.free_space_mask = None
        self.poses = None
        
        # self.stuff_classes = None    # Not used for now
        self.instance_id_to_name = {}
        self.rgb_observations = {}
        self.semantic_observations = {}
        self.semantic_prediction_masks = None
        self.semantic_prediction_instance_ids = None
        self.gt_pred_correspondences = {}

        self.keypoints = None
        self.descriptors = None
        self.keypoint_full_outputs = None

        self.G = nx.Graph()
        self.device = 'cuda'
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def load_map(self, map_path):
        """
        Loads the top-down map from a given path and creates a free space mask.
        The map is expected to be a 2D numpy array

        Arguments:
            map_path: Path to the map file (string).
        Returns:
            top_down_map: A numpy array representing the top-down map. [height, width, 3]
            free_space_mask: A boolean mask indicating free space in the map. [height, width]
        """
        
        self.top_down_map = np.load(map_path).astype(int)
        recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]])
        self.top_down_map = recolor_map[self.top_down_map]
        self.free_space_mask = np.all(self.top_down_map == [128, 128, 128], axis=-1)
        return self.top_down_map, self.free_space_mask
    
    def set_observations(self, idx, sample):
        """Saves the RGB and semantic observations for a given index."""
        self.rgb_observations[idx] = sample["rgba"][:, :, :3]  # Assuming rgba is a 4-channel image we want the first 3 channels
        self.semantic_observations[idx] = sample["semantic"]

    def set_instance_id_to_name(self, instance_id_to_name):
        """Saves the mapping of instance IDs to names."""
        self.instance_id_to_name = instance_id_to_name

    def set_poses(self, poses):
        """Saves the poses."""
        self.poses = poses
    
    def get_rgb_observation(self, idx):
        """Returns the RGB observation for a given index."""
        return self.rgb_observations.get(idx, None)
    
    def get_semantic_observation(self, idx):
        """Returns the semantic observation for a given index."""
        return self.semantic_observations.get(idx, None)
    
    def get_semantic_class_names(self):
        """Returns a list of english class names in the scene(s). E.g. ['wall', 'ceiling', 'chair']"""
        class_names = list(set(self.instance_id_to_name.values()))
        return class_names
    
    def display_rgb_observation(self, idx):
        """Displays the RGB observation for a given index."""
        img = self.rgb_observations.get(idx, None)
        if img is not None:
            plt.figure(figsize=(6, 12))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            print(f"No RGB observation found for index {idx}.")

    def display_semantic_observation(self, idx):
        """Displays the semantic observation for a given index."""
        semantic_obs = self.semantic_observations.get(idx, None)
        if semantic_obs is not None:
            # for each semantic label, assign a color
            num_classes = len(self.get_semantic_class_names())
            cmap = ListedColormap(np.random.rand(num_classes, 3))
            plt.figure(figsize=(6, 12))
            plt.imshow(semantic_obs, cmap=cmap)
            plt.axis('off')
            plt.show()
        else:   
            print(f"No semantic observation found for index {idx}.")
    
    def refine_semantic_labels(self,remove_list = ["wall","ceiling","floor","window"]):
        self.refined_instance_id_to_name = {}   # Create an empty dictionary for filtering
        for obj_label, obj_class in self.instance_id_to_name.items():
            if obj_class in remove_list:
                continue    # Skip if the object is in remove_list
            self.refined_instance_id_to_name[obj_label] = obj_class     # If not add to the new dictionary.

    def refine_semantic_predictions(self,area_threshold = 100):
        """
        Refines the semantic prediction masks by applying morphological operations.
        Also accounts for the 360 degree continuity.
        Arguments:
            area_threshold: Minimum area of the mask to be considered valid.
        Returns:
            None
            Stores the data in self.refined_prediction_masks
        """
        # Initialize the refined masks dictionary
        self.refined_prediction_masks = {}
        height, width = self.semantic_prediction_masks[0].numpy().copy().shape
        # Iterate through the semantic prediction masks
        for idx, mask in self.semantic_prediction_masks.items():
            mask = mask.numpy().copy()
            label_idx = 0
            for seg_info in self.semantic_prediction_instance_ids[idx]:
                label = seg_info["id"]
                category = seg_info["category_id"]
                # Check if the category is in the stuff classes to be excluded
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
                    if idx not in self.refined_prediction_masks:
                        self.refined_prediction_masks[idx] = {}
                    # Store the refined mask in the dictionary
                    self.refined_prediction_masks[idx][label_idx] = refined_mask
                    label_idx += 1
    
    def get_valid_keypoints(self,img_id,mask_id):
        keypoints = self.keypoints[img_id].copy()
        keypoints = np.round(keypoints).astype(int)
        mask = self.refined_prediction_masks[img_id][mask_id]
        valid = mask[keypoints[:, 1], keypoints[:, 0]] == 1 
        return keypoints[valid]
    
    def match_predicted_object_with_gt(self,img_id,mask_id):
        """
        Match a predicted object in a binary mask to its corresponding
        unique label in the ground truth segmentation mask. Uses most overlapping
        label by pixel count.

        Arguments:
        img_id: Index of the observation for the predicted mask.
        mask_id: Index of the predicted mask in the observation scene.
        """
        pred_mask = self.refined_prediction_masks[img_id][mask_id].copy()  # A binary mask where 1 indicates predicted pixels of a single object, 0 elsewhere.
        gt_mask = self.semantic_observations[img_id].copy()    # A 2D array where each pixel has a unique integer label
        overlap_gt_labels = gt_mask[pred_mask == 1]     # Only consider the ground truth pixels that the predicted mask covers
        label_counts = Counter(overlap_gt_labels)

        matched_label = label_counts.most_common(1)[0][0]   # The most common label is likely to be the correct label.
        # Create binary mask for the matched GT label
        gt_object_mask = (gt_mask == matched_label)

        # Intersection and Union
        intersection = np.logical_and(pred_mask, gt_object_mask).sum()
        union = np.logical_or(pred_mask, gt_object_mask).sum()

        iou = intersection / union if union > 0 else 0
        
        return matched_label, iou
    
    def match_predicted_object_with_gt_IoU(self,img_id,mask_id):
        """
        Match a predicted object in a binary mask to its corresponding
        unique label in the ground truth segmentation mask. Use IoU over 
        """
        # YOUR CODE WILL BE WRITTEN HERE
        return matched_label, iou
    
    def compute_all_gt_pred_correspondences(self):
        """
        Iterates through all predicted masks in self.refined_prediction_masks
        and computes the ground truth correspondence for each using the 
        match_predicted_object_with_gt method.

        Stores the result in self.gt_pred_correspondences in the form:
            self.gt_pred_correspondences[img_id][mask_id] = (matched_label, iou)
        """
        self.gt_pred_correspondences = {}

        for img_id, masks in self.refined_prediction_masks.items():
            self.gt_pred_correspondences[img_id] = {}
            for mask_id in range(len(masks)):
                matched_label, iou = self.match_predicted_object_with_gt(img_id, mask_id)
                self.gt_pred_correspondences[img_id][mask_id] = (matched_label, iou)

    def group_by_matched_label(self):
        """
        Groups all (img_id, mask_id) pairs in self.gt_pred_correspondences by their matched_label.

        Returns:
            A dictionary where keys are matched_label values, and values are lists of (img_id, mask_id).
            Example:
                {
                    12: [(0, 1), (3, 0)],
                    7: [(2, 2)],
                    ...
                }
        """
        label_to_predictions = {}

        for img_id, masks in self.gt_pred_correspondences.items():
            for mask_id, (matched_label, _) in masks.items():
                if matched_label not in label_to_predictions:
                    label_to_predictions[matched_label] = []
                label_to_predictions[matched_label].append((img_id, mask_id))

        return label_to_predictions
    
    def compare_gt_pred_correspondences(self, other_correspondences):
        """
        Compares the current self.gt_pred_correspondences with another set of correspondences.

        Args:
            other_correspondences (dict): A nested dictionary of the same structure:
                other_correspondences[img_id][mask_id] = (matched_label, iou)

        Returns:
            List of (img_id, mask_id) tuples where the correspondences differ.
        """
        conflicts = []

        for img_id in self.gt_pred_correspondences:
            if img_id not in other_correspondences:
                continue
            for mask_id in self.gt_pred_correspondences[img_id]:
                if mask_id not in other_correspondences[img_id]:
                    continue

                current_match = self.gt_pred_correspondences[img_id][mask_id]
                other_match = other_correspondences[img_id][mask_id]

                if current_match != other_match:
                    conflicts.append((img_id, mask_id))

        return conflicts
    
    def match_keypoints(self,feats0,feats1):
        """
        Matches keypoints between images using LightGlue.
        """
        feats0 = feats0.copy()
        feats1 =feats1.copy()

        feats0['keypoints'] = torch.tensor(feats0['keypoints']).to(self.device).unsqueeze(0)
        feats0['descriptors'] = torch.tensor(feats0['descriptors']).to(self.device).unsqueeze(0)
        feats0['keypoint_scores'] = torch.tensor(feats0['keypoint_scores']).to(self.device).unsqueeze(0)
        feats0['image_size'] = torch.tensor(feats0['image_size']).to(self.device).unsqueeze(0)

        feats1['keypoints'] = torch.tensor(feats1['keypoints']).to(self.device).unsqueeze(0)
        feats1['descriptors'] = torch.tensor(feats1['descriptors']).to(self.device).unsqueeze(0)
        feats1['keypoint_scores'] = torch.tensor(feats1['keypoint_scores']).to(self.device).unsqueeze(0)
        feats1['image_size'] = torch.tensor(feats1['image_size']).to(self.device).unsqueeze(0)

        matches = self.matcher({"image0": feats0, "image1": feats1})

        return matches
    
    def build_graph(self):
        k=0
        for node,f0 in self.keypoint_full_outputs.items():
            if node not in self.G.nodes:
                self.G.add_node(node)
            for neighbor in range(node+1,len(self.keypoint_full_outputs)):
                if k % 100 == 0:
                    print(f"{2*(k+1)/(len(self.keypoint_full_outputs)*(len(self.keypoint_full_outputs)-1))*100:.2f}%")
                k += 1
                if neighbor not in self.G.nodes:
                    self.G.add_node(neighbor)
                matches01 = self.match_keypoints(f0,self.keypoint_full_outputs[neighbor])
                matches = matches01["matches"][0]
                if len(matches) > 180:
                    self.G.add_edge(node,neighbor,matches=matches)

                
                    
                

    

    



        
       
