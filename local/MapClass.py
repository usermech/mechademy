import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle
import cv2

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

        self.keypoints = None
        self.descriptors = None

        self.masks = {}
        
    
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

    def refine_semantic_predictions(self,area_threshold = 100,padding=False):
        """
        Refines the semantic prediction masks by applying morphological operations.
        Also accounts for the 360 degree continuity.
        Arguments:
            area_threshold: Minimum area of the mask to be considered valid.
            padding: Whether to apply padding to the masks.
        Returns:
            refined_masks: A dictionary of refined semantic prediction masks.
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
                # Add padding on the edges if specified
                if padding:
                    binary_mask = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=1)

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
        return self.refined_prediction_masks