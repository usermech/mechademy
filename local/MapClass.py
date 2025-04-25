import numpy as np
import gzip
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class SemanticMap():
    def __init__(self):

        # top_down_map, numpy array 3 boyutlu [height, weight, 3] şeklinde. Haritanın üstten görüntüsü.
        self.top_down_map = None

        # free_space_mask, shape'i [height, weight] şeklinde. Haritada boş kısımları verir
        self.free_space_mask = None

        # poses, shape'i [num_poses, 3], kameranın konumunu verir.  
        self.poses = None
        
        # instance_id_to_name, sınıfın bir instancesinin hafızadaki adresini tutar. Dict'tir
        self.instance_id_to_name = {}

        # rgb_observations, keys, observations index'i RGB olarak saklar. Dict'tir
        self.rgb_observations = {}

        # semantic_observations, üsttekinin semantic versiyonu. Dict'tir
        self.semantic_observations = {}

        # # semantic_prediction_masks, semantik tahmin maskeleri saklar, şekil: [height, width]
        self.semantic_prediction_masks = None

        # semantic_prediction_instance_ids, instance id'lerini tutar
        self.semantic_prediction_instance_ids = None

        # keypoints, anahtar noktaların koordinatlarını tutar
        self.keypoints = None

        # descriptors, anahtar noktalarla ilişkili tanımlayıcılardır.
        self.descriptors = None

        # masks, Farklı semantic/instance sınıflarına ait binary maskeleri tutabilir. Dict'tir
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
        if self.top_down_map.ndim == 2:
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

"""
    os.chdir("C:/Users/ofsaa/Desktop/mechacademy/mechademy/local")

    semantic_map = SemanticMap()

    top_down_map, free_space_mask = semantic_map.load_map("top_down_map_last.npy")
    print("Harita yüklendi:", top_down_map.shape, free_space_mask.shape)


    print(semantic_map.set_instance_id_to_name)  
    print(semantic_map.set_instance_id_to_name({1: "wall", 2: "floor", 3: "chair"}))  
    print(semantic_map.get_semantic_class_names())
"""
