
import pickle
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch


class KeypointExtractor:
  
    cpu_device = torch.device("cpu")
    
    def __init__(self, device="cpu"):

        # SuperPoint and LightGlue
        self._device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = SuperPoint(max_num_keypoints=1024).eval().to(self._device)
        self.matcher = LightGlue(features='superpoint').eval().to(self._device)
    
    
    def extract_keypoints_and_descriptors(self, images: dict):
        """
        Extracts keypoints and descriptors from a batch of images.
        
        Args:
        - images (dict): A dictionary with keys as image indicators and values as images in np.array format.
        
        Returns:
        - keypoints_dict (dict): A dictionary with keys as image indicators and values as list of keypoints.
        - descriptors_dict (dict): A dictionary with keys as image indicators and values as descriptors.
        """
        keypoints_dict = {}
        descriptors_dict = {}
        outputs = {}

        # Iterate through the image dictionary
        for key, img in images.items():
            # Convert the image to tensor
            tensor_img = numpy_image_to_torch(img).to(self._device)

            with torch.no_grad():
                # Get keypoints and descriptors from SuperPoint
                output = self.detector.extract(tensor_img)
                # For each key in output perform .cpu() and .numpy()
                # to convert to numpy array
                keypoints = output['keypoints'].squeeze(0).cpu().numpy()
                descriptors = output['descriptors'].squeeze(0).cpu().numpy()
                keypoint_scores = output['keypoint_scores'].squeeze(0).cpu().numpy()
                image_size = output['image_size'].squeeze(0).cpu().numpy()

                outputs[key] = {'keypoints': keypoints, 'keypoint_scores': keypoint_scores, 'descriptors': descriptors, 'image_size': image_size}
                # Store in dictionaries
                keypoints_dict[key] = keypoints
                descriptors_dict[key] = descriptors


        return keypoints_dict, descriptors_dict, outputs
    
    def match_keypoints(self, outputs: dict, image1: int, image2: int):
        """
        Matches keypoints between images using LightGlue.
        
        Args:
        - outputs (dict): A dictionary with keys as image indicators and values as dictionaries containing keypoints, descriptors, etc.

        Returns:
        - matches (list): A list of matches between images.
        """
        feats = outputs[image1].copy()
        feats2 = outputs[image2].copy()
        feats['keypoints'] = torch.tensor(feats['keypoints']).to(self._device).unsqueeze(0)
        feats['descriptors'] = torch.tensor(feats['descriptors']).to(self._device).unsqueeze(0)
        feats['keypoint_scores'] = torch.tensor(feats['keypoint_scores']).to(self._device).unsqueeze(0)
        feats['image_size'] = torch.tensor(feats['image_size']).to(self._device).unsqueeze(0)

        feats2['keypoints'] = torch.tensor(feats2['keypoints']).to(self._device).unsqueeze(0)
        feats2['descriptors'] = torch.tensor(feats2['descriptors']).to(self._device).unsqueeze(0)
        feats2['keypoint_scores'] = torch.tensor(feats2['keypoint_scores']).to(self._device).unsqueeze(0)
        feats2['image_size'] = torch.tensor(feats2['image_size']).to(self._device).unsqueeze(0)

        matches = self.matcher({"image0": feats, "image1": feats2})
        
        return matches

        
if __name__ == "__main__":

    # Initialize the KeypointExtractor
    extractor = KeypointExtractor(device="cuda")
    print("KeypointExtractor initialized.")
    with open("semantic_map_00800-TEEsavR23oF.pkl", "rb") as f:
        semantic_map = pickle.load(f)
    print("Semantic map loaded.")

    print("Extracting keypoints and descriptors...")
    # Extract keypoints and descriptors
    keypoints_dict, descriptors_dict, outputs = extractor.extract_keypoints_and_descriptors(semantic_map.rgb_observations.copy())
   
    # Save the keypoints and descriptors to the semantic map object
    semantic_map.keypoints = keypoints_dict
    semantic_map.descriptors = descriptors_dict
    semantic_map.keypoint_full_outputs = outputs

    # Save the updated semantic map object
    with open("semantic_map_00800-TEEsavR23oF.pkl", "wb") as f:
        pickle.dump(semantic_map, f)
    
