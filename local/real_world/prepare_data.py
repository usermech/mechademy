import cv2
import os 
import sys
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import MapClass
from argparse import ArgumentParser

def prepare_data(image_path, output_path):
    semantic_map = MapClass.SemanticMap()
    print(os.listdir(image_path))
    file_name = os.path.split(output_path)[1]
    for idx, filename in enumerate(os.listdir(image_path)):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image = cv2.imread(os.path.join(image_path, filename))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # convert to numpy array
                image = np.array(image, dtype=np.uint8)
                sample = {"rgba":image,"semantic":None}
                semantic_map.set_observations(idx,sample)
    # Save the semantic map object
    with open(os.path.join(output_path, f'semantic_map_{file_name}.pkl'), 'wb') as f:
        pickle.dump(semantic_map, f)

if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare data for semantic map")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the prepared semantic map')
    
    args = parser.parse_args()
    
    prepare_data(args.image_path, args.output_path)
    print(f"Data prepared and saved to {args.output_path}")
