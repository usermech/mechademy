import numpy as np
import matplotlib.pyplot as plt
import habitat_sim.registry as registry
import argparse
import pickle
from scipy.ndimage import distance_transform_edt
from habitat_sim.utils.data import ImageExtractor, PoseExtractor
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
from MapClass import SemanticMap

def distance_to_closest_obstacle(x, y, distance_map):
    # Get the distance map value at (x, y)
    return distance_map[x, y]

@registry.register_pose_extractor(name="grid_pose_extractor")
class RandomPoseExtractor(PoseExtractor):
    def extract_poses(self, view, fp, param):
        height, width = view.shape
        points = []       
        step_size = 30

        for row in range(20,height-20,step_size):
            for col in range(20, width-20,step_size):
                distance = distance_to_closest_obstacle(row, col, param)
                if self._valid_point(row, col, view) and distance > 20:
                    points.append((row, col))
        poses = []
        for point in points:
            r, c = point
            point_of_interest = (r-1, c)
            pose = (point, point_of_interest, fp)
            poses.append(pose)
        return poses

@registry.register_pose_extractor(name="random_pose_extractor")
class RandomPoseExtractor(PoseExtractor):
    def extract_poses(self, view, fp,param):
        height, width = view.shape
        num_random_points = 20
        points = []

        while len(points) < num_random_points:
            row, col = np.random.randint(1, height-1), np.random.randint(1, width-1)
            distance = distance_to_closest_obstacle(row, col, param)
            if self._valid_point(row, col, view) and distance > 20:
                points.append((row, col))

        poses = []
        for point in points:
            r, c = point
            point_of_interest = (r-1, c)
            pose = (point, point_of_interest, fp)
            poses.append(pose)
        return poses
    
def parse_arguments():
    available_envs = ["00800-TEEsavR23oF", "00802-wcojb4TFT35",
                      "00803-k1cupFYWXJ6", "00808-y9hTuugGdiq",]
    
    parser = argparse.ArgumentParser(
        description="SSH and run script on remote server",
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "--env",
        type=str,
        default="00800-TEEsavR23oF",
        help="Name of the scene.\nAvailable options: " + ", ".join(available_envs)
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="grid",
        choices=["random", "grid"],
        help="Mode of pose extraction.\nAvailable options: random, grid"
    )
    return parser.parse_args()

def convert_to_map_coordinate_system(scene_poses, ref_point, meters_per_pixel):
    # Convert from scene coordinate system back to topdown map coordinate system
    startw, _, starth = ref_point
    reversed_poses = []

    for pose in scene_poses:
        pos, _, _ = pose
        c1, _, r1, = pos

        # Reverse the transformation (subtract and divide by meters_per_pixel)
        new_c1 = int((c1 - startw) / meters_per_pixel)
        new_r1 = int((r1 - starth) / meters_per_pixel)

        # Ensure the new coordinates are divisible by 10 if not round to the nearest 10
        new_r1 = int(round(new_r1 / 10) * 10)
        new_c1 = int(round(new_c1 / 10) * 10)

        # Add the reversed position and the other components back into the list
        reversed_poses.append((new_r1, new_c1))

    return reversed_poses

def load_map(local_map_path):
    map_array = np.load(local_map_path).astype(int)
    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]])
    return recolor_map[map_array]

def main():
    args = parse_arguments()

    # split the env name into two parts from '-' due to naming convention of HM3D dataset
    env_name = args.env.split('-')[1]

    # Configuration
    map_path = "/home/romer/umut/GibsonData/habitat-lab/my_scripts/maps/"
    map_path += args.env + ".npy"
    SCENE_FILEPATH =  f"/home/romer/umut/data/scene_datasets/hm3d/minival/{args.env}/{env_name}.basis.glb"
    top_down_map = load_map(map_path)

    # Make an occupancy map from the top down map where [128, 128, 128] is denoted with 1 and everything else with 0
    occupancy_map = np.zeros(top_down_map.shape[:2], dtype=np.uint8)
    occupancy_map[np.all(top_down_map == [128, 128, 128], axis=-1)] = 1
    
    distance_map = distance_transform_edt(occupancy_map)    # We will use the distance map to find the distance to the closest obstacle


    # Extract the observations from the scene
    if args.mode == 'random':
        extractor = ImageExtractor(SCENE_FILEPATH, pose_extractor_name="random_pose_extractor",output=["rgba","semantic"],shuffle=False,meters_per_pixel=0.01,position=distance_map)
    elif args.mode == 'grid':
        extractor = ImageExtractor(SCENE_FILEPATH, pose_extractor_name="grid_pose_extractor",output=["rgba","semantic"],shuffle=False,meters_per_pixel=0.01,position=distance_map)
    
    print(f'The number of extracted poses is {len(extractor)}')
 
    ref_points = list(map(lambda x: x[2], extractor.tdv_fp_ref_triples))

    reversed_poses = convert_to_map_coordinate_system(extractor.poses, ref_points[0], extractor.meters_per_pixel)

    # Initialize the semantic map object and save the observations
    semantic_map =  SemanticMap()
    semantic_map.load_map(map_path)
    semantic_map.set_instance_id_to_name(extractor.instance_id_to_name)
    semantic_map.set_poses(reversed_poses)

    for idx, sample in enumerate(extractor):
        semantic_map.set_observations(idx, sample)

    # Save semantic map object 
    with open(f"/home/romer/umut/GibsonData/habitat-lab/my_scripts/observations/semantic_map_{args.env}.pkl", 'wb') as f:
        pickle.dump(semantic_map, f)

if __name__ == "__main__":
    main()