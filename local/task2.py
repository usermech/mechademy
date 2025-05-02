import MapClass
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2

if __name__ == "__main__":
    map_path = r"C:\Users\Onur Can\Desktop\mechademy\mechademy\semantic_map_00800-TEEsavR23oF.pkl"
    with open(map_path, 'rb') as f:
        semantic_map = pickle.load(f)
    semantic_map.refine_semantic_predictions()

    try:
        img_id = 0
        while True:
            try:
                mask_id = 0
                while True:
                    most_common = semantic_map.match_predicted_object_with_gt(img_id,mask_id)
                    int_over_un = semantic_map.match_predicted_object_with_gt_IoU(img_id,mask_id)
                    if most_common != int_over_un:
                        print(f"For image {im}, two alternatives differ\nMost common {most_common}\nInter over union {int_over_un}\n")

                        rgb_img = semantic_map.rgb_observations[img_id].copy()
                        mask = semantic_map.refined_prediction_masks[img_id][mask_id]
                        kernel = np.ones((3, 3), np.uint8)
                        dilated_mask = cv2.dilate(mask,kernel,iterations=1)
                        edges = dilated_mask - mask
                        rgb_img[edges==1] = [255,0,0]
                        
                        plt.figure(figsize=(6, 12))
                        plt.imshow()
                    label_id = label_id + 1
            except KeyError:
                pass
            im = im + 1
    except KeyError:
        pass
