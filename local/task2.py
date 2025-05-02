import MapClass
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __name__ == "__main__":
    map_path = r"C:\Users\Onur Can\Desktop\mechademy\mechademy\semantic_map_00800-TEEsavR23oF.pkl"
    with open(map_path, 'rb') as f:
        semantic_map = pickle.load(f)
    semantic_map.refine_semantic_predictions()

    #print(type(semantic_map.rgb_observations))
    #print(type(semantic_map.semantic_observations))
    #print(type(semantic_map.semantic_prediction_masks))
    #print(semantic_map.instance_id_to_name)
    #for gt_mask_id in semantic_map.instance_id_to_name:
    #    print(gt_mask_id)

    #semantic_map.display_rgb_observation(1)
    #semantic_map.display_semantic_observation(1)
    
    #semantic_obs = semantic_map.semantic_prediction_masks.get(1, None)
    #num_classes = len(semantic_map.get_semantic_class_names())
    #cmap = ListedColormap(np.random.rand(num_classes, 3))
    #plt.figure(figsize=(6, 12))
    #plt.imshow(semantic_obs, cmap=cmap)
    #plt.axis('off')
    #plt.show()


    #gt_mask = semantic_map.semantic_observations[1].copy()
    #gt_object_mask = gt_mask == 767
    #plt.figure(figsize=(6, 12))
    #plt.imshow(gt_object_mask)
    #plt.axis('off')
    #plt.show()
    #print(gt_object_mask.shape)
    #print(gt_object_mask[128][30])
    

    im = 0
    try:
        while True:
            try:
                label_id = 0
                while True:
                    most_common = semantic_map.match_predicted_object_with_gt(im,label_id)
                    int_over_un = semantic_map.match_predicted_object_with_gt_IoU(im,label_id)
                    if most_common != int_over_un:
                        print(f"For image {im}, two alternatives differ\nMost common ({most_common})\nInter over union ({int_over_un})")
                    label_id = label_id + 1
            except KeyError:
                pass
            im = im + 1
    except KeyError:
        pass
